"""Streaming, deterministic heuristics. Strengths are ordinal, never probabilities."""
from collections import Counter, deque
import hashlib
import json
import math

from .contract import InputError
from . import contract as c
from .dependencies import Dependencies, combine
from .storage import HintIndex


def distance(a, b):
    return math.hypot(a['x_m'] - b['x_m'], a['y_m'] - b['y_m'])


def present(p):
    return p['source'] != 'missing'


def observed(p):
    # All producer flags are conservative blockers until a flag vocabulary is agreed.
    return p['source'] == 'observed' and not p['quality_flags']


def velocity(a, b, fps):
    return ((b['x_m'] - a['x_m']) * fps, (b['y_m'] - a['y_m']) * fps)


def norm(v):
    return math.hypot(*v)


class Detector:
    """Feed frame envelopes in order, including across arbitrary processing chunks.

    feed returns (feature, newly confirmed events). Consume events only at available_at_frame.
    finish verifies complete coverage; it does not extrapolate unfinished sequences.
    """
    def __init__(self, manifest, config=None, hints=(), track_registry=None):
        self.meta = c.manifest(manifest)
        self.cfg = c.config(config, self.meta['fps'])
        if not isinstance(hints, (list, tuple)):
            raise InputError('hints: expected list or tuple')
        self.hints = [c.hint(h, self.meta) for h in hints]
        if len({h['hint_id'] for h in self.hints}) != len(self.hints):
            raise InputError('Duplicate hint IDs')
        self.hints.sort(key=lambda h: (h['start_frame'], h['hint_id']))
        self.hint_index = HintIndex(self.hints)
        self.history = deque(maxlen=3)
        self.last_frame = -1
        self.track_teams = track_registry if track_registry is not None else {}
        self.team_counts = Counter()
        self.producer_counts = Counter()
        self.player_flag_counts = Counter()
        self.ineligible = Counter()
        self.restarts = 0
        self.candidate = None
        self.active = None
        self.loss = []
        self.disputed = False
        self.counts = Counter()
        self.source_counts = Counter()
        self.event_counts = Counter()
        self.gap_runs = self.gap_length = self.max_gap = 0

    def reset(self):
        self.history.clear()
        self.candidate = self.active = None
        self.loss = []
        self.disputed = False

    def event(self, kind, frame, support, players, team, components, reasons, flags=(), strength=None, parent=None, operation='selection'):
        deps = combine(deps=[s['deps'][operation] for s in support], frame=frame)
        context = {h['hint_id']: dict(h) for h in self.hint_index.at(frame)}
        if parent is not None:
            deps = deps.union(Dependencies.parent(parent))
            context.update({h['hint_id']: h for h in parent['context_hints']})
        # Hints are context, not a rule that changes the football decision.
        # Their interval is not support across periods; availability and IDs are carried.
        for h in context.values():
            deps = Dependencies(deps.start, deps.end, max(deps.available, h['available_at_frame']),
                                deps.points, deps.hints | {h['hint_id']}, deps.events)
        confirmed = support[-1]['frame']
        deps = Dependencies(deps.start, max(deps.end, confirmed+1), max(deps.available, confirmed),
                            deps.points, deps.hints, deps.events)
        row = {
            'contract_version': c.EVENT_VERSION, 'type': kind, 'team': team,
            'track_ids': sorted(p['track_id'] for p in players),
            'event_frame': frame, 'confirmed_at_frame': confirmed,
            'support_start_frame': deps.start, 'support_end_frame': deps.end,
            'available_at_frame': deps.available,
            'lookahead_frames': max(deps.end-1-frame, confirmed-frame),
            'availability_delay_frames': deps.available-frame,
            'strength': strength or ('high' if kind == 'possible_control_acquisition' else 'medium'),
            'components': components, 'reasons': reasons, 'quality_flags': sorted(flags),
            'context_hints': [context[k] for k in sorted(context)],
            'dependencies': deps.as_dict(), 'parent_event_id': parent['event_id'] if parent else None,
            'segment_start_frame': c.segment(self.meta, frame)[0],
        }
        identity = json.dumps([self.meta['dataset_id'], row], sort_keys=True, separators=(',', ':'))
        row['event_id'] = hashlib.sha256(identity.encode()).hexdigest()[:24]
        self.event_counts[kind] += 1
        return row

    def feed(self, envelope):
        c.envelope(envelope)
        f = c.integer(envelope['frame'], 'frame')
        if f != self.last_frame + 1 or f >= self.meta['n_frames']:
            raise InputError('Frames must be contiguous, ordered, unique and within n_frames')
        ball = c.point(envelope['ball'], f, self.meta)
        players = [c.point(p, f, self.meta, player=True) for p in envelope['players']]
        ids = [p['track_id'] for p in players]
        if len(ids) != len(set(ids)):
            raise InputError('Duplicate track ID in frame')
        for p in players:
            old_team = self.track_teams.setdefault(p['track_id'], p['team'])
            if old_team != p['team']:
                raise InputError('Track changes team without a new namespace/ID')
        players.sort(key=lambda p: p['track_id'])
        if f in self.meta.get('continuity_starts', []):
            self.reset()
            self.restarts += 1
        prev = self.history[-1] if self.history else None
        self.last_frame = f
        flags = set(ball['quality_flags'])
        if not observed(ball):
            flags.add('ball_not_observed')
        pmap = {p['track_id']: p for p in players if present(p)}
        prevmap = {p['track_id']: p for p in prev['players'] if present(p)} if prev else {}
        bvel = None
        if prev and observed(ball) and observed(prev['ball']):
            candidate_vel = velocity(prev['ball'], ball, self.meta['fps'])
            if norm(candidate_vel) > self.cfg['ball_jump_mps']:
                flags.add('ball_jump')
            elif prev['ball_ok']:
                bvel = candidate_vel
        ball_ok = observed(ball) and 'ball_jump' not in flags
        duplicate_ids, jump_ids = set(), set()
        positioned = list(pmap.values())
        for i, a in enumerate(positioned):
            for b in positioned[i + 1:]:
                if distance(a, b) <= self.cfg['duplicate_m']:
                    duplicate_ids.update([a['track_id'], b['track_id']])
        pvel, jump_deps, velocity_deps = {}, {}, {}
        for tid, p in pmap.items():
            old = prevmap.get(tid)
            used = [p,old] if old and observed(old) and observed(p) else [p]
            jump_deps[tid] = combine(points=used, frame=f)
            if old and observed(old) and observed(p):
                v = velocity(old, p, self.meta['fps'])
                if norm(v) > self.cfg['player_jump_mps']:
                    jump_ids.add(tid)
                elif tid not in prev['jump_ids']:
                    pvel[tid] = v
                    velocity_deps[tid] = jump_deps[tid].union(prev['jump_deps'][tid])
        if duplicate_ids:
            flags.add('duplicate_tracks')
        if jump_ids:
            flags.add('player_jump')
        candidates = {}
        if present(ball):
            for team in ('H', 'A'):
                pool = [p for p in positioned if p['team'] == team]
                if pool:
                    p = min(pool, key=lambda p: (distance(p, ball), p['track_id']))
                    candidates[team] = (p, distance(p, ball))
        margins = {t: candidates['A' if t == 'H' else 'H'][1] - d
                   for t, (_, d) in candidates.items() if len(candidates) == 2}
        if len(candidates) < 2:
            flags.add('insufficient_team_coverage')
        if bvel is None:
            flags.add('ball_derivative_unavailable')
        feature = {'frame': f, 'time_s': f / self.meta['fps'], 'ball_source': ball['source'],
                   'ball_speed_mps': norm(bvel) if bvel is not None else None,
                   'ball_acceleration_mps2': None, 'delta_speed_mps': None,
                   'direction_change_deg': None, 'relative_speed_mps': None,
                   'proximity_strength': 'none', 'quality_flags': sorted(flags),
                   'support_start_frame': min([ball['support_start_frame']] + [p['support_start_frame'] for p in players]),
                   'support_end_frame': max([ball['support_end_frame']] + [p['support_end_frame'] for p in players]),
                   'available_at_frame': max([ball['available_at_frame']] + [p['available_at_frame'] for p in players])}
        for t in ('H', 'A'):
            p, d = candidates.get(t, (None, None))
            feature.update({f'{t}_track_id': p['track_id'] if p else None,
                            f'{t}_distance_m': d, f'{t}_margin_m': margins.get(t),
                            f'{t}_source': p['source'] if p else 'missing',
                            f'{t}_quality_flags': sorted(set(p['quality_flags']) |
                                ({'duplicate_track'} if p['track_id'] in duplicate_ids else set()) |
                                ({'player_jump'} if p['track_id'] in jump_ids else set())) if p else []})
        if candidates:
            feature['proximity_strength'] = 'medium' if ball_ok else 'low'
        # Selection and duplicate checks examine only current-frame players.
        selection = combine(points=players, ball=[ball], frame=f)
        ball_support = [ball, prev['ball']] if prev and observed(ball) and observed(prev['ball']) else [ball]
        ball_check = combine(ball=ball_support, frame=f)
        # Velocity also depends on the previous endpoint's jump-validity check.
        motion = ball_check.union(prev['deps']['ball_check']) if bvel is not None else ball_check
        control = selection.union(motion)
        selected = min(candidates, key=lambda t: (candidates[t][1], t)) if candidates else None
        if selected:
            winner = candidates[selected][0]['track_id']
            control = control.union(velocity_deps.get(winner, jump_deps[winner]))
            for pp, _ in candidates.values():
                control = control.union(jump_deps[pp['track_id']])
        feature_deps = control  # envelope of all feature columns, not the event dependency policy
        if prev and bvel is not None and prev['bvel'] is not None:
            feature_deps = feature_deps.union(prev['deps']['motion'])
        feature.update(support_start_frame=feature_deps.start, support_end_frame=feature_deps.end,
                       available_at_frame=feature_deps.available)
        state = {'frame': f, 'ball': ball, 'players': players, 'ball_ok': ball_ok,
                 'bvel': bvel, 'jump_ids': jump_ids, 'duplicate_ids': duplicate_ids, 'jump_deps': jump_deps,
                 'candidates': candidates, 'feature': feature,
                 'deps': {'selection': selection, 'motion': motion, 'control': control, 'ball_check': ball_check}}
        events = []
        # A touch is attributed to the middle of three valid observed frames.
        if len(self.history) >= 2 and bvel is not None and prev['bvel'] is not None:
            v0 = prev['bvel']
            delta = abs(norm(bvel) - norm(v0))
            angle = None
            if min(norm(v0), norm(bvel)) >= self.cfg['direction_min_speed_mps']:
                cosine = sum(a*b for a, b in zip(v0, bvel)) / (norm(v0)*norm(bvel))
                angle = math.degrees(math.acos(max(-1, min(1, cosine))))
            feature.update(delta_speed_mps=delta, direction_change_deg=angle,
                           ball_acceleration_mps2=norm(tuple(b-a for a, b in zip(v0, bvel))) * self.meta['fps'])
            if delta >= self.cfg['touch_delta_speed_mps'] or (angle is not None and angle >= self.cfg['touch_angle_deg']):
                near = [p for p, d in prev['candidates'].values()
                        if d <= self.cfg['proximity_m'] and observed(p)
                        and p['track_id'] not in prev['duplicate_ids'] | prev['jump_ids']
                        and p['track_id'] in pmap and observed(pmap[p['track_id']])
                        and p['track_id'] not in jump_ids | duplicate_ids]
                if near:
                    teams = {p['team'] for p in near}
                    touch_deps = prev['deps']['selection'].union(selection, motion, prev['deps']['motion'],
                        *(prev['jump_deps'][pp['track_id']] for pp in near),
                        *(jump_deps[pp['track_id']] for pp in near))
                    touch_state = dict(state, deps={'touch': touch_deps})
                    events.append(self.event('possible_touch', f-1, [touch_state], near,
                                             next(iter(teams)) if len(teams) == 1 else None,
                                             {'delta_speed_mps': delta, 'direction_change_deg': angle},
                                             ['proximity_and_ball_motion_change'], operation='touch'))
        # Dispute can be weak evidence on estimated points, never an imposed winner.
        dispute = (ball['source'] in {'observed', 'estimated', 'interpolated'}
                   and not ball['quality_flags']
                   and all(p['source'] in {'observed', 'estimated', 'interpolated'} and not p['quality_flags'] for p, _ in candidates.values())
                   and len(candidates) == 2 and all(d <= self.cfg['dispute_m'] for _, d in candidates.values()) and abs(margins['H']) <= self.cfg['margin_m'] and 'ball_jump' not in flags and not ({p['track_id'] for p, _ in candidates.values()} & jump_ids))
        if dispute and not self.disputed:
            state['deps']['dispute'] = selection.union(ball_check, *(jump_deps[pp['track_id']] for pp,_ in candidates.values()))
            e = self.event('possible_dispute', f, [state], [p for p, _ in candidates.values()], None,
                           {'distance_H_m': candidates['H'][1], 'distance_A_m': candidates['A'][1]}, ['both_teams_close'], flags,
                           strength='low' if not ball_ok or any(not observed(p) for p, _ in candidates.values()) or duplicate_ids else 'medium', operation='dispute')
            events.append(e)
        self.disputed = dispute
        eligible = None
        if ball_ok and bvel is not None and len(candidates) == 2:
            t = min(candidates, key=lambda t: (candidates[t][1], t))
            p, d = candidates[t]
            tid = p['track_id']
            opponent = candidates['A' if t == 'H' else 'H'][0]
            relative = norm(tuple(b-a for a, b in zip(pvel[tid], bvel))) if tid in pvel else None
            feature['relative_speed_mps'] = relative
            if (observed(p) and observed(opponent) and tid not in duplicate_ids | jump_ids
                    and opponent['track_id'] not in duplicate_ids | jump_ids
                    and d <= self.cfg['control_m'] and margins[t] >= self.cfg['margin_m']
                    and relative is not None and relative <= self.cfg['relative_speed_mps']):
                eligible = p
        if not ball_ok:
            self.candidate = self.active = None
            self.loss = []
        if self.active:
            tid = self.active['player']['track_id']
            p = pmap.get(tid)
            if not p or not observed(p) or tid in duplicate_ids | jump_ids:
                self.active = None
                self.loss = []
            elif distance(p, ball) > self.cfg['proximity_m']:
                state['deps']['loss'] = selection.union(ball_check, jump_deps[tid])
                self.loss.append(state)
                if len(self.loss) == self.cfg['persistence_frames']:
                    events.append(self.event('possible_control_loss', self.loss[0]['frame'], self.loss,
                                             [p], p['team'], {'distance_m': distance(p, ball), 'persistence_frames': len(self.loss)},
                                             ['sustained_separation_after_control_candidate'],
                                             parent=self.active['acquisition'], operation='loss'))
                    self.active = None
                    self.loss = []
            else:
                self.loss = []
        if eligible:
            tid = eligible['track_id']
            if self.candidate and self.candidate['id'] == tid:
                self.candidate['states'].append(state)
            else:
                self.candidate = {'id': tid, 'states': [state]}
            seq = self.candidate['states']
            if len(seq) == self.cfg['persistence_frames'] and (not self.active or self.active['player']['track_id'] != tid):
                acquisition = self.event('possible_control_acquisition', seq[0]['frame'], seq, [eligible], eligible['team'],
                                         {'persistence_frames': len(seq), 'margin_m': margins[eligible['team']],
                                          'relative_speed_mps': feature['relative_speed_mps']},
                                         ['same_track_close', 'opponent_margin', 'compatible_motion'], operation='control')
                events.append(acquisition)
                self.active = {'player': eligible, 'acquisition': acquisition}
                self.loss = []
            # Bound memory while retaining the episode flag via a count above the threshold.
            if len(seq) > self.cfg['persistence_frames']:
                seq.pop(0)
        else:
            self.candidate = None
        for team in ('H', 'A'):
            self.team_counts[f'{team}_frames_with_player'] += any(p['team'] == team and present(p) for p in players)
        for producer in self.meta['producers']:
            self.producer_counts[f'{producer}_frames_with_player'] += any(p['producer'] == producer and present(p) for p in players)
        for p in players:
            self.player_flag_counts.update(f'{p["producer"]}:{flag}' for flag in p['quality_flags'])
            if present(p) and (abs(p['x_m']) > 2*self.meta['pitch_m'][0] or abs(p['y_m']) > 2*self.meta['pitch_m'][1]):
                self.counts['implausible_player_position'] += 1
        if present(ball) and (abs(ball['x_m']) > 2*self.meta['pitch_m'][0] or abs(ball['y_m']) > 2*self.meta['pitch_m'][1]):
            self.counts['implausible_ball_position'] += 1
        if eligible is None:
            self.ineligible['no_control_candidate'] += 1
            self.ineligible.update(flags)
            if len(candidates) == 2:
                for pp, dd in candidates.values():
                    if pp['quality_flags']: self.ineligible['candidate_producer_flags'] += 1
                    if pp['source'] != 'observed': self.ineligible['candidate_not_observed'] += 1
                if min(d for _, d in candidates.values()) > self.cfg['control_m']: self.ineligible['outside_control_radius'] += 1
                if max(margins.values()) < self.cfg['margin_m']: self.ineligible['insufficient_margin'] += 1
                if feature['relative_speed_mps'] is None: self.ineligible['relative_speed_unavailable'] += 1
                elif feature['relative_speed_mps'] > self.cfg['relative_speed_mps']: self.ineligible['relative_speed_excess'] += 1
        self.history.append(state)
        self.source_counts[ball['source']] += 1
        self.counts.update(flags)
        if ball['source'] == 'missing':
            if not self.gap_length:
                self.gap_runs += 1
            self.gap_length += 1
            self.max_gap = max(self.max_gap, self.gap_length)
        else:
            self.gap_length = 0
        return feature, events

    def finish(self):
        if self.last_frame + 1 != self.meta['n_frames']:
            raise InputError('Incomplete input coverage')
        return {'n_frames': self.last_frame + 1, 'ball_sources': dict(sorted(self.source_counts.items())),
                'quality_flag_frames': dict(sorted(self.counts.items())), 'events': dict(sorted(self.event_counts.items())),
                'missing_ball_runs': self.gap_runs, 'max_missing_ball_frames': self.max_gap,
                'hints': len(self.hints), 'coverage_by_team': dict(sorted(self.team_counts.items())),
                'coverage_by_producer': dict(sorted(self.producer_counts.items())),
                'player_flag_observations': dict(sorted(self.player_flag_counts.items())),
                'ineligible_reasons': dict(sorted(self.ineligible.items())), 'continuity_restarts': self.restarts,
                'persistence': {'mode': self.cfg['persistence_mode'], 'effective_frames': self.cfg['persistence_frames'],
                                'confirmation_delay_s': (self.cfg['persistence_frames']-1)/self.meta['fps'],
                                'seconds': self.cfg.get('persistence_seconds'), 'reference_fps': self.cfg.get('reference_fps')},
                'validation_level': 'heuristic_output_not_match_accuracy',
                'strength_is_calibrated_probability': False}


def detect(manifest, frames, config=None, hints=()):
    """Convenience object API. Detector.feed limits histories; its default ID registry grows with distinct IDs."""
    detector = Detector(manifest, config, hints)
    features, events = [], []
    for frame in frames:
        feature, batch = detector.feed(frame)
        features.append(feature)
        events.extend(batch)
    return features, events, detector.finish()
