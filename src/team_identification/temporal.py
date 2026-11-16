"""Causal visual team evidence. Unknown never means a new team assignment."""
from collections import Counter, deque


class TeamMemory:
    def __init__(self, window_frames=25, min_observations=5, min_support=.8, switch_observations=5, max_hold_frames=25):
        if not (window_frames >= min_observations >= 1 and 0.5 < min_support <= 1
                and switch_observations >= 2 and max_hold_frames >= 0):
            raise ValueError('Invalid team memory configuration')
        self.window, self.minimum, self.support = window_frames, min_observations, min_support
        self.switch, self.hold, self.tracks = switch_observations, max_hold_frames, {}

    def reset(self):
        self.tracks.clear()

    def retain(self, identities):
        self.tracks = {k: v for k, v in self.tracks.items() if k in identities}

    def step(self, identity, frame, team=None, role='unknown', available_at_frame=None):
        available = frame if available_at_frame is None else available_at_frame
        if type(frame) is not int or type(available) is not int or frame < 0 or available < frame:
            raise ValueError('Invalid visual frame availability')
        if (team, role) not in ((None, 'unknown'), (None, 'referee'), ('H', 'player'), ('A', 'player'), ('H', 'goalkeeper'), ('A', 'goalkeeper')):
            raise ValueError('Invalid visual team/role')
        state = self.tracks.setdefault(identity, dict(last=-1, rows=deque(), label=None, confirmed=-1, opposing=None, streak=0))
        if frame <= state['last']:
            raise ValueError('Evidence frames must strictly increase per identity')
        state['last'] = frame
        while state['rows'] and state['rows'][0][0] < frame-self.window+1:
            state['rows'].popleft()
        # Delayed evidence cannot affect an earlier output and is never backfilled.
        label = (team, role) if available == frame and role != 'unknown' else None
        if label:
            state['rows'].append((frame, label))
        votes = Counter(r[1] for r in state['rows'])
        candidate = votes.most_common(1)[0][0] if votes else None
        enough = candidate is not None and votes[candidate] >= self.minimum and votes[candidate]/sum(votes.values()) >= self.support
        if label and label != state['label']:
            state['streak'] = state['streak']+1 if state['opposing'] == label else 1
            state['opposing'] = label
        else:
            state['streak'], state['opposing'] = 0, None
        reason = 'insufficient_visual_evidence'
        if state['label'] is None and enough and label == candidate:
            state['label'], state['confirmed'] = candidate, frame
            reason = 'visual_consensus'
        elif state['label'] is not None:
            if label == state['label']:
                state['confirmed'] = frame
                reason = 'visual_confirmation'
            elif enough and candidate == label and state['streak'] >= self.switch:
                state['label'], state['confirmed'] = candidate, frame
                reason = 'repeated_visual_correction'
            elif frame-state['confirmed'] <= self.hold and state['streak'] < self.switch:
                reason = 'identity_memory'
            else:
                state['label'] = None
                reason = 'expired_or_conflicting_evidence'
        output = state['label']
        return dict(identity=identity, team=output[0] if output else None, role=output[1] if output else 'unknown',
                    reason=reason, evidence_count=len(state['rows']), available_at_frame=frame,
                    support_start_frame=min([state['confirmed']] + [r[0] for r in state['rows']]) if output else frame,
                    support_end_frame=frame+1, calibrated_probability=False)
