"""Shared visual identity overlay for possession and rendering; ATD is immutable."""
import copy
import json
from pathlib import Path
from .association import finite_xy
from .manifests import load_frozen
from .pilot import sha256

SCHEMA = 'tactical-team-identity-v1'


def apply_team_corrections(envelope, observation, pitch):
    """Explicit ATD-prior experiment: change only matched accepted team labels.

    Uncorrected ATD observations retain their original, unverified metadata.
    They are never presented as visually identified people. No visual positions
    or recovered people enter this policy, and no correction persists to a later
    frame without that frame's accepted evidence.
    """
    if observation is None:
        return envelope
    _, display = apply_overlay(envelope, observation, pitch)
    corrected = copy.deepcopy(envelope)
    accepted = {p['original_track_id']: p for p in display
                if p['original_track_id'] is not None and p['team'] in ('H', 'A')
                and p['identity_status'] == 'matched'}
    for point in corrected['players']:
        visual = accepted.get(point['track_id'])
        before = point['team']
        original_track_id = point['track_id']
        if visual:
            point['team'] = visual['team']
            if visual['team'] != before:
                point['track_id'] = f"team-corrected:{original_track_id}:{visual['team']}"
        point.setdefault('extensions', {})['team_source'] = dict(
            method='accepted_visual_correction' if visual else 'provided_atd_metadata_unverified',
            original_team=before, original_track_id=original_track_id,
            visual_identity=visual['track_id'] if visual else None)
    return corrected


def load_overlay(path, meta, video, atd_raw, homography):
    path = Path(path)
    package = json.loads(path.read_text())
    if (package.get('schema') != SCHEMA or package.get('timeline') != meta['timeline']
            or package.get('fps') != meta['fps'] or package.get('n_frames') != meta['n_frames']
            or package.get('status') != 'complete' or package.get('inference_mode') != 'full_frame'):
        raise ValueError('Complete same-timeline full-frame team overlay required')
    for key, asset in [('source_video', video), ('atd_raw', atd_raw), ('homography', homography)]:
        if asset is None or sha256(asset) != package.get(key+'_sha256'):
            raise ValueError('Team overlay source changed: '+key)
    if package.get('supervised_model'):
        from .supervised_uniforms import SupervisedUniforms
        from .four_frames import POLICY as LEGACY_POLICY
        from .spaced_frames import POLICY as SPACED_POLICY, OFFSETS
        model=path.parent/package['supervised_model']
        if sha256(model)!=package.get('supervised_model_sha256') or package.get('uniform_policy') not in (LEGACY_POLICY,SPACED_POLICY) or package.get('uniform_window_frames')!=4:
            raise ValueError('Supervised identity model/policy changed')
        expected=OFFSETS if package['uniform_policy']==SPACED_POLICY else [-3,-2,-1,0]
        if package['uniform_policy']==SPACED_POLICY or 'uniform_offsets' in package:
            if (package.get('fps')!=25 or package.get('uniform_offsets')!=expected
                    or package.get('uniform_window_span_s')!=-expected[0]/25
                    or package.get('uniform_continuity_window_frames')!=1-expected[0]):
                raise ValueError('Supervised sampling/continuity policy changed')
        artifact=json.loads(model.read_text());SupervisedUniforms(artifact)
        if not {'Barcelona','Athletic','Goleiro Barcelona','Goleiro Athletic','Arbitro'}<={r['label'] for r in artifact['examples']}:
            raise ValueError('Incomplete supervised team/role classes')
        if any(r.get('source_video_sha256')!=package['source_video_sha256'] for r in artifact['examples']):
            raise ValueError('Supervised identity model source differs')
    else:
        references = path.parent/package['frozen_references']
        if sha256(references) != package.get('frozen_references_sha256'):
            raise ValueError('Team overlay references changed')
        _, frozen = load_frozen(references)
        if frozen.get('reference_profile') != 'complete-v1':
            raise ValueError('Both own goalkeeper references required for identity overlay')
    observations = path.parent/package['observations']
    if sha256(observations) != package.get('observations_sha256'):
        raise ValueError('Team observations changed')
    grouped = {}
    previous = -1
    for line in observations.read_text().splitlines():
        row = json.loads(line)
        frame = row.get('frame')
        if type(frame) is not int or not previous < frame < meta['n_frames'] or frame < 0:
            raise ValueError('Overlay frame order invalid')
        previous = frame
        grouped[frame] = row
    if sorted(grouped) != list(range(package['start_frame'], package['end_frame_exclusive'])):
        raise ValueError('Overlay must cover its declared window without gaps')
    return grouped, package


def apply_overlay(envelope, observation, pitch):
    """Return detector-safe points plus display observations, including unknowns.

    A supplied overlay covers every current point: unverified metadata is not
    silently used as a fallback. Outside its window, legacy behavior is retained.
    """
    if observation is None:
        return envelope, None
    f = envelope['frame']
    if observation.get('frame') != f:
        raise ValueError('Team overlay frame mismatch')
    rows = observation.get('players', [])
    raw = {p['track_id']: p for p in envelope['players']}
    ids, targets = set(), set()
    display, players = [], []
    for row in rows:
        identity = row.get('identity')
        target = row.get('target_track_id')
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError('Unique visual identities required')
        ids.add(identity)
        if row.get('frame') != f or type(row.get('available_at_frame')) is not int or row['available_at_frame'] < f:
            raise ValueError('Invalid identity observation availability')
        if target is None and row.get('association_status') not in ('recovered','unassigned_person'):
            continue  # ambiguous visual proposals stay diagnostics, not extra athletes
        if target is not None:
            if target in targets or target not in raw:
                raise ValueError('Duplicate or missing identity target')
            targets.add(target)
            p = raw[target]
            if row.get('raw_xy') != [p['x_m'], p['y_m']]:
                raise ValueError('Identity target coordinate mismatch')
            point = copy.deepcopy(p)
        else:
            xy = row.get('pitch_xy')
            if not finite_xy(xy) or not all(0 <= xy[i] <= pitch[i] for i in (0, 1)):
                raise ValueError('Invalid visual recovery position')
            point = dict(frame=f, producer='visual-identity', x_m=xy[0], y_m=xy[1], source='observed', quality_flags=[])
        team, role = row.get('team'), row.get('role')
        if (team, role) not in ((None, 'unknown'), (None, 'referee'), ('H', 'player'), ('A', 'player'), ('H', 'goalkeeper'), ('A', 'goalkeeper')):
            raise ValueError('Invalid overlay team/role')
        available = row['available_at_frame'] <= f and point.get('available_at_frame', f) <= f
        accepted = (available and row.get('association_status') in ('matched', 'recovered')
                    and point['source'] == 'observed' and not point.get('quality_flags'))
        display.append(dict(point, track_id=identity, original_track_id=target, team=team if accepted else None,
                            role=role if accepted else 'unknown', identity_status=row.get('association_status'),
                            identity_reason=row.get('reason'), available_at_frame=max(row['available_at_frame'], point.get('available_at_frame', f))))
        if not accepted or team not in ('H', 'A'):
            continue
        # One producer namespace preserves local identity when a person moves
        # between an ATD-backed observation and a visually recovered position.
        original_producer=point['producer']
        point.update(team=team, track_id=identity, producer='visual-identity', available_at_frame=f,
                     support_start_frame=row.get('support_start_frame', f), support_end_frame=f+1)
        point['extensions'] = {**point.get('extensions', {}), 'team_identity': dict(copy.deepcopy(row), original_producer=original_producer)}
        players.append(point)
    for target, p in raw.items():
        if target not in targets:
            display.append(dict(p, original_track_id=target, team=None, role='unknown', identity_status='unmatched_tracking',
                                identity_reason='no_verified_visual_association'))
    return dict(envelope, players=players), display
