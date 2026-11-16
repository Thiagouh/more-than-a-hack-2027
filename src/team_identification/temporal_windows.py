"""Frozen causal sampling policies, independent of the legacy streaming policy."""
import math
import numpy as np
from .supervised_uniforms import LABELS

CONFIGURATIONS = {
    'four_consecutive': [-3, -2, -1, 0],
    'four_spaced': [-15, -10, -5, 0],
    'eight_spaced': [-25, -21, -18, -14, -11, -7, -4, 0],
    'twelve_spaced': [-50, -45, -41, -36, -32, -27, -23, -18, -14, -9, -5, 0],
}


def decide(model, identity, decision_frame, observations, offsets):
    if (not identity or type(decision_frame) is not int or decision_frame < 0
            or not offsets or any(type(f) is not int for f in offsets)
            or offsets != sorted(set(offsets)) or offsets[-1] != 0 or offsets[0] > 0):
        raise ValueError('Explicit identity, causal unique offsets ending at zero required')
    numbers = [r['frame'] for r in observations]
    if len(numbers) != len(set(numbers)):
        raise ValueError('Duplicate observations')
    rows = {r['frame']: r for r in observations}
    requested = [decision_frame + o for o in offsets]
    required = math.ceil(.75 * len(offsets))
    result = dict(identity=identity, frame=decision_frame, label='Incerto', team=None,
        role='unknown', offsets=offsets, requested_frames=requested, frame_ids=[],
        required_support=required, valid_frames=0, supporting_frames=0,
        window_span_s=-offsets[0]/25, available_at_frame=decision_frame,
        distance=None, margin=None, calibrated_probability=False)
    # Check every intervening frame: deliberate spacing is not a tracking gap.
    for f in range(requested[0], decision_frame+1):
        r = rows.get(f)
        reason = ('window_before_source' if f < 0 else
                  'missing_continuity_frame' if r is None else
                  'identity_unverified' if r.get('identity') != identity or r.get('association_verified') is not True else
                  'observation_unavailable' if type(r.get('available_at_frame')) is not int or r['available_at_frame'] < f or r['available_at_frame'] > decision_frame else None)
        if reason:
            return dict(result, reason=reason, failed_frame=f)
    predictions = []
    for f in requested:
        r = rows[f]
        p = model.predict(r['crop']) if r.get('crop') is not None else {}
        result['frame_ids'].append(f)
        predictions.append(dict(frame=f, available_at_frame=r['available_at_frame'], **p))
    result['per_frame'] = predictions
    valid = [r for r in predictions if r.get('class_distances')]
    result['valid_frames'] = len(valid)
    if len(valid) < required:
        return dict(result, reason='insufficient_valid_uniform_frames')
    classes = sorted(valid[0]['class_distances'])
    if len(classes)<2 or any(sorted(r['class_distances']) != classes for r in valid):
        raise ValueError('Class set changed')
    values = np.array([[r['class_distances'][c] for c in classes] for r in valid])
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite class distances')
    scores = dict(zip(classes, map(float, np.median(values, axis=0))))
    order = sorted(classes, key=lambda c: (scores[c], c))
    best = order[0]
    support = sum(min(r['class_distances'], key=lambda c: (r['class_distances'][c], c)) == best for r in valid)
    distance, margin = scores[best], scores[order[1]]-scores[best]
    accepted = support >= required and distance <= model.max_distance and margin >= model.min_margin
    team, role = LABELS[best] if accepted else (None, 'unknown')
    return dict(result, label=best if accepted else 'Incerto', team=team, role=role,
        supporting_frames=support, distance=distance, margin=margin, class_distances=scores,
        reason='temporal_consensus' if accepted else 'conflict_or_margin')


def check_separation(start, end, source_hash, examples, reserved_frames, training_pools, gap=750):
    """Protect the entire longest window, not just sampled frames/endpoints."""
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError('Invalid window bounds')
    if any(start < hi and end >= lo for ranges in training_pools.values() for lo, hi in ranges):
        raise ValueError('Overlap with declared training pool')
    frames = [r['frame'] for r in examples if r['source_video_sha256'] == source_hash]
    frames += list(reserved_frames)
    if any(start-gap < f < end+gap for f in frames):
        raise ValueError('Too close to historical training/test data')
