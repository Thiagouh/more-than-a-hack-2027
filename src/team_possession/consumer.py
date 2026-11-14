"""Experimental integration policy, not a validated possession model.

Acquisition sets H/A on delivery, matching active loss sets X. Touches/disputes
are context only. Last state is retained between events; ball visibility and
referee decisions require additional inputs before this can be a final model.
"""
from ..ball_events.events import delivery_key, validate_events
from ..ball_events import contract as c

POLICY_VERSION = 'experimental-event-only-v1'
VISIBILITY_POLICY_VERSION = 'experimental-event-visibility-v2'


def possession(events, meta, *, ball_observed=None, missing_ball_grace_frames=5):
    """Optional visibility is one causal bool per frame, known at that frame.

    Keep at most grace consecutive missing observations; on the next frame emit
    X and invalidate the old control. Return of visibility alone never revives it.
    Caller must delay visibility indications that are not yet available. This API
    accepts already-available indications only, not retrospective visibility.
    """
    validate_events(events, meta)
    c.integer(missing_ball_grace_frames, 'missing_ball_grace_frames')
    if ball_observed is not None:
        if not isinstance(ball_observed, (list, tuple)) or len(ball_observed) != meta['n_frames'] or any(type(v) is not bool for v in ball_observed):
            raise c.InputError('Visibility must provide one causal bool per frame')
    ordered = sorted(events, key=delivery_key)
    labels, decisions = [], []
    current, active, watermark, index = 'X', None, (-1, -1), 0
    starts = set(meta.get('continuity_starts', [])) | {0}
    segment = 0
    missing_run, gap_start, invalid_before, expirations = 0, None, 0, 0
    for frame in range(meta['n_frames']):
        if frame in starts:
            current, active, watermark, segment = 'X', None, (-1, -1), frame
            missing_run, gap_start, invalid_before = 0, None, frame
        if ball_observed is not None:
            if ball_observed[frame]:
                missing_run, gap_start = 0, None
            else:
                if missing_run == 0:
                    gap_start = frame
                missing_run += 1
                if missing_run == missing_ball_grace_frames + 1:
                    current, active = 'X', None
                    invalid_before = gap_start
                    expirations += 1
                    decisions.append(dict(frame=frame, event_id=None,
                                          reason='missing_ball_timeout', label=current))
        while index < len(ordered) and ordered[index]['available_at_frame'] <= frame:
            e = ordered[index]
            index += 1
            reason = 'context_only'
            if e['segment_start_frame'] != segment:
                reason = 'outside_current_segment'
            elif e['type'] == 'possible_control_acquisition':
                order = (e['event_frame'], e['confirmed_at_frame'])
                if ball_observed is not None and missing_run > missing_ball_grace_frames:
                    reason = 'ball_unobserved_after_grace'
                elif e['event_frame'] < invalid_before:
                    reason = 'acquisition_before_expired_gap'
                elif order < watermark:
                    reason = 'stale_acquisition'
                else:
                    current, active, watermark = e['team'], e['event_id'], order
                    reason = 'acquisition_delivered'
            elif e['type'] == 'possible_control_loss':
                if e['parent_event_id'] == active:
                    current, active = 'X', None
                    watermark = max(watermark, (e['event_frame'], e['confirmed_at_frame']))
                    reason = 'active_acquisition_lost'
                else:
                    reason = 'superseded_loss'
            decisions.append(dict(frame=frame, event_id=e['event_id'], reason=reason, label=current))
        labels.append(current)
    return dict(policy_version=VISIBILITY_POLICY_VERSION if ball_observed is not None else POLICY_VERSION,
                visibility_supplied=ball_observed is not None,
                missing_ball_grace_frames=missing_ball_grace_frames if ball_observed is not None else None,
                missing_ball_timeouts=expirations, experimental=True, labels=labels,
                decisions=decisions, pending_events=len(ordered)-index,
                retrospective_rewrites=False, dead_ball_inference=False,
                limitation='Experimental held state with optional missing-ball timeout; no referee/dead-ball model')
