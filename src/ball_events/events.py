"""Event v2 validation and temporal replay. No possession decisions."""
import heapq

from . import contract as c

EVENT_FIELDS = {'contract_version', 'event_id', 'type', 'team', 'track_ids', 'event_frame',
                'confirmed_at_frame', 'support_start_frame', 'support_end_frame', 'available_at_frame',
                'lookahead_frames', 'availability_delay_frames', 'strength', 'components', 'reasons',
                'quality_flags', 'context_hints', 'dependencies', 'parent_event_id', 'segment_start_frame', 'extensions'}


def strings(value, name):
    c.array(value, name)
    for x in value:
        c.text(x, name)
    if len(set(value)) != len(value):
        raise c.InputError(f'{name}: duplicates')
    return value


def validate_event(e, meta):
    c.fields(e, EVENT_FIELDS, EVENT_FIELDS-{'extensions'}, 'event')
    if e['contract_version'] != c.EVENT_VERSION:
        raise c.InputError('Expected events v2; regenerate legacy artifacts instead of fabricating parents')
    c.text(e['event_id'], 'event_id')
    if not isinstance(e['type'], str) or e['type'] not in c.EVENT_TYPES:
        raise c.InputError('Unknown event type')
    if e['team'] not in (None, 'H', 'A'):
        raise c.InputError('Invalid event team')
    tids = strings(e['track_ids'], 'track_ids')
    if not tids:
        raise c.InputError('Event must identify candidate tracks')
    if e['strength'] not in ('low', 'medium', 'high'):
        raise c.InputError('Invalid event strength')
    c.obj(e['components'], 'components')
    c.json_value(e['components'])
    shapes = {
        'possible_touch': ({'delta_speed_mps','direction_change_deg'}, {'delta_speed_mps'}),
        'possible_control_acquisition': ({'persistence_frames','margin_m','relative_speed_mps'}, {'persistence_frames','margin_m','relative_speed_mps'}),
        'possible_control_loss': ({'distance_m','persistence_frames'}, {'distance_m','persistence_frames'}),
        'possible_dispute': ({'distance_H_m','distance_A_m'}, {'distance_H_m','distance_A_m'}),
    }
    allowed, required = shapes[e['type']]
    c.fields(e['components'], allowed, required, 'event components')
    for key,value in e['components'].items():
        if key=='persistence_frames': c.integer(value,key,2)
        elif key=='direction_change_deg' and value is None: pass
        else:
            c.number(value,'component')
            if value<0 or key=='direction_change_deg' and value>180:
                raise c.InputError('Invalid numeric event component')
    strings(e['reasons'], 'reasons')
    if not e['reasons']: raise c.InputError('Event must explain its evidence')
    c.flags(e['quality_flags'])
    for key in ('event_frame', 'confirmed_at_frame', 'support_start_frame', 'support_end_frame',
                'available_at_frame', 'lookahead_frames', 'availability_delay_frames', 'segment_start_frame'):
        c.integer(e[key], key)
    f, confirmed, available = e['event_frame'], e['confirmed_at_frame'], e['available_at_frame']
    lo, hi = c.segment(meta, f)
    if not (lo <= e['support_start_frame'] <= f <= confirmed < hi
            and confirmed < e['support_end_frame'] <= hi
            and available >= max(confirmed, e['support_end_frame']-1)
            and e['segment_start_frame'] == lo):
        raise c.InputError('Event temporal support outside timeline or segment')
    if e['type'] in ('possible_control_acquisition','possible_control_loss') and e['components']['persistence_frames'] != confirmed-f+1:
        raise c.InputError('Persistence component disagrees with event and confirmation frames')
    if e['type']=='possible_touch' and confirmed != f+1 or e['type']=='possible_dispute' and confirmed != f:
        raise c.InputError('Event confirmation disagrees with event type')
    if e['lookahead_frames'] != max(confirmed-f, e['support_end_frame']-1-f) or e['availability_delay_frames'] != available-f:
        raise c.InputError('Event delay fields are inconsistent')
    d = e['dependencies']
    c.fields(d, {'support_start_frame', 'support_end_frame', 'available_at_frame', 'point_refs', 'hint_ids', 'event_ids'},
             {'support_start_frame', 'support_end_frame', 'available_at_frame', 'point_refs', 'hint_ids', 'event_ids'}, 'dependencies')
    for k in ('support_start_frame', 'support_end_frame', 'available_at_frame'):
        c.integer(d[k], 'dependency frame')
        if d[k] != e[k]:
            raise c.InputError('Dependency envelope disagrees with event')
    for k in ('point_refs', 'hint_ids', 'event_ids'):
        strings(d[k], k)
    for ref in d['point_refs']:
        try:
            identifier,raw_frame=ref.rsplit('@',1)
            producer=identifier.split(':',1)[0]
            point_frame=int(raw_frame)
        except (ValueError,TypeError):
            raise c.InputError('Invalid point dependency reference') from None
        c.integer(point_frame,'dependency point frame')
        if producer not in meta['producers'] or ':' not in identifier or not e['support_start_frame'] <= point_frame < e['support_end_frame']:
            raise c.InputError('Point dependency outside declared support or producer namespace')
    c.array(e['context_hints'], 'context_hints')
    for h in e['context_hints']:
        c.hint(h, meta)
        if h['available_at_frame'] > available:
            raise c.InputError('Hint not available at event delivery')
    if sorted(h['hint_id'] for h in e['context_hints']) != sorted(d['hint_ids']):
        raise c.InputError('Context and dependency hint IDs disagree')
    parent = e['parent_event_id']
    if e['type'] == 'possible_control_loss':
        c.text(parent, 'parent_event_id')
        if e['team'] is None or len(tids) != 1 or d['event_ids'] != [parent]:
            raise c.InputError('Control loss must depend on one acquisition')
    elif parent is not None or d['event_ids']:
        raise c.InputError('Only control loss may depend on an acquisition')
    if e['type'] == 'possible_control_acquisition' and (e['team'] is None or len(tids) != 1):
        raise c.InputError('Acquisition requires one track and team')
    if e['type'] == 'possible_dispute' and (e['team'] is not None or len(tids) < 2):
        raise c.InputError('Dispute cannot select winner')
    return e


def validate_parent(e, parent):
    if parent is None or parent['type'] != 'possible_control_acquisition' or parent['event_id'] != e['parent_event_id']:
        raise c.InputError('Loss predecessor missing or not an acquisition')
    if (e['track_ids'] != parent['track_ids'] or e['team'] != parent['team']
            or e['segment_start_frame'] != parent['segment_start_frame']
            or e['event_frame'] < parent['confirmed_at_frame']
            or e['confirmed_at_frame'] < parent['confirmed_at_frame']
            or e['available_at_frame'] < parent['available_at_frame']
            or e['support_start_frame'] > parent['support_start_frame']
            or e['support_end_frame'] < parent['support_end_frame']):
        raise c.InputError('Loss does not propagate acquisition causality')
    for key in ('point_refs', 'hint_ids'):
        if not set(parent['dependencies'][key]) <= set(e['dependencies'][key]):
            raise c.InputError('Loss omits transitive acquisition support')


def validate_events(events, meta):
    c.manifest(meta)
    c.array(events, 'events')
    by_id = {}
    for e in events:
        validate_event(e, meta)
        if e['event_id'] in by_id:
            raise c.InputError('Duplicate event ID')
        by_id[e['event_id']] = e
    # Every edge is loss -> acquisition, so cycles are impossible after type validation.
    for e in events:
        if e['parent_event_id']:
            if e['parent_event_id'] == e['event_id']:
                raise c.InputError('Event dependency cycle')
            validate_parent(e, by_id.get(e['parent_event_id']))
    return by_id


def delivery_key(e):
    return (e['available_at_frame'], e['confirmed_at_frame'], e['event_frame'],
            1 if e['parent_event_id'] else 0, e['event_id'])


def replay(events, meta, drain=False):
    """Receive in confirmation order, release through a queue at actual availability."""
    validate_events(events, meta)
    received = sorted(events, key=lambda e: (e['confirmed_at_frame'], bool(e['parent_event_id']), e['event_id']))
    heap, delivered, rows = [], set(), []
    stop = meta['n_frames']-1

    def release(now):
        deferred = []
        while heap and heap[0][0] <= now:
            _, _, e = heapq.heappop(heap)
            if e['parent_event_id'] and e['parent_event_id'] not in delivered:
                deferred.append(e)
                continue
            delivered.add(e['event_id'])
            rows.append(dict(event_id=e['event_id'], parent_event_id=e['parent_event_id'],
                             delivered_at_frame=now, event_frame=e['event_frame'],
                             retrospective=now > e['event_frame']))
        for e in deferred:
            heapq.heappush(heap, (e['available_at_frame'], delivery_key(e), e))
        if deferred and any(e['parent_event_id'] in delivered for e in deferred):
            release(now)

    for e in received:
        # Drain all due availability instants between confirmation instants.
        while heap and heap[0][0] < e['confirmed_at_frame']:
            release(heap[0][0])
        heapq.heappush(heap, (e['available_at_frame'], delivery_key(e), e))
        release(e['confirmed_at_frame'])
    while heap and heap[0][0] <= stop:
        release(heap[0][0])
    pending = sorted(e['event_id'] for _, _, e in heap)
    if drain:
        while heap:
            release(heap[0][0])
    return dict(contract_valid=True, received_events=len(events), delivered_within_timeline=len(events)-len(pending),
                pending_at_end=pending, drained=drain, deliveries=rows, delivery_order_valid=True,
                produced_possession_labels=False, real_front4_integration=False)


def consume(events, meta):
    result = replay(events, meta)
    result['timeline_metadata_supplied'] = True
    return result
