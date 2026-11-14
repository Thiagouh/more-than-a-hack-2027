"""Shared, strict input-v1 and event-v2 contract. Errors never include input values."""
import math
from fractions import Fraction


class InputError(ValueError):
    """A diagnostic safe to display without exposing input records."""


VERSION = 'v1'  # input contract remains v1
EVENT_VERSION = 'v2'
MAX_INTEGER = 2**53 - 1
SOURCES = {'observed', 'interpolated', 'estimated', 'missing', 'unknown'}
EVENT_TYPES = {'possible_touch', 'possible_control_acquisition', 'possible_control_loss', 'possible_dispute'}
DEFAULTS = {
    'proximity_m': 2.0, 'control_m': 1.5, 'margin_m': 1.0,
    'dispute_m': 2.0, 'persistence_frames': 3, 'relative_speed_mps': 5.0,
    'touch_delta_speed_mps': 3.0, 'touch_angle_deg': 30.0,
    'direction_min_speed_mps': 2.0, 'ball_jump_mps': 45.0,
    'player_jump_mps': 12.0, 'duplicate_m': 0.3,
}


def obj(value, name):
    if not isinstance(value, dict):
        raise InputError(f'{name}: expected object')
    return value


def array(value, name):
    if not isinstance(value, list):
        raise InputError(f'{name}: expected list')
    return value


def integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_INTEGER:
        raise InputError(f'{name}: expected supported integer >= {minimum}')
    return value


def number(value, name, positive=False):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and abs(value) <= 1e100
    except OverflowError:
        valid = False
    if not valid or (positive and value < 1e-100):
        raise InputError(f'{name}: expected finite supported number' + (' >= 1e-100' if positive else ''))
    return value


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f'{name}: expected non-empty string')
    return value


def json_value(value):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        number(value, 'JSON number')
    elif isinstance(value, list):
        for item in value:
            json_value(item)
    elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            json_value(item)
    else:
        raise InputError('Unsupported JSON extension value')


def fields(value, allowed, required, name):
    obj(value, name)
    if set(value) - set(allowed) or not set(required) <= set(value):
        raise InputError(f'{name}: missing or unknown fields; put custom metadata in extensions')
    if 'extensions' in value:
        obj(value['extensions'], 'extensions')
        json_value(value['extensions'])


def config(overrides=None, fps=None):
    overrides = {} if overrides is None else overrides
    if fps is not None: number(fps, 'fps', True)
    allowed = set(DEFAULTS) | {'persistence_mode', 'reference_fps', 'persistence_seconds'}
    fields(overrides, allowed, (), 'configuration')
    mode = overrides.get('persistence_mode', 'frames')
    if mode not in ('frames', 'duration'):
        raise InputError('persistence_mode: expected frames or duration')
    if mode == 'frames' and 'persistence_seconds' in overrides or mode == 'duration' and ('persistence_frames' in overrides or 'reference_fps' in overrides):
        raise InputError('Persistence modes are mutually exclusive')
    out = {**DEFAULTS, **overrides}
    for key in DEFAULTS:
        if key == 'persistence_frames':
            integer(out[key], key, 2)
        else:
            number(out[key], key, positive=True)
    if out['touch_angle_deg'] > 180 or out['control_m'] > out['proximity_m']:
        raise InputError('Invalid angle or control radius')
    out['persistence_mode'] = mode
    if mode == 'frames':
        reference = number(overrides.get('reference_fps', 25), 'reference_fps', True)
        out['reference_fps'] = reference
        if fps is not None and Fraction(str(fps)) != Fraction(str(reference)):
            raise InputError('Frame profile frequency mismatch: set reference_fps explicitly or use duration mode')
    else:
        seconds = number(overrides.get('persistence_seconds'), 'persistence_seconds', True)
        if fps is not None:
            count = max(2, math.ceil(Fraction(str(seconds)) * Fraction(str(fps))) + 1)
            out['persistence_frames'] = integer(count, 'effective persistence', 2)
    return out


def flags(value):
    array(value, 'quality_flags')
    for item in value:
        text(item, 'quality flag')
    if len(value) != len(set(value)):
        raise InputError('Duplicate quality flags')
    return sorted(value)


def manifest(value):
    fields(value, {'contract_version', 'dataset_id', 'timeline', 'n_frames', 'fps', 'coordinates',
                   'pitch_m', 'teams', 'producers', 'continuity_starts', 'files', 'is_clip', 'origin', 'extensions'},
           {'contract_version', 'dataset_id', 'timeline', 'n_frames', 'fps', 'coordinates', 'pitch_m', 'teams', 'producers'}, 'manifest')
    if value['contract_version'] != VERSION:
        raise InputError('Unsupported input contract_version; expected v1')
    text(value['dataset_id'], 'dataset_id')
    if value['timeline'] not in ('tactical', 'synthetic', 'external'):
        raise InputError('timeline must be tactical, synthetic or external')
    integer(value['n_frames'], 'n_frames', 1)
    number(value['fps'], 'fps', True)
    if value['coordinates'] != 'metres_corner_fixed':
        raise InputError('coordinates must be metres_corner_fixed')
    pitch = array(value['pitch_m'], 'pitch_m')
    if len(pitch) != 2:
        raise InputError('pitch_m must be [length, width]')
    for v in pitch:
        number(v, 'pitch_m', True)
    teams = obj(value['teams'], 'teams')
    if set(teams) != {'H', 'A'}:
        raise InputError('teams must explicitly map H and A')
    for v in teams.values():
        text(v, 'team name')
    producers = obj(value['producers'], 'producers')
    if not producers:
        raise InputError('producers must map IDs to versions')
    for k, v in producers.items():
        text(k, 'producer ID')
        if ':' in k:
            raise InputError('Producer IDs cannot contain colon')
        text(v, 'producer version')
    bounds = array(value.get('continuity_starts', []), 'continuity_starts')
    for v in bounds:
        integer(v, 'continuity start')
        if v >= value['n_frames']:
            raise InputError('continuity start outside timeline')
    if bounds != sorted(set(bounds)):
        raise InputError('continuity starts must be unique and sorted')
    if 'is_clip' in value and not isinstance(value['is_clip'], bool):
        raise InputError('is_clip must be boolean')
    if value.get('is_clip') and 'origin' not in value:
        raise InputError('Clip must declare origin timeline, first frame and fps')
    if 'origin' in value:
        origin = value['origin']
        fields(origin, {'timeline', 'first_frame', 'fps'}, {'timeline', 'first_frame', 'fps'}, 'origin')
        text(origin['timeline'], 'origin timeline')
        integer(origin['first_frame'], 'origin first_frame')
        number(origin['fps'], 'origin fps', True)
    if 'files' in value:
        files = value['files']
        fields(files, {'players', 'ball', 'hints'}, {'players', 'ball'}, 'files')
        for v in files.values():
            text(v, 'input path')
    return value


def segment(meta, frame):
    from bisect import bisect_right
    starts = meta.get('continuity_starts', [])
    i = bisect_right(starts, frame)
    return (starts[i-1] if i else 0, starts[i] if i < len(starts) else meta['n_frames'])


POINT_FIELDS = {'frame', 'source', 'producer', 'quality_flags', 'x_m', 'y_m',
                'support_start_frame', 'support_end_frame', 'available_at_frame', 'extensions'}
POINT_REQUIRED = {'frame', 'source', 'producer', 'quality_flags', 'x_m', 'y_m'}


def point(row, frame, meta, player=False):
    fields(row, POINT_FIELDS | ({'track_id', 'team'} if player else set()),
           POINT_REQUIRED | ({'track_id', 'team'} if player else set()), 'point')
    out = dict(row)
    if integer(out['frame'], 'point frame') != frame:
        raise InputError('Point frame does not match envelope')
    if not isinstance(out['source'], str) or out['source'] not in SOURCES:
        raise InputError('Invalid point source')
    if not isinstance(out['producer'], str) or out['producer'] not in meta['producers']:
        raise InputError('Unknown point producer')
    out['quality_flags'] = flags(out['quality_flags'])
    if out['source'] == 'missing':
        if out['x_m'] is not None or out['y_m'] is not None:
            raise InputError('Missing point must have null coordinates')
    else:
        number(out['x_m'], 'x_m')
        number(out['y_m'], 'y_m')
    start = integer(out.get('support_start_frame', frame), 'support_start_frame')
    end = integer(out.get('support_end_frame', frame+1), 'support_end_frame', 1)
    lo, hi = segment(meta, frame)
    if not lo <= start <= frame < end <= hi:
        raise InputError('Point support outside continuity segment')
    available = integer(out.get('available_at_frame', end-1), 'available_at_frame')
    if available < end-1:
        raise InputError('Point available before end of support')
    out.update(support_start_frame=start, support_end_frame=end, available_at_frame=available)
    if player:
        text(out['track_id'], 'track_id')
        if out['team'] not in ('H', 'A'):
            raise InputError('Player team must be H or A')
        out['track_id'] = f'{out["producer"]}:{out["track_id"]}'
    return out


def envelope(value):
    fields(value, {'frame', 'ball', 'players'}, {'frame', 'ball', 'players'}, 'frame envelope')
    integer(value['frame'], 'frame')
    obj(value['ball'], 'ball')
    array(value['players'], 'players')
    return value


def hint(row, meta):
    fields(row, {'hint_id', 'type', 'producer', 'start_frame', 'end_frame', 'team', 'strength', 'available_at_frame', 'extensions'},
           {'hint_id', 'type', 'producer', 'start_frame', 'end_frame', 'team', 'strength', 'available_at_frame'}, 'hint')
    out = dict(row)
    for key in ('hint_id', 'type', 'producer'):
        text(out[key], key)
    if out['producer'] not in meta['producers']:
        raise InputError('Unknown hint producer')
    start = integer(out['start_frame'], 'hint start')
    end = integer(out['end_frame'], 'hint end', 1)
    if not start < end <= meta['n_frames']:
        raise InputError('Invalid hint interval')
    integer(out['available_at_frame'], 'hint availability', start)
    if out['team'] not in (None, 'H', 'A') or out['strength'] not in ('low', 'medium', 'high'):
        raise InputError('Invalid hint team or ordinal strength')
    return out
