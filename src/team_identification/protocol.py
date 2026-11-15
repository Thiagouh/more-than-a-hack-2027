"""Versioned, image-bound visual truth and execution provenance.

Freezing records a review commitment; it cannot prove what a reviewer has seen.
"""
import importlib.metadata
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .manifests import load_image, pixel_sha256, read_json, write_json
from .pilot import binary_mask, sha256

SCHEMA_VERSION = 2
MODES = ('full_frame', 'verified_regions')
ROLES = ('player', 'goalkeeper', 'referee', 'other', 'unknown')


def runtime_provenance():
    root = Path(__file__).resolve().parents[2]
    versions = {}
    for name in ('numpy', 'pillow', 'torch', 'torchvision'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), packages=versions,
        code_revision=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        code_dirty=bool(subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()),
        code_sha256={str(path.relative_to(root)):sha256(path) for path in (
            root/'src/team_identification/protocol.py',root/'src/team_identification/evaluation.py',
            root/'src/team_identification/pilot.py',root/'src/team_identification/manifests.py',
            root/'src/ball_events/uniforms.py',root/'scripts/prepare_team_evaluation.py',
            root/'scripts/run_team_pilot.py',root/'scripts/evaluate_team_pilot.py')})


def validate_box(box, optional=False):
    if optional and box is None:
        return
    if (not isinstance(box, list) or len(box) != 4
            or any(type(v) not in (int, float) or not np.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
        raise ValueError('Invalid evaluation box')


def in_region(box, polygons):
    """Box-center inclusion in the union of reviewed polygons; boundary included."""
    if box is None:
        return True  # Empty detections remain diagnostics, never silently discarded.
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
    for polygon in polygons:
        inside = False
        for (ax, ay), (bx, by) in zip(polygon, polygon[1:]+polygon[:1]):
            cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax)
            if abs(cross) < 1e-9 and min(ax,bx) <= x <= max(ax,bx) and min(ay,by) <= y <= max(ay,by):
                return True
            if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
                inside = not inside
        if inside:
            return True
    return False


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_truth(spec):
    """Validate explicit v2 semantics, without opening images or consulting predictions."""
    if spec.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('Visual truth v2 required; review legacy roles explicitly')
    if (spec.get('reviewed') is not True or not _text(spec.get('reviewer'))
            or spec.get('annotations_complete') is not True):
        raise ValueError('Complete reviewed visual annotations required')
    if spec.get('evaluation_mode') not in MODES or spec.get('sample_kind') not in ('main', 'challenge'):
        raise ValueError('Explicit evaluation mode and sample kind required')
    if type(spec.get('synthetic',False)) is not bool:
        raise ValueError('synthetic must be boolean')
    exclusions = spec.get('excluded_segments')
    if not isinstance(exclusions, list):
        raise ValueError('Reference/development segment registry required')
    for row in exclusions:
        if (not _hash(row.get('source_sha256')) or not _text(row.get('source_segment_id'))
                or row.get('split') not in ('reference','development')):
            raise ValueError('Invalid excluded segment')
    frames = spec.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('Nonempty evaluation frames required')
    seen, groups, pixels = set(), set(), set()
    for frame in frames:
        ident = frame.get('frame')
        if type(ident) is not int or ident < 0 or ident in seen:
            raise ValueError('Unique nonnegative frame IDs required')
        seen.add(ident)
        for key in ('image_sha256', 'pixel_sha256', 'source_sha256'):
            if not _hash(frame.get(key)):
                raise ValueError('Image and source hashes required')
        if frame['pixel_sha256'] in pixels:
            raise ValueError('Repeated image pixels in evaluation')
        pixels.add(frame['pixel_sha256'])
        if any(not _text(frame.get(k)) for k in ('image', 'source_segment_id', 'group_id', 'preparation')):
            raise ValueError('Image, source segment, sampling group and preparation required')
        if any((r['source_sha256'],r['source_segment_id']) ==
               (frame['source_sha256'],frame['source_segment_id']) for r in exclusions):
            raise ValueError('Reference/development segment in evaluation')
        timestamp = frame.get('timestamp_s')
        if type(timestamp) not in (int,float) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Recording timestamp required')
        if type(frame.get('temporal_origin_known')) is not bool:
            raise ValueError('Temporal origin status required')
        source_time = frame.get('source_time_s')
        if frame['temporal_origin_known']:
            if type(source_time) not in (int, float) or not np.isfinite(source_time) or source_time < 0:
                raise ValueError('Known temporal origin requires source time')
        elif source_time is not None:
            raise ValueError('Unknown temporal origin must not invent match time')
        group = (frame['source_sha256'], frame['group_id'])
        if spec['sample_kind'] == 'main' and group in groups:
            raise ValueError('Main sample permits one frame per sampling group')
        groups.add(group)
        width, height = frame.get('width'), frame.get('height')
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError('Image dimensions required')
        polygons = frame.get('evaluation_region')
        if not isinstance(polygons, list) or not polygons:
            raise ValueError('Reviewed evaluation region required')
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError('Region must contain polygons')
            for point in polygon:
                if (not isinstance(point, list) or len(point) != 2
                        or any(type(v) not in (int,float) or not np.isfinite(v) for v in point)
                        or not (0 <= point[0] <= width and 0 <= point[1] <= height)):
                    raise ValueError('Region point outside image')
            area = sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))
            if abs(area) < 1e-9:
                raise ValueError('Region polygon has zero area')
        people = frame.get('people')
        if not isinstance(people, list):
            raise ValueError('People annotations required')
        labels = set()
        for person in people:
            label = person.get('label')
            if not _text(label) or label in labels:
                raise ValueError('Unique local person labels required')
            labels.add(label)
            validate_box(person.get('box'))
            if person['box'][2] > width or person['box'][3] > height or not in_region(person['box'], polygons):
                raise ValueError('Annotated person outside evaluation region/image')
            role, cls = person.get('role'), person.get('classification')
            if (role not in ROLES or cls not in ('H','A','referee','uncertain')
                    or type(person.get('evaluable')) is not bool
                    or type(person.get('partial')) is not bool
                    or person.get('lighting') not in ('sun','shade','mixed')
                    or person.get('occlusion') not in ('isolated','overlapped')):
                raise ValueError('Invalid visual annotation fields')
            if ((role in ('other','unknown') and (cls != 'uncertain' or person['evaluable']))
                    or (role == 'referee' and cls not in ('referee','uncertain'))
                    or (role in ('player','goalkeeper') and cls not in ('H','A','uncertain'))
                    or (person['evaluable'] and cls == 'uncertain')):
                raise ValueError('Incompatible role, classification or evaluability')
            if spec['evaluation_mode'] == 'verified_regions':
                if (person.get('mask_reviewed') is not True or not _text(person.get('mask'))
                        or not _hash(person.get('mask_sha256'))):
                    raise ValueError('Verified regions require reviewed individual masks')


def validate_assets(spec, base):
    validate_truth(spec)
    base = Path(base)
    for frame in spec['frames']:
        path = base/frame['image']
        image = load_image(path)
        if (sha256(path) != frame['image_sha256'] or pixel_sha256(image) != frame['pixel_sha256']
                or image.size != (frame['width'], frame['height'])):
            raise ValueError('Evaluation image hash/dimensions mismatch')
        if spec['evaluation_mode'] == 'verified_regions':
            for person in frame['people']:
                path = base/person['mask']
                if sha256(path) != person['mask_sha256']:
                    raise ValueError('Verified mask hash mismatch')
                with Image.open(path) as mask:
                    binary = binary_mask(mask, (image.height, image.width))
                if not binary.any():
                    raise ValueError('Verified mask must be nonempty')
                ys,xs = np.where(binary)
                x0,y0,x1,y1 = person['box']
                if xs.min() < x0 or ys.min() < y0 or xs.max()+1 > x1 or ys.max()+1 > y1:
                    raise ValueError('Verified mask escapes annotated person box')


def freeze_truth(draft, out):
    draft, out = Path(draft), Path(out)
    spec = read_json(draft)
    if spec.get('stage') != 'draft':
        raise ValueError('Freeze a reviewed draft into a new package')
    validate_assets(spec, draft.parent)
    out.mkdir(parents=True, exist_ok=False)
    (out/'assets').mkdir()
    for frame in spec['frames']:
        entries = [(frame, 'image', frame['image_sha256'])]
        if spec['evaluation_mode'] == 'verified_regions':
            entries += [(p, 'mask', p['mask_sha256']) for p in frame['people']]
        for row, key, digest in entries:
            source = draft.parent/row[key]
            destination = 'assets/'+digest+source.suffix.lower()
            shutil.copyfile(source, out/destination)
            row[key] = destination
    spec.update(stage='frozen', frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        draft_sha256=sha256(draft), environment=runtime_provenance())
    write_json(out/'frozen-truth.json', spec)


def load_truth(path):
    path = Path(path)
    spec = read_json(path)
    if spec.get('stage') != 'frozen' or not _text(spec.get('frozen_at_utc')):
        raise ValueError('Frozen visual truth package required')
    try:
        timestamp = datetime.fromisoformat(spec['frozen_at_utc'])
        valid_time = timestamp.tzinfo is not None and timestamp <= datetime.now(timezone.utc)
    except (TypeError,ValueError):
        valid_time = False
    if not valid_time:
        raise ValueError('Frozen truth timestamp must be valid and precede execution')
    validate_assets(spec, path.parent)
    return spec


def validate_splits(truth, references):
    """No source segment may cross reference/evaluation boundaries.

    The existing reference run validator additionally checks RGB pixel leakage.
    """
    evaluation = {(f['source_sha256'],f['source_segment_id']) for f in truth['frames']}
    registered = {(r['source_sha256'],r['source_segment_id']) for r in truth['excluded_segments']
                  if r['split']=='reference'}
    for row in references['references']:
        segment = (row.get('source_sha256'),row.get('source_segment_id'))
        if not _hash(segment[0]) or not _text(segment[1]):
            raise ValueError('V2 evaluation requires explicitly reviewed reference source segments')
        if segment in evaluation or segment not in registered:
            raise ValueError('Reference segment leaks or is missing from split registry')


def validate_run_truth(spec, base, truth):
    if (spec.get('schema_version') != SCHEMA_VERSION
            or sha256(Path(base)/spec['frozen_truth']) != spec.get('truth_sha256')
            or spec.get('evaluation_mode') != truth['evaluation_mode']
            or spec.get('sample_kind') != truth['sample_kind']):
        raise ValueError('Run and frozen truth binding mismatch')
    expected = {f['frame']: f for f in truth['frames']}
    if len(spec['frames']) != len(expected):
        raise ValueError('Run and truth frames differ')
    for frame in spec['frames']:
        original = expected.get(frame['frame'])
        if original is None or any(frame.get(k) != original[k] for k in ('image_sha256','pixel_sha256','timestamp_s')):
            raise ValueError('Run image/truth binding mismatch')
        image = load_image(Path(base)/frame['image'])
        if pixel_sha256(image) != original['pixel_sha256']:
            raise ValueError('Run pixel hash mismatch')


def validate_result_binding(truth, truth_path, predictions, provenance):
    if (provenance.get('schema_version') != SCHEMA_VERSION
            or provenance.get('truth_sha256') != sha256(truth_path)
            or provenance.get('evaluation_mode') != truth['evaluation_mode']
            or provenance.get('sample_kind') != truth['sample_kind']
            or provenance.get('synthetic',False) != truth.get('synthetic',False)):
        raise ValueError('Execution and frozen truth binding mismatch')
    if (type(provenance.get('model_executed')) is not bool
            or (truth['evaluation_mode']=='verified_regions' and provenance['model_executed'])
            or (truth['evaluation_mode']=='full_frame' and not truth.get('synthetic',False)
                and not provenance['model_executed'])):
        raise ValueError('Execution evidence does not match evaluation mode')
    try:
        frozen_at = datetime.fromisoformat(truth['frozen_at_utc'])
        started_at = datetime.fromisoformat(provenance['started_at_utc'])
        valid_order = (frozen_at.tzinfo is not None and started_at.tzinfo is not None
                       and started_at >= frozen_at
                       and provenance.get('truth_frozen_at_utc') == truth['frozen_at_utc'])
    except (ValueError,TypeError,KeyError):
        valid_order = False
    if not valid_order:
        raise ValueError('Execution must record frozen truth before predictions')
    expected = {f['frame']: f for f in truth['frames']}
    if len(predictions) != len(expected):
        raise ValueError('Prediction frame count mismatch')
    for frame in predictions:
        original = expected.get(frame.get('frame'))
        if (original is None or frame.get('truth_sha256') != provenance['truth_sha256']
                or frame.get('image_sha256') != original['image_sha256']
                or frame.get('pixel_sha256') != original['pixel_sha256']
                or frame.get('evaluation_mode') != truth['evaluation_mode']):
            raise ValueError('Prediction image/truth binding mismatch')
