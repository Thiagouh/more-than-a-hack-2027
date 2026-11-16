"""Validated private manifests and explicitly approved frozen reference bundles."""
import json
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from .pilot import (DESCRIPTOR_VERSION, SAM3_REVISION, TeamClassifier, binary_mask,
                    mask_stack, mask_flags, torso_evidence, sha256, reviewed_box_quality)

PARAMETERS = dict(max_distance=.35, min_margin=.10)
LABELS = {('H', 'player'), ('A', 'player'), ('H', 'goalkeeper'),
          ('A', 'goalkeeper'), (None, 'referee')}


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_image(path):
    with Image.open(path) as image:
        if image.mode == 'RGBA' and np.all(np.asarray(image)[:,:,3] == 255):
            return image.convert('RGB')
        if image.mode != 'RGB':
            raise ValueError('Images must be RGB or fully opaque RGBA')
        return image.copy()


def pixel_sha256(image):
    pixels=np.asarray(image)
    return hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()


def label(row):
    if (row.get('team'), row.get('role')) not in LABELS:
        raise ValueError('Invalid team/role')
    if not isinstance(row.get('label'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['label']):
        raise ValueError('Invalid safe annotation label')


def validate_initial(spec, image):
    if spec.get('source') not in ('broadcast', 'broadcast-screen-capture', 'tactical'):
        raise ValueError('Explicit broadcast or tactical pixels required')
    if type(spec.get('reviewed')) is not bool:
        raise ValueError('reviewed must be boolean')
    rows = spec['initial_annotations']
    if not rows or len({r['label'] for r in rows}) != len(rows):
        raise ValueError('Annotation labels must be unique and nonempty')
    for row in rows:
        label(row)
        box = row.get('box')
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in box)):
            raise ValueError('Box must contain four finite coordinates')
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError('Box outside image')
    if {r['team'] for r in rows if r['role'] == 'player'} != {'H', 'A'}:
        raise ValueError('Both outfield teams required')


def freeze(candidate_path, out, *, require_complete=False):
    candidate_path, out = Path(candidate_path), Path(out)
    spec = read_json(candidate_path)
    if spec.get('stage') != 'candidate' or spec.get('sam3_revision') != SAM3_REVISION:
        raise ValueError('Expected pinned SAM candidate manifest')
    if not re.fullmatch('[0-9a-f]{64}',spec.get('checkpoint_sha256','')):
        raise ValueError('Candidate checkpoint provenance required')
    if spec.get('mask_reviewed') is not True or not isinstance(spec.get('reviewer'),str) or not spec['reviewer'].strip():
        raise ValueError('Inspect mask review and explicitly approve it with reviewer')
    from .reference_review import PROFILE, require_complete as validate_complete
    from .goalkeeper_context import CONTEXT_PROFILE
    profile = spec.get('reference_profile', 'legacy-partial')
    if profile not in ('legacy-partial', PROFILE, CONTEXT_PROFILE):
        raise ValueError('Unknown reference profile')
    if require_complete or profile in (PROFILE, CONTEXT_PROFILE):
        validate_complete(spec)
        if profile != CONTEXT_PROFILE:
            profile = PROFILE
    base = candidate_path.parent
    # All candidates participate in overlap exclusion, including rejected references.
    rows = spec['references']
    if not rows or len({r['label'] for r in rows}) != len(rows):
        raise ValueError('Duplicate or empty references')
    images, masks = [], []
    for row in rows:
        label(row)
        image_path, mask_path = base/row['image'], base/row['mask']
        if sha256(image_path) != row['image_sha256'] or sha256(mask_path) != row['mask_sha256']:
            raise ValueError('Candidate image/mask changed after generation')
        images.append(load_image(image_path))
        with Image.open(mask_path) as mask:
            masks.append(binary_mask(mask, np.asarray(images[-1]).shape[:2]))
    # Group by actual image hash, not pathname.
    selected = []
    for i, row in enumerate(rows):
        if row.get('approved') is not True:
            continue
        if row.get('role') == 'goalkeeper' and row.get('goalkeeper_verified') is not True:
            raise ValueError('Goalkeeper references require separate verification')
        group = [j for j, other in enumerate(rows) if other['image_sha256'] == row['image_sha256']]
        stack = mask_stack([masks[j] for j in group], masks[i].shape)
        flags = mask_flags(stack)
        if flags[group.index(i)]:
            raise ValueError('Duplicate or merged reference: '+row['label'])
        if 'box' in row:
            if reviewed_box_quality(masks[i], row['box'])[1]:
                raise ValueError('Reference mask escapes reviewed box: '+row['label'])
        d, quality, reason = torso_evidence(images[i], masks[i], stack.sum(axis=0)>1)
        if reason:
            raise ValueError('Unusable reference '+row['label']+': '+reason)
        if row.get('lighting') not in ('sun', 'shade'):
            raise ValueError('Approved references require sun/shade')
        if row['role'] == 'player' and row.get('origin') != 'pre-kickoff':
            raise ValueError('Outfield references must be pre-kickoff')
        selected.append(dict(row, descriptor=d.tolist(), quality=quality,
                             image=str((base/row['image']).resolve()),
                             mask=str((base/row['mask']).resolve())))
    for team in ('H', 'A'):
        if {r['lighting'] for r in selected if r['team'] == team and r['role'] == 'player'} != {'sun', 'shade'}:
            raise ValueError('Each outfield team requires sun and shade')
    if not any(r['role'] == 'referee' for r in selected):
        raise ValueError('Approved referee reference required')
    bundle = dict(stage='frozen', reference_profile=profile, mask_reviewed=True,
                  descriptor_version=DESCRIPTOR_VERSION,
                  sam3_revision=SAM3_REVISION, parameters=PARAMETERS,
                  checkpoint_sha256=spec['checkpoint_sha256'],environment=spec.get('environment'),
                  candidate_sha256=sha256(candidate_path), reviewer=spec['reviewer'],
                  references=selected)
    if profile == CONTEXT_PROFILE:
        bundle['goalkeeper_context_policy'] = spec['goalkeeper_context_policy']
    # Retain every input mask needed to reproduce overlap exclusion.
    bundle['evidence'] = [dict(image=str((base/r['image']).resolve()),
                              mask=str((base/r['mask']).resolve()),
                              image_sha256=r['image_sha256'], mask_sha256=r['mask_sha256']) for r in rows]
    out.mkdir(parents=True, exist_ok=False)
    (out/'assets').mkdir()
    mapping={}
    for row in bundle['evidence']:
        for key in ('image','mask'):
            original=row[key]
            destination='assets/'+row[key+'_sha256']+'.png'
            if original not in mapping:
                shutil.copyfile(original,out/destination)
                mapping[original]=destination
            row[key]=mapping[original]
    for row in bundle['references']:
        row['image']=mapping[row['image']]
        row['mask']=mapping[row['mask']]
    write_json(out/'frozen-references.json', bundle)


def load_frozen(path):
    spec = read_json(path)
    if (spec.get('stage') != 'frozen' or spec.get('descriptor_version') != DESCRIPTOR_VERSION
            or spec.get('sam3_revision') != SAM3_REVISION or not spec.get('reviewer')):
        raise ValueError('Approved frozen references required')
    from .reference_review import PROFILE, require_complete as validate_complete
    from .goalkeeper_context import CONTEXT_PROFILE
    profile = spec.get('reference_profile', 'legacy-partial')
    if profile not in ('legacy-partial', PROFILE, CONTEXT_PROFILE):
        raise ValueError('Unknown reference profile')
    if profile in (PROFILE, CONTEXT_PROFILE):
        validate_complete(spec)
    base = Path(path).parent
    evidence = spec['evidence']
    if not evidence or len({r['label'] for r in spec['references']}) != len(spec['references']):
        raise ValueError('Invalid frozen reference registry')
    for row in evidence:
        for key in ('image', 'mask'):
            if sha256(base/row[key]) != row[key+'_sha256']:
                raise ValueError('Frozen asset hash mismatch')
    refs = []
    for row in spec['references']:
        label(row)
        if row.get('approved') is not True:
            raise ValueError('Unapproved frozen reference')
        if not any(all(row.get(k)==e.get(k) for k in ('image','mask','image_sha256','mask_sha256')) for e in evidence):
            raise ValueError('Reference missing from frozen evidence')
        if row['role']=='goalkeeper' and row.get('goalkeeper_verified') is not True:
            raise ValueError('Unverified goalkeeper reference')
        if row['role']=='player' and row.get('origin')!='pre-kickoff':
            raise ValueError('Outfield reference origin must be pre-kickoff')
        image = load_image(base/row['image'])
        with Image.open(base/row['mask']) as m:
            mask = binary_mask(m, np.asarray(image).shape[:2])
        if 'box' in row and reviewed_box_quality(mask, row['box'])[1]:
            raise ValueError('Frozen mask escapes reviewed box')
        group = [r for r in evidence if r['image_sha256'] == row['image_sha256']]
        masks = []
        for other in group:
            with Image.open(base/other['mask']) as m:
                masks.append(binary_mask(m, mask.shape))
        if any(mask_flags(mask_stack(masks,mask.shape))):
            # Rejecting another pair in the same image is conservative and prevents
            # forged approvals from bypassing candidate quality checks.
            indices=[i for i,m in enumerate(masks) if np.array_equal(m,mask)]
            if not indices or mask_flags(mask_stack(masks,mask.shape))[indices[0]]:
                raise ValueError('Frozen duplicate/merged reference')
        excluded = mask_stack(masks, mask.shape).sum(axis=0)>1
        d = torso_evidence(image, mask, excluded)[0]
        if d is None or not np.allclose(d, row['descriptor'], rtol=0, atol=1e-12):
            raise ValueError('Frozen descriptor mismatch')
        # Constructor checks labels and parameters; install exact frozen descriptors below.
        refs.append((image, mask, row['team'], row['role']))
    classifier = TeamClassifier(refs, **spec['parameters'])
    classifier.templates = [(r['team'], r['role'], np.array(r['descriptor'])) for r in spec['references']]
    for team in ('H', 'A'):
        if {r['lighting'] for r in spec['references'] if r['team']==team and r['role']=='player'} != {'sun','shade'}:
            raise ValueError('Frozen team lacks lighting coverage')
    if not any(r['role']=='referee' for r in spec['references']):
        raise ValueError('Frozen referee missing')
    return classifier, spec


def validate_run(spec, base, frozen):
    if spec.get('source') != 'broadcast' or not isinstance(spec.get('frames'), list) or not spec['frames']:
        raise ValueError('Nonempty broadcast frames required')
    if sha256(base/spec['frozen_references']) != spec.get('frozen_sha256'):
        raise ValueError('Frozen bundle hash mismatch')
    seen, previous = set(), -1.
    ref_hashes = {pixel_sha256(load_image(base/Path(spec['frozen_references']).parent/r['image']))
                  for r in frozen['evidence']}
    for entry in spec['frames']:
        frame, timestamp = entry['frame'], entry['timestamp_s']
        if type(frame) is not int or frame < 0 or frame in seen:
            raise ValueError('Unique nonnegative frame IDs required')
        if type(timestamp) not in (int, float) or not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ValueError('Finite increasing recording timestamps required')
        seen.add(frame)
        previous = timestamp
        path = base/entry['image']
        image=load_image(path)
        if sha256(path) != entry.get('image_sha256'):
            raise ValueError('Evaluation image hash mismatch')
        if pixel_sha256(image) in ref_hashes or entry.get('split') != 'evaluation':
            raise ValueError('Reference/evaluation leakage')
