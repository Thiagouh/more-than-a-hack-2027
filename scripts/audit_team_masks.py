"""Reinspect existing SAM candidates without loading a model or granting approval."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from src.team_identification.manifests import read_json, write_json, load_image
from src.team_identification.pilot import (
    binary_mask, sha256, mask_flags, torso_evidence, reviewed_box_quality,
    DESCRIPTOR_VERSION,
)
from scripts.prepare_team_references import mask_contact_sheet


def audit(manifest, out):
    manifest, out = Path(manifest), Path(out)
    spec = read_json(manifest)
    if spec.get('stage') != 'candidate':
        raise ValueError('Candidate manifest required')
    rows = spec['references']
    images, masks = [], []
    for row in rows:
        for key in ('image', 'mask'):
            if sha256(manifest.parent/row[key]) != row[key+'_sha256']:
                raise ValueError('Candidate asset hash mismatch')
        images.append(load_image(manifest.parent/row['image']))
        with Image.open(manifest.parent/row['mask']) as mask:
            masks.append(binary_mask(mask, np.asarray(images[-1]).shape[:2]))
    findings = []
    cores = []
    for i, row in enumerate(rows):
        group = [j for j, other in enumerate(rows)
                 if other['image_sha256'] == row['image_sha256']]
        stack = np.stack([masks[j] for j in group])
        core, quality, reason = torso_evidence(
            images[i], masks[i], stack.sum(axis=0)>1, return_mask=True)
        box_quality, box_reason = reviewed_box_quality(masks[i], row['box'])
        quality.update(box_quality)
        findings.append(dict(label=row['label'], quality=quality,
            quality_reason=mask_flags(stack)[group.index(i)] or box_reason or reason,
            approved=False, manual_review_required=True))
        cores.append(core if core is not None else np.zeros_like(masks[i]))
    out.mkdir(parents=True, exist_ok=False)
    # Each source gets its own sheet; never mix coordinates from different images.
    for digest in sorted({r['image_sha256'] for r in rows}):
        indices = [i for i,r in enumerate(rows) if r['image_sha256'] == digest]
        mask_contact_sheet(images[indices[0]], [rows[i] for i in indices],
                           [cores[i] for i in indices]).save(out/f'torso-{digest[:12]}.png')
    write_json(out/'audit.json', dict(candidate_sha256=sha256(manifest),
        descriptor_version=DESCRIPTOR_VERSION, model_executed=False,
        references_frozen=False, findings=findings))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.out)


if __name__ == '__main__':
    main()
