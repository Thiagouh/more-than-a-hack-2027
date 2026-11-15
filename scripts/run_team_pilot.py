"""Run a truth-bound full-frame or verified-region visual pilot."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from src.team_identification.pilot import Sam3, SAM3_REVISION, sha256, binary_mask, mask_stack
from src.team_identification.manifests import load_frozen, validate_run, load_image, read_json, write_json
from src.team_identification.protocol import load_truth, validate_splits, validate_run_truth, runtime_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--device',choices=['cuda','mps','cpu'],default='cpu')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    spec = read_json(args.manifest)
    base = args.manifest.parent
    if spec.get('schema_version') != 2:
        raise ValueError('Bound v2 run manifest required; legacy runs remain historical')
    truth_path = base/spec['frozen_truth']
    truth = load_truth(truth_path)
    classifier,frozen = load_frozen(base/spec['frozen_references'])
    validate_run(spec,base,frozen)
    validate_run_truth(spec,base,truth)
    validate_splits(truth,frozen)
    mode = truth['evaluation_mode']
    if mode=='full_frame':
        if not args.checkpoint or not args.checkpoint_sha256:
            parser.error('Full-frame inference requires checkpoint and SHA256')
        if args.checkpoint_sha256.lower()!=frozen['checkpoint_sha256']:
            raise ValueError('Classification must use the frozen reference checkpoint')
    elif args.checkpoint or args.checkpoint_sha256:
        parser.error('Verified-region classification does not load a model or checkpoint')
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'frames').mkdir()
    (args.out/'masks').mkdir()
    root = Path(__file__).resolve().parents[1]
    code = [Path(__file__).resolve(), *[root/p for p in (
        'src/team_identification/pilot.py','src/team_identification/manifests.py',
        'src/team_identification/protocol.py','src/ball_events/uniforms.py')]]
    provenance = dict(schema_version=2,status='started',evaluation_mode=mode,sample_kind=truth['sample_kind'],
        synthetic=truth.get('synthetic',False),
        truth_sha256=sha256(truth_path),truth_frozen_at_utc=truth['frozen_at_utc'],
        started_at_utc=datetime.now(timezone.utc).isoformat(),sam3_revision=SAM3_REVISION,
        checkpoint_sha256=frozen['checkpoint_sha256'],model_executed=mode=='full_frame',
        manifest=spec,manifest_sha256=sha256(args.manifest),frozen_references=frozen,
        environment=runtime_provenance(),code_sha256={str(p.relative_to(root)):sha256(p) for p in code},
        validation='not_evaluated')
    # Record the commitment before loading a model or generating classified output.
    write_json(args.out/'provenance.json',provenance)
    sam = Sam3(args.checkpoint,args.checkpoint_sha256,device=args.device) if mode=='full_frame' else None
    if sam:
        provenance['sam_environment'] = sam.environment
    truth_frames = {f['frame']:f for f in truth['frames']}
    with (args.out/'detections.jsonl').open('w') as journal:
        for entry in spec['frames']:
            image = load_image(base/entry['image'])
            annotations = truth_frames[entry['frame']]['people']
            if sam:
                masks,scores = sam.segment(image)
            else:
                masks = []
                for person in annotations:
                    with Image.open(truth_path.parent/person['mask']) as mask:
                        masks.append(binary_mask(mask,(image.height,image.width)))
                masks = mask_stack(masks,(image.height,image.width))
                scores = [None]*len(masks)
            # Only RGB and masks reach the classifier; truth teams/roles do not.
            rows = classifier.classify(image,masks)
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for row,mask,score in zip(rows,masks,scores):
                ys,xs = np.where(mask)
                box = [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
                name = f"{entry['frame']:08d}-{row['detection_index']:03d}.png"
                Image.fromarray(mask.astype('uint8')*255).save(args.out/'masks'/name)
                row.update(box=box,mask='masks/'+name,segmentation_score=float(score) if score is not None else None)
                if mode=='verified_regions':
                    row['region_label'] = annotations[row['detection_index']]['label']
                if box is not None:
                    color = {'H':'blue','A':'orange','referee':'yellow'}.get(row['classification'],'gray')
                    draw.rectangle(box,outline=color,width=2)
                    draw.text((box[0],max(0,box[1]-12)),row['classification'],fill=color)
            annotated.save(args.out/'frames'/f"{entry['frame']:08d}.png")
            journal.write(json.dumps(dict(frame=entry['frame'],timestamp_s=entry['timestamp_s'],
                truth_sha256=provenance['truth_sha256'],image_sha256=entry['image_sha256'],
                pixel_sha256=entry['pixel_sha256'],evaluation_mode=mode,detections=rows,
                internal_counts=dict(Counter(r['classification'] for r in rows))),allow_nan=False)+'\n')
    provenance.update(status='complete',detections_sha256=sha256(args.out/'detections.jsonl'))
    write_json(args.out/'provenance.json',provenance)


if __name__=='__main__':
    main()
