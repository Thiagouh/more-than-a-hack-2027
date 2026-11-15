"""Run independent team classification on decoded broadcast PNG frames."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from src.team_identification.pilot import Sam3,SAM3_REVISION,sha256
from src.team_identification.manifests import load_frozen,validate_run,load_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();spec=json.loads(a.manifest.read_text());base=a.manifest.parent
    classifier,frozen=load_frozen(base/spec['frozen_references'])
    validate_run(spec,base,frozen)
    if a.checkpoint_sha256.lower() != frozen['checkpoint_sha256']:
        raise ValueError('Classification must use the frozen reference checkpoint')
    sam=Sam3(a.checkpoint,a.checkpoint_sha256,device=a.device)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'frames').mkdir();(a.out/'masks').mkdir()
    with (a.out/'detections.jsonl').open('w') as journal:
        for entry in spec['frames']:
            im=load_image(base/entry['image'])
            masks,scores=sam.segment(im)
            rows=classifier.classify(im,masks)
            counts=Counter(r['team'] or r['role'] for r in rows)
            annotated=im.copy()
            draw=ImageDraw.Draw(annotated)
            for r,mask,score in zip(rows,masks,scores):
                ys,xs=np.where(mask)
                box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)] if len(xs) else None
                name=f"{entry['frame']:08d}-{r['detection_index']:03d}.png"
                Image.fromarray(mask.astype('uint8')*255).save(a.out/'masks'/name)
                r.update(box=box,mask='masks/'+name,segmentation_score=float(score))
                if box is None:continue
                color={'H':'blue','A':'orange'}.get(r['team'],'gray')
                draw.rectangle(box,outline=color,width=2)
                draw.text((box[0],max(0,box[1]-12)),{'H':'Barcelona','A':'Athletic'}.get(r['team'],r['role']),fill=color)
            annotated.save(a.out/'frames'/f"{entry['frame']:08d}.png")
            journal.write(json.dumps(dict(frame=entry['frame'],timestamp_s=entry['timestamp_s'],detections=rows,internal_counts=dict(counts)))+'\n')
    (a.out/'provenance.json').write_text(json.dumps(dict(sam3_revision=SAM3_REVISION,
        checkpoint_sha256=a.checkpoint_sha256.lower(),manifest=spec,manifest_sha256=sha256(a.manifest),
        frozen_references=frozen,environment=sam.environment,
        code_sha256={str(path.relative_to(Path(__file__).resolve().parents[1])):sha256(path)
            for path in [Path(__file__).resolve(),
                Path(__file__).resolve().parents[1]/'src/team_identification/pilot.py',
                Path(__file__).resolve().parents[1]/'src/team_identification/manifests.py',
                Path(__file__).resolve().parents[1]/'src/ball_events/uniforms.py']},
        validation='not_evaluated'),indent=2))

if __name__=='__main__':main()
