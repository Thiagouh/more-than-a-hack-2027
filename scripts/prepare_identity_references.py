"""Prepare multi-frame tactical reference proposals, without granting approval.

Use review_team_references afterward to review masks and freeze complete-v1.
"""
import argparse
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
from src.team_identification.manifests import read_json, write_json, load_image, LABELS
from src.team_identification.pilot import Sam3, SAM3_REVISION, sha256, torso_evidence, mask_flags, reviewed_box_quality
from src.team_identification.protocol import validate_box


def prepare(boxes, out, checkpoint=None, checkpoint_sha256=None, device='cpu', context_scale=1.5):
    spec=read_json(boxes)
    if spec.get('timeline')!='tactical' or spec.get('schema')!='tactical-uniform-box-proposals-v1':
        raise ValueError('Tactical box proposals required')
    extraction=boxes.parent/spec['extraction']
    if sha256(extraction)!=spec['extraction_sha256']:
        raise ValueError('Extraction manifest changed')
    source=read_json(extraction)
    frames={r['frame']:r for r in source['frames']}
    rows=spec['references']
    if not rows or len({r['label'] for r in rows})!=len(rows):
        raise ValueError('Unique nonempty reference proposals required')
    for row in rows:
        validate_box(row['box'])
        if (row.get('team'),row.get('role')) not in LABELS or row.get('lighting') not in ('sun','shade'):
            raise ValueError('Explicit proposed team, role and lighting required')
        if row['frame'] not in frames or row['frame']>=11185:
            raise ValueError('Use extracted pre-kickoff frames for initial tactical references')
        image=load_image(extraction.parent/frames[row['frame']]['image'])
        if sha256(extraction.parent/frames[row['frame']]['image'])!=frames[row['frame']]['image_sha256']:
            raise ValueError('Source frame changed')
        if row['box'][2]>image.width or row['box'][3]>image.height:
            raise ValueError('Box outside source image')
    out.mkdir(parents=True,exist_ok=False)
    (out/'assets').mkdir()
    cols=4;sheet=Image.new('RGB',(cols*320,((len(rows)+cols-1)//cols)*340),'#f0f2f3');draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        image=load_image(extraction.parent/frames[row['frame']]['image'])
        crop=image.crop(row['box']);crop.thumbnail((260,265))
        scale=min(260/crop.width,265/crop.height)
        crop=crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.NEAREST)
        x,y=(i%cols)*320,(i//cols)*340
        sheet.paste(crop,(x+(320-crop.width)//2,y+45))
        draw.text((x+8,y+5),row['label']+' | proposed '+str(row['team'])+'/'+row['role'],fill='black')
        draw.text((x+8,y+23),f"frame {row['frame']} | {row['lighting']} | PENDING REVIEW",fill='#734500')
    sheet.save(out/'uniform-review.png')
    write_json(out/'box-proposals.json',spec)
    if checkpoint is None:
        write_json(out/'review-status.json',dict(approved=False,model_executed=False,reason='Uniform boxes and roles pending review; masks not generated'))
        return
    sam=Sam3(checkpoint,checkpoint_sha256,device=device)
    references=[]
    for f in sorted({r['frame'] for r in rows}):
        entry=frames[f];image=load_image(extraction.parent/entry['image'])
        subset=[r for r in rows if r['frame']==f]
        print('Segmenting reference frame',f,flush=True)
        masks,scores=sam.segment_boxes(image,[r['box'] for r in subset],context_crop=True,context_scale=context_scale)
        image_path=out/'assets'/f'{f:08d}.png'
        shutil.copyfile(extraction.parent/entry['image'],image_path)
        overlap=masks.sum(axis=0)>1;flags=mask_flags(masks)
        for i,(row,mask,score) in enumerate(zip(subset,masks,scores)):
            mask_path=out/'assets'/f"{row['label']}.png"
            Image.fromarray(mask.astype('uint8')*255).save(mask_path)
            _,quality,reason=torso_evidence(image,mask,overlap)
            box_quality,box_reason=reviewed_box_quality(mask,row['box']);quality.update(box_quality)
            references.append(dict(row,image=str(image_path.relative_to(out)),mask=str(mask_path.relative_to(out)),
                image_sha256=sha256(image_path),mask_sha256=sha256(mask_path),approved=False,
                goalkeeper_verified=False,origin='pre-kickoff',source_sha256=source['source_video_sha256'],
                source_segment_id=f'tactical-pregame-{f//750}',quality=quality,quality_reason=flags[i] or box_reason or reason,
                segmentation_score=float(score)))
    write_json(out/'candidate-references.json',dict(stage='candidate',reference_profile='complete-v1',
        mask_reviewed=False,reviewer=None,references=references,sam3_revision=SAM3_REVISION,
        checkpoint_sha256=checkpoint_sha256,environment=sam.environment,context_scale=context_scale,initial_manifest_sha256=sha256(boxes)))
    write_json(out/'review-status.json',dict(approved=False,model_executed=True,references_frozen=False,
        reason='Review masks, torso purity, teams, lighting and both own goalkeepers before freezing'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--boxes',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--checkpoint-sha256')
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    p.add_argument('--context-scale',type=float,default=1.5)
    a=p.parse_args()
    if bool(a.checkpoint)!=bool(a.checkpoint_sha256):p.error('Checkpoint and hash must be supplied together')
    prepare(a.boxes,a.out,a.checkpoint,a.checkpoint_sha256,a.device,a.context_scale)


if __name__=='__main__':main()
