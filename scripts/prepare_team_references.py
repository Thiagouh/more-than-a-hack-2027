"""Turn reviewed pre-kickoff boxes into SAM 3 masks and frozen references."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.team_identification.pilot import Sam3, SAM3_REVISION, sha256, torso_evidence, mask_flags
from src.team_identification.manifests import validate_initial, load_image, freeze, write_json


COLORS = {'H': '#2468f2', 'A': '#f05a31', None: '#ffdd33'}


def mask_contact_sheet(image, annotations, masks):
    sheet=Image.new('RGB',(4*320,((len(annotations)+3)//4)*230),'#202020')
    draw=ImageDraw.Draw(sheet)
    pixels=np.asarray(image)
    for index,(row,mask) in enumerate(zip(annotations,masks)):
        x,y=(index%4)*320,(index//4)*230
        draw.text((x+8,y+4),row['label']+' | original / mask',fill='white')
        overlay=pixels.copy()
        overlay[mask]=(.55*overlay[mask]+.45*np.array([255,0,200])).astype(np.uint8)
        x0,y0,x1,y1=map(int,row['box'])
        region=(max(0,x0-8),max(0,y0-8),min(image.width,x1+8),min(image.height,y1+8))
        for column,source in enumerate((image,Image.fromarray(overlay))):
            crop=source.crop(region)
            crop.thumbnail((150,200))
            # Upscale only this diagnostic preview; descriptors use original RGB.
            factor=min(150/crop.width,200/crop.height)
            crop=crop.resize((int(crop.width*factor),int(crop.height*factor)))
            sheet.paste(crop,(x+column*160+(160-crop.width)//2,y+24))
    return sheet


def draw_annotations(image, annotations):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for row in annotations:
        box = row['box']
        color = COLORS[row['team']]
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] + 3, max(0, box[1] - 15)), row['label'], fill=color,
                  stroke_width=2, stroke_fill='black')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--device',choices=['cuda','mps','cpu'],default='cuda')
    parser.add_argument('--box-context-crop',action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--boxes-only', action='store_true')
    parser.add_argument('--freeze', action='store_true', help='Freeze explicitly approved candidate manifest; no GPU')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.manifest,args.out)
        return
    spec = json.loads(args.manifest.read_text())
    image_path = args.manifest.parent / spec['initial_image']
    image = load_image(image_path)
    validate_initial(spec,image)
    annotations = spec['initial_annotations']
    if len({row['label'] for row in annotations}) != len(annotations):
        raise ValueError('Annotation labels must be unique')
    if {row['team'] for row in annotations if row['role'] == 'player'} != {'H', 'A'}:
        raise ValueError('Initial player annotations must cover both teams')
    if args.boxes_only:
        args.out.mkdir(parents=True, exist_ok=False)
        draw_annotations(image, annotations).save(args.out / 'initial-box-review.png')
        return
    if spec.get('reviewed') is not True:
        raise ValueError('Set reviewed=true only after checking initial-box-review.png')
    if not args.checkpoint or not args.checkpoint_sha256:
        raise ValueError('Checkpoint path and SHA256 are required for segmentation')
    sam = Sam3(args.checkpoint, args.checkpoint_sha256,device=args.device)
    args.out.mkdir(parents=True, exist_ok=False)
    draw_annotations(image, annotations).save(args.out / 'initial-box-review.png')
    image.save(args.out/'reference-image.png')
    mask_rows,score_rows=[],[]
    for index,row in enumerate(annotations):
        print(f"Segmenting {row['label']} ({index+1}/{len(annotations)})",flush=True)
        batch,batch_scores=sam.segment_boxes(image,[row['box']],context_crop=args.box_context_crop)
        mask_rows.append(batch[0]);score_rows.append(batch_scores[0])
    masks,scores=np.asarray(mask_rows),np.asarray(score_rows)
    mask_dir = args.out / 'reference-masks'
    mask_dir.mkdir()
    references = []
    flags=mask_flags(masks)
    overlap=masks.sum(axis=0)>1
    for index,(row, mask, score) in enumerate(zip(annotations, masks, scores)):
        _,quality,reason=torso_evidence(image,mask,overlap)
        name = row['label'] + '.png'
        Image.fromarray(mask.astype('uint8') * 255).save(mask_dir / name)
        references.append(dict(image='reference-image.png', mask='reference-masks/' + name,
                               team=row['team'], role=row['role'], label=row['label'],
                               box=row['box'],
                               quality=quality,quality_reason=flags[index] or reason,
                               approved=False, lighting=row.get('lighting'),origin='pre-kickoff',
                               image_sha256=sha256(args.out/'reference-image.png'),
                               mask_sha256=sha256(mask_dir/name),
                               segmentation_score=float(score)))
    review = image.copy()
    overlay = np.asarray(review).copy()
    for row, mask in zip(annotations, masks):
        color = np.array((36, 104, 242) if row['team'] == 'H' else
                         (240, 90, 49) if row['team'] == 'A' else (255, 221, 51))
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * color).astype('uint8')
    review = Image.fromarray(overlay)
    draw_annotations(review, annotations).save(args.out / 'initial-mask-review.png')
    mask_contact_sheet(image,annotations,masks).save(args.out/'reference-contact-sheet.png')
    write_json(args.out/'candidate-references.json',dict(stage='candidate',mask_reviewed=False,
        reviewer=None,references=references,sam3_revision=SAM3_REVISION,
        checkpoint_sha256=args.checkpoint_sha256.lower(),initial_manifest_sha256=sha256(args.manifest),
        box_context_crop=args.box_context_crop,
        environment=sam.environment))


if __name__ == '__main__':
    main()
