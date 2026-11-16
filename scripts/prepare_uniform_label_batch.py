"""Prepare numbered, unlabeled RGB crops for user-supervised uniform learning.

No SAM mask or ATD team is needed to label a crop. These labels approve only
the pictured person's class, never feet, geometry or mask quality.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from src.team_identification.manifests import load_image, read_json, write_json
from src.team_identification.pilot import sha256
from src.team_identification.protocol import validate_box

CLASSES = {
    'Barcelona': ('H', 'player'), 'Athletic': ('A', 'player'),
    'Arbitro': (None, 'referee'), 'Goleiro Barcelona': ('H', 'goalkeeper'),
    'Goleiro Athletic': ('A', 'goalkeeper'), 'Outro': (None, 'other'),
    'Incerto': (None, 'unknown'),
}

# Explicit, disjoint training pools. None includes the held-out sequences at
# 16000, 16250 or 17000. The additional pool was declared before labeling round 4.
TRAINING_POOLS = {
    'pregame': [(0, 11185)],
    'development_gameplay': [(11185, 14185)],
    'additional_gameplay': [(11185, 14185), (25000, 36000), (120000, 126000)],
}
TEST_POOLS = {'held_out_test': [(16000,16005),(16250,16255),(17000,17005),(20000,20251),
                               (40000,40001),(55000,55001),(70000,70001),
                               (45000,45001),(60000,60001),(80000,80001),
                               (50000,50004),(65000,65004),(135000,135004)]}


def prepare(proposals, extraction, out):
    spec, source = read_json(proposals), read_json(extraction)
    frames = {r['frame']: r for r in source['frames']}
    rows = spec['examples']
    pool = spec.get('sampling_pool', 'pregame')
    pools = {**TRAINING_POOLS, **TEST_POOLS}
    if pool not in pools:
        raise ValueError('Use an explicitly declared training or test pool')
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids)) != len(ids) or any(type(i) is not int or i < 1 for i in ids):
        raise ValueError('Unique positive numeric example IDs required')
    images = {}
    for row in rows:
        f = row['frame']
        # Gameplay labels come only from the existing development interval;
        # the separate 16000/16250/17000 evaluation sequences remain untouched.
        if type(f) is not int or f not in frames or not any(lo <= f < hi for lo,hi in pools[pool]):
            raise ValueError('Frame outside the declared sampling pool')
        entry = frames[f]
        path = extraction.parent/entry['image']
        if sha256(path) != entry['image_sha256'] or entry['source_video_sha256'] != source['source_video_sha256']:
            raise ValueError('Source frame hash mismatch')
        image = images.setdefault(f, load_image(path))
        validate_box(row['box'])
        if row['box'][2] > image.width or row['box'][3] > image.height:
            raise ValueError('Example box outside source frame')
    out.mkdir(parents=True, exist_ok=False)
    (out/'crops').mkdir()
    cols, cell_w, cell_h = 4, 250, 285
    sheet = Image.new('RGB', (cols*cell_w, ((len(rows)+cols-1)//cols)*cell_h), '#eef1f4')
    draw = ImageDraw.Draw(sheet)
    font_path = Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
    font = ImageFont.truetype(str(font_path), 25) if font_path.exists() else ImageFont.load_default(size=25)
    examples = []
    for index, row in enumerate(rows):
        f, ident = row['frame'], row['id']
        entry, image = frames[f], images[f]
        x0,y0,x1,y1 = row['box']
        box = [int(x0),int(y0),int(x1),int(y1)]
        crop = image.crop(box)
        crop_path = out/'crops'/f'{ident:03d}.png'
        crop.save(crop_path)
        tile = crop.copy()
        factor = min(220/tile.width, 225/tile.height)
        tile = tile.resize((max(1,int(tile.width*factor)), max(1,int(tile.height*factor))), Image.Resampling.NEAREST)
        x,y = index%cols*cell_w,index//cols*cell_h
        draw.text((x+12,y+8), str(ident), font=font, fill='#152b3a')
        sheet.paste(tile,(x+(cell_w-tile.width)//2,y+45))
        examples.append(dict(id=ident, frame=f, box=box, crop=str(crop_path.relative_to(out)),
            crop_sha256=sha256(crop_path), image=str((extraction.parent/entry['image']).resolve()),
            image_sha256=entry['image_sha256'], source_video_sha256=source['source_video_sha256'],
            group_id=f'tactical-{pool}-{f//750}', split='test' if pool in TEST_POOLS else 'reference', sampling_pool=pool,
            label=None, team=None, role=None, reviewed=False, reviewer=None,
            mask_approved=False, geometry_approved=False))
    sheet.save(out/'rotular.png')
    write_json(out/'batch.json', dict(schema='uniform-supervised-label-batch-v1', status='awaiting_user_labels',
        sampling_pool=pool,
        extraction_sha256=sha256(extraction), proposals_sha256=sha256(proposals),
        contact_sheet_sha256=sha256(out/'rotular.png'), classes=list(CLASSES), examples=examples,
        training_completed=False, notes='Keep temporal groups together when separating training and evaluation.'))
    write_json(out/'answers-template.json',dict(reviewer=None,batch_sha256=sha256(out/'batch.json'),
        labels={str(i):None for i in ids}))
    return out/'rotular.png'


def record(batch, answers, out):
    spec, verdict = read_json(batch), read_json(answers)
    if spec.get('schema')!='uniform-supervised-label-batch-v1' or verdict.get('batch_sha256')!=sha256(batch):
        raise ValueError('Labels must identify the unchanged batch')
    if not isinstance(verdict.get('reviewer'),str) or not verdict['reviewer'].strip():
        raise ValueError('Explicit reviewer required')
    ids = {str(r['id']) for r in spec['examples']}
    labels = verdict.get('labels',{})
    if set(labels) != ids or any(v not in CLASSES for v in labels.values()):
        raise ValueError('Every ID requires one supported class, including Incerto or Outro')
    for row in spec['examples']:
        path = batch.parent/row['crop']
        if sha256(path)!=row['crop_sha256'] or sha256(row['image'])!=row['image_sha256']:
            raise ValueError('Labeled source/crop changed')
        label = labels[str(row['id'])]
        row.update(label=label,team=CLASSES[label][0],role=CLASSES[label][1],
                   reviewed=True,reviewer=verdict['reviewer'],crop=str(path.resolve()))
    spec.update(status='user_labeled',original_batch_sha256=sha256(batch),answers_sha256=sha256(answers))
    from datetime import datetime, timezone
    spec['reviewed_at_utc']=datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'labeled-batch.json',spec)
    return out/'labeled-batch.json'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    prep.add_argument('--proposals',type=Path,required=True);prep.add_argument('--extraction',type=Path,required=True)
    prep.add_argument('--out',type=Path,required=True)
    rec=sub.add_parser('record')
    rec.add_argument('--batch',type=Path,required=True);rec.add_argument('--answers',type=Path,required=True)
    rec.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(prepare(a.proposals,a.extraction,a.out) if a.action=='prepare' else record(a.batch,a.answers,a.out))


if __name__=='__main__':main()
