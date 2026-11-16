"""Four-frame hard-case training strips and one user label per local person."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.team_identification.manifests import read_json,write_json,load_image
from src.team_identification.pilot import sha256
from src.team_identification.protocol import validate_box
from scripts.prepare_uniform_label_batch import CLASSES,TRAINING_POOLS,TEST_POOLS


def follow(previous,image,box,radius=12):
    """Short-range RGB patch proposal, not a certified persistent identity."""
    x,y,X,Y=map(int,box);w,h=X-x,Y-y
    # Central body pixels reduce the influence of surrounding grass.
    inner=(max(1,int(.2*w)),max(1,int(.1*h)),max(2,int(.8*w)),max(2,int(.9*h)))
    reference=np.asarray(previous.crop((x+inner[0],y+inner[1],x+inner[2],y+inner[3])),dtype=float)
    pixels=np.asarray(image,dtype=float)
    best=None
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if not (0<=x+dx and 0<=y+dy and X+dx<=image.width and Y+dy<=image.height):continue
            patch=pixels[y+dy+inner[1]:y+dy+inner[3],x+dx+inner[0]:x+dx+inner[2]]
            error=float(np.mean((patch-reference)**2))
            candidate=(error,abs(dx)+abs(dy),dx,dy)
            if best is None or candidate<best:best=candidate
    if best is None:raise ValueError('No valid four-frame tracking proposal')
    return [x+best[2],y+best[3],X+best[2],Y+best[3]],best[0]


def prepare(proposals,extraction,out):
    spec,source=read_json(proposals),read_json(extraction)
    split=spec.get('split','reference')
    if split not in ('reference','test'):raise ValueError('Explicit reference or test sequence split required')
    pools=TRAINING_POOLS if split=='reference' else TEST_POOLS
    entries={r['frame']:r for r in source['frames']}
    cases=spec['sequences'];ids=[r['id'] for r in cases]
    if not cases or len(ids)!=len(set(ids)) or any(type(i)is not int or i<1 for i in ids):
        raise ValueError('Unique positive sequence IDs required')
    images={}
    for case in cases:
        validate_box(case['box'])
        for f in range(case['start_frame'],case['start_frame']+4):
            if f not in entries or not any(lo<=f<hi for ranges in pools.values() for lo,hi in ranges):
                raise ValueError('Sequence frame outside the declared split pool')
            entry=entries[f];path=extraction.parent/entry['image']
            if sha256(path)!=entry['image_sha256'] or entry['source_video_sha256']!=source['source_video_sha256']:
                raise ValueError('Four-frame extraction changed')
            if f not in images:images[f]=load_image(path)
    out.mkdir(parents=True,exist_ok=False);(out/'crops').mkdir()
    font_path=Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
    font=ImageFont.truetype(str(font_path),30) if font_path.exists() else ImageFont.load_default(size=30)
    pages=[];rows=[]
    for index,case in enumerate(cases):
        page=index//8;local=index%8
        if page==len(pages):pages.append(Image.new('RGB',(1280,4*280),'#eef1f4'))
        left,top=local%2*640,local//2*280
        frames=[];box=list(case['box']);previous=None
        for offset in range(4):
            f=case['start_frame']+offset;image=images[f]
            if previous is not None:box,error=follow(previous,image,box)
            else:error=None
            x,y,X,Y=map(int,box)
            if not (0<=x<X<=image.width and 0<=y<Y<=image.height):raise ValueError('Sequence crop outside image')
            crop=image.crop((x,y,X,Y));asset=out/'crops'/f"{case['id']}-{f}.png";crop.save(asset)
            tile=crop.copy();scale=min(145/tile.width,205/tile.height)
            tile=tile.resize((int(tile.width*scale),int(tile.height*scale)),Image.Resampling.NEAREST)
            tx=left+offset*160+(160-tile.width)//2;ty=top+45
            pages[page].paste(tile,(tx,ty))
            entry=entries[f]
            frames.append(dict(frame=f,box=[x,y,X,Y],crop=str(asset.relative_to(out)),crop_sha256=sha256(asset),
                image=str((extraction.parent/entry['image']).resolve()),image_sha256=entry['image_sha256'],
                source_video_sha256=source['source_video_sha256'],available_at_frame=f,patch_error=error))
            previous=image
        rows.append(dict(id=case['id'],identity=f"{split}-sequence-{case['id']}",split=split,frames=frames,
            challenge=case.get('challenge','difficult_pose_or_occlusion'),label=None,reviewed=False,
            association_reviewed=False,association_reviewer=None,mask_approved=False,geometry_approved=False,
            group_id=f"tactical-{split}-{case['start_frame']//750}"))
    # Draw labels after all pastes so every panel retains its case/frame IDs.
    for page_index,page_image in enumerate(pages):
        draw=ImageDraw.Draw(page_image)
        for local,row in enumerate(rows[page_index*8:(page_index+1)*8]):
            left,top=local%2*640,local//2*280
            draw.text((left+12,top+3),str(row['id']),font=font,fill='#152b3a')
            for offset,frame in enumerate(row['frames']):
                draw.text((left+offset*160+10,top+256),f"f {frame['frame']}",fill='#30495a')
    page_files=[]
    for i,page in enumerate(pages,1):
        path=out/f'rotular-{i}.png';page.save(path);page_files.append(dict(image=path.name,sha256=sha256(path)))
    write_json(out/'batch.json',dict(schema='uniform-four-frame-batch-v1',status='awaiting_user_labels',split=split,
        sequences=rows,classes=list(CLASSES),frames_per_sequence=4,fps=25,sequence_span_s=.12,
        extraction_sha256=sha256(extraction),proposals_sha256=sha256(proposals),contact_sheets=page_files,
        identity_proposal_method='Short-range RGB patch matching; requires visual continuity review before training.'))
    write_json(out/'answers-template.json',dict(batch_sha256=sha256(out/'batch.json'),reviewer=None,labels={str(i):None for i in ids}))
    return out


def record(batch,answers,out):
    spec,verdict=read_json(batch),read_json(answers)
    if spec.get('schema') not in ('uniform-four-frame-training-batch-v1','uniform-four-frame-batch-v1') or verdict.get('batch_sha256')!=sha256(batch):
        raise ValueError('Unchanged four-frame batch required')
    if not verdict.get('reviewer') or set(verdict.get('labels',{}))!={str(r['id']) for r in spec['sequences']}:
        raise ValueError('Explicit reviewer and one label per sequence required')
    if any(label not in CLASSES for label in verdict['labels'].values()):raise ValueError('Unsupported sequence label')
    examples=[]
    for sequence in spec['sequences']:
        if sequence.get('association_reviewed') is not True or not sequence.get('association_reviewer'):
            raise ValueError('Review same-person continuity before propagating one label to four frames')
        frames=sequence['frames'];numbers=[r['frame'] for r in frames]
        if len(frames)!=4 or numbers!=list(range(numbers[0],numbers[0]+4)) or sequence['split'] not in ('reference','test'):
            raise ValueError('Exactly four consecutive frames in one declared split required')
        label=verdict['labels'][str(sequence['id'])];team,role=CLASSES[label]
        for frame in frames:
            path=batch.parent/frame['crop']
            if sha256(path)!=frame['crop_sha256'] or sha256(frame['image'])!=frame['image_sha256']:
                raise ValueError('Sequence crop/source changed')
            examples.append(dict(frame,id=f"{sequence['id']}-f{frame['frame']}",sequence_id=sequence['id'],
                crop=str(path.resolve()),split=sequence['split'],group_id=sequence['group_id'],
                challenge=sequence.get('challenge'),association_reviewer=sequence['association_reviewer'],
                label=label,team=team,role=role,reviewed=True,reviewer=verdict['reviewer'],
                mask_approved=False,geometry_approved=False))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'labeled-batch.json',dict(schema='uniform-supervised-label-batch-v1',status='user_labeled',
        original_sequence_batch_sha256=sha256(batch),answers_sha256=sha256(answers),examples=examples,
        sequence_labels={str(r['id']):verdict['labels'][str(r['id'])] for r in spec['sequences']},
        independent_sequences=len(spec['sequences']),crop_count=len(examples),
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
        note='Four correlated crops per sequence; never count these as four independent validation samples.'))
    return out/'labeled-batch.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--proposals',type=Path,required=True);a.add_argument('--extraction',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('record');a.add_argument('--batch',type=Path,required=True);a.add_argument('--answers',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(prepare(a.proposals,a.extraction,a.out) if a.action=='prepare' else record(a.batch,a.answers,a.out))


if __name__=='__main__':main()
