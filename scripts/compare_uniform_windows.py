"""Prepare, freeze and score paired causal uniform-window experiments.

Preparation requires opencv-python-headless for RGB patch proposals. Proposals
never certify identity: a separate complete visual review is mandatory.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.team_identification.manifests import read_json, write_json, load_image
from src.team_identification.pilot import sha256
from src.team_identification.supervised_uniforms import SupervisedUniforms
from src.team_identification.temporal_windows import CONFIGURATIONS, decide, check_separation
from scripts.prepare_uniform_label_batch import TRAINING_POOLS, CLASSES


def history(root, source_hash):
    """Inventory actual source observations in all historical uniform manifests."""
    files=[]; frames=set()
    for path in sorted(root.glob('uniform*/**/*.json')):
        if 'uniform-temporal-round11' in str(path):continue
        try:data=read_json(path)
        except (ValueError,OSError):continue
        found=set()
        def visit(obj):
            if isinstance(obj,dict):
                if obj.get('source_video_sha256')==source_hash and type(obj.get('frame')) is int:found.add(obj['frame'])
                for v in obj.values():visit(v)
            elif isinstance(obj,list):
                for v in obj:visit(v)
        visit(data)
        if found:
            frames.update(found);files.append(dict(path=str(path.resolve()),sha256=sha256(path),frames=sorted(found)))
    return sorted(frames),files


def patch_step(previous, current, box, anchor=None):
    import cv2
    x,y,X,Y=box; w,h=X-x,Y-y
    # Body-only patch, with complete person crop retained separately.
    ix,iy,iw,ih=int(.15*w),int(.10*h),max(3,int(.7*w)),max(5,int(.45*h))
    patch=previous[y+iy:y+iy+ih,x+ix:x+ix+iw]
    sx,sy=max(0,x+ix-14),max(0,y+iy-14)
    ex,ey=min(current.shape[1],x+ix+iw+14),min(current.shape[0],y+iy+ih+14)
    response=cv2.matchTemplate(current[sy:ey,sx:ex],patch,cv2.TM_SQDIFF_NORMED)
    if anchor is not None:
        response=.35*response+.65*cv2.matchTemplate(current[sy:ey,sx:ex],anchor,cv2.TM_SQDIFF_NORMED)
    _,_,loc,_=cv2.minMaxLoc(response)
    dx,dy=sx+loc[0]-(x+ix),sy+loc[1]-(y+iy)
    error=float(response[loc[1],loc[0]])
    return [x+dx,y+dy,X+dx,Y+dy],error


def prepare(model_path, extraction, proposals, out):
    source,spec,artifact=read_json(extraction),read_json(proposals),read_json(model_path)
    reserved,inventory=history(Path('outputs'),source['source_video_sha256'])
    cases=spec['sequences'];ids=[r['id'] for r in cases]
    if not cases or len(ids)!=len(set(ids)):raise ValueError('Unique cases required')
    entries={r['frame']:r for r in source['frames']};images={}
    for case in cases:
        end=case['decision_frame']
        check_separation(end-50,end,source['source_video_sha256'],artifact['examples'],reserved,TRAINING_POOLS)
        for f in range(end-50,end+1):
            if f in images:continue
            entry=entries[f];path=extraction.parent/entry['image']
            if entry['source_video_sha256']!=source['source_video_sha256'] or sha256(path)!=entry['image_sha256']:raise ValueError('Source changed')
            if f not in images:images[f]=np.asarray(load_image(path))
    out.mkdir(parents=True,exist_ok=False);(out/'crops').mkdir();(out/'identity-review').mkdir()
    rows=[]
    for case in cases:
        end=case['decision_frame'];box=list(case['box']);identity=f"round11-person-{case['id']}";frames=[]
        strip=Image.new('RGB',(17*110,3*185),'white');draw=ImageDraw.Draw(strip)
        x,y,X,Y=box;w,h=X-x,Y-y;ix,iy,iw,ih=int(.15*w),int(.10*h),max(3,int(.7*w)),max(5,int(.45*h))
        anchor=images[end-50][y+iy:y+iy+ih,x+ix:x+ix+iw].copy()
        for index,f in enumerate(range(end-50,end+1)):
            error=None;back_error=None
            if index:
                box,error=patch_step(images[f-1],images[f],box,anchor)
                back,_=patch_step(images[f],images[f-1],box)
                back_error=max(abs(a-b) for a,b in zip(back,frames[-1]['box']))
            x,y,X,Y=box;image=Image.fromarray(images[f])
            if not (0<=x<X<=image.width and 0<=y<Y<=image.height):raise ValueError('Tracking proposal outside image')
            crop=image.crop(box);path=out/'crops'/f"{case['id']}-{f}.png";crop.save(path)
            entry=entries[f]
            frames.append(dict(frame=f,identity=identity,available_at_frame=f,box=box.copy(),
                crop=str(path.relative_to(out)),crop_sha256=sha256(path),
                image=str((extraction.parent/entry['image']).resolve()),image_sha256=entry['image_sha256'],
                source_video_sha256=source['source_video_sha256'],patch_error=error,reverse_error_px=back_error,
                association_verified=False))
            crop.thumbnail((100,145));tx=index%17*110;ty=index//17*185
            strip.paste(crop,(tx+(110-crop.width)//2,ty+23));draw.text((tx+3,ty+3),str(f),fill='black')
        review=out/'identity-review'/f"{case['id']}.png";strip.save(review)
        rows.append(dict(id=case['id'],identity=identity,decision_frame=end,split='test',label=None,
            group_id=f'tactical-round11-{end}',challenge=case.get('challenge','unreviewed'),frames=frames,
            identity_review=str(review.relative_to(out)),identity_review_sha256=sha256(review)))
    # Four representative timestamps; detailed sheets retain every frame.
    sheets=[];font=(ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',25) if Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf').exists() else ImageFont.load_default(size=25))
    for page in range(math.ceil(len(rows)/16)):
        sheet=Image.new('RGB',(1600,4*270),'#eef1f4');draw=ImageDraw.Draw(sheet)
        for i,case in enumerate(rows[page*16:(page+1)*16]):
            x=i%4*400;y=i//4*270;draw.text((x+8,y+3),str(case['id']),fill='black',font=font)
            for j,k in enumerate([0,25,40,50]):
                row=case['frames'][k];crop=load_image(out/row['crop']);crop.thumbnail((94,205))
                sheet.paste(crop,(x+j*100+(100-crop.width)//2,y+35));draw.text((x+j*100+3,y+245),f"{(k-50)/25:g}s",fill='black')
        path=out/f'rotular-{page+1}.png';sheet.save(path);sheets.append(dict(image=path.name,sha256=sha256(path)))
    write_json(out/'batch.json',dict(schema='uniform-temporal-window-batch-v1',split='test',status='awaiting_user_labels',
        sequences=rows,configurations=CONFIGURATIONS,fps=25,contact_sheets=sheets,
        model_sha256=sha256(model_path),extraction_sha256=sha256(extraction),proposals_sha256=sha256(proposals),
        historical_inventory=inventory,historical_frames=reserved,minimum_separation_frames=750,
        identity_method='RGB patch proposals with reverse check; all 51 frames require separate visual review; no ATD slots'))
    write_json(out/'identity-review-template.json',dict(batch_sha256=sha256(out/'batch.json'),reviewer=None,
        cases={str(r['id']):dict(verified_frames=[],ambiguous_frames=[],notes='') for r in rows}))
    write_json(out/'answers-template.json',dict(batch_sha256=sha256(out/'batch.json'),reviewer=None,labels={str(i):None for i in ids}))


def review_batch(batch_path, review_path, out):
    """Remove ambiguous pixel proposals from observations and public sheets."""
    batch, review = read_json(batch_path), read_json(review_path)
    if review.get('batch_sha256') != sha256(batch_path) or not review.get('reviewer'):
        raise ValueError('Bound explicit visual identity review required')
    if set(review['cases']) != {str(c['id']) for c in batch['sequences']}:
        raise ValueError('Complete identity review required')
    # Validate completely before creating an output.
    for case in batch['sequences']:
        v=review['cases'][str(case['id'])];a=set(v['verified_frames']);b=set(v['ambiguous_frames'])
        if a&b or a|b!=set(range(case['decision_frame']-50,case['decision_frame']+1)):
            raise ValueError('Every frame needs exactly one identity verdict')
    out.mkdir(parents=True,exist_ok=False);(out/'identity-review').mkdir()
    batch['proposal_batch_sha256']=sha256(batch_path)
    batch['proposal_identity_review_sha256']=sha256(review_path)
    for case in batch['sequences']:
        verified=set(review['cases'][str(case['id'])]['verified_frames'])
        for row in case['frames']:
            path=batch_path.parent/row['crop']
            if sha256(path)!=row['crop_sha256']:raise ValueError('Proposal crop changed')
            if row['frame'] in verified:
                row.update(crop=str(path.resolve()),association_verified=True)
            else:
                row.update(proposed_crop=str(path.resolve()),proposed_crop_sha256=row['crop_sha256'],
                    proposed_box=row['box'],box=None,crop=None,crop_sha256=None,association_verified=False,
                    omission_reason='visual_identity_ambiguous_or_lost')
        # Public full sequence: unverified proposals never masquerade as observations.
        strip=Image.new('RGB',(17*110,3*185),'white');draw=ImageDraw.Draw(strip)
        for i,row in enumerate(case['frames']):
            x=i%17*110;y=i//17*185;draw.text((x+3,y+3),str(row['frame']),fill='black')
            if row['crop']:
                crop=load_image(row['crop']);crop.thumbnail((100,145));strip.paste(crop,(x+(110-crop.width)//2,y+23))
            else:draw.text((x+5,y+60),'Indisponivel',fill='#666666')
        path=out/'identity-review'/f"{case['id']}.png";strip.save(path)
        case['identity_review']=str(path.relative_to(out));case['identity_review_sha256']=sha256(path)
    sheets=[];font=(ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',25) if Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf').exists() else ImageFont.load_default(size=25))
    for page in range(math.ceil(len(batch['sequences'])/16)):
        sheet=Image.new('RGB',(1600,1080),'#eef1f4');draw=ImageDraw.Draw(sheet)
        for i,case in enumerate(batch['sequences'][page*16:(page+1)*16]):
            x=i%4*400;y=i//4*270;draw.text((x+8,y+3),str(case['id']),fill='black',font=font)
            for j,k in enumerate([0,25,40,50]):
                row=case['frames'][k]
                if row['crop']:
                    crop=load_image(row['crop']);crop.thumbnail((94,205));sheet.paste(crop,(x+j*100+(100-crop.width)//2,y+35))
                else:draw.text((x+j*100+3,y+100),'Indisponivel',fill='#666666')
                draw.text((x+j*100+3,y+245),f"{(k-50)/25:g}s",fill='black')
        path=out/f'rotular-{page+1}.png';sheet.save(path);sheets.append(dict(image=path.name,sha256=sha256(path)))
    batch['contact_sheets']=sheets
    write_json(out/'batch.json',batch)
    review['proposal_batch_sha256']=review['batch_sha256'];review['batch_sha256']=sha256(out/'batch.json')
    write_json(out/'identity-reviewed.json',review)
    write_json(out/'answers-template.json',dict(batch_sha256=sha256(out/'batch.json'),reviewer=None,
        labels={str(c['id']):None for c in batch['sequences']},reviewed_at_utc=None))


def predict(model_path,batch_path,review_path,out):
    batch,review,artifact=read_json(batch_path),read_json(review_path),read_json(model_path)
    if (batch.get('schema')!='uniform-temporal-window-batch-v1' or batch.get('status')!='awaiting_user_labels'
        or batch.get('split')!='test' or batch['configurations']!=CONFIGURATIONS
        or batch['model_sha256']!=sha256(model_path) or review.get('batch_sha256')!=sha256(batch_path)
        or not review.get('reviewer')):raise ValueError('Unchanged unlabeled batch/model and explicit identity review required')
    if set(review['cases'])!={str(c['id']) for c in batch['sequences']}:raise ValueError('Complete identity review required')
    model=SupervisedUniforms(artifact);predictions={name:[] for name in CONFIGURATIONS}
    for case in batch['sequences']:
        if case['label'] is not None or case['split']!='test':raise ValueError('Labels must follow predictions')
        identity=case['identity'];end=case['decision_frame'];verdict=review['cases'][str(case['id'])]
        verified=set(verdict['verified_frames']);ambiguous=set(verdict['ambiguous_frames'])
        expected=set(range(end-50,end+1))
        if verified & ambiguous or verified|ambiguous!=expected:raise ValueError('Review every intermediate frame exactly once')
        if sha256(batch_path.parent/case['identity_review'])!=case['identity_review_sha256']:raise ValueError('Review sheet changed')
        check_separation(end-50,end,case['frames'][0]['source_video_sha256'],artifact['examples'],batch['historical_frames'],TRAINING_POOLS)
        observations=[]
        for row in case['frames']:
            path=batch_path.parent/row['crop'] if row.get('crop') else None
            if (path is not None and sha256(path)!=row['crop_sha256']) or sha256(row['image'])!=row['image_sha256']:raise ValueError('Pixels changed')
            if path is None and row['frame'] in verified:raise ValueError('Verified frame has no crop')
            observations.append(dict(row,crop=np.asarray(load_image(path)) if path else None,association_verified=row['frame'] in verified))
        for name,offsets in CONFIGURATIONS.items():
            predictions[name].append(dict(decide(model,identity,end,observations,offsets),id=case['id'],group_id=case['group_id']))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'predictions.json',dict(schema='uniform-temporal-comparison-predictions-v1',
        status='committed_before_user_labels',committed_at_utc=datetime.now(timezone.utc).isoformat(),
        model_sha256=sha256(model_path),batch_sha256=sha256(batch_path),identity_review_sha256=sha256(review_path),
        configurations=CONFIGURATIONS,predictions=predictions,independent_sequences=len(batch['sequences']),
        temporal_groups=len({c['group_id'] for c in batch['sequences']}),
        code_sha256={str(p):sha256(p) for p in [Path(__file__),Path('src/team_identification/temporal_windows.py')]},
        evaluation_unit='one person sequence; correlated frames and cases within temporal groups',
        thresholds=dict(max_distance=model.max_distance,min_margin=model.min_margin),
        future_lookahead_s=0,computational_latency_s=None,warmup_s={k:-v[0]/25 for k,v in CONFIGURATIONS.items()}))


def score(prediction_path,answers_path,out):
    pred,answers=read_json(prediction_path),read_json(answers_path)
    if answers.get('batch_sha256')!=pred['batch_sha256'] or not answers.get('reviewer'):raise ValueError('Bound user answers required')
    if datetime.fromisoformat(answers['reviewed_at_utc'])<=datetime.fromisoformat(pred['committed_at_utc']):raise ValueError('Labels must follow frozen predictions')
    cases=pred['predictions']['four_consecutive'];labels=answers['labels']
    pending={str(i) for i in answers.get('pending_cases',[])}
    exclusions=answers.get('excluded_cases',{})
    expected={str(c['id']) for c in cases}
    if not isinstance(exclusions,dict) or not set(exclusions)<=expected or any(not isinstance(v,str) or not v.strip() for v in exclusions.values()):
        raise ValueError('Exclusions require known IDs and explicit reasons')
    if pending & (set(labels)|set(exclusions)) or set(labels)|pending|set(exclusions)!=expected or any(l not in CLASSES for l in labels.values()):
        raise ValueError('Each sequence requires a supported label, explicit exclusion or pending status')
    results={}
    for name,rows in pred['predictions'].items():
        def metrics(subset):
            counts=dict(correct=0,abstentions=0,wrong=0,unevaluable=0,excluded=0,pending=0)
            for r in subset:
                truth=labels.get(str(r['id']))
                key='pending' if str(r['id']) in pending else 'excluded' if str(r['id']) in exclusions else 'unevaluable' if truth=='Incerto' else 'abstentions' if r['label']=='Incerto' else 'correct' if r['label']==truth else 'wrong'
                counts[key]+=1
            n=len(subset)-counts['unevaluable']-counts['excluded']-counts['pending'];accepted=counts['correct']+counts['wrong']
            return dict(counts,total=len(subset),evaluated=n,accuracy=counts['correct']/n if n else None,coverage=accepted/n if n else None,accepted_accuracy=counts['correct']/accepted if accepted else None)
        results[name]=dict(metrics(rows),by_category={c:metrics([r for r in rows if labels.get(str(r['id']))==c]) for c in CLASSES},
            cases=[dict(id=r['id'],truth=labels.get(str(r['id'])),prediction=r['label'],reason=r['reason'],excluded=str(r['id']) in exclusions,exclusion_reason=exclusions.get(str(r['id']))) for r in rows])
    changes={name:[dict(id=a['id'],baseline=a['label'],candidate=b['label'],truth=labels.get(str(a['id']))) for a,b in zip(cases,rows) if a['label']!=b['label'] and str(a['id']) not in exclusions and str(a['id']) not in pending] for name,rows in pred['predictions'].items()}
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'comparison.json',dict(status='partial_awaiting_labels' if pending else 'complete',pending_cases=sorted(map(int,pending)),results=results,changed_cases=changes,independent_sequences=len(cases),
        temporal_groups=pred['temporal_groups'],excluded_cases=exclusions,exclusions_selected_after_labels=bool(exclusions),
        exclusion_note='User exclusions applied identically to all configurations; source predictions and visual labels preserved. Post-label subset is exploratory.',
        configurations=CONFIGURATIONS,predictions_sha256=sha256(prediction_path),answers_sha256=sha256(answers_path),
        limitations='Exploratory manual crops; shared temporal groups; not tracking, possession or full-match accuracy. Unknown truth excluded from accuracy. No retraining.',
        future_lookahead_s=0,computational_latency_s=None,warmup_s=pred['warmup_s']))


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    a=s.add_parser('prepare')
    for key in ['model','extraction','proposals','out']:a.add_argument('--'+key,type=Path,required=True)
    a=s.add_parser('review')
    for key in ['batch','review','out']:a.add_argument('--'+key,type=Path,required=True)
    a=s.add_parser('predict')
    for key in ['model','batch','review','out']:a.add_argument('--'+key,type=Path,required=True)
    a=s.add_parser('score')
    for key in ['predictions','answers','out']:a.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.model,a.extraction,a.proposals,a.out)
    elif a.action=='review':review_batch(a.batch,a.review,a.out)
    elif a.action=='predict':predict(a.model,a.batch,a.review,a.out)
    else:score(a.predictions,a.answers,a.out)

if __name__=='__main__':main()
