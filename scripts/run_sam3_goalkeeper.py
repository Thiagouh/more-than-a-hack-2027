"""Bounded CPU SAM 3 -> goalkeeper context -> visual control observation.

Reviewed scene context is explicit. Held-ball control is manual evidence, not a
ball detector. Tactical use requires a same-source, frame-bound image manifest.
"""
import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from src.team_identification.manifests import read_json,write_json,load_image
from src.team_identification.pilot import Sam3,sha256,torso_evidence,reviewed_box_quality
from src.team_identification.goalkeeper_context import infer_goalkeeper_team
from src.team_identification.visual_geometry import projected_ball_contact,sam_ball_contact


def worker(args):
    manifest_before=sha256(args.manifest);policy_before=sha256(args.policy)
    policy=read_json(args.policy)
    code_paths=[Path(__file__).resolve(),Path(__file__).resolve().parents[1]/'src/team_identification/visual_geometry.py',
        Path(__file__).resolve().parents[1]/'src/team_identification/pilot.py',
        Path(__file__).resolve().parents[1]/'src/team_identification/goalkeeper_context.py']
    code_before={str(p):sha256(p) for p in code_paths}
    scene=read_json(args.manifest);image_path=(args.manifest.parent/scene['image']).resolve()
    if sha256(image_path)!=scene['image_sha256']:raise ValueError('Scene image changed')
    if scene.get('timeline') not in ('external','tactical'):raise ValueError('Explicit timeline required')
    if scene.get('timeline')=='tactical' and (scene.get('same_source_video_frame') is not True or
            not scene.get('source_video_sha256')):
        raise ValueError('Tactical inference requires explicit same-video frame provenance')
    if type(scene['frame']) is not int or scene['frame']<0:raise ValueError('Invalid source frame')
    image=load_image(image_path);box=np.asarray(scene['box'],float)
    if box.shape!=(4,) or not np.isfinite(box).all():raise ValueError('Invalid reviewed box')
    x,y,X,Y=box
    if not (0<=x<X<=image.width and 0<=y<Y<=image.height):raise ValueError('Box outside image')
    side=3*max(X-x,Y-y);cx,cy=(x+X)/2,(y+Y)/2
    l,t=max(0,int(cx-side/2)),max(0,int(cy-side/2))
    r,b=min(image.width,int(cx+side/2)),min(image.height,int(cy+side/2))
    crop=image.crop((l,t,r,b));local_box=[x-l,y-t,X-l,Y-t]
    started=time.monotonic();sam=Sam3(args.checkpoint,args.checkpoint_sha256,device='cpu')
    loaded=time.monotonic()
    ball_masks=None;ball_scores=None
    if scene.get('detect_ball') is True:
        # Reuse one image backbone and one model for both prompts.
        state=sam.processor.set_image(crop)
        sam.processor.set_text_prompt(prompt='person',state=state)
        a,b,c,d=local_box
        result=sam.processor.add_geometric_prompt(box=[(a+c)/(2*crop.width),(b+d)/(2*crop.height),
            (c-a)/crop.width,(d-b)/crop.height],label=True,state=state)
        candidates=result['masks'].detach().cpu().numpy()[:,0]
        scores=result['scores'].detach().cpu().numpy()
        sam.processor.reset_all_prompts(state)
        result=sam.processor.set_text_prompt(prompt='soccer ball',state=state)
        ball_masks=result['masks'].detach().cpu().numpy()[:,0]
        ball_scores=result['scores'].detach().cpu().numpy()
    else:
        candidates,scores=sam.segment(crop,local_box)
    # The same baseline candidate selection as Sam3.segment_boxes; no threshold sweep.
    region=np.zeros((crop.height,crop.width),bool)
    ax,ay,bx,by=map(int,local_box);region[ay:by,ax:bx]=True
    index=max(range(len(scores)),key=lambda i:(int((candidates[i]&region).sum())/max(1,int((candidates[i]|region).sum())),float(scores[i]))) if len(scores) else None
    mask=candidates[index] if index is not None else np.zeros_like(region)
    core,quality,reason=torso_evidence(crop,mask,return_mask=True)
    box_quality,box_reason=reviewed_box_quality(mask,local_box)
    reason=box_reason or reason
    core=np.zeros_like(mask) if core is None else core
    pixels=np.asarray(crop)[core].astype(float)
    if len(pixels):
        red,green,blue=pixels.T
        valid=(green>1.12*red)&(green>=.75*blue)&(pixels.max(axis=1)>50)&((pixels.max(axis=1)-pixels.min(axis=1))>.15*pixels.max(axis=1))
        kit_fraction=float(valid.mean())
    else:kit_fraction=0.
    evidence=dict(scene['context'])
    # Manual context alone cannot promote a failed SAM silhouette or non-green/teal core.
    evidence['goalkeeper_candidate']=evidence.get('goalkeeper_candidate') is True and reason is None and mask.any().item()
    evidence['distinctive_kit']=evidence.get('distinctive_kit') is True and kit_fraction>=.60
    decision=infer_goalkeeper_team(policy,evidence)
    Image.fromarray(mask.astype('uint8')*255).save(args.out/'sam-mask-crop.png')
    Image.fromarray(core.astype('uint8')*255).save(args.out/'sam-torso-crop.png')
    crop.save(args.out/'context-crop.png')
    crop_rgb=np.asarray(crop).copy();crop_rgb[mask]=(.5*crop_rgb[mask]+.5*np.array([255,0,255])).astype('uint8')
    Image.fromarray(crop_rgb).save(args.out/'sam-review.png')
    reviewer=scene.get('held_ball_reviewer','')
    manual_control=scene.get('held_ball_reviewed') is True and isinstance(reviewer,str) and bool(reviewer.strip())
    contact=projected_ball_contact(mask,scene,(l,t))
    ball_contact=(sam_ball_contact(mask,ball_masks,ball_scores) if ball_masks is not None else
        dict(accepted=False,reason='sam_ball_not_requested',producer='SAM3 soccer ball'))
    if ball_contact['accepted']:
        ball_file=args.out/'sam-ball-crop.png'
        Image.fromarray(ball_masks[ball_contact['candidate_index']].astype('uint8')*255).save(ball_file)
        ball_contact.update(mask_file=str(ball_file.resolve()),mask_sha256=sha256(ball_file))
    observations=[]
    if decision['team'] is not None:
        observations.append(dict(observation_id='sam3-gk-'+scene['image_sha256'][:16],frame=scene['frame'],
            available_at_frame=scene['frame'],timeline=scene['timeline'],image_sha256=scene['image_sha256'],
            team=decision['team'],role='goalkeeper',classification_method='sam3_goalkeeper_context',
            sam3_executed=True,mask_quality_ok=reason is None,context_valid=True,
            control_evidence='reviewed_held_ball' if manual_control else ('ball_body_proximity' if contact['accepted'] or ball_contact['accepted'] else 'none'),
            ball_control_observed=manual_control or contact['accepted'] or ball_contact['accepted'],reviewer=reviewer))
    if (sha256(args.manifest)!=manifest_before or sha256(args.policy)!=policy_before
            or sha256(image_path)!=scene['image_sha256']
            or any(sha256(path)!=digest for path,digest in code_before.items())):
        raise RuntimeError('Scene, policy, RGB or inference code changed during SAM execution')
    write_json(args.out/'visual-observations.json',dict(schema='sam3-visual-control-v1',frame=scene['frame'],timeline=scene['timeline'],
        same_source_video_frame=scene.get('same_source_video_frame',False),source_video_sha256=scene.get('source_video_sha256'),
        manifest_sha256=manifest_before,policy_sha256=policy_before,code_sha256=code_before,image=str(image_path),image_sha256=scene['image_sha256'],
        sam3_executed=True,checkpoint_sha256=args.checkpoint_sha256,environment=sam.environment,
        model_load_seconds=loaded-started,inference_seconds=time.monotonic()-loaded,
        quality=quality,box_quality=box_quality,quality_reason=reason,kit_fraction=kit_fraction,
        mask_file=str((args.out/'sam-mask-crop.png').resolve()),mask_sha256=sha256(args.out/'sam-mask-crop.png'),
        context_crop_origin=[l,t],mask_reviewed=False,mask_reviewer=None,
        kit_threshold=.60,kit_heuristic_calibrated=False,decision=decision,
        held_ball_manual_review=manual_control,projected_ball_contact=contact,sam_ball_contact=ball_contact,observations=observations,
        scene_context_reviewed=True,script_sha256=sha256(__file__)))


def bounded(args):
    import psutil
    args.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','scripts.run_sam3_goalkeeper','--worker','--manifest',str(args.manifest),
        '--checkpoint',str(args.checkpoint),'--checkpoint-sha256',args.checkpoint_sha256,
        '--policy',str(args.policy),'--out',str(args.out)]
    started=time.monotonic();peak=0;status='complete'
    with (args.out/'worker.log').open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);monitored=psutil.Process(p.pid)
        while p.poll() is None:
            try:peak=max(peak,monitored.memory_info().rss)
            except psutil.NoSuchProcess:break
            elapsed=time.monotonic()-started
            if elapsed>=60 or peak>7_000_000_000:
                status='time_limit' if elapsed>=60 else 'memory_limit';p.kill();break
            time.sleep(.05)
        p.wait()
    if p.returncode and status=='complete':status='worker_failed'
    write_json(args.out/'resources.json',dict(status=status,wall_seconds=time.monotonic()-started,
        peak_sampled_rss_bytes=peak,time_limit_seconds=60,rss_limit_bytes=7_000_000_000,returncode=p.returncode))
    if status!='complete':raise RuntimeError('Bounded SAM run: '+status)
    print(args.out/'visual-observations.json')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--policy',type=Path,default=Path('configs/team-pilot-goalkeeper-context.json'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true')
    args=p.parse_args();worker(args) if args.worker else bounded(args)


if __name__=='__main__':main()
