"""Source-bound green-keeper association to existing ATD points, no invented XY.

Restricted to the reviewed first-half home-goal window. RGB threshold is an
uncalibrated appearance cue; team comes from reviewed goal/period context.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from itertools import islice
import numpy as np
from src.ball_events.inputs import atd
from src.ball_events.artifacts import sha,write_json
from scripts.prepare_tactical_visual import homographies,project_observed


def associate(rgb,matrix,pitch,points):
    # Field ROI is restricted to the reviewed home goal, not the image's left half.
    corners=[]
    for x,y in ((0,18),(22,18),(22,50),(0,50)):
        q=matrix@np.array([x/pitch[0],y/pitch[1],1.]);corners.append(q[:2]/q[2])
    corners=np.asarray(corners)
    l=max(0,int(corners[:,0].min()));r=min(rgb.shape[1],int(corners[:,0].max())+1)
    t=max(0,int(corners[:,1].min()));b=min(rgb.shape[0],int(corners[:,1].max())+1)
    if l>=r or t>=b:return dict(selected=None,reason='goal_roi_outside_image',components=[])
    crop=rgb[t:b,l:r];red=crop[:,:,0].astype(float);green=crop[:,:,1].astype(float);blue=crop[:,:,2].astype(float)
    cue=(green>=150)&(green>=1.25*red)&(green>=1.15*blue)&(green-red>=45)
    ys,xs=np.where(cue);remaining=set(zip(xs.tolist(),ys.tolist()));components=[]
    while remaining:
        seed=remaining.pop();stack=[seed];component=[seed]
        while stack:
            x,y=stack.pop()
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)):
                q=(x+dx,y+dy)
                if q in remaining:remaining.remove(q);stack.append(q);component.append(q)
        if len(component)<8:continue
        xy=np.asarray(component);x0,y0=xy.min(axis=0);x1,y1=xy.max(axis=0)+1;w=x1-x0;h=y1-y0
        if not (2<=w<=50 and 20<=h<=90 and w/h<=1.):continue
        components.append(dict(box=[int(x0+l),int(y0+t),int(x1+l),int(y1+t)],pixels=len(component)))
    choices=[]
    for component in components:
        x0,y0,x1,y1=component['box'];height=y1-y0;width=x1-x0
        for p in points:
            xy=project_observed(p,matrix,pitch,p['frame'])
            if xy is None:continue
            if not x0-.4*width<=xy[0]<=x1+.4*width or not y1-.15*height<=xy[1]<=y1+.3*height+3:continue
            cost=float(np.hypot(xy[0]-(x0+x1)/2,xy[1]-(y1+2)))
            choices.append(dict(track_id=p['track_id'],raw_xy=[p['x_m'],p['y_m']],pixel=xy,box=component['box'],cost_px=cost))
    choices.sort(key=lambda r:(r['cost_px'],r['track_id']))
    if not choices:return dict(selected=None,reason='no_existing_point_matches_kit_feet',components=components)
    if len(choices)>1 and choices[1]['track_id']!=choices[0]['track_id'] and choices[1]['cost_px']-choices[0]['cost_px']<2:
        return dict(selected=None,reason='ambiguous_raw_point_association',components=components)
    return dict(selected=choices[0],reason='reviewed_goal_context_rgb_feet_association',components=components)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);video=Path('data/tactical/BAR-ATH_tactical.mp4');start,end=11185,14185
    meta,fac,*_=atd('data');hs=homographies(Path('data/tactical/BAR-ATH_tactical_homography.csv'),range(start,end))
    points={r['frame']:[p for p in r['players'] if p['track_id'] in ('24','46')] for r in islice(fac(),end) if r['frame']>=start}
    command=['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),'-vf',f'select=between(n\\,{start}\\,{end-1})',
        '-fps_mode','passthrough','-frames:v',str(end-start),'-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    diagnostics=[];assignments=[];before=sha(video);code_hash=sha(__file__)
    with (a.out/'decode.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log)
        try:
            for f in range(start,end):
                data=proc.stdout.read(1920*1080*3)
                if len(data)!=1920*1080*3:raise ValueError('Incomplete original frame decode')
                rgb=np.frombuffer(data,dtype=np.uint8).reshape(1080,1920,3)
                result=associate(rgb,hs[f]['matrix'],meta['pitch_m'],points[f]) if hs[f]['valid'] else dict(selected=None,reason='invalid_homography',components=[])
                diagnostics.append(dict(frame=f,**result))
                if result['selected']:
                    assignments.append(dict(frame=f,available_at_frame=f,team='H',role='goalkeeper',**result['selected'],
                        image_rgb_sha256=hashlib.sha256(data).hexdigest(),quality='heuristic_rgb_cue_plus_reviewed_goal_context_not_calibrated'))
            if proc.stdout.read(1):raise ValueError('Unexpected extra source frame')
        finally:proc.stdout.close();returncode=proc.wait()
    if returncode:raise RuntimeError('Original frame decoder failed')
    if sha(video)!=before or sha(__file__)!=code_hash:raise ValueError('Source/code changed during association')
    evidence=[]
    for folder,fs in [('outputs/possession-v4-audit-source',[11518,11689,11873,11918]),('outputs/context25-goalkeeper-source',[12400,12650])]:
        for f in fs:
            image=Path(folder)/f'{f:08d}.png';evidence.append(dict(frame=f,image=str(image.resolve()),image_sha256=sha(image)))
    policy=Path('configs/team-pilot-goalkeeper-context.json')
    write_json(a.out/'tracking-review.json',dict(schema='reviewed-tracking-quality-v1',timeline='tactical',
        reviewer='Codex RGB/projection review and user confirmed defended end; offline, no human certification',
        source_video_sha256=before,atd_raw_sha256=sha('data/tactical/BAR-ATH_tactical_atd.txt'),
        homography_sha256=sha('data/tactical/BAR-ATH_tactical_homography.csv'),goalkeeper_policy=str(policy.resolve()),goalkeeper_policy_sha256=sha(policy),
        algorithm_sha256=code_hash,thresholds=dict(green_min=150,green_red_ratio=1.25,green_blue_ratio=1.15,green_red_difference=45,minimum_component_pixels=8,minimum_component_height=20),
        evidence=evidence,frame_assignments=assignments,rules=[dict(rule_id='reviewed-home-goalkeeper-role',track_id='24',
            start_frame=start,end_frame_exclusive=end,available_at_frame=start,action='goalkeeper_role',team='H',
            reason='Reviewed H24 home goalkeeper role; this does not certify the raw position of every frame.'),dict(rule_id='reviewed-A46-left-goal-conflict',track_id='46',
            start_frame=start,end_frame_exclusive=end,available_at_frame=start,action='reject_observation',x_max_m=meta['pitch_m'][0]/2,
            reason='Reviewed first-half home-goal tracklet: A46 is absent/spurious or coincides with the green home goalkeeper, not evidence of an Athletic player.'),
            dict(rule_id='reviewed-away-keeper-right',track_id='46',start_frame=13679,end_frame_exclusive=end,available_at_frame=13679,
            action='goalkeeper_role',team='A',x_min_m=meta['pitch_m'][0]/2,reason='Away goalkeeper return on defended right end; original team retained.')],
        limitations=['Only this reviewed first-half window','No new coordinates: selects existing observed ATD point or abstains',
        'Green cue can fail; no automatic uniform reference approval','Offline reviewed availability is not live inference latency']))
    write_json(a.out/'association-diagnostics.json',diagnostics)
    from collections import Counter
    print(json.dumps(dict(frames=end-start,assignments=len(assignments),selected_tracks=dict(Counter(r['track_id'] for r in assignments)),abstentions=dict(Counter(r['reason'] for r in diagnostics if r['selected'] is None))),indent=2))


if __name__=='__main__':main()
