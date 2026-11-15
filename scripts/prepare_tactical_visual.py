"""Extract exact tactical frames and project only contemporaneous observed ATD.

This prepares reviewable inputs, not approved boxes, goalkeeper roles or control.
The original video is required: screen recordings and broadcast aren't substituted.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from src.team_identification.manifests import write_json
from src.team_identification.pilot import sha256


def video_info(video):
    data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],text=True))
    if len(data.get('streams',[]))!=1:raise ValueError('One video stream required')
    return data['streams'][0]


def extract_exact(video,frames,out,*,expected_frames=199319,expected_fps=25,expected_size=None):
    from fractions import Fraction
    frames=list(frames)
    if not frames or len(frames)!=len(set(frames)) or any(type(f) is not int or not 0<=f<expected_frames for f in frames):
        raise ValueError('Unique original frame indices within video required')
    if not video.is_file():raise FileNotFoundError('Original tactical video not materialized: '+str(video))
    if expected_size is not None and video.stat().st_size!=expected_size:
        raise ValueError('Video size differs from original Drive metadata')
    info=video_info(video)
    if int(info.get('nb_frames',-1))!=expected_frames or Fraction(info['r_frame_rate'])!=Fraction(str(expected_fps)):
        raise ValueError('Video timeline differs from tactical source; no resampling allowed')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source_hash=sha256(video)
    frames=sorted(frames)
    runs=[]
    for f in frames:
        if runs and f==runs[-1][1]+1:runs[-1][1]=f
        else:runs.append([f,f])
    select='+'.join(f'eq(n\\,{lo})' if lo==hi else f'between(n\\,{lo}\\,{hi})' for lo,hi in runs)
    # Decode on the original counter: seeking by recording seconds is not used.
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(video),
        '-vf','select='+select,'-fps_mode','passthrough','-frames:v',str(len(frames)),
        str(out/'selected-%06d.png')],check=True)
    rows=[]
    for i,f in enumerate(frames,1):
        path=out/f'selected-{i:06d}.png'
        if not path.is_file():raise ValueError('Incomplete exact-frame extraction')
        renamed=out/f'{f:08d}.png';path.rename(renamed)
        with Image.open(renamed) as image:
            if image.size!=(info['width'],info['height']):raise ValueError('Extracted image dimensions changed')
        rows.append(dict(frame=f,image=renamed.name,image_sha256=sha256(renamed),
            source_video_sha256=source_hash,timeline='tactical',same_source_video_frame=True,
            extraction='ffmpeg original decoded frame counter; no scaling or fps conversion'))
    write_json(out/'extraction.json',dict(schema='tactical-frame-extraction-v1',video=str(video.resolve()),
        source_video_sha256=source_hash,video_info=info,frames=rows,script_sha256=sha256(__file__)))
    return rows


def homographies(path,frames):
    required=set(frames);rows={};previous=-1
    with path.open(newline='') as stream:
        for raw in csv.DictReader(stream):
            f=int(raw['frame'])
            if f!=previous+1:raise ValueError('Homography frames must be contiguous from zero')
            previous=f
            if f not in required:continue
            matrix=np.array([float(raw[f'h{i}{j}']) for i in range(3) for j in range(3)]).reshape(3,3)
            error=float(raw['reprojection_error'])
            if raw['valid']!='1' or not np.isfinite(matrix).all() or not math.isfinite(error):
                rows[f]=dict(valid=False,matrix=None,reprojection_error=None)
            else:rows[f]=dict(valid=True,matrix=matrix,reprojection_error=error)
    if set(rows)!=required:raise ValueError('Homography lacks requested frames')
    return rows


def project_observed(point,matrix,pitch,frame):
    if (point['source']!='observed' or point.get('quality_flags') or
            point.get('available_at_frame',frame)>frame):return None
    q=matrix@np.array([point['x_m']/pitch[0],point['y_m']/pitch[1],1.])
    if not np.isfinite(q).all() or abs(q[2])<1e-9:return None
    xy=q[:2]/q[2]
    return [float(v) for v in xy] if np.isfinite(xy).all() else None


def prepare(video,frames,out,data_dir):
    from itertools import islice
    from games import asset_paths
    from src.ball_events.inputs import atd
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    paths=asset_paths(data_dir=data_dir)
    hs=homographies(Path(paths['homography']),frames)
    meta,factory,*_=atd(data_dir)
    if any(not r['valid'] for r in hs.values()):raise ValueError('Requested frame has no valid homography')
    rows=extract_exact(video,frames,out,expected_frames=meta['n_frames'],expected_fps=meta['fps'],expected_size=5222307334)
    targets={r['frame']:r for r in rows}
    for envelope in islice(factory(),max(frames)+1):
        f=envelope['frame']
        if f not in targets:continue
        entry=targets[f];matrix=hs[f]['matrix']
        image=Image.open(out/entry['image']).convert('RGB');draw=ImageDraw.Draw(image)
        candidates=[]
        for i,p in enumerate(envelope['players']):
            xy=project_observed(p,matrix,meta['pitch_m'],f)
            if xy is None or not (0<=xy[0]<image.width and 0<=xy[1]<image.height):continue
            # No team labels or persistent identities enter visual candidate review.
            name=f'appearance-{len(candidates)+1:02d}'
            candidates.append(dict(label=name,foot_pixel=xy,box=None,box_reviewed=False))
            x,y=xy;draw.ellipse((x-3,y-3,x+3,y+3),fill='white',outline='black');draw.text((x+5,y),name,fill='yellow')
        ball=project_observed(envelope['ball'],matrix,meta['pitch_m'],f)
        if ball and not (0<=ball[0]<image.width and 0<=ball[1]<image.height):ball=None
        if ball:
            x,y=ball;draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=2)
        image.save(out/f'{f:08d}-projection-review.png')
        half=1 if KICKOFF_VIDEO_FRAMES['1st Half']<=f<KICKOFF_VIDEO_FRAMES['2nd Half'] else (2 if f>=KICKOFF_VIDEO_FRAMES['2nd Half'] else None)
        write_json(out/f'{f:08d}-scene-draft.json',dict(entry,image=str((out/entry['image']).resolve()),
            source_video=str(video.resolve()),half=half,reviewed=False,reviewer=None,box=None,
            candidates=candidates,context=dict(half=half,goalkeeper_candidate=False,distinctive_kit=False,
                near_defended_goal=False,period_verified=False,pitch_end_verified=False,orientation_verified=False,
                play_origin_verified=False,pitch_end=None,view=None,kit_family=None,notes='Pending RGB and projection review'),
            held_ball_reviewed=False,held_ball_reviewer=None,
            ball_observation=dict(frame=f,image_sha256=entry['image_sha256'],source='observed' if ball else 'missing',
                xy=ball,available_at_frame=f,producer='same-frame observed ATD + homography',
                homography_sha256=sha256(paths['homography']),atd_raw_sha256=sha256(paths['atd_raw']),
                homography_valid=True,reprojection_error=hs[f]['reprojection_error'],projection_reviewed=False),
            homography=[float(v) for v in matrix.ravel()]))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,default=Path('data/tactical/BAR-ATH_tactical.mp4'))
    p.add_argument('--frames',type=int,nargs='+',required=True);p.add_argument('--data-dir',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();prepare(a.video,a.frames,a.out,a.data_dir)


if __name__=='__main__':main()
