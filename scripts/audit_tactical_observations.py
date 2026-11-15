"""Compare same-frame RGB, raw ATD, projections and rendering eligibility."""
import argparse
import json
from itertools import islice
from pathlib import Path
from PIL import Image,ImageDraw
from src.ball_events.inputs import atd
from src.ball_events.artifacts import sha,write_json
from src.team_possession.render_observations import render_status
from scripts.prepare_tactical_visual import homographies,project_observed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extractions',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    entries={}
    for folder in a.extractions:
        ex=json.loads((folder/'extraction.json').read_text())
        for r in ex['frames']:entries[r['frame']]=(folder,r,ex['source_video_sha256'])
    meta,factory,*_=atd('data');hpath=Path('data/tactical/BAR-ATH_tactical_homography.csv');hs=homographies(hpath,entries)
    results=[]
    for env in islice(factory(),max(entries)+1):
        f=env['frame']
        if f not in entries:continue
        folder,entry,video_hash=entries[f];path=folder/entry['image']
        if sha(path)!=entry['image_sha256']:raise ValueError('Extracted source changed')
        image=Image.open(path).convert('RGB');draw=ImageDraw.Draw(image);points=[]
        for pt in env['players']+[dict(env['ball'],track_id='ball',team=None)]:
            status=render_status(pt,f,meta['pitch_m']);xy=project_observed(pt,hs[f]['matrix'],meta['pitch_m'],f) if hs[f]['valid'] else None
            row=dict(pt,render_status=status,projected_pixel=xy);points.append(row)
            if xy and 0<=xy[0]<image.width and 0<=xy[1]<image.height:
                color={'H':'cyan','A':'orange',None:'red'}[pt['team']];x,y=xy
                draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                draw.text((x+5,y+2),str(pt['team'] or '')+pt['track_id'],fill=color)
        name=f'{f:08d}-team-projection.png';image.save(a.out/name)
        results.append(dict(frame=f,image=str(path.resolve()),image_sha256=sha(path),source_video_sha256=video_hash,
            homography_valid=hs[f]['valid'],reprojection_error=hs[f]['reprojection_error'],
            overlay=name,observations=points,raw_observed_players=sum(p['source']=='observed' for p in env['players']),
            omitted_valid_in_pitch=[p['track_id'] for p in points if p['source']=='observed'
                and 0<=p['x_m']<=meta['pitch_m'][0] and 0<=p['y_m']<=meta['pitch_m'][1]
                and p['render_status'] not in ('drawn','drawn_with_quality_warning')]))
    write_json(a.out/'audit.json',dict(timeline='tactical',atd_raw_sha256=sha('data/tactical/BAR-ATH_tactical_atd.txt'),
        homography_sha256=sha(hpath),code_sha256=sha(__file__),frames=results,
        limitations=['projection of elevated ball is not a verified pixel observation',
                    'team labels from ATD; not visual classifications','visual inspection required to diagnose causes']))


if __name__=='__main__':main()
