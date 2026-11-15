"""Bounded, baseline CPU SAM person proposals; never auto-approve recoveries."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from src.team_identification.pilot import sha256
from src.team_identification.manifests import write_json


def worker(a):
    import numpy as np
    from PIL import Image,ImageDraw
    from src.team_identification.pilot import Sam3
    before=sha256(a.manifest);scene=json.loads(a.manifest.read_text())
    image_path=Path(scene['image']);expected=scene['image_sha256']
    if sha256(image_path)!=expected:raise ValueError('Image changed')
    image=Image.open(image_path).convert('RGB');model=Sam3(a.checkpoint,a.checkpoint_sha256,device='cpu')
    masks,scores=model.segment(image)
    preview=image.copy();draw=ImageDraw.Draw(preview);rows=[]
    for i,(mask,score) in enumerate(zip(masks,scores)):
        ys,xs=np.where(mask)
        if not len(xs):continue
        box=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        name=f'person-{i:03d}';path=a.out/(name+'-mask.png')
        Image.fromarray(mask.astype('uint8')*255).save(path)
        draw.rectangle(box,outline='magenta',width=2);draw.text((box[0],box[1]-12),name,fill='yellow')
        rows.append(dict(observation_id=name,frame=scene['frame'],source_frame=scene['frame'],
            available_at_frame=scene['frame'],image=str(image_path.resolve()),image_sha256=expected,
            mask=str(path.resolve()),mask_sha256=sha256(path),box=box,sam_score=float(score),
            sam3_executed=True,mask_reviewed=False,feet_verified=False,homography_verified=False,
            projection_reviewed=False,team=None,pitch_xy=None,reviewer=None,
            reason='automatic_proposal_pending_review'))
    preview.save(a.out/'proposals.png')
    if sha256(a.manifest)!=before or sha256(image_path)!=expected:raise ValueError('Input changed during inference')
    write_json(a.out/'proposals.json',dict(schema='tactical-reviewed-evidence-v1',timeline='tactical',
        same_source_video_frame=scene['same_source_video_frame'],source_video_sha256=scene['source_video_sha256'],
        source_manifest_sha256=before,checkpoint_sha256=a.checkpoint_sha256,
        code_sha256=sha256(__file__),pilot_code_sha256=sha256('src/team_identification/pilot.py'),
        environment=model.environment,recoveries=rows,game_evidence=[]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--checkpoint-sha256',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.worker:return worker(a)
    import psutil
    a.out.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','scripts.run_sam3_recovery','--worker','--manifest',str(a.manifest),
             '--checkpoint',str(a.checkpoint),'--checkpoint-sha256',a.checkpoint_sha256,'--out',str(a.out)]
    start=time.monotonic();peak=0;status='complete'
    with (a.out/'worker.log').open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);monitor=psutil.Process(proc.pid)
        while proc.poll() is None:
            try:peak=max(peak,monitor.memory_info().rss)
            except psutil.NoSuchProcess:break
            if time.monotonic()-start>=60 or peak>7_000_000_000:
                status='time_limit' if time.monotonic()-start>=60 else 'memory_limit';proc.kill();break
            time.sleep(.05)
        proc.wait()
    if proc.returncode and status=='complete':status='worker_failed'
    write_json(a.out/'resources.json',dict(status=status,wall_seconds=time.monotonic()-start,
        peak_sampled_rss_bytes=peak,time_limit_seconds=60,rss_limit_bytes=7_000_000_000,returncode=proc.returncode))
    if status!='complete':raise RuntimeError('Bounded SAM recovery: '+status)


if __name__=='__main__':main()
