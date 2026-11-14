"""Reproducible synthetic full-length benchmark. No match assets, network or ground truth.

Runs detector and file pipeline in separate processes. CSV generation time is separate;
RSS is process peak (including imports/setup). Synthetic input files are removed afterward.
"""
import argparse
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.ball_events import Detector
from src.ball_events.artifacts import run,write_json
from src.ball_events.demo import metadata,point
from src.ball_events.inputs import standardized


def synthetic(n,players,pattern='churn'):
    for f in range(n):
        generation=f//250  # churn all IDs every ten seconds at 25 Hz
        objects=[point(f,10.+(i%11)*7.,y=10.+(i//11)*30.,track_id=f'g{generation}-p{i}',team='H' if i<players//2 else 'A') for i in range(players)]
        phase=f%40
        x=10. if phase<8 else 10.+.4*(phase-7) if phase<16 else 13.2 if phase<24 else max(10.,13.2-.4*(phase-23)) if phase<32 else 10.
        ball=point(f,x if pattern=='events' else 10.,y=10.,source='observed' if pattern=='events' or f%2==0 else 'estimated')
        yield dict(frame=f,ball=ball,players=objects)


def inputs(folder,n,players,pattern):
    m=metadata(n);m['dataset_id']='benchmark-churn-alternating-v2'
    m['files']={'ball':'ball.csv','players':'players.csv'}
    fields=['frame','x_m','y_m','source','producer','quality_flags']
    with (folder/'ball.csv').open('w',newline='') as bh,(folder/'players.csv').open('w',newline='') as ph:
        bw=csv.DictWriter(bh,fieldnames=fields,lineterminator='\n');pw=csv.DictWriter(ph,fieldnames=fields+['track_id','team'],lineterminator='\n')
        bw.writeheader();pw.writeheader()
        for frame in synthetic(n,players,pattern):
            for p,w in [(frame['ball'],bw),*((p,pw) for p in frame['players'])]:
                w.writerow(dict(p,quality_flags='[]'))
    write_json(folder/'manifest.json',m)
    return folder/'manifest.json'


def worker(args):
    n,players=args.frames,args.players
    generation=0.; temporary_peak=0
    if args.mode=='detector':
        d=Detector(metadata(n));begin=time.perf_counter()
        for frame in synthetic(n,players,args.pattern): d.feed(frame)
        q=d.finish();elapsed=time.perf_counter()-begin
        count=sum(q['events'].values());ids=len(d.track_teams)
    else:
        from unittest.mock import patch
        from src.ball_events.storage import RunStore
        original=RunStore.close
        def measure(store):
            nonlocal temporary_peak
            path=Path(store.db.execute('PRAGMA database_list').fetchone()[2])
            temporary_peak=max(temporary_peak,sum(p.stat().st_size for p in path.parent.glob('run.sqlite*')))
            original(store)
        with tempfile.TemporaryDirectory(prefix='frente5-benchmark-') as temp:
            folder=Path(temp); begin=time.perf_counter();path=inputs(folder,n,players,args.pattern)
            generation=time.perf_counter()-begin
            m,f,h,files,status=standardized(path);begin=time.perf_counter()
            with patch('src.ball_events.artifacts.RunStore.close',measure):
                run(m,f,h,files,args.out/'pipeline',hypothesis='Measure synthetic full file pipeline with ID churn and alternating sources; no accuracy claim',hints_status=status)
            elapsed=time.perf_counter()-begin
        q=json.loads((args.out/'pipeline/quality.json').read_text());count=sum(q['events'].values());ids=None
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':rss*=1024
    result=dict(mode=args.mode,n_frames=n,players_per_frame=players,pattern=args.pattern,pattern_definition='all IDs change every 250 frames; churn alternates ball source; events uses observed ball movement/holds',
                elapsed_s=elapsed,csv_generation_s=generation,peak_process_rss_bytes=rss,
                temporary_sqlite_and_journal_bytes=temporary_peak,event_count=count,retained_registry_ids=ids,
                output_bytes=sum(p.stat().st_size for p in (args.out/'pipeline').rglob('*') if p.is_file()) if args.mode=='pipeline' else 0,
                limitation='Synthetic computational measurement, not match accuracy; RSS includes process setup.')
    write_json(args.out/(args.mode+'.json'),result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=199319);parser.add_argument('--players',type=int,default=22)
    parser.add_argument('--pattern',choices=['churn','events'],default='churn')
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--mode',choices=['detector','pipeline'])
    args=parser.parse_args()
    if args.frames<1 or args.players<2:parser.error('frames >= 1 and players >= 2 required')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode:worker(args)
    else:
        for mode in ('detector','pipeline'):
            subprocess.run([sys.executable,__file__,'--frames',str(args.frames),'--players',str(args.players),'--out',str(args.out),'--mode',mode,'--pattern',args.pattern],check=True)
        write_json(args.out/'summary.json',{mode:json.loads((args.out/(mode+'.json')).read_text()) for mode in ('detector','pipeline')})
        print('Synthetic scale reports saved. No match quality measured.')


if __name__=='__main__':main()
