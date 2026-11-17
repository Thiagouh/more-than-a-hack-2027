"""Build La MasIA's offline submission with Frente 4's frozen football rules.

No ground truth, trained match classifier, reviewed corrections, or video input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from games import require, GAMES, GAME
from evaluation import possession_metrics as pm
from src.frente4.pipeline import run


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--out-dir',type=Path,required=True)
    parser.add_argument('--ball-quality',action='store_true',help='Explicit physics-only filtering experiment; no learned model')
    control=parser.add_mutually_exclusive_group()
    control.add_argument('--control-events',type=Path,help='Hash-bound raw-ATD Frente 5 bundle')
    control.add_argument('--include-control-events',action='store_true',help='Regenerate and bind automatic Frente 5 events')
    args=parser.parse_args()
    if args.out_dir.exists():raise ValueError('Use a new output directory')
    inputs=require(['atd_meta','atd_raw','smart_tagging'],GAME,args.data_dir)
    config=Path('configs/frente4_v1.json')
    code=[Path(__file__),Path('games.py'),Path('starter/load_smart_tagging.py'),
          *sorted(Path('src/frente4').glob('*.py'))]
    if args.control_events or args.include_control_events:
        code.extend([Path('configs/frente5_v1.json'),Path('starter/load_atd.py'),*sorted(Path('src/ball_events').glob('*.py'))])
    source_hashes={str(p):sha(p) for p in [*map(Path,inputs),config,*code]}
    method='frente4-physics-quality-v2' if args.ball_quality else 'frente4-frozen-offline-rules'
    if args.control_events or args.include_control_events:method+='-supported-events-v1'
    provenance=dict(status='started',method=method,
        ball_quality_enabled=args.ball_quality,
        control_events_enabled=bool(args.control_events or args.include_control_events),
        team='La MasIA',upstream_commit='288d5a0a1aa37324850436188655ce777263b056',
        upstream_pr='https://github.com/Thiagouh/more-than-a-hack-2027/pull/1',
        ground_truth_used=False,training_on_provided_game=False,human_corrections=False,
        configuration=json.loads(config.read_text()),source_sha256=source_hashes,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    args.out_dir.mkdir(parents=True)
    manifest=args.out_dir/'provenance.json'
    manifest.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    started=time.perf_counter()
    event_bundle=args.control_events
    if args.include_control_events:
        event_bundle=args.out_dir/'automatic-control-events'
        subprocess.run([sys.executable,'-m','src.ball_events','detect','--source','atd',
            '--data-dir',str(args.data_dir),'--config','configs/frente5_v1.json',
            '--hypothesis','Automatic control evidence with source-bound support; no visual corrections or references.',
            '--out',str(event_bundle)],check=True)
    events_path=None
    if event_bundle:
        from src.ball_events.inputs import atd
        from src.ball_events.contract import config as detector_config
        from src.frente4.event_support import load_automatic_bundle
        meta,_,_,event_inputs,_=atd(args.data_dir)
        expected={key:sha(path) for key,path in event_inputs.items()}
        expected_configuration=detector_config(json.loads(Path('configs/frente5_v1.json').read_text()),meta['fps'])
        _,binding=load_automatic_bundle(event_bundle,meta,expected,expected_configuration)
        events_path=event_bundle/'events.jsonl'
        provenance['automatic_control_binding']=binding
        source_hashes.update({str(p):sha(p) for p in [events_path,event_bundle/'manifest.json']})
        manifest.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    temporary=args.out_dir/'prediction.partial.csv'
    _,diagnostic=run(str(temporary),str(args.data_dir),str(config),events_path=str(events_path) if events_path else None,
                     cache_dir=str(args.out_dir/'cache'),ball_quality=args.ball_quality,event_support_only=events_path is not None)
    reason=pm.validate_submission(pm.read_label_csv(temporary),GAMES[GAME]['n_video_frames'])
    if reason:raise ValueError(reason)
    if any(sha(p)!=digest for p,digest in source_hashes.items()):
        raise ValueError('Source changed during generation')
    final=args.out_dir/'La MasIA.csv'
    temporary.rename(final)
    temporary.with_suffix('.diagnostico.json').rename(args.out_dir/'diagnostico.json')
    provenance.update(status='complete',prediction_sha256=sha(final),
                      n_frames=GAMES[GAME]['n_video_frames'],elapsed_s=time.perf_counter()-started,
                      counts=diagnostic['summary']['counts'])
    manifest.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(provenance,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
