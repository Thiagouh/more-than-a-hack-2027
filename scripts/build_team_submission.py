"""Build La MasIA's offline submission with Frente 4's frozen football rules.

No ground truth, trained match classifier, reviewed corrections, or video input.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
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
    args=parser.parse_args()
    if args.out_dir.exists():raise ValueError('Use a new output directory')
    inputs=require(['atd_meta','atd_raw','smart_tagging'],GAME,args.data_dir)
    config=Path('configs/frente4_v1.json')
    code=[Path(__file__),Path('games.py'),Path('starter/load_smart_tagging.py'),
          *sorted(Path('src/frente4').glob('*.py'))]
    source_hashes={str(p):sha(p) for p in [*map(Path,inputs),config,*code]}
    provenance=dict(status='started',method='frente4-frozen-offline-rules',
        team='La MasIA',upstream_commit='288d5a0a1aa37324850436188655ce777263b056',
        upstream_pr='https://github.com/Thiagouh/more-than-a-hack-2027/pull/1',
        ground_truth_used=False,training_on_provided_game=False,human_corrections=False,
        configuration=json.loads(config.read_text()),source_sha256=source_hashes,
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    args.out_dir.mkdir(parents=True)
    manifest=args.out_dir/'provenance.json'
    manifest.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    started=time.perf_counter()
    temporary=args.out_dir/'prediction.partial.csv'
    _,diagnostic=run(str(temporary),str(args.data_dir),str(config),
                     cache_dir=str(args.out_dir/'cache'))
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
