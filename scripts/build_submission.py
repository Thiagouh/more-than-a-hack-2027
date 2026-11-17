"""Generate full-match algorithmic possession CSV without opening any ground truth."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time
from src.ball_events.inputs import atd
from src.team_possession.collective import CollectiveModel, configuration, POLICY_VERSION
from src.team_possession.restarts import load_kickoffs
from src.team_identification.pilot import sha256
from evaluation import possession_metrics as pm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path('data'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--end-frame',type=int,help='Diagnostic prefix only; omit for submission')
    args=parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory')
    meta,factory,hints,files,status=atd(args.data_dir)
    end=args.end_frame or meta['n_frames']
    if not 0<end<=meta['n_frames']:
        raise ValueError('Invalid end frame')
    original_n_frames=meta['n_frames']
    if end<original_n_frames:
        meta=dict(meta,n_frames=end,continuity_starts=[f for f in meta.get('continuity_starts',[]) if f<end])
        hints=[h for h in hints if h.get('available_at_frame',h.get('frame',0))<end]
    kickoffs,kpath=load_kickoffs(args.data_dir,meta)
    if kpath:files['kickoff_tags']=kpath
    detector=dict(persistence_frames=3,acquisition_persistence_frames=3,
                  loss_persistence_frames=3,relative_speed_mps=5.)
    cfg=configuration(fps=meta['fps'])
    model=CollectiveModel(meta,detector,cfg,hints,kickoffs=kickoffs)
    args.out.mkdir(parents=True)
    provenance=dict(policy=POLICY_VERSION,detector=detector,collective=cfg,
        team_source='provided ATD metadata',visual_evidence=False,
        ground_truth_used=False,original_n_frames=original_n_frames,end_frame_exclusive=end,
        input_sha256={k:sha256(p) for k,p in files.items()},
        code_sha256={str(p):sha256(p) for p in [Path(__file__),*sorted(Path('src').rglob('*.py'))]},
        kickoffs=kickoffs)
    (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    counts=Counter();started=time.perf_counter()
    from itertools import islice
    with (args.out/'prediction.partial.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['frame','label'])
        for row in islice(factory(),end):
            state,_,_,_=model.feed(row)
            label=state['label'];counts[label]+=1
            writer.writerow([row['frame'],label])
            if (row['frame']+1)%1000==0:
                handle.flush()
                print(json.dumps(dict(frames=row['frame']+1,elapsed_s=round(time.perf_counter()-started,2),counts=counts)),flush=True)
    model.finish()
    if {k:sha256(p) for k,p in files.items()} != provenance['input_sha256']:
        raise ValueError('Inputs changed during inference')
    if {str(p):sha256(p) for p in [Path(__file__),*sorted(Path('src').rglob('*.py'))]} != provenance['code_sha256']:
        raise ValueError('Code changed during inference; rerun from a frozen source state')
    partial=args.out/'prediction.partial.csv'
    error=pm.validate_submission(pm.read_label_csv(partial),end)
    if error:raise ValueError(error)
    partial.rename(args.out/'prediction.csv')
    summary=dict(status='complete',submission_ready=end==original_n_frames,
        n_frames=end,counts=counts,elapsed_s=time.perf_counter()-started,
        prediction_sha256=sha256(args.out/'prediction.csv'))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
