"""Frozen 1/2/3/4/5 comparison on the SAME public window and synthetic cases.

Processes the full tracking prefix to retain causal history before the window.
Only acquisition changes; loss=3, relative speed=5, missing-ball grace=5.
"""
import argparse
import copy
import csv
from itertools import islice
import json
import math
from pathlib import Path

from evaluation import possession_metrics as pm
from src.ball_events.artifacts import identity, sha, write_json
from src.ball_events.inputs import atd
from src.team_possession import possession
from scripts.compare_persistence import run_profile, scenarios, config
from src.ball_events.demo import metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',type=int,default=[1,2,3,4,5])
    args=parser.parse_args()
    profiles=sorted(set(args.profiles))
    if not profiles or any(p<1 for p in profiles):raise ValueError('Profiles must be positive integers')
    baseline=json.loads((args.baseline/'protocol.json').read_text())
    meta,factory,hints,files,status=atd(args.data_dir)
    original=copy.deepcopy(meta)
    start,end=baseline['start_frame'],baseline['end_frame_exclusive']
    if baseline['missing_ball_grace_frames']!=5 or meta['fps']!=25:raise ValueError('Baseline protocol mismatch')
    if {k:sha(v) for k,v in files.items()}!=baseline['detector_input_provenance']['input_sha256']:
        raise ValueError('Tracking differs from baseline')
    # Prefix 0..end retains identical causal state; no detection beyond the window is needed.
    meta['n_frames']=end
    meta['continuity_starts']=[f for f in meta.get('continuity_starts',[]) if f<end]
    args.out.mkdir(parents=True,exist_ok=False)
    cases=scenarios()
    frozen=identity(files,meta,{'profiles':profiles,'loss_frames':3,'grace':5},hints,status)
    write_json(args.out/'protocol.json',dict(profiles=profiles,start_frame=start,end_frame_exclusive=end,
        processed_tracking_prefix=[0,end],original_n_frames=original['n_frames'],fps=25,
        loss_frames=3,relative_speed_mps=5,missing_ball_grace_frames=5,
        hypotheses='Lower persistence may preserve brief controls but allow transient false acquisitions; no parameter selection on public GT.',
        provenance=frozen,synthetic_expectations=[{k:v for k,v in c.items() if k!='frames'} for c in cases]))
    observed=[]
    with files['atd_raw'].open() as handle:
        for line in islice(handle,end):
            xy=line.strip().split(':')[2].split(',')[:2]
            observed.append(all(math.isfinite(float(v)) for v in xy))
    if len(observed)!=end:raise ValueError('Incomplete tracking prefix')
    results={};hashes=set();synthetic={}
    for case in cases:
        synthetic[case['name']]={'expected':case['expected_teams'],'profiles':{}}
        for p in profiles:
            result=run_profile(metadata(len(case['frames'])),lambda case=case:iter(case['frames']),[],p)
            acquisitions=[e for e in result['events'] if e['type']=='possible_control_acquisition']
            synthetic[case['name']]['profiles'][str(p)]=dict(observed=[e['team'] for e in acquisitions],
                expectation_met=[e['team'] for e in acquisitions]==case['expected_teams'],
                acquisitions=[{k:e[k] for k in ('team','event_frame','confirmed_at_frame','available_at_frame')} for e in acquisitions],
                input_stream_sha256=result['input_stream_sha256'])
    write_json(args.out/'synthetic-results.json',synthetic)
    for p in profiles:
        print(f'Processing acquisition={p}, prefix={end} frames',flush=True)
        result=run_profile(meta,lambda:islice(factory(),end),hints,p)
        hashes.add(result['input_stream_sha256'])
        result['possession']=possession(result['events'],meta,ball_observed=observed,missing_ball_grace_frames=5)
        folder=args.out/str(p);folder.mkdir()
        write_json(folder/'result.json',result)
        labels=result['possession']['labels']
        with (folder/'prediction.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['frame','label']);writer.writerows(enumerate(labels))
        reason=pm.validate_submission(pm.read_label_csv(folder/'prediction.csv'),end)
        if reason:raise ValueError(reason)
        results[str(p)]=dict(configuration=result['configuration'],quality=result['quality'],
            input_stream_sha256=result['input_stream_sha256'],labels=labels[start:end],
            acquisition_events=sum(e['type']=='possible_control_acquisition' and start<=e['event_frame']<end for e in result['events']))
        if p in (3,5):
            previous=json.loads((args.baseline/f'consumer-{p}.json').read_text())['labels'][start:end]
            if results[str(p)]['labels']!=previous:raise ValueError('Baseline replay changed unexpectedly')
    if len(hashes)!=1:raise ValueError('Tracking inputs differ between profiles')
    if frozen!=identity(files,meta,{'profiles':profiles,'loss_frames':3,'grace':5},hints,status):
        raise ValueError('Input/code changed during experiment')
    # Open the SAME reference only after every fixed prediction has been written.
    gt=pm.label_codes(pm.read_label_csv(args.ground_truth))[start:end]
    previous_summary=json.loads((args.baseline/'summary.json').read_text())
    if sha(args.ground_truth)!=previous_summary['ground_truth_sha256'] or any(gt==pm.X):
        raise ValueError('Ground truth mismatch')
    write_json(args.out/'reference-sha256.json',dict(sha256=sha(args.ground_truth),source='same frozen public reference as baseline'))
    summary={}
    for p,r in results.items():
        scored=pm.evaluate(gt,pm.label_codes(r['labels']),bootstrap_n=0)
        write_json(args.out/p/'official-metrics.json',scored)
        summary[p]=dict(acquisition_events=r['acquisition_events'],
            label_counts={l:r['labels'].count(l) for l in pm.LABELS},
            metrics={k:scored[k] for k in ('macro_f1','spell_f1_mean','accuracy','n_gt_spells','n_pred_spells')},
            configuration=r['configuration'],input_stream_sha256=r['input_stream_sha256'])
    write_json(args.out/'summary.json',summary)
    with (args.out/'comparison-by-frame.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['frame','reference',*[f'prediction_{p}' for p in profiles]])
        writer.writerows([start+i,pm.LABELS[int(gt[i])],*[results[str(p)]['labels'][i] for p in profiles]] for i in range(end-start))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
