"""Frozen public-window check of 3/5 acquisition with independent missing-ball grace.

Uses saved detector events, the SAME raw ATD hashes, and official metrics. No
re-detection or parameter search. Ground truth is loaded only after protocol and
both predictions are written. This is an integration check, not model selection.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from evaluation import possession_metrics as pm
from src.ball_events.artifacts import write_json, sha
from src.team_possession import possession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--tracking',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--start',type=int,default=11185)
    parser.add_argument('--frames',type=int,default=3000)
    parser.add_argument('--grace',type=int,default=5)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    base=args.run/'same-tracking-real'
    provenance=json.loads((base/'provenance.json').read_text())
    meta=provenance['input_metadata']
    if provenance['input_sha256']['atd_raw']!=sha(args.tracking):
        raise ValueError('Tracking hash differs from frozen paired detector inputs')
    end=args.start+args.frames
    if not 0<=args.start<end<=meta['n_frames']:raise ValueError('Invalid fixed window')
    args.out.mkdir(parents=True,exist_ok=False)
    protocol=dict(start_frame=args.start,end_frame_exclusive=end,fps=meta['fps'],
        acquisition_profiles=[3,5],loss_frames=3,relative_speed_mps=5,
        missing_ball_grace_frames=args.grace,
        timeout_rule='Hold the first grace missing frames; X on grace+1; require fresh acquisition after an expired gap.',
        acquisition_frozen=True,ground_truth_used_for_parameter_selection=False,
        source='Public GT used by notebook 03_first_submission; generated two-minute window, not an identified demo video.',
        timing='Causal delivery; no retrospective rewrites; preserve full-match history before clipping.',
        detector_input_provenance=provenance,
        script_sha256=sha(__file__),consumer_sha256=sha(Path(__file__).resolve().parents[1]/'src/team_possession/consumer.py'))
    write_json(args.out/'protocol.json',protocol)
    observed=[]
    with args.tracking.open() as handle:
        for line in handle:
            xy=line.strip().split(':')[2].split(',')[:2]
            observed.append(all(math.isfinite(float(v)) for v in xy))
    if len(observed)!=meta['n_frames']:raise ValueError('Visibility length mismatch')
    results={};labels={}
    for p in ('3','5'):
        source=base/p/'result.json'
        previous=json.loads(source.read_text())
        cfg=previous['configuration']
        if (cfg['acquisition_persistence_frames'],cfg['loss_persistence_frames'],cfg['relative_speed_mps'])!=(int(p),3,5.):
            raise ValueError('Frozen profile differs from declared protocol')
        result=possession(previous['events'],meta,ball_observed=observed,missing_ball_grace_frames=args.grace)
        write_json(args.out/f'consumer-{p}.json',result)
        full=result['labels'];labels[p]=full[args.start:end]
        with (args.out/f'prediction-{p}.csv').open('w',newline='') as handle:
            w=csv.writer(handle);w.writerow(['frame','label']);w.writerows(enumerate(full))
        reason=pm.validate_submission(pm.read_label_csv(args.out/f'prediction-{p}.csv'),meta['n_frames'])
        if reason:raise ValueError(reason)
        results[p]=dict(frozen_events_sha256=sha(source),
            full_match_label_counts={l:full.count(l) for l in ('H','A','D','X')},
            window_label_counts={l:labels[p].count(l) for l in ('H','A','D','X')},
            missing_ball_timeouts=result['missing_ball_timeouts'],
            acquisition_events=sum(e['type']=='possible_control_acquisition' and args.start<=e['event_frame']<end for e in previous['events']))
    # Reference is opened AFTER both fixed predictions; never feeds the consumer.
    reference=pm.read_label_csv(args.ground_truth)
    reason=pm.validate_submission(reference,meta['n_frames'])
    if reason:raise ValueError(reason)
    gt=pm.label_codes(reference)[args.start:end]
    if any(v==pm.X for v in gt):raise ValueError('Window must be wholly inside published reference')
    for p in ('3','5'):
        codes=pm.label_codes(labels[p])
        scored=pm.evaluate(gt,codes,bootstrap_n=0)
        write_json(args.out/f'official-metrics-{p}.json',scored)
        results[p]['metrics']={k:scored[k] for k in ('macro_f1','spell_f1_mean','accuracy','n_abstain_on_graded','n_gt_spells','n_pred_spells')}
    summary=dict(protocol_window={k:protocol[k] for k in ('start_frame','end_frame_exclusive','fps','missing_ball_grace_frames')},
        ground_truth_sha256=sha(args.ground_truth),
        reference_counts={l:int(sum(gt==i)) for i,l in enumerate(pm.LABELS)},
        profiles=results,window_different_frames=sum(a!=b for a,b in zip(labels['3'],labels['5'])),
        full_match_different_frames=sum(a!=b for a,b in zip(json.loads((args.out/'consumer-3.json').read_text())['labels'],json.loads((args.out/'consumer-5.json').read_text())['labels'])),
        window_missing_ball_frames=sum(not x for x in observed[args.start:end]),
        notebook_demo_video_identified=False,
        limitation='Small public integration window; no dead-ball model; not evidence of full-match ranking.')
    write_json(args.out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
