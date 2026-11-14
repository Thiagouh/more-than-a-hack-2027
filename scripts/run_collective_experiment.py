"""Freeze collective metrics, test synthetic passes, then compare a fixed public window.

No reference labels enter either predictor. No sweep/selection on public scores.
"""
import argparse
import copy
import csv
from itertools import islice
import json
from pathlib import Path

from evaluation import possession_metrics as pm
from src.ball_events.artifacts import identity,write_json,canonical,sha
from src.ball_events.inputs import atd
from src.team_possession.collective import CollectiveModel,configuration,UNITS,METRICS_VERSION,POLICY_VERSION
from src.team_possession.consumer import possession
from src.team_possession.demo import cases
from src.team_possession.evaluation import evaluate_collective


def synthetic_case(case,collective_cfg=None):
    model=CollectiveModel(case['manifest'],collective_config=collective_cfg)
    states=[];events=[];visible=[]
    for row in case['frames']:
        state,_,batch,_=model.feed(row)
        states.append(state);events.extend(batch);visible.append(row['ball']['source']=='observed')
    quality=model.finish()
    baseline=possession(events,case['manifest'],ball_observed=visible,missing_ball_grace_frames=5)
    reference_args=dict(passes=case['passes'],recoveries=case['recoveries'])
    result=evaluate_collective(states,events,25,**reference_args)
    result['baseline_pass_metrics']=evaluate_collective(
        [dict(s,label=baseline['labels'][s['frame']]) for s in states],events,25,passes=case['passes'])['pass_metrics']
    return dict(expected=case['expected'],evaluation=result,quality=quality,states=states)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    previous_protocol=json.loads((args.baseline/'protocol.json').read_text())
    start,end=previous_protocol['start_frame'],previous_protocol['end_frame_exclusive']
    meta,factory,hints,files,status=atd(args.data_dir)
    n_original=meta['n_frames'];meta=copy.deepcopy(meta);meta['n_frames']=end
    meta['continuity_starts']=[f for f in meta.get('continuity_starts',[]) if f<end]
    expected=previous_protocol['detector_input_provenance']['input_sha256']
    if {k:sha(p) for k,p in files.items()}!=expected:raise ValueError('Tracking differs from frozen baseline')
    args.out.mkdir(parents=True,exist_ok=False)
    cfg=configuration(fps=meta['fps'])
    detector_cfg=dict(persistence_frames=3,acquisition_persistence_frames=3,loss_persistence_frames=3,relative_speed_mps=5.)
    provenance=identity(files,meta,{'detector':detector_cfg,'collective':cfg},hints,status)
    examples=cases()
    variants=[('default',{}),('horizon_1s',{'forecast_horizon_s':1.}),('reaction_0_1s',{'reaction_time_s':.1}),
              ('reaction_0_3s',{'reaction_time_s':.3}),('speed_6mps',{'max_speed_mps':6.}),('speed_10mps',{'max_speed_mps':10.})]
    protocol=dict(start_frame=start,end_frame_exclusive=end,original_n_frames=n_original,fps=meta['fps'],
        loss_frames=3,missing_ball_grace_frames=5,profiles=['anterior','coletivo'],
        model_names={'anterior':'Anterior','coletivo':'Coletivo'},
        collective_configuration=cfg,detector_configuration=detector_cfg,
        parameters_calibrated=False,parameter_selection_on_public_reference=False,
        synthetic_sensitivity_variants=variants,
        synthetic_references=[{k:v for k,v in case.items() if k!='frames'} for case in examples],
        provenance=provenance,metrics_version=METRICS_VERSION,policy_version=POLICY_VERSION,
        hypothesis='Preserve a known team through plausible observed passes; predicted interception does not confirm turnover.',
        scope='Same public 120-second window, causal tracking prefix; not full-match accuracy validation.')
    write_json(args.out/'protocol.json',protocol)
    write_json(args.out/'metric-catalog.json',dict(version=METRICS_VERSION,units=UNITS))
    print('Testing synthetic passes and parameter sensitivity',flush=True)
    synthetic={}
    for name,overrides in variants:
        synthetic[name]={case['name']:synthetic_case(case,overrides) for case in examples}
    write_json(args.out/'synthetic.json',synthetic)
    model=CollectiveModel(meta,detector_cfg,cfg,hints)
    labels=[];states=[];events=[];visible=[]
    print(f'Processing real tracking prefix 0..{end-1}',flush=True)
    with (args.out/'measurements.jsonl').open('w') as journal:
        for row in islice(factory(),end):
            snapshot,_,batch,measurement=model.feed(row)
            labels.append(snapshot['label']);events.extend(batch)
            visible.append(row['ball']['source']=='observed')
            if row['frame']>=start:
                states.append(snapshot);journal.write(canonical(measurement)+'\n')
    quality=model.finish()
    legacy=possession(events,meta,ball_observed=visible,missing_ball_grace_frames=5)
    previous=json.loads((args.baseline/'consumer-3.json').read_text())['labels'][start:end]
    if legacy['labels'][start:end]!=previous:raise ValueError('Baseline changed unexpectedly')
    old=dict(configuration=detector_cfg,events=events,possession=legacy,quality=quality['detector'])
    new=dict(configuration=detector_cfg,collective_configuration=cfg,events=events,
             possession=dict(labels=labels,states=states,policy_version=POLICY_VERSION,experimental=True),quality=quality)
    for name,result in [('anterior',old),('coletivo',new)]:
        folder=args.out/name;folder.mkdir()
        write_json(folder/'result.json',result)
        with (folder/'prediction.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['frame','label']);writer.writerows(enumerate(result['possession']['labels']))
        reason=pm.validate_submission(pm.read_label_csv(folder/'prediction.csv'),end)
        if reason:raise ValueError(reason)
    if identity(files,meta,{'detector':detector_cfg,'collective':cfg},hints,status)!=provenance:
        raise ValueError('Input/code changed during experiment')
    # Only now open reference labels; the model never accepts them as inputs.
    previous_summary=json.loads((args.baseline/'summary.json').read_text())
    gt_hash=sha(args.ground_truth)
    if gt_hash!=previous_summary['ground_truth_sha256']:raise ValueError('Reference differs from baseline')
    gt=pm.label_codes(pm.read_label_csv(args.ground_truth))[start:end]
    if any(gt==pm.X):raise ValueError('Window is not fully covered by the public reference')
    summary=dict(ground_truth_sha256=gt_hash,models={},synthetic_file='synthetic.json',
        public_pass_references_available=False,public_meeting_references_available=False,
        public_recovery_reference_available=False,forecast_parameters_calibrated=False)
    for name,predicted in [('anterior',legacy['labels'][start:end]),('coletivo',labels[start:end])]:
        metrics=pm.evaluate(gt,pm.label_codes(predicted),bootstrap_n=0)
        summary['models'][name]=dict(label_counts={t:predicted.count(t) for t in pm.LABELS},
            metrics={k:metrics[k] for k in ('macro_f1','spell_f1_mean','accuracy','n_gt_spells','n_pred_spells')})
        write_json(args.out/name/'official-metrics.json',metrics)
    assessment=evaluate_collective(states,events,meta['fps'],reference_labels=[pm.LABELS[int(v)] for v in gt])
    write_json(args.out/'collective-assessment.json',assessment)
    summary['states']=assessment['state_counts']
    summary['different_frames']=sum(a!=b for a,b in zip(legacy['labels'][start:end],labels[start:end]))
    write_json(args.out/'summary.json',summary)
    write_json(args.out/'reference-sha256.json',dict(sha256=gt_hash))
    with (args.out/'comparison-by-frame.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['frame','reference','anterior','coletivo','state','reason'])
        writer.writerows((start+i,pm.LABELS[int(v)],legacy['labels'][start+i],labels[start+i],states[i]['state'],states[i]['reason']) for i,v in enumerate(gt))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
