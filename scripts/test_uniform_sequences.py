"""Commit and score held-out four-frame cases, once per athlete sequence."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import numpy as np
from src.team_identification.manifests import read_json,write_json,load_image
from src.team_identification.pilot import sha256
from src.team_identification.supervised_uniforms import SupervisedUniforms
from src.team_identification.four_frames import FourFrameUniforms,POLICY
from scripts.prepare_uniform_label_batch import TRAINING_POOLS
from scripts.test_supervised_uniforms import score as score_cases


def sequence_digest(frames):
    return hashlib.sha256('|'.join(r['crop_sha256'] for r in frames).encode()).hexdigest()


def predict(model_path,batch_path,out):
    artifact,batch=read_json(model_path),read_json(batch_path)
    model=SupervisedUniforms(artifact);stream=FourFrameUniforms(model)
    if batch.get('split')!='test' or batch.get('status')!='awaiting_user_labels':
        raise ValueError('Unlabeled held-out sequence batch required')
    cases=batch['sequences']
    if len({r['id'] for r in cases})!=len(cases):raise ValueError('Unique test sequence IDs required')
    predictions=[];baselines=[]
    for case in cases:
        if (case.get('split')!='test' or case.get('reviewed') or case.get('label') is not None
                or case.get('association_reviewed') is not True or not case.get('association_reviewer')):
            raise ValueError('Review person continuity, but do not supply class labels before prediction')
        rows=case['frames'];frames=[r['frame'] for r in rows]
        if len(rows)!=4 or frames!=list(range(frames[0],frames[0]+4)):
            raise ValueError('Exactly four consecutive frames required')
        history=[]
        for row in rows:
            f=row['frame']
            if any(lo<=f<hi for ranges in TRAINING_POOLS.values() for lo,hi in ranges):
                raise ValueError('Test overlaps declared training pool')
            if any(r['source_video_sha256']==row['source_video_sha256'] and abs(r['frame']-f)<750 for r in artifact['examples']):
                raise ValueError('Test too close to actual training example')
            path=batch_path.parent/row['crop']
            if sha256(path)!=row['crop_sha256'] or sha256(row['image'])!=row['image_sha256']:
                raise ValueError('Sequence pixels changed')
            crop=np.asarray(load_image(path))
            history.append(stream.feed(case['identity'],f,crop,available_at_frame=row.get('available_at_frame',f)))
        common=dict(id=case['id'],frame=frames[-1],frame_ids=frames,group_id=case['group_id'],crop_sha256=sequence_digest(rows))
        predictions.append(dict(common,**{k:v for k,v in history[-1].items() if k not in common},per_frame=history))
        baselines.append(dict(common,**model.predict(crop)))
        stream.reset()
    out.mkdir(parents=True,exist_ok=False)
    binding=dict(schema='uniform-four-frame-test-predictions-v1',status='committed_before_user_labels',
        model_sha256=sha256(model_path),batch_sha256=sha256(batch_path),policy=POLICY,
        committed_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_unit='One reviewed person sequence, not four independent samples',
        scope='Manual person crops only; not automatic tracking or possession accuracy.')
    write_json(out/'predictions.json',dict(binding,predictions=predictions))
    write_json(out/'single-frame-baseline.json',dict(binding,predictions=baselines,
        baseline='Same frozen classifier using the final frame alone; committed before labels'))
    return binding


def score(predictions_path,labeled_path,out):
    predictions,labeled=read_json(predictions_path),read_json(labeled_path)
    if labeled.get('original_sequence_batch_sha256')!=predictions.get('batch_sha256'):
        raise ValueError('Sequence labels differ from committed batch')
    groups={}
    for row in labeled['examples']:groups.setdefault(row['sequence_id'],[]).append(row)
    examples=[]
    for ident,rows in groups.items():
        rows=sorted(rows,key=lambda r:r['frame'])
        if len(rows)!=4 or len({r['label'] for r in rows})!=1 or not all(r['split']=='test' and r['reviewed'] for r in rows):
            raise ValueError('Four consistently labeled test frames required per case')
        examples.append(dict(id=ident,label=rows[0]['label'],reviewed=True,split='test',crop_sha256=sequence_digest(rows)))
    out.mkdir(parents=True,exist_ok=False)
    normalized=out/'sequence-labels.json'
    write_json(normalized,dict(status='user_labeled',original_batch_sha256=labeled['original_sequence_batch_sha256'],
        reviewed_at_utc=labeled['reviewed_at_utc'],examples=examples,original_labels_sha256=sha256(labeled_path)))
    report=score_cases(predictions_path,normalized,out/'four-frames')
    baseline_path=predictions_path.parent/'single-frame-baseline.json'
    if baseline_path.is_file():
        baseline=read_json(baseline_path)
        if any(baseline.get(k)!=predictions.get(k) for k in ('model_sha256','batch_sha256','committed_at_utc')):
            raise ValueError('Single-frame baseline binding differs')
        one=score_cases(baseline_path,normalized,out/'single-frame')
        write_json(out/'comparison.json',dict(four_frames=report['metrics'],single_frame=one['metrics'],
            same_cases=True,training_unchanged=True,independent_sequences=len(examples),
            limitations='Small exploratory test; neighboring crops are correlated.'))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('predict');a.add_argument('--model',type=Path,required=True);a.add_argument('--batch',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--predictions',type=Path,required=True);a.add_argument('--labeled',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args();predict(a.model,a.batch,a.out) if a.action=='predict' else print(score(a.predictions,a.labeled,a.out)['metrics'])


if __name__=='__main__':main()
