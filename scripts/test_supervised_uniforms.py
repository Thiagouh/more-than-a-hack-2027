"""Commit predictions before user test labels; score the fixed predictions later."""
import argparse
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from src.team_identification.manifests import load_image,read_json,write_json
from src.team_identification.pilot import sha256
from src.team_identification.supervised_uniforms import SupervisedUniforms
from scripts.prepare_uniform_label_batch import TRAINING_POOLS


def predict(model_path,batch_path,out):
    artifact,batch=read_json(model_path),read_json(batch_path)
    model=SupervisedUniforms(artifact)
    if batch.get('sampling_pool')!='held_out_test' or batch.get('status')!='awaiting_user_labels':
        raise ValueError('Unlabeled held-out test batch required')
    rows=[]
    for row in batch['examples']:
        if row.get('split')!='test' or row.get('reviewed') or row.get('label') is not None:
            raise ValueError('Test labels must not be available when committing predictions')
        f=row['frame']
        if any(lo<=f<hi for ranges in TRAINING_POOLS.values() for lo,hi in ranges):
            raise ValueError('Test frame overlaps a declared training interval')
        if any(r['source_video_sha256']==row['source_video_sha256'] and abs(r['frame']-f)<750 for r in artifact['examples']):
            raise ValueError('Test frame too close to a training example')
        path=batch_path.parent/row['crop']
        if sha256(path)!=row['crop_sha256'] or sha256(row['image'])!=row['image_sha256']:
            raise ValueError('Test pixels changed')
        rows.append(dict(id=row['id'],frame=f,group_id=row['group_id'],crop_sha256=row['crop_sha256'],
                         **model.predict(np.asarray(load_image(path)))))
    out.mkdir(parents=True,exist_ok=False)
    result=dict(schema='uniform-supervised-test-predictions-v1',status='committed_before_user_labels',
        model_sha256=sha256(model_path),batch_sha256=sha256(batch_path),
        committed_at_utc=datetime.now(timezone.utc).isoformat(),predictions=rows,
        metrics=None,scope='Classification of manually selected person crops only; not full-frame detection or possession.')
    write_json(out/'predictions.json',result)
    return result


def score(predictions_path,labeled_path,out):
    predictions,labeled=read_json(predictions_path),read_json(labeled_path)
    if (predictions.get('status')!='committed_before_user_labels' or labeled.get('status')!='user_labeled'
            or labeled.get('original_batch_sha256')!=predictions.get('batch_sha256')):
        raise ValueError('Test labels must match the original committed batch')
    if datetime.fromisoformat(predictions['committed_at_utc'])>=datetime.fromisoformat(labeled['reviewed_at_utc']):
        raise ValueError('Test predictions must precede user review')
    truth={r['id']:r for r in labeled['examples']}
    if len(truth)!=len(labeled['examples']) or set(truth)!={r['id'] for r in predictions['predictions']}:
        raise ValueError('Test IDs differ')
    totals=Counter();classes={};confusion={};comparisons=[]
    for row in predictions['predictions']:
        target=truth[row['id']]
        if target['split']!='test' or not target['reviewed'] or target['crop_sha256']!=row['crop_sha256']:
            raise ValueError('Invalid test label binding')
        accepted=row['label']!='Incerto';evaluable=target['label']!='Incerto'
        correct=evaluable and row['label']==target['label']
        count=Counter(examples=1,evaluable=int(evaluable),accepted=int(accepted and evaluable),correct=int(correct),
                      abstained=int(evaluable and not accepted),wrong_accepted=int(evaluable and accepted and not correct))
        totals.update(count);classes.setdefault(target['label'],Counter()).update(count)
        confusion.setdefault(target['label'],Counter())[row['label']]+=1
        comparisons.append(dict(id=row['id'],predicted=row['label'],reference=target['label'],correct=correct))
    def ratios(c):
        return dict(c,accuracy=c['correct']/c['evaluable'] if c['evaluable'] else None,
                    coverage=c['accepted']/c['evaluable'] if c['evaluable'] else None,
                    accuracy_accepted=c['correct']/c['accepted'] if c['accepted'] else None)
    result=dict(metrics=ratios(totals),by_class={k:ratios(v) for k,v in classes.items()},
        confusion={k:dict(v) for k,v in confusion.items()},comparisons=comparisons,
        temporal_group_count=len({r['group_id'] for r in predictions['predictions']}),
        predictions_sha256=sha256(predictions_path),labels_sha256=sha256(labeled_path),model_sha256=predictions['model_sha256'],
        retrained_on_test=False,limitations='Small exploratory crop classification test; no full-match, detection or possession accuracy claim.')
    out.mkdir(parents=True,exist_ok=False);write_json(out/'evaluation.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('predict');a.add_argument('--model',type=Path,required=True);a.add_argument('--batch',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--predictions',type=Path,required=True);a.add_argument('--labeled',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    predict(a.model,a.batch,a.out) if a.action=='predict' else print(score(a.predictions,a.labeled,a.out)['metrics'])


if __name__=='__main__':main()
