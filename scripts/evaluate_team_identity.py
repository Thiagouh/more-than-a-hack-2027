"""Evaluate identity against frozen visual truth; never used by inference."""
import argparse
import json
from pathlib import Path
from src.team_identification.identity_evaluation import evaluate_identity, promotion_gate, evaluate_atd_correspondences
from src.team_identification.manifests import load_frozen, write_json
from src.team_identification.pilot import sha256
from src.team_identification.protocol import load_truth, validate_splits


def load_run(path, truth):
    package=json.loads(path.read_text())
    observations=path.parent/package['observations']
    if package.get('status')!='complete' or package.get('inference_mode')!='full_frame' or sha256(observations)!=package.get('observations_sha256'):
        raise ValueError('Complete unchanged full-frame predictions required')
    from datetime import datetime
    started=datetime.fromisoformat(package['started_at_utc'])
    frozen_at=datetime.fromisoformat(truth['frozen_at_utc'])
    if started.tzinfo is None or started<frozen_at:
        raise ValueError('Visual truth must be frozen before evaluation inference')
    if package.get('supervised_model'):
        from src.team_identification.supervised_uniforms import SupervisedUniforms
        from src.team_identification.four_frames import POLICY
        model_path=path.parent/package['supervised_model']
        if sha256(model_path)!=package.get('supervised_model_sha256') or package.get('uniform_policy')!=POLICY or package.get('uniform_window_frames')!=4:
            raise ValueError('Supervised identity model/policy changed')
        artifact=json.loads(model_path.read_text());SupervisedUniforms(artifact)
        for frame in truth['frames']:
            if any(r['source_video_sha256']==frame['source_sha256'] and abs(r['frame']-frame['frame'])<750 for r in artifact['examples']):
                raise ValueError('Evaluation overlaps supervised training temporal neighborhood')
    else:
        references=path.parent/package['frozen_references']
        if sha256(references)!=package.get('frozen_references_sha256'):
            raise ValueError('Frozen references changed')
        _,frozen=load_frozen(references)
        validate_splits(truth,frozen)
    wanted={f['frame']:f for f in truth['frames']}
    rows=[]
    for line in observations.read_text().splitlines():
        row=json.loads(line)
        if row['frame'] in wanted:
            frame=wanted[row['frame']]
            from src.team_identification.manifests import load_image
            import hashlib
            image=load_image(Path(frame['_base'])/frame['image'])
            if (row['image_rgb_sha256']!=hashlib.sha256(image.tobytes()).hexdigest()
                    or frame['source_sha256']!=package['source_video_sha256']):
                raise ValueError('Prediction RGB/source differs from reviewed truth')
            for detection in row['detections']:
                if detection.get('role')=='other':
                    detection.update(role='unknown',classification='uncertain')
            rows.append(row)
    return evaluate_identity(truth,rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--overlay',type=Path,required=True)
    p.add_argument('--baseline-overlay',type=Path)
    p.add_argument('--baseline-atd',type=Path,help='Data directory for reviewed-correspondence original ATD baseline')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.baseline_overlay and a.baseline_atd:p.error('Choose one baseline')
    truth=load_truth(a.truth)
    for frame in truth['frames']:frame['_base']=str(a.truth.parent)
    report=load_run(a.overlay,truth)
    report.update(truth_sha256=sha256(a.truth),overlay_sha256=sha256(a.overlay),promotion=dict(accepted=False,reason='baseline_comparison_required'))
    if a.baseline_overlay:
        baseline=load_run(a.baseline_overlay,truth)
        report['baseline']=baseline
        report['promotion']=promotion_gate(report,baseline)
    elif a.baseline_atd:
        from itertools import islice
        from src.ball_events.inputs import atd
        meta,factory,_,files,_=atd(a.baseline_atd)
        package=json.loads(a.overlay.read_text())
        if sha256(files['atd_raw'])!=package['atd_raw_sha256'] or sha256(files['atd_meta'])!=package['atd_meta_sha256']:
            raise ValueError('Baseline ATD differs from candidate source')
        wanted={f['frame'] for f in truth['frames']}
        baseline=evaluate_atd_correspondences(truth,[r for r in islice(factory(),max(wanted)+1) if r['frame'] in wanted])
        report['baseline']=baseline
        report['promotion']=promotion_gate(report,baseline)
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'evaluation.json',report)


if __name__=='__main__':main()
