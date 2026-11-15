"""Write independent goalkeeper hypotheses from hash-bound reviewed context.

Does not rewrite SAM decisions, references, evaluation reports or possession.
"""
import argparse
import json
from pathlib import Path
from src.team_identification.goalkeeper_context import infer_goalkeeper_team,validate_policy
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256


def apply(detections, context, policy_path, out):
    hints,policy=read_json(context),read_json(policy_path)
    validate_policy(policy)
    if (hints.get('detections_sha256') != sha256(detections)
            or not isinstance(hints.get('reviewer'),str) or not hints['reviewer'].strip()):
        raise ValueError('Context must bind the unchanged detections and identify reviewer')
    frames=[json.loads(line) for line in detections.read_text().splitlines() if line.strip()]
    indexed={}
    for frame in frames:
        for detection in frame['detections']:
            key=(frame['frame'],detection['detection_index'])
            if key in indexed:raise ValueError('Duplicate frame/detection key')
            indexed[key]=(frame,detection)
    results=[];seen=set()
    for entry in hints['candidates']:
        key=(entry['frame'],entry['detection_index'])
        if key in seen or key not in indexed:raise ValueError('Unknown or duplicate context candidate')
        seen.add(key)
        frame,visual=indexed[key]
        if not frame.get('image_sha256') or entry.get('image_sha256') != frame['image_sha256']:
            raise ValueError('Context image binding mismatch')
        hypothesis=infer_goalkeeper_team(policy,entry)
        # Keep visual decisions intact; disagreements remain explicit hypotheses.
        results.append(dict(frame=key[0],detection_index=key[1],image_sha256=frame['image_sha256'],
            visual_decision=visual,context_evidence=entry,contextual_hypothesis=hypothesis,
            conflicts_with_visual=(hypothesis['team'] is not None and
                visual.get('classification') in ('H','A','referee') and
                (visual.get('classification') != hypothesis['team'] or visual.get('role') != 'goalkeeper'))))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'contextual-goalkeepers.json',dict(schema='goalkeeper-context-hypotheses-v1',
        status='experimental',reviewer=hints['reviewer'],detections_sha256=sha256(detections),
        context_sha256=sha256(context),policy_sha256=sha256(policy_path),policy=policy,
        script_sha256=sha256(__file__),rule_sha256=sha256(Path(__file__).resolve().parents[1]/'src/team_identification/goalkeeper_context.py'),
        visual_decisions_overwritten=False,references_frozen=False,model_executed=False,results=results))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detections',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=Path('configs/team-pilot-goalkeeper-context.json'))
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();apply(args.detections,args.context,args.policy,args.out)


if __name__=='__main__':main()
