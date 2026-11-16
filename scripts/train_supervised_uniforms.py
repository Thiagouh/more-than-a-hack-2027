"""Freeze a first supervised uniform baseline from confirmed crop batches."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from src.team_identification.manifests import load_image, read_json, write_json
from src.team_identification.pilot import sha256
from src.team_identification.supervised_uniforms import fit


def train(batches, out):
    examples=[];inputs={};seen=set()
    for batch in batches:
        spec=read_json(batch)
        if spec.get('status')!='user_labeled' or spec.get('schema')!='uniform-supervised-label-batch-v1':
            raise ValueError('User-labeled crop batch required')
        inputs[str(batch.resolve())]=sha256(batch)
        for row in spec['examples']:
            crop=batch.parent/row['crop']
            if sha256(crop)!=row['crop_sha256'] or sha256(row['image'])!=row['image_sha256']:
                raise ValueError('Training source/crop changed')
            key=(row['source_video_sha256'],row['frame'],tuple(row['box']))
            if key in seen:raise ValueError('Duplicate training crop')
            seen.add(key)
            examples.append(dict(row,example_id=f"{batch.parent.name}:{row['id']}",rgb=np.asarray(load_image(crop))))
    # Existing project defaults, frozen before examining test labels or scores.
    model=fit(examples,max_distance=.35,min_margin=.10)
    model.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),input_sha256=inputs,
        code_sha256={str(p):sha256(p) for p in [Path(__file__),Path('src/team_identification/supervised_uniforms.py'),Path('src/ball_events/uniforms.py')]})
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'model.json',model)
    write_json(out/'training-summary.json',{k:model[k] for k in ('version','class_counts','frozen_at_utc','method','limitations')})
    return model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(train(a.batches,a.out)['class_counts'])


if __name__=='__main__':main()
