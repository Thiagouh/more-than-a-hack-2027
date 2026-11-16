"""Frozen supervised exemplar baseline for user-labeled RGB person crops.

Uses the existing HSV torso descriptor and Hellinger nearest-class distances.
No pretrained network is fine-tuned, no ATD labels or pitch positions are used,
and scores are distances rather than calibrated probabilities.
"""
from collections import Counter
import math
import numpy as np
from src.ball_events.uniforms import histogram

VERSION = 'user-supervised-torso-nearest-v1'
LABELS = {'Barcelona': ('H','player'), 'Athletic': ('A','player'),
          'Goleiro Barcelona': ('H','goalkeeper'), 'Goleiro Athletic': ('A','goalkeeper'),
          'Arbitro': (None,'referee'), 'Outro': (None,'other')}


class SupervisedUniforms:
    def __init__(self, artifact):
        if artifact.get('version') != VERSION or artifact.get('status') != 'frozen':
            raise ValueError('Frozen supervised uniform baseline required')
        self.max_distance = artifact['max_distance']
        self.min_margin = artifact['min_margin']
        if any(isinstance(v,bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in (self.max_distance,self.min_margin)):
            raise ValueError('Finite distance/margin thresholds within [0,1] required')
        self.examples = artifact['examples']
        if len({r['label'] for r in self.examples}) < 2:
            raise ValueError('At least two reviewed classes required')
        for row in self.examples:
            d = np.asarray(row['descriptor'])
            if row['label'] not in LABELS or d.shape != (200,) or not np.isfinite(d).all() or (d < 0).any() or not np.isclose(d.sum(),1.):
                raise ValueError('Invalid supervised exemplar descriptor')

    def predict(self, crop):
        d = histogram(crop)
        if d is None:
            return dict(label='Incerto',team=None,role='unknown',reason='crop_too_small',distance=None,margin=None)
        scores, nearest = {}, {}
        for row in self.examples:
            label = row['label']
            distance = float(np.linalg.norm(np.sqrt(d)-np.sqrt(row['descriptor']))/math.sqrt(2))
            if label not in scores or distance < scores[label]:
                scores[label], nearest[label] = distance, row['example_id']
        ranked = sorted(scores,key=lambda label:(scores[label],label))
        best = ranked[0]
        margin = scores[ranked[1]]-scores[best]
        accepted = scores[best] <= self.max_distance and margin >= self.min_margin
        label = best if accepted else 'Incerto'
        team,role = LABELS[label] if accepted else (None,'unknown')
        return dict(label=label,team=team,role=role,reason='supervised_uniform_match' if accepted else 'distance_or_margin',
            distance=scores[best],margin=margin,nearest_example=nearest[best],
            class_distances=scores,calibrated_probability=False)


def fit(examples, max_distance=.35, min_margin=.10):
    rows=[]
    for row in examples:
        if row.get('split') != 'reference' or row.get('reviewed') is not True or not row.get('reviewer'):
            raise ValueError('Only explicit reviewed training references may fit the model')
        if row['label']=='Incerto':
            continue
        if row['label'] not in LABELS:
            raise ValueError('Unsupported training label')
        d=histogram(row['rgb'])
        if d is None:
            raise ValueError('Reviewed training crop too small')
        rows.append({k:v for k,v in row.items() if k!='rgb'} | dict(descriptor=d.tolist()))
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Unique supervised training IDs required')
    model=dict(version=VERSION,status='frozen',examples=rows,max_distance=max_distance,min_margin=min_margin,
        method='Nearest reviewed exemplar per class using existing HSV torso histograms and Hellinger distance',
        class_counts=dict(Counter(r['label'] for r in rows)),
        parameters_selected_on_test=False,calibrated_probabilities=False,
        limitations=['Manual person crops; not detection, association, tracking or possession accuracy.',
                     'Other class has limited negative examples; open-set recognition is not validated.'])
    SupervisedUniforms(model)
    return model
