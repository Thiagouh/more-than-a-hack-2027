"""Evaluate reviewed visual labels, never possession ground truth."""
from collections import Counter
import math


def validate_box(box, optional=False):
    if optional and box is None:
        return
    if (not isinstance(box,list) or len(box)!=4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in box)
            or not 0 <= box[0] < box[2] or not 0 <= box[1] < box[3]):
        raise ValueError('Invalid evaluation box')


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def evaluate(truth, predictions, threshold=.5):
    """Maximum-cardinality one-to-one box matching, class-blind, IoU >= threshold."""
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be within (0,1]')
    if truth.get('reviewed') is not True or not truth.get('reviewer'):
        raise ValueError('Reviewed visual evaluation labels required')
    gt_frames = truth['frames']
    if len({f['frame'] for f in gt_frames}) != len(gt_frames):
        raise ValueError('Duplicate evaluation frame')
    pred = {}
    for frame in predictions:
        if frame['frame'] in pred:
            raise ValueError('Duplicate prediction frame')
        pred[frame['frame']] = frame['detections']
    if set(pred) != {f['frame'] for f in gt_frames}:
        raise ValueError('Predictions and reviewed frames must match exactly')
    totals, strata = Counter(), {}
    for frame in gt_frames:
        people, detections = frame['people'], pred[frame['frame']]
        if len({p['label'] for p in people}) != len(people):
            raise ValueError('Duplicate person label')
        for person in people:
            validate_box(person['box'])
            if (person.get('classification') not in ('H','A','referee','uncertain')
                    or type(person.get('evaluable')) is not bool
                    or person.get('lighting') not in ('sun','shade','mixed')
                    or person.get('occlusion') not in ('isolated','overlapped')):
                raise ValueError('Invalid visual evaluation annotation')
        for detection in detections:
            validate_box(detection.get('box'),optional=True)
            if detection.get('classification') not in ('H','A','referee','uncertain'):
                raise ValueError('Invalid prediction classification')
        edges = [[j for _,j in sorted(
            [(-iou(d['box'],p['box']),j) for j,p in enumerate(people)
             if iou(d['box'],p['box']) >= threshold])]
                 for d in detections]
        owner = {}
        def assign(i, visited):
            for j in edges[i]:
                if j in visited:
                    continue
                visited.add(j)
                if j not in owner or assign(owner[j], visited):
                    owner[j] = i
                    return True
            return False
        for i in range(len(detections)):
            assign(i,set())
        matched = set(owner.values())
        totals['detections'] += len(detections)
        totals['accepted'] += sum(d['classification'] != 'uncertain' for d in detections)
        totals['uncertain_detections'] += sum(d['classification'] == 'uncertain' for d in detections)
        totals['duplicates'] += sum(bool(edges[i]) for i in range(len(detections)) if i not in matched)
        totals['unmatched_detections'] += len(detections)-len(matched)
        totals['people'] += len(people)
        totals['missed_people'] += len(people)-len(owner)
        for j,p in enumerate(people):
            d = detections[owner[j]] if j in owner else None
            accepted = d is not None and d['classification'] != 'uncertain'
            correct = accepted and p['classification'] != 'uncertain' and d['classification'] == p['classification']
            totals['correct_accepted'] += bool(correct)
            evaluable_athlete = p['evaluable'] and p['classification'] in ('H','A')
            totals['evaluable_athletes'] += evaluable_athlete
            totals['covered_athletes'] += bool(evaluable_athlete and accepted)
            for key in (p['lighting'],p['occlusion']):
                c = strata.setdefault(key,Counter())
                c['people'] += 1
                c['accepted'] += bool(accepted)
                c['correct_accepted'] += bool(correct)
                c['evaluable_athletes'] += evaluable_athlete
                c['covered_athletes'] += bool(evaluable_athlete and accepted)
    def metrics(c):
        return dict(c, precision_accepted=c['correct_accepted']/c['accepted'] if c['accepted'] else None,
                    coverage=c['covered_athletes']/c['evaluable_athletes'] if c['evaluable_athletes'] else None)
    result = metrics(totals)
    result['strata'] = {k:metrics(v) for k,v in strata.items()}
    result['iou_threshold'] = threshold
    result['frame_count'] = len(gt_frames)
    result['targets_observed'] = (result['precision_accepted'] is not None and result['coverage'] is not None
                                  and result['precision_accepted'] >= .95 and result['coverage'] >= .8)
    result['possession_integration_authorized'] = False
    return result
