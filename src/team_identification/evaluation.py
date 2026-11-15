"""Class-blind visual evaluation with role metrics and grouped uncertainty."""
from collections import Counter
import numpy as np
from .protocol import validate_box, validate_truth, in_region

BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 0
RATIOS = {'precision_accepted': ('correct_accepted','accepted'),
          'precision_team_and_role': ('correct_team_and_role','accepted'),
          'coverage': ('covered_athletes','evaluable_athletes')}
COUNT_KEYS = ('detections','accepted','correct_accepted','correct_team_and_role',
              'uncertain_detections','duplicates','unmatched_detections','people',
              'missed_people','evaluable_athletes','covered_athletes','outside_region_detections')


def iou(a, b):
    if a is None or b is None:
        return 0.
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def metrics(counter):
    result = {key: int(counter[key]) for key in COUNT_KEYS}
    for name,(num,den) in RATIOS.items():
        result[name] = counter[num]/counter[den] if counter[den] else None
    return result


def grouped_intervals(groups, temporal_known):
    """Percentile cluster bootstrap; people within a group are not independent."""
    result = dict(method='group_percentile_bootstrap',confidence=.95,
        repetitions=BOOTSTRAP_REPETITIONS,seed=BOOTSTRAP_SEED,group_count=len(groups),
        intervals={name: None for name in RATIOS})
    if len(groups) < 10 or not temporal_known:
        result['warning'] = 'fewer_than_10_groups' if len(groups)<10 else 'unknown_temporal_origin'
        return result
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {name: [] for name in RATIOS}
    # Vectorization keeps 2,000 replicates inexpensive even for many strata.
    counts = list(groups.values())
    result['denominator_groups'] = {name: sum(c[den]>0 for c in counts) for name,(_,den) in RATIOS.items()}
    weights = rng.multinomial(len(counts), np.full(len(counts),1/len(counts)),BOOTSTRAP_REPETITIONS)
    for name,(num,den) in RATIOS.items():
        numerator = weights @ np.array([c[num] for c in counts])
        denominator = weights @ np.array([c[den] for c in counts])
        samples[name] = (numerator[denominator>0]/denominator[denominator>0]
                         if result['denominator_groups'][name]>=10 else np.array([]))
    result['intervals'] = {name: np.quantile(values,[.025,.975]).tolist() if len(values) else None
                           for name,values in samples.items()}
    result['valid_repetitions'] = {name: len(values) for name,values in samples.items()}
    result['warning'] = 'Exploratory; require 10 denominator-supporting groups per metric; zero errors can yield degenerate intervals'
    return result


def evaluate(truth, predictions, threshold=.5):
    """Maximum-cardinality matching, class-blind, deterministic local IoU order.

    The CLI verifies image/package bindings before calling this pure scorer.
    """
    validate_truth(truth)
    if type(threshold) not in (int,float) or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be within (0,1]')
    pred = {}
    for frame in predictions:
        if frame['frame'] in pred:
            raise ValueError('Duplicate prediction frame')
        pred[frame['frame']] = frame['detections']
    if set(pred) != {f['frame'] for f in truth['frames']}:
        raise ValueError('Predictions and reviewed frames must match exactly')
    totals, strata, role_counts, groups, stratum_groups = Counter(), {}, Counter(), {}, {}
    confusion = {role: Counter() for role in ('player','goalkeeper','referee','other','unknown')}
    for role in ('player','goalkeeper','goalkeeper_H','goalkeeper_A','referee','other','unknown'):
        strata['role:'+role] = Counter()
    for frame in truth['frames']:
        people = frame['people']
        detections, outside = [], 0
        for detection in pred[frame['frame']]:
            validate_box(detection.get('box'), optional=True)
            cls,role = detection.get('classification'),detection.get('role')
            if (cls not in ('H','A','referee','uncertain')
                    or role not in ('player','goalkeeper','referee','unknown')
                    or (cls in ('H','A') and role not in ('player','goalkeeper'))
                    or (cls=='referee' and role!='referee')
                    or (cls=='uncertain' and role!='unknown')
                    or (detection.get('box') is None and cls!='uncertain')):
                raise ValueError('Invalid prediction classification/role')
            box = detection.get('box')
            if box and (box[2]>frame['width'] or box[3]>frame['height']):
                raise ValueError('Prediction box outside image')
            if truth['evaluation_mode']=='verified_regions' or in_region(box,frame['evaluation_region']):
                detections.append(detection)
            else:
                outside += 1
        if truth['evaluation_mode']=='verified_regions':
            labels = [d.get('region_label') for d in detections]
            if len(set(labels))!=len(labels) or set(labels)!={p['label'] for p in people} or outside:
                raise ValueError('Verified-region predictions must match every frozen region exactly')
            edges = [[j for j,p in enumerate(people) if p['label']==d['region_label']] for d in detections]
        else:
            edges = [[j for _,j in sorted([(-iou(d.get('box'),p['box']),j)
                        for j,p in enumerate(people) if iou(d.get('box'),p['box'])>=threshold])]
                     for d in detections]
        owner = {}
        def assign(i, visited):
            for j in edges[i]:
                if j in visited:
                    continue
                visited.add(j)
                if j not in owner or assign(owner[j],visited):
                    owner[j] = i
                    return True
            return False
        for i in range(len(detections)):
            assign(i,set())
        matched = set(owner.values())
        count = Counter(detections=len(detections),outside_region_detections=outside,
            accepted=sum(d['classification']!='uncertain' for d in detections),
            uncertain_detections=sum(d['classification']=='uncertain' for d in detections),
            duplicates=sum(bool(edges[i]) for i in range(len(detections)) if i not in matched),
            unmatched_detections=len(detections)-len(matched),people=len(people),
            missed_people=len(people)-len(owner))
        group = (frame['source_sha256'],frame['group_id'])
        for j,person in enumerate(people):
            d = detections[owner[j]] if j in owner else None
            accepted = d is not None and d['classification']!='uncertain'
            correct = bool(accepted and person['classification']!='uncertain'
                           and d['classification']==person['classification'])
            joint = bool(correct and d['role']==person['role'])
            athlete = person['evaluable'] and person['role'] in ('player','goalkeeper')
            contribution = Counter(people=1,accepted=int(accepted),correct_accepted=int(correct),
                detections=int(d is not None),
                correct_team_and_role=int(joint),evaluable_athletes=int(athlete),
                covered_athletes=int(athlete and accepted),missed_people=int(d is None),
                uncertain_detections=int(d is not None and not accepted))
            for key in ('correct_accepted','correct_team_and_role','evaluable_athletes','covered_athletes'):
                count[key] += contribution[key]
            role = person['role']
            confusion[role][d['role'] if d else 'missed'] += 1
            if athlete:
                role_counts[role] += 1
                if role=='goalkeeper':
                    role_counts['goalkeeper_'+person['classification']] += 1
            keys = [person['lighting'],person['occlusion'],'role:'+role,
                    'role:'+role+'/lighting:'+person['lighting'],
                    'role:'+role+'/occlusion:'+person['occlusion']]
            if role=='goalkeeper' and person['classification'] in ('H','A'):
                team_key = 'role:goalkeeper_'+person['classification']
                keys += [team_key,team_key+'/lighting:'+person['lighting'],team_key+'/occlusion:'+person['occlusion']]
            for key in keys:
                strata.setdefault(key,Counter()).update(contribution)
                stratum_groups.setdefault(key,{}).setdefault(group,Counter()).update(contribution)
        totals.update(count)
        groups.setdefault(group,Counter()).update(count)
    known = all(f['temporal_origin_known'] for f in truth['frames'])
    result = metrics(totals)
    result.update(schema_version=2,evaluation_mode=truth['evaluation_mode'],sample_kind=truth['sample_kind'],
        synthetic=truth.get('synthetic',False),
        iou_threshold=threshold,frame_count=len(truth['frames']),
        strata={key: metrics(value) for key,value in strata.items()},
        role_confusion={key: dict(value) for key,value in confusion.items()},
        strata_scope='Annotated people only; unassociated extras belong to global precision',
        uncertainty=grouped_intervals(groups,known),
        strata_uncertainty={key: grouped_intervals(
            {g: stratum_groups.get(key,{}).get(g,Counter()) for g in groups},known) for key in strata},
        possession_integration_authorized=False)
    diversity = {key: strata.get(key,Counter())['evaluable_athletes'] for key in ('sun','shade','isolated','overlapped')}
    requirements = dict(independent_groups=len(groups)>=30,accepted_decisions=totals['accepted']>=300,
        goalkeeper_appearances=role_counts['goalkeeper']>=30,goalkeeper_H=role_counts['goalkeeper_H']>=10,
        goalkeeper_A=role_counts['goalkeeper_A']>=10,temporal_origin_known=known,
        diversity=all(diversity.values()),main_sample=truth['sample_kind']=='main',
        real_evidence=not truth.get('synthetic',False))
    result['sampling'] = dict(group_count=len(groups),evaluable_appearances_by_role=dict(role_counts),
        diversity_counts=diversity,requirements=requirements,
        sufficient_for_exploratory_analysis=all(requirements.values()),
        warning='Operational minima are not a statistical power guarantee')
    result['point_targets_observed'] = (result['precision_accepted'] is not None and result['coverage'] is not None
        and result['precision_accepted']>=.95 and result['coverage']>=.8)
    result['targets_observed'] = result['point_targets_observed']
    return result
