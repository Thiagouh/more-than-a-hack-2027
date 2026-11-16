"""Direct team metrics plus independently reviewed association/identity errors."""
from collections import Counter
from .evaluation import evaluate, iou


def evaluate_identity(truth, predictions):
    report = evaluate(truth, predictions)
    by_frame = {r['frame']: r for r in predictions}
    counts, strata, previous = Counter(), {}, {}
    for frame in sorted(truth['frames'], key=lambda r: r['frame']):
        detections = by_frame[frame['frame']]['detections']
        people = frame['people']
        edges = [[j for _,j in sorted((-iou(d.get('box'),p['box']),j) for j,p in enumerate(people)
                                      if iou(d.get('box'),p['box']) >= .5)] for d in detections]
        owner = {}
        def assign(i, visited):
            for j in edges[i]:
                if j not in visited:
                    visited.add(j)
                    if j not in owner or assign(owner[j],visited):
                        owner[j]=i
                        return True
            return False
        for i in range(len(detections)):
            assign(i,set())
        for j, person in enumerate(people):
            if not person['evaluable'] or person['role'] not in ('player','goalkeeper'):
                continue
            d = detections[owner[j]] if j in owner else None
            c = Counter(athletes=1, missed=int(d is None))
            classified = bool(d and d['classification'] in ('H','A'))
            c['classified'] = int(classified)
            c['team_errors'] = int(classified and d['classification'] != person['classification'])
            c['uncertain'] = int(d is not None and not classified)
            c['identity_reviewed'] = int(bool(person.get('person_id')))
            if person.get('association_reviewed') is True:
                c['association_reviewed'] = 1
                assigned = bool(d and d.get('association_status') in ('matched','recovered'))
                c['associated'] = int(assigned)
                c['association_errors'] = int(assigned and d.get('target_track_id') != person.get('expected_track_id'))
            if person.get('person_id') and d and d.get('identity'):
                key = (frame['source_sha256'], frame['group_id'], person['person_id'])
                if key in previous:
                    c['identity_transitions'] = 1
                    c['identity_switches'] = int(previous[key] != d['identity'])
                previous[key] = d['identity']
            counts.update(c)
            strata.setdefault(person['role'], Counter()).update(c)
    def summarize(c):
        result = {k:int(c[k]) for k in ('athletes','classified','team_errors','uncertain','missed',
            'association_reviewed','associated','association_errors','identity_reviewed','identity_transitions','identity_switches')}
        result['team_accuracy_accepted'] = 1-c['team_errors']/c['classified'] if c['classified'] else None
        result['team_coverage'] = c['classified']/c['athletes'] if c['athletes'] else None
        return result
    report['identity'] = summarize(counts)
    report['identity_by_role'] = {k:summarize(v) for k,v in strata.items()}
    report['association_validation_complete'] = counts['association_reviewed'] == counts['athletes'] and counts['athletes'] > 0
    report['identity_tracking_review_complete'] = counts['identity_reviewed'] == counts['athletes'] and counts['athletes'] > 0
    return report


def promotion_gate(candidate, baseline):
    a,b = candidate['identity'],baseline['identity']
    if (not candidate['association_validation_complete'] or not baseline['association_validation_complete']
            or not candidate.get('identity_tracking_review_complete') or not baseline.get('identity_tracking_review_complete')
            or a['athletes'] != b['athletes']):
        return dict(accepted=False,reason='incomplete_or_incomparable_review')
    checks = dict(fewer_team_errors=a['team_errors'] < b['team_errors'],
                  coverage_preserved=a['classified'] >= b['classified'],
                  associations_not_worse=a['association_errors'] <= b['association_errors'],
                  association_coverage_preserved=a['associated'] >= b['associated'],
                  identity_switches_not_worse=a['identity_switches'] <= b['identity_switches'])
    return dict(accepted=all(checks.values()), checks=checks,
                reason='direct_identification_only; possession requires separate regression comparison')


def evaluate_atd_correspondences(truth, envelopes):
    """ATD labels at reviewed correspondences: oracle association, not detection."""
    predictions=[]
    by_frame={r['frame']:r for r in envelopes}
    for frame in truth['frames']:
        raw={p['track_id']:p for p in by_frame[frame['frame']]['players']}
        detections=[]
        for person in frame['people']:
            if person.get('association_reviewed') is not True:
                raise ValueError('ATD baseline requires every person/slot correspondence reviewed')
            slot=person.get('expected_track_id')
            p=raw.get(slot)
            if p is None or p['source']!='observed' or p.get('quality_flags') or p.get('available_at_frame',frame['frame'])>frame['frame']:
                continue
            detections.append(dict(box=person['box'],classification=p['team'],role='player',identity='atd-slot-'+slot,
                association_status='matched',target_track_id=slot))
        predictions.append(dict(frame=frame['frame'],detections=detections))
    report=evaluate_identity(truth,predictions)
    report['baseline_method']='ATD labels at reviewed person/slot correspondences; oracle association, not detection metrics'
    return report
