"""Evaluation with explicit independent references; unavailable is not zero accuracy."""
from collections import Counter, defaultdict
import math

from ..ball_events import contract as c
from ..ball_events.evaluation import matching, stats, MATCHING_VERSION


def error_summary(values):
    result=stats(values,len(values))
    result.update(mae=sum(abs(v) for v in values)/len(values) if values else None,
                  rmse=math.sqrt(sum(v*v for v in values)/len(values)) if values else None,
                  anticipations=sum(v<0 for v in values))
    return result


def pass_scores(states, passes):
    """Pass intervals are half-open ABSOLUTE frames; CSV possession is not a pass reference."""
    if passes is None:
        return dict(available=False,reason='independent_pass_reference_not_provided')
    c.array(passes,'pass reference')
    by_frame={s['frame']:s for s in states}
    records=[]
    for ref in passes:
        c.obj(ref,'pass record')
        start=c.integer(ref['start_frame'],'pass start');end=c.integer(ref['end_frame_exclusive'],'pass end',1)
        if start>=end or ref['team'] not in ('H','A'):raise c.InputError('Invalid reference pass interval or team')
        if any(f not in by_frame for f in range(start,end)):raise c.InputError('Reference pass outside evaluated frames')
        labels=[by_frame[f]['label'] for f in range(start,end)]
        opposite='A' if ref['team']=='H' else 'H'
        records.append(dict(team=ref['team'],kind=ref.get('kind','unknown'),role=ref.get('role','unknown'),
            aerial=ref.get('aerial','unknown'),coverage=ref.get('coverage','unknown'),
            frames=len(labels),correct=labels.count(ref['team']),abstain=labels.count('X'),
            wrong_team=labels.count(opposite),whole_pass_correct=all(l==ref['team'] for l in labels)))
    def summarize(items):
        frames=sum(r['frames'] for r in items);n=len(items)
        return dict(reference_passes=n,reference_frames=frames,
            retention=sum(r['correct'] for r in items)/frames if frames else None,
            integral_pass_retention=sum(r['whole_pass_correct'] for r in items)/n if n else None,
            false_turnovers_per_100_passes=100*sum(r['wrong_team']>0 for r in items)/n if n else None,
            abstention=sum(r['abstain'] for r in items)/frames if frames else None)
    strata={}
    for field in ('team','kind','role','aerial','coverage'):
        groups=defaultdict(list)
        for row in records:groups[str(row[field])].append(row)
        strata[field]={label:summarize(items) for label,items in groups.items()}
    return dict(available=True,overall=summarize(records),strata=strata,records=records)


def accepted_recoveries(states, events):
    by_id={e['event_id']:e for e in events};previous=None;segment=None;result=[]
    for row in states:
        if row.get('segment_start_frame',0)!=segment:
            previous=None;segment=row.get('segment_start_frame',0)
        for event_id in row.get('released_acquisitions',[]):
            event=by_id[event_id]
            if previous is not None and event['team']!=previous:
                result.append(event)
            previous=event['team']
        if previous is None and row.get('origin_team') in ('H','A'):previous=row['origin_team']
    return result


def recovery_scores(states, events, reference, fps, tolerance_s=.2):
    c.number(tolerance_s,'recovery matching tolerance',positive=True)
    if reference is None:return dict(available=False,reason='independent_recovery_reference_not_provided')
    c.array(reference,'recovery reference')
    for row in reference:
        c.obj(row,'recovery');c.text(row['reference_id'],'reference_id');c.number(row['time_s'],'recovery time')
        if row['team'] not in ('H','A'):raise c.InputError('Invalid reference recovery team')
    if len({r['reference_id'] for r in reference})!=len(reference):raise c.InputError('Duplicate recovery reference ID')
    predicted=accepted_recoveries(states,events)
    pairs=matching(predicted,reference,fps,tolerance_s)
    stats_by_team={}
    for team in ('H','A'):
        ps=[p for p in predicted if p['team']==team];rs=[r for r in reference if r['team']==team]
        matched=[(p,r) for p,r in pairs if p['team']==team]
        stats_by_team[team]=dict(predicted=len(ps),reference=len(rs),matched=len(matched),
            precision=len(matched)/len(ps) if ps else None,recall=len(matched)/len(rs) if rs else None)
    return dict(available=True,matching_version=MATCHING_VERSION,tolerance_s=tolerance_s,
        predicted=len(predicted),reference=len(reference),matched=len(pairs),
        precision=len(pairs)/len(predicted) if predicted else None,recall=len(pairs)/len(reference) if reference else None,
        event_time_error_s=error_summary([p['event_frame']/fps-r['time_s'] for p,r in pairs]),
        confirmation_delay_s=error_summary([p['confirmed_at_frame']/fps-r['time_s'] for p,r in pairs]),
        delivery_delay_s=error_summary([p['available_at_frame']/fps-r['time_s'] for p,r in pairs]),
        by_team=stats_by_team,pairs=[dict(event_id=p['event_id'],reference_id=r['reference_id']) for p,r in pairs])


def meeting_error_scores(states, reference, fps):
    if reference is None:return dict(available=False,reason='independent_meeting_reference_not_provided')
    by_frame={s['frame']:s for s in states};errors=[];missing=0;groups=defaultdict(list)
    for row in reference:
        c.obj(row,'meeting reference');frame=c.integer(row['prediction_frame'],'prediction frame')
        c.number(row['reference_time_s'],'reference meeting time')
        if row['team'] not in ('H','A') or frame not in by_frame:raise c.InputError('Invalid meeting reference team/frame')
        snapshot=by_frame[frame]
        key='origin_team_meeting_time_s' if row['team']==snapshot['origin_team'] else 'opponent_meeting_time_s'
        predicted=snapshot['metrics'][key]
        if predicted is None:missing+=1;continue
        error=frame/fps+predicted-row['reference_time_s'];errors.append(error)
        groups[row['team']].append(error)
    return dict(available=True,reference=len(reference),unavailable_predictions=missing,
        errors_s=error_summary(errors),by_team={t:error_summary(es) for t,es in groups.items()})


def evaluate_collective(states, events, fps, reference_labels=None, passes=None, recoveries=None, meetings=None, tolerance_s=.2):
    c.number(fps,'fps',positive=True)
    c.array(states,'states')
    if any(b['frame']!=a['frame']+1 for a,b in zip(states,states[1:])):raise c.InputError('Evaluation frames must be contiguous')
    errors=[];seen=set()
    for s in states:
        mf=s.get('metric_frame')
        error=s['metrics'].get('trajectory_error_m')
        if mf is not None and mf not in seen and error is not None:errors.append(error);seen.add(mf)
    errors_summary=error_summary(errors)
    result=dict(state_counts=dict(Counter(s['state'] for s in states)),
        label_counts=dict(Counter(s['label'] for s in states)),
        trajectory_error_m=dict(one_step_horizon_s=1/fps,**errors_summary),
        pass_metrics=pass_scores(states,passes),recovery_metrics=recovery_scores(states,events,recoveries,fps,tolerance_s),
        meeting_error=meeting_error_scores(states,meetings,fps),reference_was_model_input=False)
    if reference_labels is None:result['possession_metrics']=dict(available=False,reason='possession_reference_not_provided')
    else:
        from evaluation import possession_metrics as pm
        if len(reference_labels)!=len(states):raise c.InputError('Reference and prediction length differ')
        result['possession_metrics']=dict(available=True,official=pm.evaluate(pm.label_codes(reference_labels),pm.label_codes([s['label'] for s in states]),bootstrap_n=0))
    return result


def forecast_error_scores(states, reference_positions, fps, horizons_s=(.04,.4,.8,1.2,1.6,2.)):
    """Post-hoc forecast consistency against FUTURE reference positions.

    If supplied positions are from the tracker, this is not calibrated physical
    accuracy. This function never changes predictions or fills missing samples.
    """
    if reference_positions is None:
        return dict(available=False,reason='position_reference_not_provided')
    by_horizon={}
    for horizon in horizons_s:
        c.number(horizon,'forecast evaluation horizon',positive=True)
        steps=round(horizon*fps)
        if steps<1 or abs(steps/fps-horizon)>1e-9:
            raise c.InputError('Forecast evaluation horizon must align to frames')
        errors=[];missing=0;eligible=0
        for state in states:
            if state.get('metric_frame')!=state['frame']:continue
            forecast=state.get('forecast_xy',[])
            if len(forecast)<steps:continue
            eligible+=1
            point=reference_positions.get(state['frame']+steps)
            if point is None:
                missing+=1;continue
            c.array(point,'reference position')
            if len(point)!=2:raise c.InputError('Expected XY reference position')
            for value in point:c.number(value,'reference position')
            errors.append(math.dist(forecast[steps-1],point))
        by_horizon[str(horizon)]=dict(predictions=eligible,missing_reference=missing,error_m=error_summary(errors))
    return dict(available=True,reference_kind='explicit_xy_reference',by_horizon=by_horizon)
