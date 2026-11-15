"""Finite, source-reviewed ATD quality overlay; raw coordinates stay immutable."""
import copy
import json
from pathlib import Path
from ..ball_events import contract as c
from ..ball_events.artifacts import sha


def load_tracking_review(path,meta,video):
    path=Path(path);policy=json.loads(path.read_text())
    if (policy.get('schema')!='reviewed-tracking-quality-v1' or policy.get('timeline')!=meta['timeline']
            or video is None or sha(video)!=policy.get('source_video_sha256')):
        raise c.InputError('Tracking review must match the source video')
    c.text(policy.get('reviewer'),'tracking reviewer')
    if sha(path.parent/policy['goalkeeper_policy'])!=policy['goalkeeper_policy_sha256']:
        raise c.InputError('Goalkeeper side policy changed')
    for r in policy['evidence']:
        c.integer(r['frame'],'review frame')
        if sha(path.parent/r['image'])!=r['image_sha256']:raise c.InputError('Review image changed')
    for rule in policy['rules']:
        for key in ('rule_id','track_id','reason'):c.text(rule.get(key),key)
        for key in ('start_frame','end_frame_exclusive','available_at_frame'):c.integer(rule.get(key),key)
        if rule['start_frame']>=rule['end_frame_exclusive'] or rule['available_at_frame']>rule['start_frame']:
            raise c.InputError('Review requires bounded, declared offline availability')
        if rule.get('action') not in ('reject_observation','goalkeeper_role'):raise c.InputError('Unknown tracking review action')
        if rule['action']=='goalkeeper_role' and rule.get('team') not in ('H','A'):raise c.InputError('Invalid goalkeeper team')
        if rule.get('x_max_m') is not None:c.number(rule['x_max_m'],'review x bound',positive=True)
    if policy.get('atd_raw_sha256')!=sha('data/tactical/BAR-ATH_tactical_atd.txt'):
        raise c.InputError('Tracking review ATD changed')
    if policy.get('homography_sha256')!=sha('data/tactical/BAR-ATH_tactical_homography.csv'):
        raise c.InputError('Tracking review homography changed')
    assignments={}
    for row in policy.get('frame_assignments',[]):
        for key in ('frame','available_at_frame'):c.integer(row[key],key)
        if row['available_at_frame']<row['frame'] or row['frame'] in assignments or row.get('team')!='H':
            raise c.InputError('Invalid goalkeeper association')
        if row.get('track_id') not in ('24','46') or len(row.get('raw_xy',[]))!=2:raise c.InputError('Invalid source goalkeeper point')
        assignments[row['frame']]=row
    policy['assignments_by_frame']=assignments
    return policy


def apply_tracking_review(envelope,policy):
    if policy is None:return envelope,[]
    result=copy.deepcopy(envelope);f=envelope['frame'];diagnostics=[]
    for rule in policy['rules']:
        if not rule['start_frame']<=f<rule['end_frame_exclusive'] or rule['available_at_frame']>f:continue
        for p in result['players']:
            if p['track_id']!=rule['track_id'] or p['source']!='observed' or p.get('available_at_frame',f)>f:continue
            if rule.get('x_max_m') is not None and p['x_m']>rule['x_max_m']:continue
            if rule.get('x_min_m') is not None and p['x_m']<rule['x_min_m']:continue
            if rule['action']=='reject_observation':
                p['quality_flags']=sorted(set(p.get('quality_flags',[]))|{'reviewed_unreliable_goalkeeper_observation'})
            elif p['team']!=rule['team']:
                # Role verification alone never silently rewrites a team's identity.
                diagnostics.append(dict(rule_id=rule['rule_id'],track_id=p['track_id'],action='abstain_team_conflict',reason=rule['reason']));continue
            diagnostics.append(dict(frame=f,rule_id=rule['rule_id'],track_id=p['track_id'],action=rule['action'],
                team=rule.get('team'),role='goalkeeper' if rule['action']=='goalkeeper_role' else None,
                reason=rule['reason'],available_at_frame=f,source='offline_rgb_review',raw_xy=[p['x_m'],p['y_m']]))
    assignment=policy.get('assignments_by_frame',{}).get(f)
    if assignment and assignment['available_at_frame']<=f:
        target=next((p for p in envelope['players'] if p['track_id']==assignment['track_id']
            and p['source']=='observed' and not p.get('quality_flags') and p.get('available_at_frame',f)<=f),None)
        if target and abs(target['x_m']-assignment['raw_xy'][0])<1e-6 and abs(target['y_m']-assignment['raw_xy'][1])<1e-6:
            for p in result['players']:
                if p['track_id'] in ('24','46') and p['source']=='observed' and p['x_m']<52.65:
                    p['quality_flags']=sorted(set(p['quality_flags'])|{'reviewed_unreliable_goalkeeper_observation'})
                    if p['track_id']!=target['track_id']:
                        diagnostics.append(dict(frame=f,rule_id='goalkeeper-conflicting-point',track_id=p['track_id'],
                            action='reject_observation',reason='Competing point does not match the current green keeper feet',raw_xy=[p['x_m'],p['y_m']]))
            replacement=dict(target,track_id='reviewed-gk-H',team='H',quality_flags=[])
            result['players'].append(replacement)
            diagnostics.append(dict(frame=f,rule_id='rgb-goalkeeper-association',track_id=target['track_id'],
                canonical_track_id='atd:reviewed-gk-H',action='goalkeeper_assignment',team='H',role='goalkeeper',
                raw_xy=[target['x_m'],target['y_m']],reason=assignment['quality'],
                source='same_frame_rgb_cue_and_reviewed_goal_context',available_at_frame=f,
                image_rgb_sha256=assignment['image_rgb_sha256'],box=assignment['box'],cost_px=assignment['cost_px']))
    return result,diagnostics
