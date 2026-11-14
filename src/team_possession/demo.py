"""Independent synthetic pass references; never inferred from model predictions."""
import math
from ..ball_events.demo import metadata, point


def case(name='short',distance=8.,speed=10.,two_receivers=False,interception=False,gap=None,missing_defender=False,deflection=False):
    landing=7+math.ceil(distance/(speed/25))
    n=landing+18
    frames=[]
    receiver=20+distance+(12 if interception else 0)
    for f in range(n):
        x=20 if f<8 else 20+min(distance,(f-7)*speed/25)
        y=20.+(min(2.,max(0,f-18)*.15) if deflection else 0)
        players=[point(f,20,track_id='sender',team='H'),point(f,receiver,y=22 if deflection else 20,track_id='receiver',team='H'),
                 point(f,20+distance if interception else 45,y=20 if interception else 45,track_id='a1',team='A'),
                 point(f,75,y=50,track_id='a2',team='A')]
        if two_receivers:
            players[1]['y_m']=19
            players.append(point(f,receiver,y=21,track_id='receiver2',team='H'))
        if missing_defender and 12<=f<=landing:
            players=[p for p in players if p['team']=='H']
        ball=point(f,x,y=y)
        if gap and gap[0]<=f<gap[1]:ball=point(f,None,source='missing')
        frames.append(dict(frame=f,ball=ball,players=players))
    meta=metadata(n);meta['dataset_id']='collective-synthetic-'+name
    passes=[] if interception else [dict(start_frame=8,end_frame_exclusive=landing+1,team='H',kind='long' if distance>12 else 'short',role='player',aerial='unknown',coverage='low' if missing_defender else 'observed')]
    recoveries=[dict(reference_id='recovery-a',time_s=(landing+1)/25,team='A')] if interception else []
    return dict(name=name,manifest=meta,frames=frames,passes=passes,recoveries=recoveries,
        expected='adversary_only_after_control' if interception else 'uncertainty_with_missing_coverage' if missing_defender or gap else 'preserve_H',
        limitations=['synthetic_not_match_accuracy','no_ball_height'])


def cases():
    return [case(),case('long',distance=18,speed=8),case('two_receivers',two_receivers=True),
        case('interception',interception=True),case('deflection_without_opponent_control',deflection=True),
        case('short_occlusion',gap=(14,17)),case('long_occlusion',gap=(14,25)),
        case('missing_defender',missing_defender=True)]
