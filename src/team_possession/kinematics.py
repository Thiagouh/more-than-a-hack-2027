"""Causal metric primitives. Arrival is an opportunity, never a control probability."""
from collections import deque
import math

from ..ball_events.dependencies import Dependencies, combine


def xy(p):
    return (p['x_m'], p['y_m'])


def length(v):
    return math.hypot(*v)


def add(p, v, t):
    return (p[0]+v[0]*t, p[1]+v[1]*t)


def angle_difference(a, b):
    return abs((a-b+180) % 360-180)


class MotionWindow:
    def __init__(self, fps):
        self.fps = fps
        self.samples = deque(maxlen=5)
        self.errors = deque(maxlen=5)
        self.previous_fit = None

    def clear(self):
        self.samples.clear()
        self.errors.clear()
        self.previous_fit = None

    def update(self, point, deps, valid):
        f = point['frame']
        if not valid:
            self.clear()
            return None, None
        if self.samples and self.samples[-1][0]['frame'] != f-1:
            self.clear()
        previous = self.previous_fit
        error = None
        if previous:
            prediction = add(xy(previous['point']), previous['velocity'],
                             (f-previous['point']['frame'])/self.fps)
            value = math.dist(prediction, xy(point))
            error = (value, previous['deps'].union(deps))
            self.errors.append(error)
        self.samples.append((point, deps))
        if len(self.samples) < 3:
            return None, error
        times = [(p['frame']-f)/self.fps for p,_ in self.samples]
        mean_t = sum(times)/len(times)
        denom = sum((t-mean_t)**2 for t in times)
        means={key:sum(p[key] for p,_ in self.samples)/len(times) for key in ('x_m','y_m')}
        velocity = tuple(sum((t-mean_t)*(p[key]-means[key]) for t,(p,_) in zip(times,self.samples))/denom
                         for key in ('x_m','y_m'))
        fit = dict(point=point, velocity=velocity, samples=len(times),
                   deps=combine(deps=[d for _,d in self.samples], frame=f))
        self.previous_fit = fit
        return fit, error


def time_to_reach(position, velocity, target, cfg):
    """Reaction at current velocity, then directed acceleration and capped speed.

    Directional speed is signed: motion away must first be reversed. Orthogonal
    velocity is discarded AFTER reaction (optimistic turning approximation).
    """
    radius = cfg['control_radius_m']
    if math.dist(position,target) <= radius:
        return 0.
    reaction = cfg['reaction_time_s']
    # Passive motion can enter the control radius before reaction completes.
    relative=(position[0]-target[0],position[1]-target[1])
    vv=sum(v*v for v in velocity)
    pv=sum(p*v for p,v in zip(relative,velocity))
    discriminant=pv*pv-vv*(sum(p*p for p in relative)-radius*radius)
    if vv>0 and discriminant>=0:
        entry=(-pv-math.sqrt(discriminant))/vv
        if 0<=entry<=reaction:return entry
    reacted = add(position,velocity,reaction)
    delta = (target[0]-reacted[0],target[1]-reacted[1])
    distance = length(delta)
    if distance <= radius:
        return reaction
    distance -= radius
    vmax, acceleration = cfg['max_speed_mps'],cfg['acceleration_mps2']
    if length(velocity) > vmax+1e-9:
        return None  # Explicitly unavailable; do not silently clip the observation.
    direction = (delta[0]/length(delta),delta[1]/length(delta))
    initial = sum(a*b for a,b in zip(velocity,direction))
    ramp_time = (vmax-initial)/acceleration
    ramp_distance = initial*ramp_time+.5*acceleration*ramp_time*ramp_time
    if distance <= ramp_distance:
        travel = (-initial+math.sqrt(initial*initial+2*acceleration*distance))/acceleration
    else:
        travel = ramp_time+(distance-ramp_distance)/vmax
    return reaction+travel


def arrival_metrics(ball, ball_fit, players, fits, cfg, meta):
    """Returns per-team opportunities, candidate explanations and a forecast.

    Horizon censoring and missing motion have distinct reasons. Exposure is a
    fraction of sampled forecast times, NOT an interception probability.
    """
    reason = 'ball_motion_unavailable' if ball_fit is None else None
    result = {'teams':{}, 'candidates':[], 'forecast_xy':[], 'censored_candidates':[]}
    for team in ('H','A'):
        result['teams'][team] = dict(meeting_time_s=None,reason=reason or 'insufficient_coverage',
            candidates_count=0,exposure_fraction=None,approach_distance_m=None,relative_speed_mps=None)
    if reason:
        return result
    horizon = cfg['forecast_horizon_s']
    steps = math.floor(horizon*meta['fps']+1e-9)
    times = [i/meta['fps'] for i in range(1,steps+1)]
    if not times:
        return result
    # The trajectory ends at the field boundary, not beyond an invented continuation.
    forecast = [(tau,add(xy(ball),ball_fit['velocity'],tau)) for tau in times]
    forecast = [(tau,p) for tau,p in forecast if 0<=p[0]<=meta['pitch_m'][0] and 0<=p[1]<=meta['pitch_m'][1]]
    result['forecast_xy'] = [list(p) for _,p in forecast]
    if not forecast:
        for team in result['teams'].values():team['reason']='forecast_outside_pitch'
        return result
    for team in ('H','A'):
        pool = [p for p in players if p['team']==team]
        valid = [p for p in pool if p['track_id'] in fits and length(fits[p['track_id']]['velocity'])<=cfg['max_speed_mps']]
        info=result['teams'][team]
        if not pool:continue
        if not valid:
            info['reason']='motion_unavailable' if not any(p['track_id'] in fits for p in pool) else 'speed_outside_arrival_model'
            continue
        exposed=[False]*len(forecast)
        approaches=[]
        for p in valid:
            velocity=fits[p['track_id']]['velocity']
            meeting=None
            for i,(tau,q) in enumerate(forecast):
                tti=time_to_reach(xy(p),velocity,q,cfg)
                if tti is not None and tti<=tau+1e-9:
                    exposed[i]=True
                    if meeting is None:meeting=tau
            if meeting is None:
                result['censored_candidates'].append(dict(track_id=p['track_id'],team=team,reason='no_meeting_within_horizon'))
                continue
            approach=min(math.dist(add(xy(p),velocity,tau),q) for tau,q in forecast)
            relative=length(tuple(a-b for a,b in zip(ball_fit['velocity'],velocity)))
            approaches.append((meeting,p['track_id'],approach,relative))
            result['candidates'].append(dict(track_id=p['track_id'],team=team,position_xy=list(xy(p)),
                meeting_time_s=meeting,approach_distance_m=approach,relative_speed_mps=relative))
        info['exposure_fraction']=sum(exposed)/len(forecast)
        info['candidates_count']=len(approaches)
        info['reason']='no_meeting_within_horizon'
        if approaches:
            meeting,selected,_,relative=min(approaches)
            selected_point=next(p for p in valid if p['track_id']==selected)
            info.update(meeting_time_s=meeting,reason=None,relative_candidate_distance_m=math.dist(xy(ball),xy(selected_point)),
                        approach_distance_m=min(a[2] for a in approaches),relative_speed_mps=relative)
    result['candidates'].sort(key=lambda p:(p['meeting_time_s'],p['track_id']))
    return result
