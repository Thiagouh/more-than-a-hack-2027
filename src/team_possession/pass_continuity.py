"""Observed receiver continuity, bounded by the existing last-control clock."""
import math


class PassContinuity:
    def __init__(self):
        self.candidates={};self.previous_ball=None;self.team=None

    def reset(self):
        self.__init__()

    def step(self,frame,origin,players,ball,velocity,cfg,checks,arrivals):
        if origin!=self.team:self.reset();self.team=origin
        result=dict(accepted=False,checks=dict(checks),candidates=[],abandoned=[],reason='requirements_not_met')
        if not all(checks.values()) or velocity is None:
            result['abandoned']=list(self.candidates);self.candidates={};self.previous_ball=None
            return result
        speed=math.hypot(*velocity)
        own={p['track_id']:p for p in players if p['team']==origin and p['source']=='observed'
             and not p.get('quality_flags') and p.get('available_at_frame',frame)<=frame}
        next_candidates={}
        for tid,p in own.items():
            dx,dy=p['x_m']-ball['x_m'],p['y_m']-ball['y_m'];distance=math.hypot(dx,dy)
            along=(dx*velocity[0]+dy*velocity[1])/speed if speed else -1
            lateral=abs(dx*velocity[1]-dy*velocity[0])/speed if speed else math.inf
            tau=along/speed if speed else math.inf
            old=self.candidates.get(tid)
            approaching=(old is not None and distance<old['distance_m'] and old['frame']==frame-1)
            ray=(cfg['control_radius_m']<along and 0<tau<=cfg['forecast_horizon_s']
                 and lateral<=cfg['control_radius_m'])
            supported=any(c['track_id']==tid and c['team']==origin for c in arrivals)
            # A retained receiver must stay observed and get closer every frame.
            # No missing defender is interpreted as an infinite advantage.
            if (approaching or ray or supported) and distance>cfg['control_radius_m']:
                next_candidates[tid]=dict(track_id=tid,frame=frame,distance_m=distance,
                    started_frame=old['started_frame'] if old else frame,
                    previous_distance_m=old['distance_m'] if old else None,previous_frame=old['frame'] if old else None,
                    approaching=approaching,ray_supported=ray,current_arrival_supported=supported,
                    lateral_m=lateral,ray_time_s=tau if tau>=0 and math.isfinite(tau) else None)
        result['abandoned']=sorted(set(self.candidates)-set(next_candidates))
        self.candidates=next_candidates;self.previous_ball=(ball['x_m'],ball['y_m'])
        result.update(accepted=bool(next_candidates),candidates=list(next_candidates.values()),
                      reason='observed_receiver_continuity' if next_candidates else 'no_observed_approaching_receiver')
        return result
