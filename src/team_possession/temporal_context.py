"""Bounded causal context of independent evidence, never a vote of filled labels."""
from collections import deque
from ..ball_events.dependencies import Dependencies,combine


class TemporalContext:
    def __init__(self,frames=25,recent_frames=5):
        self.frames=frames;self.recent_frames=recent_frames;self.history=deque(maxlen=frames)
        self.team=None;self.last_frame=None

    def reset(self):
        self.history.clear();self.team=None;self.last_frame=None

    def step(self,frame,origin,base_label,base_reason,control,current,checks,age_s,
             max_age_s,dead=False,barrier=False):
        if (self.last_frame is not None and frame!=self.last_frame+1) or origin!=self.team or dead or barrier:
            self.reset()
        self.team=origin;self.last_frame=frame
        # Facts come from current tracking / explicit visual control or geometry.
        # A contextual output is never fed back as fresh support.
        evidence_team=None;strength=0.
        if control in ('H','A'):evidence_team=control;strength=1.
        elif base_label in ('H','A') and base_reason in (
                'plausible_same_team_pass','observed_receiver_continuity','observed_sender_outgoing_launch',
                'sustained_probable_reception','visual_reviewed_goalkeeper_control','visual_control_corroborated',
                'kickoff_tracking_confirmed','kickoff_center_setup_inferred','verified_restart_tracking_confirmed'):
            evidence_team=base_label;strength=.6
        if not dead and not barrier:
            self.history.append(dict(frame=frame,team=evidence_team,strength=strength,
                support=current['support'] if current else None,control=control))
        self.history=deque((r for r in self.history if frame-self.frames<r['frame']<=frame),maxlen=self.frames)
        weights={'H':0.,'A':0.};facts=[]
        for r in self.history:
            if r['team'] not in weights:continue
            age=frame-r['frame'];weight=(self.frames-age)/self.frames
            if age<self.recent_frames:weight*=2.
            weights[r['team']]+=weight*r['strength'];facts.append(r)
        total=sum(weights.values());fraction=weights.get(origin,0.)/total if total else None
        control_age=min((frame-r['frame'] for r in facts if r['control']==origin),default=None)
        result=dict(accepted=False,window_frames=self.frames,recent_frames=self.recent_frames,
            buffered_frames=len(self.history),independent_evidence_frames=len(facts),weighted_support=weights,
            origin_support_fraction=fraction,recent_control_age_frames=control_age,
            base_label=base_label,base_reason=base_reason,checks=dict(checks),support=None,
            confidence='insufficient',probabilities_calibrated=False,reason='base_decision_preserved')
        if base_label!='X' or origin not in ('H','A'):return result
        allowed=base_reason in ('no_supported_receiver','no_control_or_pass_motion','trajectory_not_supported')
        requirements=dict(eligible_unknown_reason=allowed,origin_within_limit=age_s is not None and age_s<=max_age_s,
            enough_independent_frames=len(facts)>=self.recent_frames,
            no_recent_opponent_control=not any(r['control'] in ('H','A') and r['control']!=origin and frame-r['frame']<self.recent_frames for r in facts),
            consistent_history=fraction is not None and fraction>=.75,
            **checks)
        result['checks']=requirements
        # Slow/temporarily unmodelled motion needs a nearby observed teammate and
        # a recent control, not just an old majority or a stationary ball.
        if base_reason in ('no_control_or_pass_motion','trajectory_not_supported'):
            requirements['recent_control']=control_age is not None and control_age<=self.recent_frames
            requirements['nearby_own_player']=current is not None and current['distances'][origin] is not None and current['distances'][origin]<=2.
        if not all(requirements.values()):
            result['reason']='context_requirements_not_met';return result
        support=[]
        for r in facts:
            if r['support']:
                d=r['support'];support.append(Dependencies(d['support_start_frame'],d['support_end_frame'],d['available_at_frame'],
                    frozenset(d['point_refs']),frozenset(d['hint_ids']),frozenset(d['event_ids'])))
        result.update(accepted=True,confidence='context_supported_not_observed_control',reason='causal_context_continuity',
            support=combine(deps=support,frame=frame).as_dict() if support else None)
        return result
