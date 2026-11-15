"""Versioned causal collective-possession metrics and experimental pass policy.

No ground truth is accepted here. Predictions are opportunities, not probabilities.
"""
from collections import Counter, deque
import heapq
import math
import copy

from ..ball_events import contract as c
from ..ball_events.detector import Detector, observed
from ..ball_events.dependencies import Dependencies, combine
from .restarts import validate_kickoffs, kickoff_setup
from .visual_control import VisualControl
from .game_state import GameState
from .pass_continuity import PassContinuity
from .temporal_context import TemporalContext
from .tracking_review import apply_tracking_review
from .kinematics import MotionWindow, length, angle_difference, arrival_metrics

METRICS_VERSION = 'collective-metrics-v2'
POLICY_VERSION = 'collective-context25-v5-experimental'
DEFAULTS = dict(forecast_horizon_s=2.0,reaction_time_s=.2,acceleration_mps2=3.,max_speed_mps=8.,
    control_radius_m=1.5,trajectory_error_limit_m=.75,min_motion_fraction=.5,
    min_observed_players_per_team=2,max_pass_without_control_s=4.,dispute_margin_s=.12,
    minimum_pass_speed_mps=2.,direction_min_speed_mps=2.,missing_ball_grace_frames=5,
    reception_sustain_s=.20,reception_horizon_s=.60,reception_advantage_s=.20,context_frames=25,context_recent_frames=5)
UNITS = {
    'origin_team':'team','last_supported_control_age_s':'s','origin_support_fraction':'fraction',
    'origin_valid_observation_fraction':'fraction','release_interval':'frame interval','ball_speed_fitted_mps':'m/s','ball_heading_deg':'degrees',
    'trajectory_error_m':'m','trajectory_error_rms_m':'m','speed_change_mps':'m/s','heading_change_deg':'degrees',
    'origin_team_meeting_time_s':'s','receiver_candidates_count':'count','receiver_approach_distance_m':'m',
    'receiver_relative_speed_mps':'m/s','opponent_meeting_time_s':'s','meeting_time_advantage_s':'s',
    'opponent_exposure_fraction':'fraction','opponent_motion_change_evidence':'evidence record',
    'opponent_control_streak_frames':'frames','ball_missing_run_frames':'frames',
    'observed_players_by_team':'counts','valid_motion_fraction_by_team':'fractions','evidence_age_s':'s',
    'context_observed_frames':'count','context_support_fraction':'fraction',
    'forecast_horizon_s':'s','censored_candidates':'records','quality_flags':'flags'}


def configuration(overrides=None, fps=25):
    overrides={} if overrides is None else overrides
    c.fields(overrides,set(DEFAULTS),(),'collective configuration')
    cfg={**DEFAULTS,**overrides}
    for key,value in cfg.items():
        if key in ('missing_ball_grace_frames','min_observed_players_per_team','context_frames','context_recent_frames'):
            c.integer(value,key,0 if key=='missing_ball_grace_frames' else 1)
        else:
            c.number(value,key,positive=True)
    if not 0<cfg['min_motion_fraction']<=1 or not 1/fps<=cfg['forecast_horizon_s']<=5:
        raise c.InputError('Invalid coverage fraction or forecast horizon')
    if not cfg['context_recent_frames']<=cfg['context_frames']<=100:raise c.InputError('Invalid context window')
    if cfg['reaction_time_s']>cfg['max_pass_without_control_s'] or cfg['direction_min_speed_mps']>cfg['max_speed_mps']:
        raise c.InputError('Inconsistent collective motion parameters')
    if cfg['reception_horizon_s']>cfg['forecast_horizon_s']:
        raise c.InputError('Reception horizon exceeds forecast horizon')
    return cfg


class Measurements:
    def __init__(self, meta, cfg):
        self.meta,self.cfg=meta,cfg
        self.ball=MotionWindow(meta['fps'])
        self.players={}
        self.streak={'H':0,'A':0}
        self.missing=0

    def reset(self):
        self.ball.clear();self.players.clear();self.streak={'H':0,'A':0};self.missing=0

    def step(self, detector_state, feature):
        s=detector_state;f=s['frame'];ball=s['ball'];players=s['players']
        previous=self.ball.previous_fit
        if not s['ball_ok']:previous=None
        fit,error=self.ball.update(ball,s['deps']['ball_check'],s['ball_ok'])
        fits={};valid=[];counts={t:0 for t in ('H','A')}
        present_ids=set()
        for p in players:
            tid=p['track_id'];present_ids.add(tid)
            ok=observed(p) and tid not in s['jump_ids']|s['duplicate_ids']
            if ok:
                valid.append(p);counts[p['team']]+=1
            window=self.players.setdefault(tid,MotionWindow(self.meta['fps']))
            pfit,_=window.update(p,s['jump_deps'].get(tid,combine(points=[p],frame=f)),ok)
            if pfit is not None:fits[tid]=pfit
        for tid in set(self.players)-present_ids:del self.players[tid]
        eligible=feature['control_eligible_team']
        for t in self.streak:self.streak[t]=self.streak[t]+1 if eligible==t else 0
        self.missing=self.missing+1 if ball['source']!='observed' else 0
        deps=combine(points=players,ball=[ball],frame=f).union(s['deps']['control'],s['deps']['ball_check'])
        if fit:deps=deps.union(fit['deps'])
        if previous:deps=deps.union(previous['deps'])
        for pfit in fits.values():deps=deps.union(pfit['deps'])
        for _,d in self.ball.errors:deps=deps.union(d)
        arrivals=arrival_metrics(ball,fit,valid,fits,self.cfg,self.meta)
        outside=s['ball_ok'] and not (0<=ball['x_m']<=self.meta['pitch_m'][0] and 0<=ball['y_m']<=self.meta['pitch_m'][1])
        if outside:
            arrivals['forecast_xy']=[];arrivals['candidates']=[]
            for team in arrivals['teams'].values():
                team.update(meeting_time_s=None,reason='ball_outside_pitch',candidates_count=0,
                            exposure_fraction=None,approach_distance_m=None,relative_speed_mps=None)
        speed=length(fit['velocity']) if fit else None
        heading=math.degrees(math.atan2(fit['velocity'][1],fit['velocity'][0]))%360 if fit and speed>=self.cfg['direction_min_speed_mps'] else None
        old_speed=length(previous['velocity']) if previous else None
        old_heading=math.degrees(math.atan2(previous['velocity'][1],previous['velocity'][0]))%360 if previous and old_speed>=self.cfg['direction_min_speed_mps'] else None
        values={key:None for key in UNITS}
        values.update(ball_speed_fitted_mps=speed,ball_heading_deg=heading,
            trajectory_error_m=error[0] if error else None,
            trajectory_error_rms_m=math.sqrt(sum(e*e for e,_ in self.ball.errors)/len(self.ball.errors)) if self.ball.errors else None,
            speed_change_mps=speed-old_speed if speed is not None and old_speed is not None else None,
            heading_change_deg=angle_difference(heading,old_heading) if heading is not None and old_heading is not None else None,
            ball_missing_run_frames=self.missing,observed_players_by_team=counts,
            valid_motion_fraction_by_team={t:sum(p['team']==t and p['track_id'] in fits and length(fits[p['track_id']]['velocity'])<=self.cfg['max_speed_mps'] for p in valid)/counts[t] if counts[t] else None for t in counts},
            forecast_horizon_s=self.cfg['forecast_horizon_s'],censored_candidates=arrivals['censored_candidates'],
            quality_flags=sorted(set(feature['quality_flags'])|({'ball_outside_pitch'} if outside else set())|{'ball_height_unavailable','arrival_turning_approximation'}))
        reasons={key:'no_confirmed_origin' for key in UNITS if values[key] is None}
        for key in ('ball_speed_fitted_mps','trajectory_error_m','trajectory_error_rms_m','speed_change_mps'):
            if values[key] is None:reasons[key]='insufficient_consecutive_motion_history'
        for key in ('ball_heading_deg','heading_change_deg'):
            if values[key] is None:reasons[key]='low_speed_or_motion_unavailable'
        row=dict(metrics_version=METRICS_VERSION,frame=f,time_s=f/self.meta['fps'],
            segment_start_frame=c.segment(self.meta,f)[0],values=values,unavailable_reasons=reasons,
            arrivals=arrivals,control_eligible_team=eligible,control_streak=dict(self.streak),
            instant_ball_speed_mps=feature['ball_speed_mps'],
            distances={t:feature[f'{t}_distance_m'] for t in ('H','A')},
            selected_quality={t:feature[f'{t}_quality_flags'] for t in ('H','A')},
            selected_sources={t:feature[f'{t}_source'] for t in ('H','A')},
            ball_valid=s['ball_ok'] and not outside,
            valid_control_observation=s['ball_ok'] and len(s['candidates'])==2,
            support=deps.as_dict(),available_at_frame=deps.available)
        return row


class CollectiveModel:
    """Streaming facade: same Detector events, separate collective state and metrics.

    Inputs may contain delayed points. Entire metric records are queued by their
    dependency availability; a late historical row never overwrites a newer row.
    """
    def __init__(self, meta, detector_config=None, collective_config=None, hints=(), kickoffs=(), tracking_review=None):
        self.meta=c.manifest(copy.deepcopy(meta))
        self.tracking_review=tracking_review
        self.kickoffs=validate_kickoffs(kickoffs,self.meta)
        self.visual_control=VisualControl(self.meta)
        self.kickoff_prior=None
        self.game_state=GameState(self.meta);self.pass_continuity=PassContinuity()
        self.reception=None;self.reception_previous=None;self.reception_support=None
        self.cfg=configuration(collective_config,meta['fps'])
        self.temporal_context=TemporalContext(self.cfg['context_frames'],self.cfg['context_recent_frames'])
        self.detector=Detector(meta,detector_config,hints)
        self.measurements=Measurements(meta,self.cfg)
        self.pending=[];self.serial=0;self.latest=None;self.history={}
        self.origin=None;self.origin_event=None;self.last_supported=None;self.release=None;self.last_control_support=None
        self.watermark=(-1,-1);self.invalid_before=0;self.unavailable_run=0;self.gap_start=None
        self.last_label='X';self.last_frame=-1;self.last_touch=None
        self.counts=Counter();self.state_counts=Counter();self.decision_counts=Counter()

    def enqueue(self, available, kind, value):
        self.serial+=1
        heapq.heappush(self.pending,(available,0 if kind=='event' else 1,self.serial,kind,value))

    def feed(self, envelope, visual_observations=(), game_evidence=(), recovered_observations=()):
        c.envelope(envelope)
        envelope,tracking_diagnostics=apply_tracking_review(envelope,self.tracking_review)
        previous_label=self.last_label
        from .recovery import apply_recoveries
        recovered_players,recovery_diagnostics=apply_recoveries(recovered_observations,envelope,self.meta)
        self.visual_control.validate(visual_observations,envelope['frame'])
        f=envelope['frame'];fps=self.meta['fps']
        # A future coordinate must not alter even a jump/duplicate branch now.
        # Operationally mask it as unavailable; never replay it into past labels.
        unavailable=[]
        def available_point(raw,player=False):
            checked=c.point(raw,f,self.meta,player=player)
            if checked['available_at_frame']<=f:return raw
            unavailable.append(dict(track_id=checked.get('track_id','ball'),declared_available_at_frame=checked['available_at_frame']))
            masked={k:raw[k] for k in ('frame','producer')}
            if player:masked.update(track_id=raw['track_id'],team=raw['team'])
            masked.update(x_m=None,y_m=None,source='missing',quality_flags=['input_not_yet_available'],
                support_start_frame=f,support_end_frame=f+1,available_at_frame=f,
                extensions={'unavailable_input':{'declared_available_at_frame':checked['available_at_frame']}})
            return masked
        envelope=dict(frame=f,ball=available_point(envelope['ball']),
                      players=[available_point(p,True) for p in envelope['players']])
        if (f in self.meta.get('continuity_starts',[]) or f==0 or
                any(r['frame']==f and r['available_at_frame']<=f for r in self.kickoffs)):
            self.measurements.reset();self.latest=None;self.history.clear()
            self.visual_control=VisualControl(self.meta)
            self.game_state.reset();self.pass_continuity.reset();self.temporal_context.reset()
            self.kickoff_prior=None
            self.reception=None;self.reception_previous=None;self.reception_support=None
            self.origin=self.origin_event=self.last_supported=self.release=None
            self.last_control_support=None
            self.watermark=(-1,-1);self.invalid_before=f;self.unavailable_run=0;self.gap_start=None;self.last_label='X';self.last_touch=None
        feature,events=self.detector.feed(envelope)
        state=self.detector.history[-1]
        row=self.measurements.step(state,feature)
        row['unavailable_inputs']=unavailable
        self.counts['unavailable_input_points']+=len(unavailable)
        if unavailable:row['values']['quality_flags']=sorted(set(row['values']['quality_flags'])|{'input_not_yet_available'})
        self.enqueue(row['available_at_frame'],'metric',row)
        for event in events:self.enqueue(event['available_at_frame'],'event',event)
        # Unavailable is not treated as observed merely because its raw row was supplied.
        ball=state['ball']
        visible=ball['available_at_frame']<=f and state['ball_ok']
        if not visible:
            if not self.unavailable_run:self.gap_start=f
            self.unavailable_run+=1
            if self.unavailable_run==self.cfg['missing_ball_grace_frames']+1:
                self.origin=self.origin_event=self.last_supported=self.release=None
                self.last_control_support=None
                self.invalid_before=f+1;self.counts['expired_ball_gaps']+=1
            if self.unavailable_run>self.cfg['missing_ball_grace_frames']:
                self.invalid_before=f+1
        else:
            self.unavailable_run=0;self.gap_start=None
        segment=c.segment(self.meta,f)[0]
        released=[]
        while self.pending and self.pending[0][0]<=f:
            _,_,_,kind,item=heapq.heappop(self.pending)
            if item['segment_start_frame']!=segment:continue
            if kind=='metric':
                if item['frame']>=f-4:self.history[item['frame']]=item
                if self.latest is None or item['frame']>self.latest['frame']:self.latest=item
            elif item['type']=='possible_control_acquisition':
                order=(item['event_frame'],item['confirmed_at_frame'])
                if (self.unavailable_run<=self.cfg['missing_ball_grace_frames'] and item['event_frame']>=self.invalid_before and order>=self.watermark):
                    self.origin=item['team'];self.origin_event=item;self.last_supported=item['confirmed_at_frame']
                    self.last_control_support=item['dependencies']
                    self.release=None;self.watermark=order
                    released.append(item['event_id'])
            elif item['type']=='possible_touch':self.last_touch=item
            # Control loss informs the record but is NOT a collective turnover.
        self.history={k:v for k,v in self.history.items() if k>=f-4}
        current=self.latest
        values={key:None for key in UNITS};reasons={key:'metric_not_yet_available' for key in UNITS}
        arrivals={'teams':{},'candidates':[],'forecast_xy':[],'censored_candidates':[]}
        if current:
            values={**current['values']};reasons={**current['unavailable_reasons']};arrivals=current['arrivals']
            values['evidence_age_s']=(f-current['support']['support_start_frame'])/fps;reasons.pop('evidence_age_s',None)
        fresh=current is not None and current['frame']==f
        observed_control=current['control_eligible_team'] if fresh else None
        kickoff_diagnostic={'reason':'no_kickoff_at_frame'}
        kickoff_applied=False
        for kickoff in self.kickoffs:
            if kickoff['frame']>f or kickoff['available_at_frame']>f:continue
            if f!=max(kickoff['frame'],kickoff['available_at_frame']):continue
            kickoff_diagnostic=dict(kickoff,reason='late_kickoff_ignored')
            if kickoff['frame']!=f:continue
            team=kickoff['team']
            if team is None:
                setup=kickoff_setup(dict(envelope,players=[p for p in state['players'] if p['track_id'] not in state['jump_ids']|state['duplicate_ids']]),self.meta)
                kickoff_diagnostic.update(setup)
                if setup['team'] is not None:
                    team=setup['team']
                    self.origin=team;self.origin_event=None;self.last_supported=f;self.release=None
                    self.last_control_support=combine(points=state['players'],deps=[Dependencies.hint(dict(
                        start_frame=f,available_at_frame=f,hint_id=kickoff['source_id']))],frame=f).as_dict()
                    kickoff_applied=True
                continue
            other='A' if team=='H' else 'H'
            distance=current['distances'][team] if fresh else None
            other_distance=current['distances'][other] if fresh else None
            compatible=(fresh and visible and current['ball_valid'] and distance is not None
                and distance<=self.detector.cfg['proximity_m']
                and current['selected_sources'][team]=='observed' and not current['selected_quality'][team]
                and other_distance is not None and other_distance-distance>=self.detector.cfg['margin_m'])
            kickoff_diagnostic['reason']='kickoff_tracking_confirmed' if compatible else 'kickoff_tracking_conflict_or_missing'
            if compatible:
                self.origin=team;self.origin_event=None;self.last_supported=f;self.release=None
                self.last_control_support=combine(deps=[Dependencies.hint(dict(start_frame=f,
                    available_at_frame=f,hint_id=kickoff['source_id']))],frame=f).as_dict()
                self.last_control_support=combine(deps=[self._dependencies(self.last_control_support),
                    self._dependencies(current['support'])],frame=f).as_dict()
                kickoff_applied=True
        if kickoff_applied:
            self.kickoff_prior=dict(frame=f,team=self.origin,source_id=kickoff_diagnostic['source_id'])
        visual,visual_support=self.visual_control.step(visual_observations,f,observed_control,
            priority=bool(released) or kickoff_applied)
        if visual['accepted']:
            self.origin=visual['team'];self.origin_event=None;self.last_supported=f;self.release=None
            self.last_control_support=visual_support
        game=self.game_state.step(f,ball,state['ball_ok'],game_evidence,kickoff=kickoff_applied)
        restart_tracking_confirmed=False
        if game['dead'] or game['transition'] in ('dead_ball_evidence_expired','verified_restart'):
            self.origin=self.origin_event=self.last_supported=self.release=None
            self.last_control_support=None;self.kickoff_prior=None
            self.reception=None;self.reception_previous=None;self.reception_support=None
            self.pass_continuity.reset();self.invalid_before=f+1
        restart=game.get('restart')
        if (restart and restart.get('team') in ('H','A') and fresh and visible and current['ball_valid']
                and observed_control==restart['team']):
            self.origin=restart['team'];self.last_supported=f;self.invalid_before=f
            self.last_control_support=current['support'];restart_tracking_confirmed=True
        recovery=self._reception_step(current if fresh and visible and not game['dead'] else None,f, bool(released) or kickoff_applied or visual['accepted'])
        recovered=recovery['accepted']
        if recovered:
            self.origin=recovery['team'];self.origin_event=None;self.last_supported=f;self.release=None
            self.last_control_support=self.reception_support
        if self.origin and fresh and observed_control==self.origin and f>=self.invalid_before and not kickoff_applied and not recovered and not visual['accepted']:
            self.last_supported=f;self.release=None;self.last_control_support=current['support']
        if self.origin:
            opposite='A' if self.origin=='H' else 'H'
            values['origin_team']=self.origin
            values['last_supported_control_age_s']=(f-self.last_supported)/fps
            values['origin_support_fraction']=sum(m['control_eligible_team']==self.origin and f-4<=m['frame']<=f for m in self.history.values())/min(5,f-segment+1)
            values['origin_valid_observation_fraction']=sum(m['valid_control_observation'] for m in self.history.values())/min(5,f-segment+1)
            values['release_interval']=self.release
            if current:
                own=arrivals['teams'][self.origin];other=arrivals['teams'][opposite]
                values.update(origin_team_meeting_time_s=own['meeting_time_s'],receiver_candidates_count=own['candidates_count'] if own['reason'] in (None,'no_meeting_within_horizon') else None,
                    receiver_approach_distance_m=own['approach_distance_m'],
                    receiver_relative_speed_mps=own['relative_speed_mps'] if own.get('relative_candidate_distance_m') is not None and own['relative_candidate_distance_m']<=self.cfg['control_radius_m'] else None,
                    opponent_meeting_time_s=other['meeting_time_s'],
                    meeting_time_advantage_s=other['meeting_time_s']-own['meeting_time_s'] if own['meeting_time_s'] is not None and other['meeting_time_s'] is not None else None,
                    opponent_exposure_fraction=other['exposure_fraction'],opponent_control_streak_frames=current['control_streak'][opposite])
                reasons['origin_team_meeting_time_s']=own['reason'];reasons['opponent_meeting_time_s']=other['reason']
                if values['meeting_time_advantage_s'] is None:reasons['meeting_time_advantage_s']='one_or_both_team_times_unavailable'
                if values['receiver_relative_speed_mps'] is None:reasons['receiver_relative_speed_mps']='receiver_not_currently_near_or_motion_unavailable'
                distance=current['distances'][self.origin]
                if fresh and self.release is None and distance is not None and distance>self.detector.cfg['proximity_m'] and current['selected_sources'][self.origin]=='observed' and not current['selected_quality'][self.origin]:
                    self.release=dict(start_frame=self.last_supported,end_frame=f,width_ms=1000*(f-self.last_supported)/fps)
                    values['release_interval']=self.release
                touch=self.last_touch
                near=current['distances'][opposite] is not None and current['distances'][opposite]<=self.detector.cfg['proximity_m']
                change=(values['trajectory_error_m'] is not None and values['trajectory_error_m']>self.cfg['trajectory_error_limit_m']) or (values['heading_change_deg'] is not None and values['heading_change_deg']>=self.detector.cfg['touch_angle_deg']) or (values['speed_change_mps'] is not None and abs(values['speed_change_mps'])>=self.detector.cfg['touch_delta_speed_mps'])
                values['opponent_motion_change_evidence']=dict(nearby_opponent=near,motion_change=change,
                    touch_event_ids=[touch['event_id']] if touch and touch['team']==opposite and 0<=f-touch['event_frame']<=2 else [],
                    confirms_turnover=False)
                if all(values[k] is None for k in ('trajectory_error_m','heading_change_deg','speed_change_mps')):
                    values['opponent_motion_change_evidence']=None
                    reasons['opponent_motion_change_evidence']='ball_motion_unavailable'
                if values['receiver_candidates_count'] is None:reasons['receiver_candidates_count']=own['reason']
        for key,value in values.items():
            if value is not None:reasons.pop(key,None)
            elif key=='release_interval' and self.origin:reasons[key]='no_release_detected'
            elif key not in reasons:reasons[key]='unavailable'
        label,phase,reason='X','incerteza','no_confirmed_origin'
        if self.origin:
            if not visible:
                if self.unavailable_run<=self.cfg['missing_ball_grace_frames'] and self.last_label==self.origin:
                    label=self.origin;reason='short_ball_gap_hold'
                else:reason='ball_unavailable'
            elif not fresh:reason='metrics_not_currently_available'
            elif not current['ball_valid']:reason='ball_quality_or_pitch_boundary'
            elif observed_control==self.origin:
                label,phase,reason=self.origin,'controle','current_control_evidence'
            elif values['last_supported_control_age_s']>self.cfg['max_pass_without_control_s']:
                reason='origin_support_expired'
            else:
                counts=values['observed_players_by_team'];fractions=values['valid_motion_fraction_by_team']
                coverage=all(counts[t]>=self.cfg['min_observed_players_per_team'] and fractions[t] is not None and fractions[t]>=self.cfg['min_motion_fraction'] for t in ('H','A'))
                errors=[values['trajectory_error_m'],values['trajectory_error_rms_m']]
                coherent=all(e is not None and e<=self.cfg['trajectory_error_limit_m'] for e in errors)
                moving=max(values['ball_speed_fitted_mps'] or 0.,current['instant_ball_speed_mps'] or 0.)>=self.cfg['minimum_pass_speed_mps']
                if not current['ball_valid']:reason='ball_quality_or_pitch_boundary'
                elif not coverage:reason='insufficient_team_motion_coverage'
                elif not coherent:reason='trajectory_not_supported'
                elif not moving:reason='no_control_or_pass_motion'
                elif values['origin_team_meeting_time_s'] is None:reason='no_supported_receiver'
                else:
                    label,phase,reason=self.origin,'passe','plausible_same_team_pass'
                    advantage=values['meeting_time_advantage_s']
                    if advantage is not None and advantage<=self.cfg['dispute_margin_s']:
                        phase,reason='disputa','opponent_interception_opportunity'
        counts=values.get('observed_players_by_team') or {}
        fractions=values.get('valid_motion_fraction_by_team') or {}
        other='A' if self.origin=='H' else 'H'
        motion=values.get('opponent_motion_change_evidence') or {}
        continuity_checks=dict(origin_supported=self.origin is not None,
            current_observed_ball=bool(visible and fresh and current['ball_valid']),
            within_control_age=bool(self.origin and values['last_supported_control_age_s']<=self.cfg['max_pass_without_control_s']),
            coverage=all(counts.get(t,0)>=self.cfg['min_observed_players_per_team'] and
                fractions.get(t) is not None and fractions[t]>=self.cfg['min_motion_fraction'] for t in ('H','A')),
            coherent=all(values.get(k) is not None and values[k]<=self.cfg['trajectory_error_limit_m']
                         for k in ('trajectory_error_m','trajectory_error_rms_m')),
            moving=(values.get('ball_speed_fitted_mps') or 0)>=self.cfg['minimum_pass_speed_mps'],
            no_opponent_control=observed_control in (None,self.origin),
            no_opponent_contact=not (motion.get('nearby_opponent') and
                (motion.get('motion_change') or motion.get('touch_event_ids'))),
            live=not game['dead'])
        fit=self.measurements.ball.previous_fit
        continuity=self.pass_continuity.step(f,self.origin,
            [p for p in state['players'] if p['track_id'] not in state['jump_ids']|state['duplicate_ids']
                and p['track_id'] not in {r['track_id'] for r in recovered_players}]+recovered_players,
            ball,fit['velocity'] if fit and fresh else None,self.cfg,continuity_checks,arrivals['candidates'])
        if label=='X' and reason=='no_supported_receiver' and continuity['accepted']:
            label,phase,reason=self.origin,'passe','observed_receiver_continuity'
        # A just-released ball may leave the passer before a receiver enters the
        # finite forecast. This is a launch hypothesis, not an arbitrary X fill.
        launch=dict(accepted=False,reason='not_a_recent_observed_release')
        if label=='X' and reason=='no_supported_receiver' and all(continuity_checks.values()):
            age=values['last_supported_control_age_s']
            own=[p for p in state['players'] if p['team']==self.origin and p['source']=='observed'
                 and not p.get('quality_flags') and p['track_id'] not in state['jump_ids']|state['duplicate_ids']]
            sender=min(own,key=lambda p:math.hypot(p['x_m']-ball['x_m'],p['y_m']-ball['y_m']),default=None)
            if sender and age<=self.cfg['missing_ball_grace_frames']/fps:
                delta=(ball['x_m']-sender['x_m'],ball['y_m']-sender['y_m'])
                velocity=fit['velocity'] if fit else (0.,0.)
                other_distance=current['distances'][other]
                launch=dict(accepted=bool(length(delta)<=self.detector.cfg['proximity_m'] and
                    sum(a*b for a,b in zip(delta,velocity))>0 and other_distance is not None and
                    other_distance>self.detector.cfg['proximity_m']),sender_track_id=sender['track_id'],
                    age_s=age,max_age_s=self.cfg['missing_ball_grace_frames']/fps,
                    sender_distance_m=length(delta),reason='observed_sender_outgoing_launch')
            if launch['accepted']:label,phase,reason=self.origin,'saida_de_passe','observed_sender_outgoing_launch'
        continuity['launch']=launch
        prior=self.kickoff_prior
        if prior and not kickoff_applied:
            other='A' if prior['team']=='H' else 'H'
            opponent_near=(fresh and current['distances'][other] is not None
                and current['distances'][other]<=self.cfg['control_radius_m'])
            opponent_touch=(self.last_touch is not None and self.last_touch['team']==other
                and self.last_touch['event_frame']>=prior['frame'])
            expired=(self.origin!=prior['team'] or (f-prior['frame'])/fps>self.cfg['max_pass_without_control_s']
                or self.unavailable_run>self.cfg['missing_ball_grace_frames']
                or bool(released) or visual['accepted'] or recovered or observed_control is not None
                or opponent_near or opponent_touch or (fresh and visible and not current['ball_valid']))
            if expired:self.kickoff_prior=None
            elif visible and fresh and label=='X':
                label,phase,reason=prior['team'],'kickoff','kickoff_opening_prior'
        if kickoff_applied:
            label,phase,reason=self.origin,'controle',kickoff_diagnostic['reason']
        elif visual['accepted']:
            label,phase,reason=self.origin,'controle',visual['reason']
        elif recovered:
            label,phase,reason=self.origin,'recepcao_provavel','sustained_probable_reception'
        if game['dead']:
            label,phase,reason='D','bola_morta',game['active']['reason']
        elif restart_tracking_confirmed:
            label,phase,reason=self.origin,'controle','verified_restart_tracking_confirmed'
        elif game['transition'] in ('dead_ball_evidence_expired','verified_restart'):
            label,phase,reason='X','incerteza',game['transition']
        other='A' if self.origin=='H' else 'H'
        recent_touch=bool(self.last_touch and self.last_touch['team']==other and 0<=f-self.last_touch['event_frame']<=2)
        distance_other=current['distances'][other] if fresh else None
        opponent_time=arrivals['teams'].get(other,{}).get('meeting_time_s')
        errors=[values.get('trajectory_error_m'),values.get('trajectory_error_rms_m')]
        context_checks=dict(current_observed_ball=bool(visible and fresh and current['ball_valid']),
            coverage=continuity_checks['coverage'],
            trajectory_not_contradicted=all(e is None or e<=self.cfg['trajectory_error_limit_m'] for e in errors),
            no_opponent_control=observed_control in (None,self.origin),
            opponent_observed_and_separated=distance_other is not None and distance_other>self.detector.cfg['proximity_m'],
            no_recent_opponent_touch=not recent_touch,
            no_imminent_interception=opponent_time is None or opponent_time>self.cfg['reception_horizon_s'])
        context=self.temporal_context.step(f,self.origin,label,reason,observed_control,current if fresh else None,
            context_checks,values.get('last_supported_control_age_s'),self.cfg['max_pass_without_control_s'],
            dead=game['dead'],barrier=bool(game['transition']) or self.unavailable_run>self.cfg['missing_ball_grace_frames'])
        if context['accepted']:
            label,phase,reason=self.origin,'continuidade_contextual','causal_context_continuity'
        values['context_observed_frames']=context['independent_evidence_frames']
        values['context_support_fraction']=context['origin_support_fraction']
        for key in ('context_observed_frames','context_support_fraction'):
            if values[key] is not None:reasons.pop(key,None)
        support=current['support'] if current else None
        parts=[d for d in (support,self.last_control_support,self.origin_event['dependencies'] if self.origin_event else None) if d]
        if context['accepted'] and context['support']:parts.append(context['support'])
        for review in tracking_diagnostics:
            parts.append(Dependencies.hint(dict(start_frame=f,available_at_frame=f,hint_id=review['rule_id'])).as_dict())
        game_source=game.get('active') or game.get('restart')
        if game_source:
            parts.append(Dependencies.hint(dict(start_frame=game_source['frame'],
                available_at_frame=game_source['available_at_frame'],hint_id=game_source['evidence_id'])).as_dict())
        for recovered in recovery_diagnostics:
            if recovered['status'] in ('accepted','accepted_team_correction'):
                parts.append(Dependencies.hint(dict(start_frame=f,available_at_frame=recovered['available_at_frame'],
                    hint_id=recovered['observation_id'])).as_dict())
        if parts:
            combined=combine(deps=[Dependencies(d['support_start_frame'],d['support_end_frame'],d['available_at_frame'],
                frozenset(d['point_refs']),frozenset(d['hint_ids']),frozenset(d['event_ids'])) for d in parts],frame=f)
            support=combined.as_dict()
        self.last_label=label;self.last_frame=f
        self.state_counts[phase]+=1;self.decision_counts[reason]+=1
        validity={key:dict(valid=value is not None,reason=reasons.get(key)) for key,value in values.items()}
        fractions=values.get('valid_motion_fraction_by_team')
        if fractions is not None:
            validity['valid_motion_fraction_by_team']['components']={t:dict(valid=v is not None,reason=None if v is not None else 'no_observed_players') for t,v in fractions.items()}
        snapshot=dict(frame=f,segment_start_frame=segment,label=label,state=phase,reason=reason,control_observed_team=observed_control,
            origin_team=self.origin,metrics_version=METRICS_VERSION,policy_version=POLICY_VERSION,
            metrics=values,metric_validity=validity,metric_frame=current['frame'] if current else None,
            support=support,available_at_frame=f,origin_event_id=self.origin_event['event_id'] if self.origin_event else None,
            origin_support=self.origin_event['dependencies'] if self.origin_event else None,
            candidate_players=arrivals['candidates'],forecast_xy=arrivals['forecast_xy'] if fresh and visible else [],
            ball_unavailable_run_frames=self.unavailable_run,released_acquisitions=released,
            kickoff=kickoff_diagnostic,kickoff_opening_prior=copy.deepcopy(self.kickoff_prior),reception=recovery,visual_control=visual,
            game_state=game,pass_continuity=continuity,recovered_observations=recovery_diagnostics,
            temporal_context=context,tracking_review=tracking_diagnostics,
            transition=dict(previous_label=previous_label,label=label,changed=previous_label!=label,reason=reason,
                game_evidence_ids=[r['evidence_id'] for r in game_evidence if r['frame']==f and r['available_at_frame']<=f]),
            experimental=True,probabilities_calibrated=False)
        return snapshot,feature,events,row

    @staticmethod
    def _dependencies(d):
        return Dependencies(d['support_start_frame'],d['support_end_frame'],d['available_at_frame'],
            frozenset(d['point_refs']),frozenset(d['hint_ids']),frozenset(d['event_ids']))

    def _reception_step(self, current, frame, confirmed=False):
        cfg=self.cfg
        result=dict(accepted=False,team=None,track_id=None,duration_s=0.,checks={})
        previous=self.reception_previous
        self.reception_previous=None
        if current is None or not self.origin or confirmed:
            self.reception=None;self.reception_support=None
            result['reason']='confirmed_acquisition_priority' if confirmed else 'no_current_evidence_or_origin'
            return result
        team='A' if self.origin=='H' else 'H'
        candidates=[p for p in current['arrivals']['candidates'] if p['team']==team]
        candidate=min(candidates,key=lambda p:(p['meeting_time_s'],p['track_id'])) if candidates else None
        if candidate is None:
            self.reception=None;self.reception_support=None
            result['reason']='no_reception_candidate'
            return result
        key=(team,candidate['track_id'])
        self.reception_previous=(frame,key,candidate['distance_m'])
        values=current['values'];counts=values['observed_players_by_team'];fractions=values['valid_motion_fraction_by_team']
        opponent=current['arrivals']['teams'][self.origin]['meeting_time_s']
        checks=dict(ball_valid=current['ball_valid'],
            coverage=all(counts[t]>=cfg['min_observed_players_per_team'] and fractions[t] is not None
                and fractions[t]>=cfg['min_motion_fraction'] for t in ('H','A')),
            trajectory=all(values[k] is not None and values[k]<=cfg['trajectory_error_limit_m']
                for k in ('trajectory_error_m','trajectory_error_rms_m')),
            moving=(values['ball_speed_fitted_mps'] or 0)>=cfg['minimum_pass_speed_mps'],
            meeting=candidate['meeting_time_s']<=cfg['reception_horizon_s'],
            advantage=opponent is not None and opponent-candidate['meeting_time_s']>=cfg['reception_advantage_s'],
            approach=candidate['approach_distance_m']<=cfg['control_radius_m'],
            closing=previous is not None and previous[0]==frame-1 and previous[1]==key
                and candidate['distance_m']<previous[2]-1e-6)
        result.update(team=team,track_id=key[1],checks=checks,candidate=dict(candidate))
        if not all(checks.values()):
            self.reception=None;self.reception_support=None
            result['reason']='reception_requirements_not_met'
            return result
        if self.reception is None or self.reception['key']!=key or self.reception['last_frame']!=frame-1:
            self.reception=dict(key=key,start_frame=frame,last_frame=frame)
            self.reception_support=current['support']
        else:
            self.reception['last_frame']=frame
            self.reception_support=combine(deps=[self._dependencies(self.reception_support),
                self._dependencies(current['support'])],frame=frame).as_dict()
        duration=(frame-self.reception['start_frame']+1)/self.meta['fps']
        result.update(duration_s=duration,accepted=duration+1e-9>=cfg['reception_sustain_s'],
            reason='sustained_probable_reception' if duration+1e-9>=cfg['reception_sustain_s'] else 'accumulating_reception_evidence')
        return result

    def finish(self):
        return dict(detector=self.detector.finish(),policy_version=POLICY_VERSION,metrics_version=METRICS_VERSION,
            state_counts=dict(self.state_counts),reason_counts=dict(self.decision_counts),
            kickoffs=self.kickoffs,pending_records=len(self.pending),counts=dict(self.counts),configuration=self.cfg,
            limitations=['no_ball_height','optimistic_turning_after_reaction','observed_coverage_not_roster_coverage',
                        'uncalibrated_physical_parameters','dead_ball_requires_verified_evidence'])
