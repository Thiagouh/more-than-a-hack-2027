"""Causal dead-ball evidence. No possession reference or low-speed shortcut."""
import copy
import math
from ..ball_events import contract as c


def validate_evidence(rows, meta):
    seen=set()
    for r in rows:
        c.obj(r,'game evidence')
        for k in ('evidence_id','source_id','reason'):c.text(r.get(k),k)
        for k in ('frame','available_at_frame','valid_until_frame'):c.integer(r.get(k),k)
        if r['evidence_id'] in seen:raise c.InputError('Duplicate game evidence')
        seen.add(r['evidence_id'])
        if (r['kind'] not in ('interruption','restart','geometry') or r['frame']>=meta['n_frames']
                or r['available_at_frame']<r['frame'] or r['valid_until_frame']<r['frame']
                or r['valid_until_frame']-r['frame']>10*meta['fps']):
            raise c.InputError('Invalid game evidence bounds')
        if r.get('verified') is not True:c.text(r.get('reason'),'abstention reason')
        if r.get('verified') is True:
            c.text(r.get('reviewer'),'evidence reviewer')
            if r['kind']=='restart' and r.get('team') not in (None,'H','A'):
                raise c.InputError('Invalid restart team')
            if r['kind']=='geometry':
                c.number(r.get('boundary_uncertainty_m'),'boundary uncertainty')
                if not 0<=r['boundary_uncertainty_m']<=1:raise c.InputError('Invalid boundary uncertainty')
    return copy.deepcopy(list(rows))


class GameState:
    def __init__(self,meta):
        self.meta=meta;self.active=None;self.previous=None;self.last_frame=None

    def reset(self):
        self.active=None;self.previous=None;self.last_frame=None

    def step(self,frame,ball,ball_ok,evidence=(),kickoff=False):
        rows=validate_evidence(evidence,self.meta)
        diagnostics=[];transition=None;restart=None
        if self.last_frame is not None and frame!=self.last_frame+1:self.reset()
        self.last_frame=frame
        if self.active and frame>self.active['valid_until_frame']:
            transition='dead_ball_evidence_expired';self.active=None
        current=[]
        for r in rows:
            if r['frame']!=frame or r['available_at_frame']>frame:
                diagnostics.append(dict(evidence_id=r['evidence_id'],reason='late_or_unavailable_not_replayed'));continue
            if r.get('verified') is not True:
                diagnostics.append(dict(evidence_id=r['evidence_id'],reason='unverified'));continue
            current.append(r)
        # A conflict never chooses a team or silently resumes play.
        stops=[r for r in current if r['kind']=='interruption']
        starts=[r for r in current if r['kind']=='restart']
        if stops and starts:
            diagnostics.append(dict(reason='conflicting_stop_and_restart'));starts=[];stops=[]
        if kickoff:
            self.active=None;transition='explicit_kickoff_restart'
        elif starts and len({r.get('team') for r in starts})==1:
            restart=starts[0];self.active=None;transition='verified_restart'
        elif stops:
            self.active=stops[0];transition='verified_interruption'
        geometry=next((r for r in current if r['kind']=='geometry'),None)
        observed=(ball_ok and ball['source']=='observed' and not ball.get('quality_flags')
                  and ball.get('available_at_frame',frame)<=frame)
        if geometry and observed and self.previous and not starts and not kickoff:
            old=self.previous
            inside=lambda p:0<=p['x_m']<=self.meta['pitch_m'][0] and 0<=p['y_m']<=self.meta['pitch_m'][1]
            # Require the whole ball beyond a reviewed boundary, consecutive observations,
            # an inside-to-outside path and a plausible displacement (not a tracking jump).
            margin=.11+geometry['boundary_uncertainty_m']
            outside=(ball['x_m'] < -margin or ball['y_m'] < -margin or
                     ball['x_m']>self.meta['pitch_m'][0]+margin or ball['y_m']>self.meta['pitch_m'][1]+margin)
            speed=math.hypot(ball['x_m']-old['x_m'],ball['y_m']-old['y_m'])*self.meta['fps']
            if inside(old) and outside and speed<=45:
                self.active=dict(geometry,kind='ball_out',reason='reviewed_geometry_observed_boundary_crossing')
                transition='observed_ball_out'
            else:diagnostics.append(dict(reason='boundary_crossing_not_supported',speed_mps=speed))
        self.previous=copy.deepcopy(ball) if observed else None
        return dict(dead=self.active is not None,active=copy.deepcopy(self.active),restart=restart,
                    transition=transition,diagnostics=diagnostics,
                    evidence_mode='reviewed_offline' if current else 'no_new_verified_evidence')
