"""Physics-only observation filtering; no interpolation, fitting, or team labels.

Thresholds follow Frente 2's existing cleaning proposal (PR #2): 40 m/s,
two adjacent frames around jumps, five metres beyond the pitch. The boundary
margin for a lost-outside trigger is supplied by Frente 4's existing policy.
This is offline filtering, not a claim of causal availability.
"""
import numpy as np

DEFAULTS = dict(max_speed_mps=40.0, jump_padding_frames=2, reference_fps=25.0,
                outside_limit_m=5.0,max_excursion_s=2.0)


def clean_ball_observations(tr, boundary_margin_m=.5):
    xy=np.asarray(tr['ball_xy'])
    pitch=np.asarray(tr['pitch'],dtype=float)
    fps=float(tr['fps'])
    if (xy.ndim!=2 or xy.shape[1]!=2 or xy.dtype.kind not in 'fi'
            or len(xy)!=tr['n_frames'] or pitch.shape!=(2,)
            or not np.isfinite(pitch).all() or (pitch<=0).any()
            or not np.isfinite(fps) or fps<=0
            or not np.isfinite(boundary_margin_m) or boundary_margin_m<0):
        raise ValueError('Aligned numeric tracking and positive pitch/fps required')
    finite=np.isfinite(xy).all(axis=1)
    partial=np.isfinite(xy).any(axis=1)&~finite
    limit=DEFAULTS['outside_limit_m']
    outside=finite&((xy[:,0]<-limit)|(xy[:,0]>pitch[0]+limit)
                    |(xy[:,1]<-limit)|(xy[:,1]>pitch[1]+limit))
    jump=np.zeros(len(xy),dtype=bool)
    indices=np.empty(0,dtype=int);speed=np.empty(0,dtype=float)
    if len(xy)>1:
        with np.errstate(invalid='ignore',over='ignore'):
            speed=np.linalg.norm(np.diff(xy.astype(float),axis=0),axis=1)*fps
        indices=np.flatnonzero(finite[1:]&finite[:-1]&(speed>DEFAULTS['max_speed_mps']))
        jump[indices]=True;jump[indices+1]=True
    original_jump=jump.copy()
    padding=max(1,round(DEFAULTS['jump_padding_frames']*fps/DEFAULTS['reference_fps']))
    for shift in range(1,padding+1):
        jump[shift:]|=original_jump[:-shift]
        jump[:-shift]|=original_jump[shift:]
    excursion=np.zeros(len(xy),dtype=bool)
    for left,right in zip(indices[:-1],indices[1:]):
        # A bounded observed island, bracketed by two impossible jumps, may be
        # another tracked object. Require stable context on both sides and a
        # return closer to the original trajectory than either impossible jump.
        if (left<2 or right+3>=len(xy) or (right-left)/fps>DEFAULTS['max_excursion_s']
                or not finite[left-2:right+4].all()
                or outside[left] or outside[right+1]
                or not (speed[left-2:left]<=DEFAULTS['max_speed_mps']).all()
                or not (speed[right+1:right+3]<=DEFAULTS['max_speed_mps']).all()):
            continue
        direct=float(np.linalg.norm(xy[right+1]-xy[left]))
        outbound=float(np.linalg.norm(xy[left+1]-xy[left]))
        inbound=float(np.linalg.norm(xy[right+1]-xy[right]))
        if direct<min(outbound,inbound) and direct*fps/(right+1-left)<=DEFAULTS['max_speed_mps']:
            excursion[left+1:right+1]=True
    removed=finite&(outside|jump|excursion)
    cleaned=xy.copy() if xy.dtype.kind=='f' else xy.astype(float)
    cleaned[~finite|removed]=np.nan
    kept=np.isfinite(cleaned).all(axis=1)
    beyond=kept&((cleaned[:,0]<-boundary_margin_m)
                 |(cleaned[:,0]>pitch[0]+boundary_margin_m)
                 |(cleaned[:,1]<-boundary_margin_m)
                 |(cleaned[:,1]>pitch[1]+boundary_margin_m))
    starts=np.zeros(len(xy),dtype=bool)
    if len(xy)>1:starts[1:]=~kept[1:]&kept[:-1]&beyond[:-1]
    output=dict(tr,ball_xy=cleaned)
    summary=dict(method='physics-only-ball-quality-v2',parameters=dict(DEFAULTS),
        effective_jump_padding_frames=padding,
        boundary_margin_m=float(boundary_margin_m),
        original_finite_observations=int(finite.sum()),kept_observations=int(kept.sum()),
        removed_observations=int(removed.sum()),partial_coordinate_frames=int(partial.sum()),
        removed_jump_observations=int((finite&jump).sum()),
        removed_excursion_observations=int((finite&excursion).sum()),
        additional_excursion_observations=int((finite&excursion&~jump&~outside).sum()),
        removed_extreme_outside_observations=int(outside.sum()),
        lost_outside_starts=int(starts.sum()),interpolated_observations=0,
        fitted_on_match=False,availability='offline; adjacent frames can invalidate earlier observations')
    return output,starts,summary
