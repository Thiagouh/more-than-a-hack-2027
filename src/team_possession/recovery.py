"""Reviewed visual observations stay separate from immutable ATD detector inputs."""
import math
from ..ball_events import contract as c


def apply_recoveries(rows,envelope,meta):
    frame=envelope['frame'];accepted=[];diagnostics=[];seen=set()
    for r in rows:
        for key in ('observation_id','reviewer','reason','image_sha256','mask_sha256'):
            c.text(r.get(key),'recovery '+key)
        for key in ('frame','available_at_frame'):c.integer(r.get(key),key)
        if r['available_at_frame']<r['frame']:raise c.InputError('Recovery available before source image')
        correction=r.get('kind')=='team_correction'
        tid=r.get('target_track_id') if correction else 'visual-'+r['observation_id']
        target=next((p for p in envelope['players'] if p['track_id']==tid and p['source']=='observed'
                     and not p.get('quality_flags') and p.get('available_at_frame',frame)<=frame),None)
        status='accepted'
        xy=r.get('pitch_xy')
        if r['observation_id'] in seen:status='duplicate_observation'
        elif r['frame']!=frame or r['available_at_frame']>frame:status='late_or_unavailable'
        elif any(r.get(k) is not True for k in ('sam3_executed','mask_reviewed','feet_verified','homography_verified','projection_reviewed')):status='unverified_geometry_or_mask'
        elif r.get('team') not in ('H','A'):status='uncertain_team'
        elif not isinstance(xy,list) or len(xy)!=2 or any(type(v) not in (float,int) or not math.isfinite(v) for v in xy):status='unavailable_position'
        elif not (0<=xy[0]<=meta['pitch_m'][0] and 0<=xy[1]<=meta['pitch_m'][1]):status='outside_pitch'
        elif correction and (target is None or math.hypot(target['x_m']-xy[0],target['y_m']-xy[1])>=1.5):status='correction_target_not_matched'
        elif not correction and any(p['source']=='observed' and p.get('available_at_frame',frame)<=frame
                 and math.hypot(p['x_m']-xy[0],p['y_m']-xy[1])<1.5 for p in envelope['players']+accepted):status='existing_or_ambiguous_track'
        seen.add(r['observation_id'])
        if status=='accepted':
            if correction:xy=[target['x_m'],target['y_m']]
            canonical_id=f"{target['producer']}:{tid}" if correction else 'visual-recovery:'+tid
            accepted.append(dict(frame=frame,track_id=canonical_id,team=r['team'],x_m=xy[0],y_m=xy[1],
                source='observed',producer='visual-recovery',quality_flags=[],available_at_frame=r['available_at_frame']))
        diagnostics.append(dict(r,status=('accepted_team_correction' if correction and status=='accepted' else status),used_for_control=False,
            integration='receiver_evidence_only; raw detector unchanged'))
    return accepted,diagnostics


def load_evidence_package(path,meta,video,kind):
    """Hash-bind reviewed game/recovery evidence to exact extracted source frames."""
    import json
    import re
    from pathlib import Path
    from ..ball_events.artifacts import sha
    from .game_state import validate_evidence
    path=Path(path);package=json.loads(path.read_text())
    if (package.get('schema')!='tactical-reviewed-evidence-v1' or package.get('timeline')!='tactical'
            or package.get('same_source_video_frame') is not True or video is None
            or package.get('source_video_sha256')!=sha(video)):
        raise c.InputError('Evidence must match tactical source video')
    if kind=='game_evidence':validate_evidence(package[kind],meta)
    ids=[r['observation_id'] for r in package[kind]] if kind=='recoveries' else []
    if len(ids)!=len(set(ids)):raise c.InputError('Duplicate recovery IDs in package')
    grouped={}
    homography_rows=None
    if kind=='recoveries' and package[kind]:
        from scripts.prepare_tactical_visual import homographies
        hpath=path.parent/package['homography_file']
        if sha(hpath)!=package['homography_sha256']:raise c.InputError('Recovery homography changed')
        homography_rows=homographies(hpath,[r['frame'] for r in package[kind]])
    for r in package[kind]:
        c.integer(r.get('frame'),'evidence frame')
        if r['frame']>=meta['n_frames']:raise c.InputError('Evidence outside run')
        for key in ('image','mask') if kind=='recoveries' else ('image',):
            asset=path.parent/r[key]
            if not re.fullmatch('[0-9a-f]{64}',r.get(key+'_sha256','')) or sha(asset)!=r[key+'_sha256']:
                raise c.InputError('Evidence asset hash mismatch')
        if r.get('source_frame')!=r['frame']:raise c.InputError('Evidence frame mismatch')
        if kind=='game_evidence':validate_evidence([r],meta)
        elif all(r.get(k) is True for k in ('feet_verified','homography_verified','projection_reviewed')):
            proposal_path=path.parent/r['proposal_package']
            if sha(proposal_path)!=r['proposal_package_sha256']:raise c.InputError('SAM proposal changed')
            proposals=json.loads(proposal_path.read_text())
            matches=[p for p in proposals.get('recoveries',[]) if p.get('mask_sha256')==r['mask_sha256']
                     and p.get('image_sha256')==r['image_sha256'] and p.get('frame')==r['frame']
                     and p.get('sam3_executed') is True]
            if len(matches)!=1 or proposals.get('source_video_sha256')!=package['source_video_sha256']:
                raise c.InputError('Recovery does not match a unique SAM source proposal')
            import numpy as np
            from PIL import Image
            h=homography_rows[r['frame']]
            if not h['valid']:raise c.InputError('Invalid recovery homography')
            pixel=r.get('foot_pixel')
            if not isinstance(pixel,list) or len(pixel)!=2 or any(type(v) not in (int,float) or not math.isfinite(v) for v in pixel):
                raise c.InputError('Invalid reviewed feet')
            with Image.open(path.parent/r['image']) as image:image_size=image.size
            with Image.open(path.parent/r['mask']) as mask:
                mask_size=mask.size;pixels=np.asarray(mask).copy()
            if mask_size!=image_size:raise c.InputError('Recovery mask must use full source image')
            if pixels.ndim!=2 or not np.isin(pixels,[0,1,255]).all() or not pixels.any():
                raise c.InputError('Invalid recovery mask')
            box=r.get('box')
            if not isinstance(box,list) or len(box)!=4 or not (box[0]-3<=pixel[0]<=box[2]+3 and box[1]-3<=pixel[1]<=box[3]+3):
                raise c.InputError('Reviewed feet do not match recovery box')
            if not (0<=pixel[0]<image_size[0] and 0<=pixel[1]<image_size[1]):raise c.InputError('Feet outside image')
            try:q=np.linalg.solve(h['matrix'],np.array([*pixel,1.]))
            except np.linalg.LinAlgError:raise c.InputError('Singular recovery homography') from None
            if not np.isfinite(q).all() or abs(q[2])<1e-9:raise c.InputError('Invalid inverse projection')
            xy=q[:2]/q[2]*np.array(meta['pitch_m'])
            if not np.isfinite(xy).all() or r.get('pitch_xy') is None or np.linalg.norm(xy-r['pitch_xy'])>1e-5:
                raise c.InputError('Recovery coordinates differ from reviewed projection')
        grouped.setdefault(r['frame'],[]).append(r)
    return grouped
