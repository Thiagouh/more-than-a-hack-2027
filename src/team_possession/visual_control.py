"""Causal bridge from SAM appearances to collective possession, never to ATD IDs.

Team identification alone cannot establish control. Held-ball manual review is
explicit; automatic 2D ball/body proximity requires consecutive corroboration.
"""
import re
from ..ball_events import contract as c
from ..ball_events.dependencies import combine,Dependencies


class VisualControl:
    def __init__(self,meta):
        self.meta=meta;self.previous=None;self.count=0;self.start=None

    def validate(self,records,frame):
        if not isinstance(records,(list,tuple)):raise c.InputError('Visual observations must be a list')
        ids=set()
        for r in records:
            required={'observation_id','frame','available_at_frame','timeline','image_sha256',
                'team','role','classification_method','sam3_executed','mask_quality_ok',
                'context_valid','control_evidence','ball_control_observed','reviewer'}
            c.fields(r,required|{'reference_bundle_sha256'},required,'visual control')
            for key in ('frame','available_at_frame'):c.integer(r[key],'visual '+key)
            if r['frame']>frame or r['available_at_frame']<r['frame']:
                raise c.InputError('Invalid visual observation availability')
            if r['timeline']!=self.meta['timeline']:raise c.InputError('Visual observation timeline mismatch')
            if r['team'] not in ('H','A') or r['role'] not in ('player','goalkeeper'):
                raise c.InputError('Visual control must identify an athlete and team')
            c.text(r['observation_id'],'visual observation id')
            if r['observation_id'] in ids:raise c.InputError('Duplicate visual observation')
            ids.add(r['observation_id'])
            if not isinstance(r['image_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',r['image_sha256']):
                raise c.InputError('Visual control image hash required')
            if r['classification_method'] not in ('sam3_uniform','sam3_goalkeeper_context'):
                raise c.InputError('Unknown visual classification method')
            if r['classification_method']=='sam3_goalkeeper_context' and r['role']!='goalkeeper':
                raise c.InputError('Goalkeeper context cannot assign an outfield role')
            if r['classification_method']=='sam3_uniform' and not re.fullmatch('[0-9a-f]{64}',r.get('reference_bundle_sha256','')):
                raise c.InputError('Frozen uniform reference hash required')
            if r['control_evidence'] not in ('none','reviewed_held_ball','ball_body_proximity'):
                raise c.InputError('Unknown visual control evidence')
            for key in ('sam3_executed','mask_quality_ok','context_valid','ball_control_observed'):
                if type(r[key]) is not bool:raise c.InputError('Explicit visual booleans required')
            if r['control_evidence']=='reviewed_held_ball':
                c.text(r['reviewer'],'held-ball reviewer')
                if r['role']!='goalkeeper':raise c.InputError('Held-ball evidence requires a goalkeeper')

    def step(self,records,frame,observed_team=None,priority=False):
        valid=[r for r in records if r['frame']==frame and r['available_at_frame']<=frame
            and all(r[k] for k in ('sam3_executed','mask_quality_ok','context_valid','ball_control_observed'))
            and r['control_evidence']!='none']
        teams={r['team'] for r in valid}
        diagnostic=dict(accepted=False,team=None,reason='no_eligible_visual_control',observation_ids=[])
        if priority or len(teams)!=1:
            self.previous=None;self.count=0
            if priority:diagnostic['reason']='kickoff_or_tracking_acquisition_priority'
            elif len(teams)>1:diagnostic['reason']='conflicting_visual_teams'
            return diagnostic,None
        team=next(iter(teams))
        if observed_team is not None and observed_team!=team:
            self.previous=None;self.count=0
            return dict(diagnostic,reason='visual_tracking_control_conflict'),None
        count=min(3,self.count+1) if self.previous==(frame-1,team) else 1
        self.start=frame-count+1
        self.previous=(frame,team);self.count=count
        manual=any(r['control_evidence']=='reviewed_held_ball' for r in valid)
        ids=sorted(r['observation_id'] for r in valid)
        if count==1:self.sequence=[]
        self.sequence=(self.sequence+[ids])[-3:]
        if not manual and count<3:
            return dict(diagnostic,team=team,reason='visual_control_pending_corroboration',streak_frames=count,observation_ids=ids),None
        start=frame if manual else self.start
        # Carry the cumulative sources of a corroborated sequence, not just its last frame.
        hints=ids if manual else sorted({i for batch in self.sequence for i in batch})
        support=combine(deps=[Dependencies.hint(dict(start_frame=start,available_at_frame=frame,
            hint_id=source)) for source in hints],frame=frame).as_dict()
        reason='visual_reviewed_goalkeeper_control' if manual else 'visual_control_corroborated'
        return dict(diagnostic,accepted=True,team=team,reason=reason,streak_frames=count,
            observation_ids=ids,manual_ball_review=manual),support


def load_visual_package(path,meta,source_video=None):
    """Load same-source observations; broadcast/sample seconds never become tactical frames."""
    from pathlib import Path
    from ..team_identification.manifests import read_json
    from ..team_identification.pilot import sha256
    path=Path(path);spec=read_json(path)
    if spec.get('schema')!='sam3-visual-control-v1' or spec.get('sam3_executed') is not True:
        raise c.InputError('Executed SAM visual observation package required')
    if spec.get('timeline')!=meta['timeline']:raise c.InputError('Visual package timeline mismatch')
    if meta['timeline']=='tactical':
        if source_video is None or spec.get('same_source_video_frame') is not True:
            raise c.InputError('Same tactical video/frame binding required')
        if sha256(source_video)!=spec.get('source_video_sha256'):
            raise c.InputError('Visual package source video hash mismatch')
        if spec.get('mask_reviewed') is not True or not isinstance(spec.get('mask_reviewer'),str) or not spec['mask_reviewer'].strip():
            raise c.InputError('Tactical SAM mask requires explicit post-inference review')
        mask=Path(spec.get('mask_file',''))
        if not mask.is_absolute():mask=path.parent/mask
        if not mask.is_file() or sha256(mask)!=spec.get('mask_sha256'):
            raise c.InputError('Reviewed tactical SAM mask changed')
    image=Path(spec['image'])
    if not image.is_absolute():image=path.parent/image
    if sha256(image)!=spec.get('image_sha256'):
        raise c.InputError('Visual package image changed')
    grouped={};validator=VisualControl(meta)
    for observation in spec['observations']:
        frame=observation['frame']
        c.integer(frame,'visual observation frame')
        if meta['timeline']=='tactical' and frame!=spec.get('frame'):
            raise c.InputError('Visual observation is not the package source frame')
        if not 0<=frame<meta['n_frames']:raise c.InputError('Visual observation outside run')
        if observation['image_sha256']!=spec['image_sha256']:
            raise c.InputError('Visual observation image binding mismatch')
        validator.validate([observation],frame)
        grouped.setdefault(frame,[]).append(observation)
    return grouped
