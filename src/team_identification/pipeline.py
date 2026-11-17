"""Mask -> uniform -> feet -> local identity -> ATD association, without truth."""
import math
import numpy as np
from .association import LocalIdentities, ImageIdentities, associate, feet_to_pitch, finite_xy
from .temporal import TeamMemory
from .four_frames import FourFrameUniforms, POLICY as LEGACY_POLICY
from .spaced_frames import SpacedFourFrameUniforms, POLICY as SPACED_POLICY

DEFAULTS = dict(max_distance_m=1.5, min_margin_m=.25, max_reprojection_error_px=2.,
                identity_max_gap_frames=25, identity_max_speed_mps=10.,
                memory_window_frames=25, memory_min_observations=5,
                memory_min_support=.8, memory_switch_observations=5, memory_max_hold_frames=25)


class IdentityPipeline:
    def __init__(self, meta, classifier, config=None, *, uniform_policy=SPACED_POLICY, identity_method="pitch"):
        if identity_method not in ("pitch","image-overlap"):
            raise ValueError("Unknown identity method")
        self.identity_method=identity_method
        self.meta, self.classifier = meta, classifier
        if uniform_policy not in (LEGACY_POLICY,SPACED_POLICY):
            raise ValueError('Unknown supervised temporal policy')
        stream=SpacedFourFrameUniforms if uniform_policy==SPACED_POLICY else FourFrameUniforms
        if callable(getattr(classifier,'predict',None)) and meta['fps']!=25:
            raise ValueError('Supervised temporal offsets require the original 25 fps timeline')
        self.four_frames = stream(classifier) if callable(getattr(classifier,'predict',None)) else None
        self.config = dict(DEFAULTS)
        if config:
            if set(config)-set(DEFAULTS):
                raise ValueError('Unknown identity configuration')
            self.config.update(config)
        c = self.config
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in c.values()):
            raise ValueError('Finite numeric identity parameters required')
        for key in ('identity_max_gap_frames','memory_window_frames','memory_min_observations','memory_switch_observations','memory_max_hold_frames'):
            if type(c[key]) is not int:
                raise ValueError('Integer frame/observation parameters required')
        if not 0<c['min_margin_m']<c['max_distance_m'] or c['max_reprojection_error_px']<=0:
            raise ValueError('Invalid geometry thresholds')
        self.identities = LocalIdentities(meta['fps'], c['identity_max_gap_frames'], c['identity_max_speed_mps'], c['min_margin_m'])
        if identity_method=="image-overlap":
            self.identities=ImageIdentities()
        self.memory = TeamMemory(c['memory_window_frames'], c['memory_min_observations'], c['memory_min_support'],
                                 c['memory_switch_observations'], c['memory_max_hold_frames'])

    def step(self, envelope, image, masks, homography):
        f = envelope['frame']
        reset = f in self.meta.get('continuity_starts', [])
        if reset:
            self.memory.reset()
            if self.four_frames:self.four_frames.reset()
        masks = list(masks)
        if self.four_frames:
            from .pilot import mask_flags,torso_evidence
            stack=np.asarray(masks,dtype=bool) if masks else np.zeros((0,image.height,image.width),bool)
            flags=mask_flags(stack) if masks else []
            classifications=[]
            overlap=stack.sum(axis=0)>1
            rgb=np.asarray(image)
            for i,mask in enumerate(masks):
                _,_,reason=torso_evidence(rgb,mask,overlap)
                classifications.append(dict(detection_index=i,team=None,role='unknown',classification='uncertain',
                    reason=flags[i] or reason or 'pending_four_frame_uniforms'))
        else:
            classifications = self.classifier.classify(image, masks) if masks else []
        if len(classifications)!=len(masks):
            raise ValueError('Every person mask requires a classification result')
        detections = []
        for mask, cls in zip(masks, classifications):
            from .pilot import mask_bounds
            bounds = mask_bounds(mask)
            box = list(bounds) if bounds else None
            # Bottom silhouette is an estimated foot contact, never human verification.
            feet, xy = None, None
            if box and box[3]-box[1] >= 20 and box[2]-box[0] >= 10 and cls.get('reason') in ('uniform_match', 'distance_or_margin','pending_four_frame_uniforms'):
                x0,y0,x1,y1 = box
                ys,xs = np.where(mask[y0:y1,x0:x1])
                ys += y0
                xs += x0
                bottom = ys >= np.quantile(ys, .95)
                feet = [float(np.median(xs[bottom])), float(ys.max())]
                if homography.get('valid'):
                    xy = feet_to_pitch(feet, homography['matrix'], self.meta['pitch_m'], homography['reprojection_error'],
                                       self.config['max_reprojection_error_px'])
            detections.append(dict(cls, box=box, foot_pixel=feet, pitch_xy=xy,
                                   feet_method='estimated_mask_bottom', geometry_valid=xy is not None))
        return self._complete(envelope,image,detections,reset)

    def step_boxes(self, envelope, image, boxes, homography):
        if not self.four_frames:
            raise ValueError('Box observations require supervised crop uniforms')
        f=envelope['frame'];reset=f in self.meta.get('continuity_starts',[])
        if reset:
            self.memory.reset();self.four_frames.reset()
        detections=[]
        for i,raw in enumerate(boxes):
            values=np.asarray(raw,dtype=float)
            if values.shape!=(4,) or not np.isfinite(values).all():
                raise ValueError('Finite box coordinates required')
            x,y,X,Y=values
            x=max(0,int(np.floor(x)));y=max(0,int(np.floor(y)))
            X=min(image.width,int(np.ceil(X)));Y=min(image.height,int(np.ceil(Y)))
            if not 0<=x<X<=image.width or not 0<=y<Y<=image.height:
                raise ValueError('Invalid person box')
            box=[x,y,X,Y];feet=[(x+X-1)/2.,float(Y-1)];xy=None
            valid_size=Y-y>=20 and X-x>=10
            if valid_size and homography.get('valid'):
                xy=feet_to_pitch(feet,homography['matrix'],self.meta['pitch_m'],
                    homography['reprojection_error'],self.config['max_reprojection_error_px'])
            detections.append(dict(detection_index=i,team=None,role='unknown',classification='uncertain',
                reason='pending_four_frame_uniforms' if valid_size else 'small_mask',
                box=box,foot_pixel=feet,pitch_xy=xy,feet_method='estimated_box_bottom',
                geometry_valid=xy is not None,geometry_confidence='uncalibrated_box_approximation'))
        return self._complete(envelope,image,detections,reset)

    def _complete(self,envelope,image,detections,reset):
        f=envelope['frame']
        unsafe={'duplicate_mask','merged_or_contained_mask','merged_or_non_upright_mask'}
        identity_inputs=[d['box'] if d['box'] and d['reason'] not in unsafe else None for d in detections] if self.identity_method=='image-overlap' else [d['pitch_xy'] for d in detections]
        identities = self.identities.step(f, identity_inputs, reset=reset)
        self.memory.retain(set(self.identities.tracks))
        if self.four_frames:self.four_frames.retain(set(self.identities.tracks))
        for d, identity in zip(detections, identities):
            d['identity'] = identity
            d['visual_team'], d['visual_role'] = d['team'], d['role']
            if identity:
                if self.four_frames:
                    result=self.four_frames.feed(identity,f,np.asarray(image.crop(d['box'])))
                    d.update(result)
                    d['four_frame_uniforms']=dict(result)
                else:
                    d.update(self.memory.step(identity, f, d['team'], d['role']))
            else:
                d.update(team=None, role='unknown', reason='invalid_geometry')
            d['classification'] = d['team'] or ('referee' if d['role']=='referee' else 'uncertain')
        points = [p for p in envelope['players'] if p['source'] == 'observed' and not p.get('quality_flags')
                  and p.get('available_at_frame', f) <= f and finite_xy([p['x_m'], p['y_m']])
                  and 0 <= p['x_m'] <= self.meta['pitch_m'][0] and 0 <= p['y_m'] <= self.meta['pitch_m'][1]]
        # Confirmed officials cannot consume player tracks.
        indices = [i for i, d in enumerate(detections) if d['identity'] and d['geometry_valid'] and d['role'] not in ('referee','other')]
        matches = associate([detections[i]['pitch_xy'] for i in indices], [[p['x_m'], p['y_m']] for p in points],
                            self.config['max_distance_m'], self.config['min_margin_m'])
        players = []
        for i, match in zip(indices, matches):
            d = detections[i]
            target = points[match['point_index']] if match['point_index'] is not None else None
            nearby = any(math.dist(d['pitch_xy'], [p['x_m'], p['y_m']]) <= self.config['max_distance_m']*2 for p in points)
            status = match['status']
            # An ambiguous match must not become an extra athlete. Recovery only
            # uses accepted uniform evidence, valid geometry and no nearby track.
            if status == 'unmatched' and not nearby and d['team'] in ('H', 'A'):
                status = 'recovered'
            elif status == 'unmatched' and not nearby:
                status = 'unassigned_person'
            row = dict(frame=f, identity=d['identity'], target_track_id=target['track_id'] if target else None,
                       raw_xy=[target['x_m'], target['y_m']] if target else None, pitch_xy=d['pitch_xy'],
                       team=d['team'], role=d['role'], association_status=status, association=match,
                       reason=d['reason'], available_at_frame=f, support_start_frame=d.get('support_start_frame', f),
                       support_end_frame=f+1, foot_pixel=d['foot_pixel'], geometry_method='automatic_box_bottom_and_homography' if d['feet_method']=='estimated_box_bottom' else 'automatic_mask_feet_and_homography')
            if self.four_frames:row['four_frame_uniforms']=d.get('four_frame_uniforms')
            players.append(row)
            d['association_status'], d['target_track_id'] = status, row['target_track_id']
        return dict(frame=f, players=players, detections=detections,
                    unresolved_tracking_ids=sorted(p['track_id'] for p in points if not any(r['target_track_id']==p['track_id'] for r in players)),
                    parameters_calibrated=False)
