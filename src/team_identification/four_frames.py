"""Causal four-frame interpretation of one geometrically associated person.

Four consecutive frames at 25 Hz span 120 ms. Distances are combined before
the decision; uncertain individual frames are not silently discarded as votes.
"""
from collections import deque
import numpy as np
from .supervised_uniforms import LABELS

POLICY = 'supervised-four-consecutive-frames-v1'


class FourFrameUniforms:
    def __init__(self, model):
        self.model, self.tracks = model, {}

    def reset(self):
        self.tracks.clear()

    def retain(self, identities):
        self.tracks = {k:v for k,v in self.tracks.items() if k in identities}

    def feed(self, identity, frame, crop, *, available_at_frame=None, association_verified=True):
        if not isinstance(identity,str) or not identity or type(frame) is not int or frame < 0:
            raise ValueError('Explicit local identity and nonnegative frame required')
        available = frame if available_at_frame is None else available_at_frame
        if type(available) is not int or available < frame:
            raise ValueError('Invalid four-frame availability')
        history = self.tracks.setdefault(identity, deque(maxlen=4))
        if history and frame <= history[-1]['frame']:
            raise ValueError('Four-frame observations must strictly increase')
        if not association_verified or (history and frame != history[-1]['frame']+1):
            history.clear()
        prediction = (self.model.predict(crop) if association_verified and available==frame and crop is not None else {})
        history.append(dict(frame=frame, available=available, prediction=prediction))
        result = dict(identity=identity, frame=frame, label='Incerto', team=None, role='unknown',
            policy=POLICY, window_frames=4, frames_observed=len(history),
            frame_ids=[r['frame'] for r in history], available_at_frame=frame,
            support_start_frame=history[0]['frame'], support_end_frame=frame+1,
            distance=None, margin=None, calibrated_probability=False,
            reason='association_unverified' if not association_verified else 'four_frame_warmup')
        if not association_verified:
            history.clear()
            return result
        valid = [r for r in history if r['prediction'].get('class_distances')]
        result['valid_frames'] = len(valid)
        if len(history)<4:
            return result
        if len(valid)<3:
            return dict(result,reason='insufficient_valid_uniform_frames')
        classes = sorted(valid[0]['prediction']['class_distances'])
        if any(sorted(r['prediction']['class_distances'])!=classes for r in valid):
            raise ValueError('Uniform model/classes changed within a four-frame sequence')
        values = np.array([[r['prediction']['class_distances'][c] for c in classes] for r in valid])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite four-frame distance')
        scores = np.median(values,axis=0)
        order = sorted(range(len(classes)),key=lambda i:(scores[i],classes[i]))
        best = classes[order[0]]
        distance,margin=float(scores[order[0]]),float(scores[order[1]]-scores[order[0]])
        votes=sum(min(r['prediction']['class_distances'],key=lambda c:(r['prediction']['class_distances'][c],c))==best for r in valid)
        accepted=(votes>=3 and distance<=self.model.max_distance and margin>=self.model.min_margin)
        team,role=LABELS[best] if accepted else (None,'unknown')
        result.update(label=best if accepted else 'Incerto',team=team,role=role,distance=distance,margin=margin,
            class_distances={c:float(scores[i]) for i,c in enumerate(classes)},supporting_frames=votes,
            reason='four_frame_consensus' if accepted else 'four_frame_conflict_or_margin',
            per_frame_labels=[r['prediction'].get('label','Incerto') for r in history])
        return result
