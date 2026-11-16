"""Selected causal policy: four samples over 0.6 s, with every-frame identity checks."""
from collections import deque
from .temporal_windows import decide

POLICY = 'supervised-four-spaced-frames-0.6s-v1'
OFFSETS = [-15, -10, -5, 0]


class SpacedFourFrameUniforms:
    def __init__(self, model):
        self.model, self.tracks = model, {}

    def reset(self):
        self.tracks.clear()

    def retain(self, identities):
        self.tracks = {k:v for k,v in self.tracks.items() if k in identities}

    def feed(self, identity, frame, crop, *, available_at_frame=None, association_verified=True):
        if not isinstance(identity,str) or not identity or type(frame) is not int or frame<0:
            raise ValueError('Explicit local identity and nonnegative frame required')
        available=frame if available_at_frame is None else available_at_frame
        if type(available) is not int or available<frame:
            raise ValueError('Invalid observation availability')
        history=self.tracks.setdefault(identity,deque(maxlen=16))
        if history and frame<=history[-1]['frame']:
            raise ValueError('Observations must strictly increase')
        if not association_verified or (history and frame!=history[-1]['frame']+1):
            history.clear()
        # Cache descriptors rather than RGB images; unavailable pixels are never read.
        prediction=self.model.predict(crop) if association_verified and available==frame and crop is not None else {}
        history.append(dict(identity=identity,frame=frame,available_at_frame=available,
            association_verified=association_verified,crop=prediction))
        class CachedScores:
            max_distance=self.model.max_distance
            min_margin=self.model.min_margin
            @staticmethod
            def predict(scores):return scores
        result=decide(CachedScores(),identity,frame,list(history),OFFSETS)
        result.update(policy=POLICY,window_frames=4,continuity_window_frames=16,
            frames_observed=len(history),support_start_frame=max(history[0]['frame'],frame-15),
            support_end_frame=frame+1)
        if not association_verified:
            history.clear();result['reason']='association_unverified'
        elif len(history)<16:
            result['reason']='spaced_four_frame_warmup'
        return result
