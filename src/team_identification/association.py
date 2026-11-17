"""Class-blind, one-to-one geometry and conservative local identities.

Distances are metres, never classification probabilities. ATD slot/team labels
are deliberately not inputs to geometric matching or identity continuity.
"""
import math
import numpy as np


def finite_xy(value):
    return (isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2
            and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                    and math.isfinite(v) for v in value))


def feet_to_pitch(pixel, matrix, pitch, reprojection_error, max_error_px=2.):
    """Inverse of the delivered normalized-pitch -> image homography."""
    if not finite_xy(pixel) or not math.isfinite(reprojection_error) or not 0 <= reprojection_error <= max_error_px:
        return None
    h = np.asarray(matrix, dtype=float)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return None
    try:
        q = np.linalg.solve(h, [*pixel, 1.])
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(q).all() or abs(q[2]) < 1e-9:
        return None
    xy = (q[:2] / q[2] * np.asarray(pitch)).tolist()
    return xy if all(0 <= xy[i] <= pitch[i] for i in range(2)) else None


def _assignment(cost):
    """Rectangular Hungarian assignment (rows <= columns), deterministic ties."""
    n, m = cost.shape
    u, v, p, way = np.zeros(n+1), np.zeros(m+1), np.zeros(m+1, int), np.zeros(m+1, int)
    for i in range(1, n+1):
        p[0] = i
        minimum, used, j0 = np.full(m+1, np.inf), np.zeros(m+1, bool), 0
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], np.inf, 0
            for j in range(1, m+1):
                if used[j]:
                    continue
                cur = cost[i0-1, j-1] - u[i0] - v[j]
                if cur < minimum[j]:
                    minimum[j], way[j] = cur, j0
                if minimum[j] < delta:
                    delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
    result = [None] * n
    for j in range(1, m+1):
        if p[j]:
            result[p[j]-1] = j-1
    return result, sum(cost[i, j] for i, j in enumerate(result))


def associate(people_xy, points_xy, max_distance_m=1.5, min_margin_m=.25):
    """Global maximum-cardinality/minimum-distance match with ambiguity checks.

    Removing each selected edge and solving again checks the global alternative,
    including swaps and unmatched people. An ambiguous edge is never forced.
    """
    if not all(math.isfinite(v) for v in (max_distance_m,min_margin_m)) or not max_distance_m > 0 or not 0 < min_margin_m < max_distance_m:
        raise ValueError('Positive distance and smaller positive margin required')
    n, m = len(people_xy), len(points_xy)
    if not n:
        return []
    penalty = (n+1) * (max_distance_m+1)
    cost = np.full((n, m+n), penalty, float)
    for i, xy in enumerate(people_xy):
        for j, point in enumerate(points_xy):
            if finite_xy(xy) and finite_xy(point):
                d = math.dist(xy, point)
                cost[i, j] = d if d <= max_distance_m else penalty*4
            else:
                cost[i, j] = penalty*4
    chosen, total = _assignment(cost)
    rows = []
    for i, j in enumerate(chosen):
        if j >= m or cost[i, j] > max_distance_m:
            rows.append(dict(point_index=None, status='unmatched' if finite_xy(people_xy[i]) else 'invalid_geometry', distance_m=None, margin_m=None))
            continue
        alternative = cost.copy()
        alternative[i, j] = penalty*4
        _, other_total = _assignment(alternative)
        margin = float(other_total-total)
        accepted = margin >= min_margin_m
        rows.append(dict(point_index=j if accepted else None, status='matched' if accepted else 'ambiguous',
                         distance_m=float(cost[i, j]), margin_m=margin))
    return rows


class ImageIdentities:
    """Conservative consecutive-frame box overlap, independent of noisy feet.

    No prediction-only continuity or re-identification through gaps. Ambiguous
    global assignments retire the old identity instead of selecting by color.
    """
    def __init__(self, min_iou=.3, min_margin=.1):
        self.min_iou, self.min_margin = min_iou, min_margin
        self.tracks = {}; self.serial = 0; self.last_frame = -1

    def step(self, frame, boxes, reset=False):
        if type(frame) is not int or frame <= self.last_frame:
            raise ValueError('Identity frames must strictly increase')
        if reset or frame != self.last_frame+1:
            self.tracks.clear()
        self.last_frame = frame
        keys = sorted(self.tracks); n, m = len(boxes), len(self.tracks)
        penalty = float(n+1)
        cost = np.full((n,m+n),penalty)
        for i,box in enumerate(boxes):
            if box is None:continue
            x,y,X,Y=box; area=(X-x)*(Y-y)
            for j,key in enumerate(keys):
                a,b,A,B=self.tracks[key]
                inter=max(0,min(X,A)-max(x,a))*max(0,min(Y,B)-max(y,b))
                union=area+(A-a)*(B-b)-inter
                iou=inter/union if union else 0
                if iou>=self.min_iou:cost[i,j]=1-iou
        chosen,total=_assignment(cost) if n else ([],0)
        result=[]; next_tracks={}
        for i,(box,j) in enumerate(zip(boxes,chosen)):
            if box is None:
                result.append(None);continue
            key=None
            if j<m and cost[i,j]<penalty:
                alternative=cost.copy();alternative[i,j]=penalty*4
                _,other=_assignment(alternative)
                if other-total>=self.min_margin:key=keys[j]
            if key is None:
                self.serial+=1;key=f'image-person-{self.serial:06d}'
            next_tracks[key]=list(box);result.append(key)
        self.tracks=next_tracks
        return result


class LocalIdentities:
    """Causal motion association; slots are not identities. Ambiguity retires IDs."""
    def __init__(self, fps=25, max_gap_frames=25, max_speed_mps=10., margin_m=.25):
        if not all(math.isfinite(v) for v in (fps,max_speed_mps,margin_m)) or fps <= 0 or max_gap_frames < 1 or max_speed_mps <= 0 or margin_m<=0:
            raise ValueError('Invalid identity configuration')
        self.fps, self.max_gap, self.max_speed, self.margin = fps, max_gap_frames, max_speed_mps, margin_m
        self.serial, self.last_frame, self.tracks = 0, -1, {}

    def reset(self):
        self.tracks.clear()  # serial deliberately survives cuts

    def step(self, frame, positions, reset=False):
        if type(frame) is not int or frame <= self.last_frame:
            raise ValueError('Identity frames must strictly increase')
        self.last_frame = frame
        if reset:
            self.reset()
        self.tracks = {k: v for k, v in self.tracks.items() if frame-v['frame'] <= self.max_gap}
        ids = sorted(self.tracks)
        predictions = []
        for key in ids:
            t = self.tracks[key]
            dt = (frame-t['frame'])/self.fps
            predictions.append([t['xy'][a]+t['velocity'][a]*dt for a in (0, 1)])
        # Per-track physical gates prevent a stale ID from jumping across the pitch.
        max_gate = max([.5+self.max_speed*(frame-self.tracks[k]['frame'])/self.fps for k in ids] or [1.5])
        matches = associate(positions, predictions, max_gate, min(self.margin, max_gate/2))
        result, next_tracks = [], {}
        for xy, match in zip(positions, matches):
            if not finite_xy(xy):
                result.append(None)
                continue
            key = ids[match['point_index']] if match['point_index'] is not None else None
            old = self.tracks.get(key)
            if old and math.dist(old['xy'], xy) > .5+self.max_speed*(frame-old['frame'])/self.fps:
                key, old = None, None
            if key is None:
                self.serial += 1
                key = f'person-{self.serial:06d}'
            velocity = [(xy[a]-old['xy'][a])*self.fps/(frame-old['frame']) for a in (0, 1)] if old else [0., 0.]
            speed = math.hypot(*velocity)
            if speed > self.max_speed:
                velocity = [v*self.max_speed/speed for v in velocity]
            next_tracks[key] = dict(frame=frame, xy=list(xy), velocity=velocity)
            result.append(key)
        # Unobserved/ambiguous identities are retired rather than silently reused.
        # Only a wholly missing frame permits bounded recovery on the next frame.
        self.tracks = next_tracks if positions else self.tracks
        return result
