"""Local RGB crop classification. Scores are distances, never probabilities.

Instantiate once per video source/continuity segment; reset at cuts or periods.
Reference images and fitted patterns must remain outside public version control.
"""
from collections import Counter, deque
import math

import numpy as np

from .contract import InputError, integer, number, text


def histogram(rgb):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype.kind not in 'uif':
        raise InputError('RGB crop must be a numeric H x W x 3 array in 0..255')
    if not np.isfinite(image).all() or (image < 0).any() or (image > 255).any():
        raise InputError('RGB values must be finite and in 0..255')
    h, w = image.shape[:2]
    if h < 20 or w < 10:
        return None
    return pixel_histogram(image[int(.20*h):int(.55*h), int(.20*w):int(.80*w)].reshape(-1, 3))


def pixel_histogram(rgb):
    """Histogram of every supplied RGB pixel, without spatial resampling."""
    pixels = np.asarray(rgb)
    if pixels.ndim != 2 or pixels.shape[1] != 3 or not len(pixels):
        raise InputError('Pixels must be a nonempty N x 3 array')
    if pixels.dtype.kind not in 'uif' or not np.isfinite(pixels).all() or (pixels < 0).any() or (pixels > 255).any():
        raise InputError('RGB values must be finite and in 0..255')
    pixels = pixels.astype(float)/255
    maximum, minimum = pixels.max(axis=1), pixels.min(axis=1)
    delta = maximum-minimum
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 0)
    hue = np.zeros_like(delta)
    nz = delta > 0
    # Exclusive channels avoid overwriting hue at tied maxima.
    channel = pixels.argmax(axis=1)
    for k in range(3):
        mask = nz & (channel == k)
        hue[mask] = ((pixels[mask, (k+1) % 3]-pixels[mask, (k+2) % 3])/delta[mask]+2*k) % 6
    hue /= 6
    # Soft binning makes nearby colours continuous across bin boundaries and hue wrap.
    hs = np.zeros((24, 8))
    hx = hue*24
    sx = saturation*7
    hi, si = np.floor(hx).astype(int), np.floor(sx).astype(int)
    for dh in (0, 1):
        for ds in (0, 1):
            weight = (hx-hi if dh else 1-(hx-hi))*(sx-si if ds else 1-(sx-si))
            np.add.at(hs, ((hi+dh) % 24, np.minimum(si+ds, 7)), weight)
    value = np.zeros(8)
    vx = maximum*7
    vi = np.floor(vx).astype(int)
    np.add.at(value, vi, 1-(vx-vi))
    np.add.at(value, np.minimum(vi+1, 7), vx-vi)
    return np.r_[.5*hs.ravel()/len(pixels), .5*value/len(pixels)]


class UniformClassifier:
    """One source, explicit (team, role) reference labels, causal per-track votes.

    Pattern input: {'team': 'H'|'A'|None, 'role': 'player'|'goalkeeper'|'referee',
                    'crops': [RGB arrays, ...]}.
    Multiple entries per label are independent templates, not competing labels.
    Call forget(track_id) for retired tracks to bound the registry.
    """
    def __init__(self, source, patterns, max_distance=.35, min_margin=.10):
        text(source, 'video source')
        number(max_distance, 'max_distance', True)
        number(min_margin, 'min_margin')
        if max_distance > 1 or not 0 <= min_margin <= 1:
            raise InputError('Distance and margin must be within 0..1')
        self.source, self.max_distance, self.min_margin = source, max_distance, min_margin
        self.patterns = []
        for pattern in patterns:
            team, role = pattern['team'], pattern['role']
            if role not in ('player', 'goalkeeper', 'referee') or (role == 'referee' and team is not None) or (role != 'referee' and team not in ('H', 'A')):
                raise InputError('Each player/goalkeeper requires H/A; referees require null team')
            histograms = [histogram(crop) for crop in pattern['crops']]
            if not histograms or any(h is None for h in histograms):
                raise InputError('Reference crops must be at least 20 x 10')
            self.patterns.append(((team, role), np.mean(histograms, axis=0)))
        if len({label for label, _ in self.patterns}) < 2:
            raise InputError('At least two distinct labels are required to measure ambiguity')
        self.history = {}

    def forget(self, track_id):
        self.history.pop(track_id, None)

    def reset(self):
        self.history.clear()

    def classify(self, rgb, track_id, frame, available_at_frame=None):
        text(track_id, 'track_id')
        integer(frame, 'frame')
        available = frame if available_at_frame is None else integer(available_at_frame, 'available_at_frame')
        if available < frame:
            raise InputError('Crop cannot be available before its frame')
        history = self.history.setdefault(track_id, deque())
        if history and frame <= history[-1]['frame']:
            raise InputError('Crop frames must increase within each track')
        while history and history[0]['frame'] < frame-4:
            history.popleft()
        hist = histogram(rgb)
        scores = {}
        if hist is not None:
            for label, template in self.patterns:
                d = float(np.linalg.norm(np.sqrt(hist)-np.sqrt(template))/math.sqrt(2))
                scores[label] = min(scores.get(label, 1.), d)
        ordered = sorted(scores.items(), key=lambda item: (item[1], str(item[0])))
        distance = ordered[0][1] if ordered else None
        margin = ordered[1][1]-distance if len(ordered) > 1 else None
        label = ordered[0][0] if ordered and distance <= self.max_distance and margin >= self.min_margin else None
        history.append(dict(frame=frame, available=available, label=label))
        accepted = [entry['label'] for entry in history if entry['label'] is not None]
        votes = Counter(accepted)
        stable = votes.most_common(1)[0][0] if len(accepted) >= 3 else None
        if stable is not None and (votes[stable]/len(accepted) < .8 or label != stable):
            stable = None
        return dict(source=self.source, team=stable[0] if stable else None,
                    role=stable[1] if stable else 'unknown', distance=distance, margin=margin,
                    observations=len(history), accepted_observations=len(accepted),
                    support_start_frame=history[0]['frame'], support_end_frame=frame+1,
                    available_at_frame=max(entry['available'] for entry in history),
                    calibrated_probability=False)


def resolve_player(player, visual=None):
    """Return (detector point or None, diagnostic). ATD team takes precedence.

    Call even without a visual result for ATD-only inputs. Preserve diagnostics for
    excluded detections in the adapter's own report. Never pass unresolved points
    into the H/A-only detector contract.
    """
    out = dict(player)
    team = out.get('team')
    if team not in (None, 'H', 'A'):
        raise InputError('Metadata team must be H, A or null')
    diagnostic = dict(team_source='metadata' if team else 'unknown', visual_conflict=False)
    if visual is not None:
        frame = integer(out['frame'], 'frame')
        start = integer(visual['support_start_frame'], 'visual support start')
        end = integer(visual['support_end_frame'], 'visual support end', 1)
        available = integer(visual['available_at_frame'], 'visual availability')
        if not start <= frame or end != frame+1 or available < frame:
            raise InputError('Visual evidence must end at the player frame')
        diagnostic['visual'] = dict(visual)
        if team:
            diagnostic['visual_conflict'] = (visual['team'] is not None and visual['team'] != team) or visual['role'] == 'referee'
        elif visual['role'] in ('player', 'goalkeeper') and visual['team'] in ('H', 'A'):
            team = visual['team']
            diagnostic['team_source'] = 'uniform'
            out['support_start_frame'] = min(out.get('support_start_frame', frame), start)
            out['support_end_frame'] = max(out.get('support_end_frame', frame+1), end)
            out['available_at_frame'] = max(out.get('available_at_frame', frame), available)
    diagnostic['excluded'] = team is None
    if team is None:
        return None, diagnostic
    out['team'] = team
    out['extensions'] = {**out.get('extensions', {}), 'team_identification': diagnostic}
    return out, diagnostic
