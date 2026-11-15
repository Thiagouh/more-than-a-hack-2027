"""Shared rendering eligibility and explicit omission reasons."""
import math


def render_status(point,frame,pitch):
    if point.get('source')=='missing':return 'missing'
    if point.get('available_at_frame',frame)>frame:return 'not_yet_available'
    x,y=point.get('x_m'),point.get('y_m')
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y)):return 'invalid_coordinates'
    if not (0<=x<=pitch[0] and 0<=y<=pitch[1]):return 'outside_pitch'
    return 'drawn' if point.get('source')=='observed' and not point.get('quality_flags') else 'drawn_with_quality_warning'
