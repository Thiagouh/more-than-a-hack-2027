"""Frame-bound projected-ball/body corroboration; not independent ball detection."""
import math
import numpy as np


def projected_ball_contact(mask,scene,crop_origin):
    result=dict(accepted=False,reason='no_reviewed_projected_ball',producer='ATD projected into tactical RGB')
    ball=scene.get('ball_observation')
    if (scene.get('timeline')!='tactical' or scene.get('same_source_video_frame') is not True
            or not isinstance(ball,dict) or ball.get('source')!='observed'
            or ball.get('projection_reviewed') is not True or ball.get('homography_valid') is not True
            or ball.get('quality_flags',[]) or ball.get('image_sha256')!=scene.get('image_sha256')
            or type(ball.get('frame')) is not int or ball.get('frame')!=scene.get('frame')
            or type(ball.get('available_at_frame')) is not int
            or ball['available_at_frame']!=scene['frame']):
        return result
    error=ball.get('reprojection_error')
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=2:
        return dict(result,reason='homography_error_or_unknown')
    xy=np.asarray(ball.get('xy'),dtype=float)
    if xy.shape!=(2,) or not np.isfinite(xy).all():return dict(result,reason='ball_pixel_invalid')
    mask=np.asarray(mask)
    if mask.ndim!=2 or mask.dtype!=bool:raise ValueError('Binary person mask required')
    ys,xs=np.where(mask)
    if not len(xs):return dict(result,reason='empty_person_mask')
    x,y=xy-np.asarray(crop_origin)
    x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    # A conservative upper-body pixel region. Hands may extend beyond torso core.
    if not (x0-2<=x<x1+2 and y0+.1*(y1-y0)<=y<y0+.7*(y1-y0)):
        return dict(result,reason='ball_outside_upper_body')
    px,py=int(round(x)),int(round(y));h,w=mask.shape
    if not (0<=px<w and 0<=py<h):return dict(result,reason='ball_outside_image')
    nearby=mask[max(0,py-2):min(h,py+3),max(0,px-2):min(w,px+3)]
    if not nearby.any():return dict(result,reason='ball_not_adjacent_to_silhouette')
    return dict(result,accepted=True,reason='projected_ball_near_upper_body',pixel=[float(x),float(y)],
        independent_ball_detector=False,requires_three_consecutive_observations=True)


def sam_ball_contact(person_mask,ball_masks,scores):
    """Filter SAM soccer-ball candidates by size and upper-body adjacency.

    SAM scores select candidates after the official confidence filter; they are
    not approval or a calibrated possession probability.
    """
    person=np.asarray(person_mask)
    balls=np.asarray(ball_masks);scores=np.asarray(scores)
    if person.ndim!=2 or person.dtype!=bool:raise ValueError('Binary person mask required')
    if balls.shape!=(len(scores),*person.shape) or balls.dtype!=bool or not np.isfinite(scores).all():
        raise ValueError('Ball candidates must match crop dimensions')
    ys,xs=np.where(person)
    if not len(xs):return dict(accepted=False,reason='empty_person_mask',producer='SAM3 soccer ball')
    x0,x1,y0,y1=xs.min(),xs.max()+1,ys.min(),ys.max()+1
    candidates=[]
    for i,mask in enumerate(balls):
        by,bx=np.where(mask);area=len(bx)
        if not 2<=area<=.10*int(person.sum()):continue
        w,h=bx.max()-bx.min()+1,by.max()-by.min()+1
        if not .5<=w/h<=2:continue
        x,y=float(bx.mean()),float(by.mean())
        if not (x0-2<=x<x1+2 and y0+.1*(y1-y0)<=y<y0+.7*(y1-y0)):continue
        px,py=int(round(x)),int(round(y));hh,ww=person.shape
        if not person[max(0,py-2):min(hh,py+3),max(0,px-2):min(ww,px+3)].any():continue
        candidates.append((float(scores[i]),i,[x,y],area))
    if not candidates:return dict(accepted=False,reason='no_valid_sam_ball_near_upper_body',producer='SAM3 soccer ball')
    score,index,pixel,area=max(candidates)
    return dict(accepted=True,reason='sam_ball_near_upper_body',producer='SAM3 soccer ball',
        candidate_index=index,pixel=pixel,ball_pixels=area,segmentation_score=score,
        independent_ball_detector=True,requires_three_consecutive_observations=True,calibrated_probability=False)
