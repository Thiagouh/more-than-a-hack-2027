"""Frozen masked-uniform references and frame-local SAM 3 adapter."""
import hashlib
import importlib.util
import re
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from src.ball_events.uniforms import pixel_histogram

SAM3_REVISION = 'eef6839bcfc3458dd52ea80535dc6bd62aea86b1'
DESCRIPTOR_VERSION = 'masked-hsv-v3-core'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binary_mask(mask, shape):
    mask = np.asarray(mask)
    if mask.shape != shape or mask.dtype.kind not in 'buif' or (mask.dtype.kind != 'b' and not np.isin(mask, [0, 1, 255]).all()):
        raise ValueError('Expected binary mask matching image dimensions')
    return mask if mask.dtype.kind == 'b' else mask != 0


def mask_stack(masks, shape):
    masks = list(masks)
    return np.stack([binary_mask(m, shape) for m in masks]) if masks else np.zeros((0, *shape), bool)


def interior_mask(mask, radius):
    """Conservative square erosion, with the image exterior treated as unknown."""
    padded = np.pad(mask, radius, constant_values=False)
    core = np.ones_like(mask)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):
            core &= padded[dy:dy+mask.shape[0], dx:dx+mask.shape[1]]
    return core


def reviewed_box_quality(mask, box):
    """Measure leakage without clipping a bad mask into an apparently good one."""
    values = np.asarray(box)
    if (values.shape != (4,) or values.dtype.kind not in 'uif'
            or not np.isfinite(values).all()):
        raise ValueError('Invalid reviewed box')
    x0,y0,x1,y1 = values
    if not (0 <= x0 < x1 <= mask.shape[1] and 0 <= y0 < y1 <= mask.shape[0]):
        raise ValueError('Reviewed box outside image')
    left,top = int(np.floor(x0)),int(np.floor(y0))
    right,bottom = int(np.ceil(x1)),int(np.ceil(y1))
    area = int(mask.sum())
    outside = area-int(mask[top:bottom,left:right].sum())
    fraction = outside/area if area else 0.
    return dict(outside_box_pixels=outside, outside_box_fraction=fraction), (
        'mask_escapes_reviewed_box' if fraction > .1 else None)


def torso_evidence(image, mask, excluded=None, *, return_mask=False):
    image = np.asarray(image)
    if (image.ndim != 3 or image.shape[2] != 3 or image.dtype.kind not in 'uif'
            or not np.isfinite(image).all() or (image < 0).any() or (image > 255).any()):
        raise ValueError('Expected finite RGB H x W x 3 in 0..255')
    mask = binary_mask(mask, image.shape[:2])
    excluded = np.zeros_like(mask) if excluded is None else binary_mask(excluded, mask.shape)
    quality = dict(mask_pixels=int(mask.sum()), torso_pixels=0, valid_torso_pixels=0,
                   boundary_torso_pixels=0, core_radius=0,
                   removed_torso_pixels=0, valid_torso_fraction=0.,
                   overlap_pixels=int((mask & excluded).sum()))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None, quality, 'empty_mask'
    x0,x1,y0,y1 = xs.min(),xs.max()+1,ys.min(),ys.max()+1
    h,w = y1-y0,x1-x0
    if h < 20 or w < 10:
        return None, quality, 'small_mask'
    if w/h > .85:
        return None, quality, 'merged_or_non_upright_mask'
    # Outside this tight box the original mask is identically false. Erosion
    # with false padding is therefore exactly equivalent on the smaller array.
    original_shape=mask.shape
    bounds=(x0,y0,x1,y1)
    image=image[y0:y1,x0:x1]
    mask=mask[y0:y1,x0:x1]
    excluded=excluded[y0:y1,x0:x1]
    torso = np.zeros_like(mask)
    torso[int(.2*h):int(.55*h),int(.2*w):int(.8*w)] = True
    torso &= mask
    radius = max(1, int(np.ceil(.05*w)))
    core = torso & interior_mask(mask, radius)
    valid = core & ~excluded
    total, count = int(torso.sum()), int(valid.sum())
    quality.update(torso_pixels=total, valid_torso_pixels=count,
                   boundary_torso_pixels=int((torso & ~core).sum()), core_radius=radius,
                   removed_torso_pixels=total-count,
                   valid_torso_fraction=count/total if total else 0.)
    if count < 24 or quality['valid_torso_fraction'] < .5:
        return None, quality, 'insufficient_torso'
    if return_mask:
        full=np.zeros(original_shape,dtype=bool)
        x0,y0,x1,y1=bounds
        full[y0:y1,x0:x1]=valid
        return full, quality, None
    return pixel_histogram(image[valid]), quality, None


def descriptor(image, mask, excluded=None):
    return torso_evidence(image, mask, excluded)[0]


def mask_flags(masks):
    flags = [None] * len(masks)
    areas = [int(m.sum()) for m in masks]
    bounds=[]
    for mask in masks:
        ys,xs=np.nonzero(mask)
        bounds.append((int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)) if len(xs) else None)
    for i in range(len(masks)):
        for j in range(i):
            if bounds[i] is None or bounds[j] is None:continue
            left=max(bounds[i][0],bounds[j][0]);top=max(bounds[i][1],bounds[j][1])
            right=min(bounds[i][2],bounds[j][2]);bottom=min(bounds[i][3],bounds[j][3])
            if left>=right or top>=bottom:continue
            intersection = int((masks[i][top:bottom,left:right] & masks[j][top:bottom,left:right]).sum())
            if not min(areas[i], areas[j]):
                continue
            if intersection / (areas[i]+areas[j]-intersection) >= .85:
                flags[i] = flags[j] = 'duplicate_mask'
            elif intersection / min(areas[i], areas[j]) >= .8:
                flags[i] = flags[j] = 'merged_or_contained_mask'
    return flags


class TeamClassifier:
    def __init__(self, references, max_distance=.35, min_margin=.10):
        for value in (max_distance, min_margin):
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Thresholds must be finite within 0..1')
        self.templates=[]
        self.max_distance,self.min_margin=max_distance,min_margin
        for image,mask,team,role in references:
            if (team,role) not in [('H','player'),('A','player'),('H','goalkeeper'),('A','goalkeeper'),(None,'referee')]:
                raise ValueError('Invalid reference label')
            d=descriptor(image,mask)
            if d is None:raise ValueError('Reference torso too small')
            self.templates.append((team,role,d))
        if not {'H','A'} <= {t for t,r,_ in self.templates if r == 'player'}:raise ValueError('Both outfield teams required')

    def classify(self,image,masks):
        masks=mask_stack(masks,np.asarray(image).shape[:2])
        if len(masks)==0:return []
        overlap=masks.sum(axis=0)>1
        flags=mask_flags(masks)
        rows=[]
        for i,mask in enumerate(masks):
            d,quality,reason=torso_evidence(image,mask,overlap)
            reason=flags[i] or reason
            scores={}
            if d is not None and reason is None:
                for team,role,ref in self.templates:
                    distance=float(np.linalg.norm(np.sqrt(d)-np.sqrt(ref))/np.sqrt(2))
                    scores[team,role]=min(scores.get((team,role),1.),distance)
            ranked=sorted(scores.items(),key=lambda p:(p[1],str(p[0])))
            distance=ranked[0][1] if ranked else None
            margin=ranked[1][1]-distance if len(ranked)>1 else None
            accepted=distance is not None and margin is not None and distance<=self.max_distance and margin>=self.min_margin
            team,role=ranked[0][0] if accepted else (None,'unknown')
            rows.append(dict(detection_index=i,team=team,role=role,distance=distance,margin=margin,
                classification=team or ('referee' if role == 'referee' else 'uncertain'),
                reason=reason or ('uniform_match' if accepted else 'distance_or_margin'),
                **quality,calibrated_probability=False))
        return rows


def verify_installation(package_file):
    root=Path(package_file).resolve().parent
    revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True)
    if revision != SAM3_REVISION or dirty.strip():
        raise RuntimeError('SAM 3 requires the pinned, unmodified Git checkout')
    return revision


class Sam3:
    def __init__(self,checkpoint,expected_sha256,device='cuda'):
        if device not in ('cuda','mps','cpu'):
            raise ValueError('Device must be cuda, mps or cpu')
        if not isinstance(expected_sha256,str) or not re.fullmatch('[0-9a-fA-F]{64}',expected_sha256):
            raise ValueError('Expected SHA256 must contain 64 hex characters')
        if sha256(checkpoint)!=expected_sha256.lower():raise ValueError('Checkpoint hash mismatch')
        import torch
        package=importlib.util.find_spec('sam3')
        if package is None:raise RuntimeError('Install SAM 3 in the selected Python environment')
        verify_installation(package.origin)
        if device != 'cuda':
            from .mac_compat import enable
            enable(package.origin,device)
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if device == 'cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
        if device == 'mps' and not torch.backends.mps.is_available():raise RuntimeError('Apple MPS GPU required')
        self.environment=dict(torch=torch.__version__,cuda=torch.version.cuda,device=device,
            experimental=device!='cuda',gpu=torch.cuda.get_device_name(0) if device=='cuda' else device)
        if device != 'cuda':
            from . import mac_compat
            self.environment.update(compatibility=mac_compat.VERSION,compatibility_sha256=sha256(mac_compat.__file__))
        self.processor=Sam3Processor(build_sam3_image_model(checkpoint_path=str(checkpoint),
            load_from_HF=False,device=device),device=device)

    def segment(self,image,box=None):
        state=self.processor.set_image(image)
        if box is None:
            out=self.processor.set_text_prompt(prompt='person',state=state)
        else:
            self.processor.set_text_prompt(prompt='person',state=state)
            x0,y0,x1,y1=box;w,h=image.size
            out=self.processor.add_geometric_prompt(box=[(x0+x1)/(2*w),(y0+y1)/(2*h),(x1-x0)/w,(y1-y0)/h],label=True,state=state)
        masks=mask_stack(out['masks'].detach().cpu().numpy()[:,0],(image.height,image.width))
        scores=out['scores'].detach().cpu().numpy()
        if len(masks)!=len(scores) or not np.isfinite(scores).all():raise ValueError('Invalid SAM outputs')
        return masks,scores

    def segment_boxes(self, image, boxes, context_crop=False, context_scale=3.):
        """Return the best mask for each reviewed box, independently."""
        if not np.isfinite(context_scale) or context_scale < 1:
            raise ValueError('Context scale must be finite and >= 1')
        masks, scores = [], []
        for box in boxes:
            if context_crop:
                x0,y0,x1,y1=box
                side=context_scale*max(x1-x0,y1-y0)
                cx,cy=(x0+x1)/2,(y0+y1)/2
                left,top=max(0,int(cx-side/2)),max(0,int(cy-side/2))
                right,bottom=min(image.width,int(cx+side/2)),min(image.height,int(cy+side/2))
                candidates,candidate_scores=self.segment(image.crop((left,top,right,bottom)),
                    [x0-left,y0-top,x1-left,y1-top])
                restored=np.zeros((len(candidates),image.height,image.width),bool)
                restored[:,top:bottom,left:right]=candidates
                candidates=restored
            else:
                candidates, candidate_scores = self.segment(image, box)
            if not len(candidate_scores):
                masks.append(np.zeros((image.height,image.width),bool))
                scores.append(0.)
                continue
            region=np.zeros((image.height,image.width),bool)
            x0,y0,x1,y1=map(int,box)
            region[y0:y1,x0:x1]=True
            alignment=[int((m & region).sum())/max(1,int((m | region).sum())) for m in candidates]
            best = max(range(len(candidates)),key=lambda i:(alignment[i],float(candidate_scores[i])))
            masks.append(candidates[best])
            scores.append(float(candidate_scores[best]))
        return mask_stack(masks,(image.height,image.width)), np.asarray(scores)
