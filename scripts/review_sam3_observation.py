"""Apply an explicit hash-bound review to a SAM control candidate, without inference."""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256,binary_mask


def review(package,verdict_path,out):
    spec=read_json(package);verdict=read_json(verdict_path)
    if spec.get('schema')!='sam3-visual-control-v1' or spec.get('sam3_executed') is not True:
        raise ValueError('Executed SAM candidate required')
    if (verdict.get('package_sha256')!=sha256(package) or verdict.get('image_sha256')!=spec['image_sha256']
            or verdict.get('mask_sha256')!=spec.get('mask_sha256')
            or type(verdict.get('approved')) is not bool
            or not isinstance(verdict.get('reviewer'),str) or not verdict['reviewer'].strip()
            or not isinstance(verdict.get('notes'),str) or not verdict['notes'].strip()):
        raise ValueError('Explicit post-inference asset-bound mask review required')
    image=Path(spec['image'])
    if not image.is_absolute():image=package.parent/image
    if sha256(image)!=spec['image_sha256']:raise ValueError('Candidate source image changed')
    mask=Path(spec['mask_file'])
    if not mask.is_absolute():mask=package.parent/mask
    if sha256(mask)!=spec['mask_sha256']:raise ValueError('Candidate mask changed')
    with Image.open(mask) as m:
        pixels=np.asarray(m)
        if pixels.ndim!=2:raise ValueError('Single-channel binary mask required')
        binary_mask(pixels,pixels.shape)
        if verdict['approved'] and not pixels.any():raise ValueError('Cannot approve empty mask')
    ball=spec.get('sam_ball_contact',{})
    if verdict['approved'] and ball.get('accepted'):
        ball_path=Path(ball['mask_file'])
        if (verdict.get('ball_visible_correct') is not True or verdict.get('ball_mask_sha256')!=ball['mask_sha256']
                or sha256(ball_path)!=ball['mask_sha256']):
            raise ValueError('SAM ball mask needs explicit asset-bound correctness review')
    if verdict['approved'] and spec.get('quality_reason') is not None:
        raise ValueError('Visual review does not override automatic mask blockers')
    if verdict.get('held_ball_verified') is True:
        if not verdict['approved'] or not spec['observations'] or any(o['role']!='goalkeeper' for o in spec['observations']):
            raise ValueError('Reviewed held ball requires an approved classified SAM goalkeeper')
        for observation in spec['observations']:
            observation.update(control_evidence='reviewed_held_ball',ball_control_observed=True,reviewer=verdict['reviewer'])
        spec['held_ball_manual_review']=True
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    spec.update(mask_file=str(mask.resolve()),mask_reviewed=verdict['approved'],mask_reviewer=verdict['reviewer'],
        mask_review_notes=verdict['notes'],mask_review_sha256=sha256(verdict_path),candidate_sha256=sha256(package))
    if not verdict['approved']:
        spec['observations']=[]
    write_json(out/'visual-observations.json',spec)
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--review',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();review(a.package,a.review,a.out)


if __name__=='__main__':main()
