"""Prepare a hash-bound step-2 visual review and optionally freeze a complete bundle.

Never infers approval from SAM scores or geometry. All candidates, including
rejected masks, remain available for shared-pixel exclusion.
"""
import argparse
import copy
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

from scripts.audit_team_masks import audit
from src.team_identification.manifests import read_json, write_json, freeze, load_frozen, load_image
from src.team_identification.pilot import sha256, binary_mask, torso_evidence
from src.team_identification.protocol import runtime_provenance
from src.team_identification.reference_review import PROFILE, completeness, text
from src.team_identification.goalkeeper_context import CONTEXT_PROFILE,validate_policy


def render_panels(spec, base, out):
    """Display original RGB, silhouette overlay and only descriptor RGB pixels."""
    rows=spec['references']
    loaded=[]
    for row in rows:
        image=load_image(base/row['image'])
        with Image.open(base/row['mask']) as m:
            mask=binary_mask(m,(image.height,image.width))
        loaded.append((image,mask))
    out.mkdir()
    for i,row in enumerate(rows):
        image,mask=loaded[i]
        group=[loaded[j][1] for j,r in enumerate(rows) if r['image_sha256']==row['image_sha256']]
        core,_,reason=torso_evidence(image,mask,np.stack(group).sum(axis=0)>1,return_mask=True)
        core=np.zeros_like(mask) if core is None else core
        pixels=np.asarray(image)
        overlay=pixels.copy()
        overlay[mask]=(.5*overlay[mask]+.5*np.array([255,0,255])).astype('uint8')
        rgb_core=np.where(core[:,:,None],pixels,210).astype('uint8')
        x,y,X,Y=map(int,row['box'])
        region=(max(0,x-8),max(0,y-8),min(image.width,X+8),min(image.height,Y+8))
        w,h=(region[2]-region[0])*5,(region[3]-region[1])*5
        sheet=Image.new('RGB',(w*3,h+25),'white');draw=ImageDraw.Draw(sheet)
        for j,(tile,title) in enumerate(zip((image,Image.fromarray(overlay),Image.fromarray(rgb_core)),
                ('original','SAM mask','descriptor RGB'))):
            sheet.paste(tile.crop(region).resize((w,h),Image.Resampling.NEAREST),(j*w,25))
            draw.text((j*w+3,5),title,fill='black')
        sheet.save(out/f"{row['label']}.png")


def review(manifest, out, decisions=None, freeze_complete=False, goalkeeper_policy=None):
    spec = read_json(manifest)
    if spec.get('stage') != 'candidate':
        raise ValueError('SAM candidate manifest required')
    policy = read_json(goalkeeper_policy) if goalkeeper_policy else None
    if policy is not None:
        validate_policy(policy)
    verdict = read_json(decisions) if decisions else None
    if verdict:
        if verdict.get('candidate_sha256') != sha256(manifest) or not text(verdict.get('reviewer')):
            raise ValueError('Review must identify reviewer and unchanged candidate manifest')
        entries = verdict.get('references',[])
        if len(entries) != len({e['label'] for e in entries}) or {e['label'] for e in entries} != {r['label'] for r in spec['references']}:
            raise ValueError('One decision per candidate required, including rejected masks')
        if any(e.get('decision') not in ('approve','reject','pending') or not text(e.get('notes')) for e in entries):
            raise ValueError('Explicit decision and notes required')
    if out.exists():
        raise FileExistsError(out)
    # Audit checks image/mask hashes and renders actual descriptor pixels before review.
    audit(manifest,out/'audit')
    render_panels(spec,manifest.parent,out/'panels')
    checked = copy.deepcopy(spec)
    checked.update(reference_profile=CONTEXT_PROFILE if policy else PROFILE,reviewer=verdict['reviewer'] if verdict else None,
                   mask_reviewed=bool(verdict and all(e['decision'] != 'pending' for e in verdict['references'])))
    if policy:
        checked['goalkeeper_context_policy'] = policy
    entries = {e['label']:e for e in verdict['references']} if verdict else {}
    template = []
    for row in checked['references']:
        original = next(r for r in spec['references'] if r['label']==row['label'])
        for key in ('image','mask'):
            row[key] = str((manifest.parent/row[key]).resolve())
        entry = entries.get(row['label'],dict(label=row['label'],decision='pending',notes='Awaiting visual review',
            image_sha256=row['image_sha256'],mask_sha256=row['mask_sha256'],
            torso_verified=False,lighting_verified=False,team_and_role_verified=False,halo_visible=None,
            team=row['team'],role=row['role'],lighting=row.get('lighting'),
            source_sha256=row.get('source_sha256'),source_segment_id=row.get('source_segment_id')))
        if any(entry.get(k) != original.get(k) for k in ('image_sha256','mask_sha256')):
            raise ValueError('Review asset binding mismatch: '+row['label'])
        template.append(entry)
        row['approved'] = entry['decision'] == 'approve'
        row['visual_review'] = dict(entry,reviewer=verdict['reviewer'] if verdict else None)
        # Source/lighting corrections require explicit verified review; no guessed provenance.
        for key in ('source_sha256','source_segment_id','lighting','goalkeeper_verified'):
            if key in entry:
                row[key] = entry[key]
    candidate = out/'reviewed-candidates.json'
    write_json(candidate,checked)
    if not verdict:
        write_json(out/'review-draft.json',dict(candidate_sha256=sha256(manifest),reviewer=None,references=template))
    report = completeness(checked)
    report.update(candidate_sha256=sha256(manifest),reviewed_candidate_sha256=sha256(candidate),
        decisions_sha256=sha256(decisions) if decisions else None,
        goalkeeper_policy_sha256=sha256(goalkeeper_policy) if goalkeeper_policy else None,model_executed=False,
        references_frozen=False,provenance=runtime_provenance(),
        review_script_sha256=sha256(__file__),contract_sha256=sha256(Path(__file__).resolve().parents[1]/'src/team_identification/reference_review.py'))
    report['status'] = 'ready_for_complete_freeze' if report['ready'] else 'incomplete_no_freeze'
    write_json(out/'review-report.json',report)
    if freeze_complete and report['ready']:
        try:
            freeze(candidate,out/'frozen',require_complete=True)
            _,bundle = load_frozen(out/'frozen/frozen-references.json')
        except ValueError as error:
            report.update(status='invalid_reference_no_freeze',ready=False)
            report['blockers'].append(str(error))
        else:
            report.update(status='contextual_references_frozen' if policy else 'complete_frozen',references_frozen=True,
                frozen_sha256=sha256(out/'frozen/frozen-references.json'),frozen_count=len(bundle['references']))
        write_json(out/'review-report.json',report)
    print(report['status'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--decisions',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--freeze-complete',action='store_true')
    parser.add_argument('--goalkeeper-policy',type=Path,help='Use explicit contextual goalkeeper strategy instead of requiring own GK templates')
    args = parser.parse_args()
    review(args.manifest,args.out,args.decisions,args.freeze_complete,args.goalkeeper_policy)


if __name__ == '__main__':main()
