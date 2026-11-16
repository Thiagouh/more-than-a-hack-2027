"""Convert completed identity review annotations into existing v2 visual truth.

The default writes a draft. --freeze uses the existing asset/review checks.
"""
import argparse
from pathlib import Path
from src.team_identification.manifests import read_json, write_json, load_image, pixel_sha256
from src.team_identification.protocol import freeze_truth


def prepare(annotations, split, out, freeze=False):
    spec=read_json(annotations)
    if spec.get('schema')!='team-identity-review-v1':
        raise ValueError('Identity review annotations required')
    selected=[r for r in spec['frames'] if r['split']==split]
    if not selected:
        raise ValueError('Empty review split')
    reviewers={r.get('reviewer') for r in selected}
    complete=all(r.get('reviewed') is True and r.get('annotations_complete') is True and r.get('reviewer') for r in selected)
    exclusions=[dict(source_sha256=spec['source_video_sha256'],source_segment_id=f'tactical-pregame-{f//750}',split='reference')
                for f in sorted({r['frame'] for r in spec['frames'] if r['split']=='reference_candidates'})]
    if split=='evaluation':
        exclusions += [dict(source_sha256=spec['source_video_sha256'],source_segment_id=g,split='development')
                       for g in sorted({r['group_id'] for r in spec['frames'] if r['split']=='development'})]
    frames=[]
    for row in selected:
        path=(annotations.parent/row['image']).resolve()
        image=load_image(path)
        frames.append(dict(row,image=str(path),width=image.width,height=image.height,
            pixel_sha256=pixel_sha256(image),source_sha256=spec['source_video_sha256'],
            source_segment_id=row['group_id'],timestamp_s=row['frame']/spec['fps'],
            temporal_origin_known=True,source_time_s=row['frame']/spec['fps'],
            preparation='Fixed original tactical frames; manually complete people/association review before freezing',
            evaluation_region=[[[0,0],[image.width,0],[image.width,image.height],[0,image.height]]]))
    truth=dict(schema_version=2,stage='draft',reviewed=complete,reviewer=', '.join(sorted(r for r in reviewers if r)) or None,
        annotations_complete=complete,evaluation_mode='full_frame',sample_kind='challenge',synthetic=False,
        excluded_segments=exclusions,frames=frames,identity_split=split,
        limitations='Three held-out sequences are exploratory; no full-match accuracy claim.')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'truth-draft.json',truth)
    if freeze:
        if not complete:raise ValueError('Every selected frame requires complete human review')
        freeze_truth(out/'truth-draft.json',out/'frozen')
    return truth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--split',choices=['development','evaluation'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze',action='store_true')
    a=p.parse_args();prepare(a.annotations,a.split,a.out,a.freeze)


if __name__=='__main__':main()
