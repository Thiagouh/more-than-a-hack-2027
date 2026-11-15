"""Prepare annotation independently, freeze truth, then bind an evaluation run."""
import argparse
from pathlib import Path
from src.team_identification.manifests import load_frozen, load_image, pixel_sha256, write_json, validate_run
from src.team_identification.pilot import sha256
from src.team_identification.protocol import freeze_truth, load_truth, runtime_provenance, validate_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--truth',type=Path,help='Already frozen truth; create a bound run manifest')
    parser.add_argument('--freeze',type=Path,metavar='DRAFT',help='Freeze reviewed v2 annotation without references')
    parser.add_argument('--frames-dir',type=Path)
    parser.add_argument('--recording',type=Path)
    parser.add_argument('--seconds',type=int,nargs='+',default=[16,19,22,25,28,31])
    parser.add_argument('--mode',choices=['full_frame','verified_regions'],default='full_frame')
    parser.add_argument('--sample-kind',choices=['main','challenge'],default='main')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.freeze:
        if args.truth or args.references or args.frames_dir or args.recording:
            parser.error('--freeze is a separate stage')
        freeze_truth(args.freeze,args.out)
        return
    if args.truth:
        if not args.references or args.frames_dir or args.recording:
            parser.error('--truth requires --references; frame preparation is a separate stage')
        truth = load_truth(args.truth)
        _, frozen = load_frozen(args.references)
        validate_splits(truth,frozen)
        spec = dict(schema_version=2,source='broadcast',evaluation_mode=truth['evaluation_mode'],
            sample_kind=truth['sample_kind'],frozen_truth=str(args.truth.resolve()),truth_sha256=sha256(args.truth),
            frozen_references=str(args.references.resolve()),frozen_sha256=sha256(args.references),
            frames=[dict(frame=f['frame'],timestamp_s=f['timestamp_s'],
                image=str((args.truth.parent/f['image']).resolve()),image_sha256=f['image_sha256'],
                pixel_sha256=f['pixel_sha256'],split='evaluation') for f in truth['frames']],
            environment=runtime_provenance())
        validate_run(spec,args.out,frozen)
        args.out.mkdir(parents=True,exist_ok=False)
        write_json(args.out/'run-manifest.json',spec)
        return
    if args.references or not args.frames_dir or not args.recording:
        parser.error('Draft preparation requires --frames-dir and --recording, without references')
    recording_hash = sha256(args.recording)
    frames = []
    for second in sorted(set(args.seconds)):
        if second < 0:
            parser.error('Recording seconds must be nonnegative')
        path = args.frames_dir/f'{second}.png'
        image = load_image(path)
        frames.append(dict(frame=second,timestamp_s=second,image=str(path.resolve()),
            image_sha256=sha256(path),pixel_sha256=pixel_sha256(image),width=image.width,height=image.height,
            source_sha256=recording_hash,source_segment_id=None,group_id=None,
            temporal_origin_known=False,source_time_s=None,
            preparation='Existing PNG recording sample; record extraction settings and cuts during review',
            evaluation_region=[],people=[]))
    if not frames:
        raise ValueError('Evaluation frames required')
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'visual-truth-draft.json',dict(schema_version=2,stage='draft',
        reviewed=False,reviewer=None,annotations_complete=False,
        evaluation_mode=args.mode,sample_kind=args.sample_kind,excluded_segments=[],
        frame_id_semantics='recording sample ID, not original match frame',
        timestamp_semantics='screen recording seconds; source contains cuts',
        instructions='Review region polygons, source segments/groups, temporal origin and preparation. Annotate every person in region: label, box, role, classification, evaluable, partial, lighting, occlusion. Verified regions also require mask, mask_sha256, mask_reviewed. Record reference/development segments in excluded_segments. Freeze before producing predictions.',
        environment=runtime_provenance(),frames=frames))


if __name__ == '__main__':
    main()
