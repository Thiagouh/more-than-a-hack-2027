"""Create a hashed held-out recording manifest and an unreviewed annotation template."""
import argparse
from pathlib import Path
from src.team_identification.manifests import load_frozen, load_image, write_json
from src.team_identification.pilot import sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',type=Path,required=True)
    parser.add_argument('--frames-dir',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--seconds',type=int,nargs='+',default=[16,19,22,25,28,31])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    _,frozen=load_frozen(args.references)
    frames=[]
    for second in sorted(set(args.seconds)):
        path=args.frames_dir/f'{second}.png'
        load_image(path)
        digest=sha256(path)
        if digest in {r['image_sha256'] for r in frozen['evidence']}:
            raise ValueError('Reference image in evaluation')
        frames.append(dict(frame=second,timestamp_s=second,image=str(path.resolve()),
                           image_sha256=digest,split='evaluation'))
    if not frames:raise ValueError('Evaluation frames required')
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/'run-manifest.json',dict(source='broadcast',
        frame_id_semantics='recording sample ID, not original match frame',
        timestamp_semantics='screen recording seconds; source contains cuts',
        recording_sha256=sha256(args.recording),
        frozen_references=str(args.references.resolve()),frozen_sha256=sha256(args.references),frames=frames))
    write_json(args.out/'visual-truth-draft.json',dict(reviewed=False,reviewer=None,
        instructions='Annotate every visible person independently of model output. Each person needs label, box, classification H/A/referee/uncertain, evaluable boolean, lighting sun/shade/mixed, occlusion isolated/overlapped.',
        frames=[dict(frame=f['frame'],image=f['image'],people=[]) for f in frames]))


if __name__=='__main__':main()
