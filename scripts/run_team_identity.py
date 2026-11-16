"""Run causal full-frame identity with approved same-source uniform references.

No ground-truth input. Outputs are experimental until separately evaluated.
"""
import argparse
from itertools import islice
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
import copy
import numpy as np
from PIL import Image
from src.ball_events.inputs import atd
from src.team_identification.manifests import load_frozen, write_json
from src.team_identification.overlay import SCHEMA
from src.team_identification.pilot import Sam3, sha256
from src.team_identification.pipeline import IdentityPipeline
from src.team_identification.spaced_frames import POLICY as SPACED_POLICY, OFFSETS
from src.team_identification.four_frames import POLICY as LEGACY_POLICY
from scripts.prepare_tactical_visual import homographies, video_info


def run(args):
    meta, factory, _, files, _ = atd(args.data_dir)
    if not 0 <= args.start_frame < args.end_frame <= meta['n_frames']:
        raise ValueError('Invalid original-frame window')
    video = args.data_dir/'tactical/BAR-ATH_tactical.mp4'
    video_hash = sha256(video)
    model_path=getattr(args,'supervised_model',None)
    if model_path:
        from src.team_identification.supervised_uniforms import SupervisedUniforms
        artifact=json.loads(model_path.read_text());classifier=SupervisedUniforms(artifact)
        required={'Barcelona','Athletic','Goleiro Barcelona','Goleiro Athletic','Arbitro'}
        if not required<={r['label'] for r in artifact['examples']} or any(r.get('source_video_sha256')!=video_hash for r in artifact['examples']):
            raise ValueError('Complete same-source supervised uniform classes required')
        uniform_binding=dict(supervised_model=str(model_path.resolve()),supervised_model_sha256=sha256(model_path),
            uniform_policy=getattr(args,'uniform_policy',SPACED_POLICY),uniform_window_frames=4)
        offsets=OFFSETS if uniform_binding['uniform_policy']==SPACED_POLICY else [-3,-2,-1,0]
        uniform_binding.update(uniform_offsets=offsets,uniform_window_span_s=-offsets[0]/25,
            uniform_continuity_window_frames=1-offsets[0])
    else:
        classifier, frozen = load_frozen(args.references)
        if frozen.get('reference_profile') != 'complete-v1':
            raise ValueError('Complete reviewed uniform references, including both goalkeepers, required')
        if any(r.get('source_sha256') != video_hash for r in frozen['references']):
            raise ValueError('Use tactical references; cross-camera transfer has not been validated')
        if args.checkpoint_sha256 != frozen['checkpoint_sha256']:
            raise ValueError('Checkpoint differs from frozen reference segmentation')
        uniform_binding=dict(frozen_references=str(args.references.resolve()),frozen_references_sha256=sha256(args.references))
    info = video_info(video)
    from fractions import Fraction
    if int(info['nb_frames']) != meta['n_frames'] or Fraction(info['r_frame_rate']) != meta['fps']:
        raise ValueError('Video and tracking timelines differ')
    hpath = args.data_dir/'tactical/BAR-ATH_tactical_homography.csv'
    hs = homographies(hpath, range(args.start_frame, args.end_frame))
    config = json.loads(args.config.read_text()) if args.config else None
    from src.team_possession.restarts import load_kickoffs
    kickoffs,_=load_kickoffs(args.data_dir,meta)
    pipeline_meta=copy.deepcopy(meta)
    period_resets=[r['frame'] for r in kickoffs if r.get('half',1)>1]
    pipeline_meta['continuity_starts']=sorted(set(meta.get('continuity_starts',[]))|set(period_resets))
    pipeline = IdentityPipeline(pipeline_meta, classifier, config, uniform_policy=getattr(args,'uniform_policy',SPACED_POLICY))
    args.out.mkdir(parents=True, exist_ok=False)
    package = dict(schema=SCHEMA, timeline=meta['timeline'], fps=meta['fps'], n_frames=meta['n_frames'],
        status='started', validation='not_evaluated', inference_mode='full_frame', start_frame=args.start_frame,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        end_frame_exclusive=args.end_frame, source_video_sha256=video_hash,
        atd_raw_sha256=sha256(files['atd_raw']), atd_meta_sha256=sha256(files['atd_meta']), homography_sha256=sha256(hpath),
        **uniform_binding,
        observations='observations.jsonl', configuration=pipeline.config, parameters_calibrated=False,
        period_reset_frames=period_resets,
        evidence_latency='offline replay; model runtime not measured',
        code_sha256={p.name: sha256(p) for p in (Path(__file__), *sorted(Path('src/team_identification').glob('*.py')))})
    write_json(args.out/'overlay.json', package)
    sam = Sam3(args.checkpoint, args.checkpoint_sha256, device=args.device)
    # Exact decode counter, bounded to the requested interval; no timestamp seek.
    select = f'between(n\\,{args.start_frame}\\,{args.end_frame-1})'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video), '-vf', 'select='+select,
               '-fps_mode', 'passthrough', '-frames:v', str(args.end_frame-args.start_frame),
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    with (args.out/'decode.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            with (args.out/'observations.jsonl').open('w') as journal:
                for envelope in islice(factory(), args.start_frame, args.end_frame):
                    size = info['width']*info['height']*3
                    payload = proc.stdout.read(size)
                    if len(payload) != size:
                        raise ValueError('Incomplete video decode')
                    image = Image.frombytes('RGB', (info['width'], info['height']), payload)
                    masks, _ = sam.segment(image)
                    row = pipeline.step(envelope, image, masks, hs[envelope['frame']])
                    mask_dir=args.out/'masks'/f"{envelope['frame']:08d}"
                    mask_dir.mkdir(parents=True)
                    for d,mask in zip(row['detections'],masks):
                        if d['box'] is None:continue
                        x,y,X,Y=d['box']
                        asset=mask_dir/f"{d['detection_index']:03d}.png"
                        Image.fromarray(np.asarray(mask[y:Y,x:X],dtype='uint8')*255).save(asset)
                        d.update(mask=str(asset.relative_to(args.out)),mask_sha256=sha256(asset),mask_crop_box=d['box'])
                    by_identity={d['identity']:d for d in row['detections'] if d.get('identity')}
                    for player in row['players']:
                        detection=by_identity[player['identity']]
                        player['mask']=detection.get('mask')
                        player['mask_sha256']=detection.get('mask_sha256')
                    import hashlib
                    row['image_rgb_sha256'] = hashlib.sha256(payload).hexdigest()
                    journal.write(json.dumps(row, allow_nan=False)+'\n')
                    if envelope['frame'] % 25 == 0:
                        journal.flush()
                        print('frame', envelope['frame'], flush=True)
            proc.stdout.close()
            if proc.wait() != 0:
                raise ValueError('Video decode failed')
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
    package.update(status='complete', completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   observations_sha256=sha256(args.out/'observations.jsonl'), sam_environment=sam.environment)
    write_json(args.out/'overlay.json', package)
    return package


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    uniforms=p.add_mutually_exclusive_group(required=True)
    uniforms.add_argument('--references', type=Path)
    uniforms.add_argument('--supervised-model',type=Path,help='Frozen supervised crop model; interpret four samples over 0.6 seconds per local identity')
    p.add_argument('--uniform-policy',choices=[SPACED_POLICY,LEGACY_POLICY],default=SPACED_POLICY,help='Default: offsets -15,-10,-5,0; legacy consecutive policy remains available')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--start-frame', type=int, default=11160, help='Includes 25-frame warmup before the public comparison')
    p.add_argument('--end-frame', type=int, default=14185)
    p.add_argument('--config', type=Path)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
