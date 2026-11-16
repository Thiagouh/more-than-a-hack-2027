"""Prepare exact-source review images; never approve annotations automatically."""
import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scripts.prepare_tactical_visual import extract_exact, homographies, project_observed
from src.ball_events.inputs import atd
from src.team_identification.manifests import write_json
from src.team_identification.pilot import sha256
from src.team_possession.render_observations import render_status


def prepare(data_dir, out):
    video = data_dir/'tactical/BAR-ATH_tactical.mp4'
    meta, factory, _, files, _ = atd(data_dir)
    pregame = list(range(0, 11250, 750))
    development = [11185, 11186, 11187, 11188, 11189, 11210, 11435, 12400, 14184]
    # Separate sequences, beyond the development window and a 30 s guard band.
    evaluation = [16000, 16001, 16002, 16003, 16004, 16250, 16251, 16252, 16253, 16254,
                  17000, 17001, 17002, 17003, 17004]
    frames = sorted(set(pregame+development+evaluation))
    out.mkdir(parents=True, exist_ok=False)
    entries = extract_exact(video, frames, out/'frames', expected_frames=meta['n_frames'], expected_fps=meta['fps'])
    hs = homographies(data_dir/'tactical/BAR-ATH_tactical_homography.csv', frames)
    selected = {e['frame']: e for e in entries}
    audit, annotations = [], []
    for envelope in islice(factory(), max(frames)+1):
        f = envelope['frame']
        if f not in selected:
            continue
        image = Image.open(out/'frames'/selected[f]['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        points = []
        for p in envelope['players']:
            status = render_status(p, f, meta['pitch_m'])
            xy = project_observed(p, hs[f]['matrix'], meta['pitch_m'], f) if hs[f]['valid'] else None
            points.append(dict(p, render_status=status, projected_pixel=xy))
            if xy and 0 <= xy[0] < image.width and 0 <= xy[1] < image.height:
                x, y = xy
                color = '#2676ff' if p['team'] == 'H' else '#ff7032'
                draw.ellipse((x-4, y-4, x+4, y+4), fill=color, outline='white')
                draw.text((x+5, y), p['team']+p['track_id'], fill='white', stroke_width=2, stroke_fill='black')
        image.save(out/'frames'/f'{f:08d}-tracking.png')
        counts = Counter(p['team'] for p in points if p['render_status'].startswith('drawn'))
        audit.append(dict(frame=f, observed_by_metadata_team=dict(counts), homography_valid=hs[f]['valid'], points=points))
        split = 'development' if f in development else 'evaluation' if f in evaluation else 'reference_candidates'
        annotations.append(dict(frame=f, image='frames/'+selected[f]['image'], image_sha256=selected[f]['image_sha256'],
            split=split, group_id=f'{split}-{f//250}', reviewed=False, reviewer=None,
            annotations_complete=False, people=[], notes='Annotate every visible athlete and referee; do not copy ATD team labels.'))
    for name, fs in [('pregame', pregame), ('development', development), ('evaluation', evaluation)]:
        sheet = Image.new('RGB', (3*640, ((len(fs)+2)//3)*385), '#20252a')
        draw = ImageDraw.Draw(sheet)
        for i, f in enumerate(fs):
            image = Image.open(out/'frames'/f'{f:08d}.png').convert('RGB')
            image.thumbnail((640, 360))
            x, y = (i%3)*640, (i//3)*385
            sheet.paste(image, (x, y+25))
            draw.text((x+10, y+5), f'frame {f} | recording {f/meta["fps"]:.2f}s | {name}', fill='white')
        sheet.save(out/f'{name}-contact-sheet.jpg', quality=92)
    write_json(out/'tracking-audit.json', audit)
    write_json(out/'annotations-draft.json', dict(schema='team-identity-review-v1', status='pending_human_review',
        timeline='tactical', fps=meta['fps'], source_video_sha256=entries[0]['source_video_sha256'],
        atd_raw_sha256=sha256(files['atd_raw']), atd_meta_sha256=sha256(files['atd_meta']),
        homography_sha256=sha256(data_dir/'tactical/BAR-ATH_tactical_homography.csv'), frames=annotations))
    write_json(out/'review-status.json', dict(approved=False, references_frozen=False, evaluation_complete=False,
        reason='Review required: uniform categories, all visible people, feet and identity correspondences.',
        development_window=[11185, 14185], evaluation_sequences=[[16000,16005],[16250,16255],[17000,17005]],
        reference_candidates='pregame-contact-sheet.jpg', original_tracking_preserved=True))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(prepare(a.data_dir, a.out))


if __name__ == '__main__':
    main()
