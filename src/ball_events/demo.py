"""Small synthetic fixtures: no match data, no manually annotated match labels."""
import copy
import csv
from pathlib import Path

from .artifacts import canonical, write_json


def metadata(n_frames=30):
    return dict(contract_version='v1', dataset_id='synthetic-v1', timeline='synthetic',
                fps=25, n_frames=n_frames, pitch_m=[105.3, 68.0], coordinates='metres_corner_fixed',
                teams={'H': 'Synthetic Home', 'A': 'Synthetic Away'}, producers={'synthetic': '1'},
                continuity_starts=[])


def point(frame, x, y=20.0, source='observed', **extra):
    return dict(frame=frame, x_m=x, y_m=y if x is not None else None, source=source,
                producer='synthetic', quality_flags=[], **extra)


def frames(n_frames=30):
    for f in range(n_frames):
        # Stable H candidate, pass, one-frame deflection, contest, persistent A candidate,
        # then occlusion/estimated observation. All constructed by code.
        x = 20.0 if f < 8 else 20 + (f-7)*0.4 if f < 13 else 22-(f-13)*0.3 if f < 17 else 20.8
        b = point(f, x)
        home = point(f, 20.0, track_id='h1', team='H')
        away = point(f, 26.0 if f < 14 else 21.7, track_id='a1', team='A')
        if f >= 18:
            home['x_m'], away['x_m'] = 18.0, 21.0
        if 24 <= f < 27:
            b = point(f, None, source='missing')
        elif f >= 27:
            b = point(f, x, source='estimated')
        yield dict(frame=f, ball=b, players=[home, away])


def write_inputs(out, scenario='original'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if scenario not in ('original','causal'): raise ValueError('Unknown synthetic scenario')
    meta = metadata(32 if scenario=='causal' else 30)
    meta['dataset_id'] = 'synthetic-causal-v2' if scenario=='causal' else meta['dataset_id']
    meta['files'] = {'players': 'players.csv', 'ball': 'ball.csv', 'hints': 'hints.jsonl'}
    common = ['frame', 'x_m', 'y_m', 'source', 'producer', 'quality_flags', 'available_at_frame']
    with (out / 'ball.csv').open('w', newline='') as bh, (out / 'players.csv').open('w', newline='') as ph:
        bw = csv.DictWriter(bh, fieldnames=common, lineterminator='\n')
        pw = csv.DictWriter(ph, fieldnames=common+['track_id', 'team'], lineterminator='\n')
        bw.writeheader()
        pw.writeheader()
        for frame in (causal_frames() if scenario=='causal' else frames()):
            for row, writer in [(frame['ball'], bw), *((p, pw) for p in frame['players'])]:
                row = copy.deepcopy(row)
                row['quality_flags'] = canonical(row['quality_flags'])
                writer.writerow(row)
    hint = dict(hint_id='synthetic-restart', type='restart_hint', start_frame=0, end_frame=4,
                available_at_frame=2, team='H', strength='low', producer='synthetic')
    (out / 'hints.jsonl').write_text(canonical(hint)+'\n', encoding='utf-8')
    write_json(out / 'manifest.json', meta)
    return out / 'manifest.json'


def causal_frames():
    """Delay acquisition past the end, then separation, dispute, estimates and missing ball."""
    xs=[20.]*8+[20.4,20.8,21.2,21.6,22.,22.4,22.8,23.2]+[26.]*6+[23.]*10
    for f,x in enumerate(xs):
        home=point(f,20. if f<22 else 22.,track_id='h1',team='H')
        away=point(f,26. if f<22 else 24.,track_id='a1',team='A')
        if f==1: away['available_at_frame']=100
        ball=point(f,x)
        if 27<=f<30: ball=point(f,x,source='estimated')
        if f>=30: ball=point(f,None,source='missing')
        yield dict(frame=f,ball=ball,players=[home,away])
