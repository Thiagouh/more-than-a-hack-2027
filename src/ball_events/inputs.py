"""Local-only adapters. File rows are streamed, with strict ordering and null handling."""
import csv
import json
from pathlib import Path

from .contract import InputError
from . import contract as c


def read_json(path):
    def bad_constant(_):
        raise InputError('JSON contains non-finite value')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=bad_constant)


def csv_rows(path, required):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)) or not required <= set(reader.fieldnames):
            raise InputError(f'CSV schema invalid: {Path(path).name}')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise InputError('CSV row has incorrect field count')
            try:
                for key in ('frame', 'support_start_frame', 'support_end_frame', 'available_at_frame'):
                    if key in row and row[key] != '':
                        row[key] = int(row[key])
                    elif key in row:
                        del row[key]
                for key in ('x_m', 'y_m'):
                    row[key] = None if row[key] == '' else float(row[key])
                if not row.get('quality_flags'):
                    raise InputError('quality_flags must explicitly contain a JSON list')
                row['quality_flags'] = json.loads(row['quality_flags'])
                if 'extensions' in row:
                    row['extensions'] = json.loads(row['extensions'])
            except (ValueError, TypeError):
                raise InputError('CSV has invalid number or quality_flags JSON') from None
            yield row


def standardized(path):
    path = Path(path).resolve()
    meta = c.manifest(read_json(path))
    files = meta.get('files', {})
    c.obj(files, 'files')
    if set(files) - {'players', 'ball', 'hints'} or not {'players', 'ball'} <= set(files):
        raise InputError('files must identify players, ball and optional hints')
    for value in files.values(): c.text(value, 'input path')
    paths = {k: (path.parent / v).resolve() for k, v in files.items()}
    for key in ('players', 'ball'):
        if not paths[key].is_file():
            raise InputError(f'Missing required {key} file')
    hints = []
    status = 'not_provided'
    if 'hints' in paths:
        if paths['hints'].is_file():
            hints = [c.hint(json.loads(line), meta) for line in paths['hints'].read_text(encoding='utf-8').splitlines() if line.strip()]
            status = 'provided'
        else:
            status = 'optional_file_missing'
    required = {'frame', 'x_m', 'y_m', 'source', 'producer', 'quality_flags'}

    def frames():
        balls = iter(csv_rows(paths['ball'], required))
        players = iter(csv_rows(paths['players'], required | {'track_id', 'team'}))
        p = next(players, None)
        for frame in range(meta['n_frames']):
            b = next(balls, None)
            if b is None or b.get('frame') != frame:
                raise InputError('Ball CSV must cover every frame exactly once in order')
            batch = []
            while p is not None and p.get('frame') == frame:
                batch.append(p)
                p = next(players, None)
            if p is not None and (not isinstance(p.get('frame'), int) or p['frame'] < frame):
                raise InputError('Player CSV frames must be ordered and within timeline')
            yield {'frame': frame, 'ball': b, 'players': batch}
        if next(balls, None) is not None or p is not None:
            raise InputError('CSV contains extra frames')

    fingerprints = {'input_manifest': path, **{k: p for k, p in paths.items() if p.is_file()}}
    return meta, frames, hints, fingerprints, status


def atd(data_dir=None):
    from games import GAMES, GAME
    from starter.load_atd import atd_paths, iter_aligned_points, inspect_atd_metadata
    try:
        meta_path, raw_path = atd_paths(data_dir=data_dir)
    except OSError:
        raise InputError('Assets ATD ausentes: confira --data-dir ou MTAH_DATA_DIR; veja docs/DADOS-LOCAIS.md') from None
    game = GAMES[GAME]
    meta = dict(contract_version='v1', dataset_id=GAME, timeline='tactical', fps=game['fps'],
                n_frames=game['n_video_frames'], pitch_m=list(game['pitch']),
                coordinates='metres_corner_fixed', teams={'H': 'Barcelona', 'A': 'Athletic Club'},
                producers={'atd': 'delivered-epts'}, continuity_starts=[])
    diagnostics = inspect_atd_metadata(meta_path, meta['pitch_m'], meta['fps'], meta['teams'])
    meta['extensions'] = {'atd_diagnostics': {k: v for k, v in diagnostics.items() if k != 'slots'}}
    return (meta, lambda: iter_aligned_points(meta_path, raw_path, meta['n_frames'], meta['pitch_m'], meta['fps'], meta['teams']),
            [], {'atd_meta': Path(meta_path), 'atd_raw': Path(raw_path)}, 'not_provided')
