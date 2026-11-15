"""Explicit kickoff evidence, separate from detector hints and possession labels."""
from ..ball_events import contract as c


def validate_kickoffs(records, meta):
    result = []
    for raw in records:
        required = {'half', 'frame', 'team', 'available_at_frame', 'source_id'}
        c.fields(raw, required, required, 'kickoff')
        row = dict(raw)
        if row['half'] not in (1, 2) or isinstance(row['half'], bool):
            raise c.InputError('kickoff half must be 1 or 2')
        c.integer(row['frame'], 'kickoff frame')
        c.integer(row['available_at_frame'], 'kickoff availability')
        c.text(row['source_id'], 'kickoff source')
        if row['frame'] >= meta['n_frames'] or row['team'] not in (None, 'H', 'A'):
            raise c.InputError('Invalid kickoff frame or team')
        result.append(row)
    if len({r['half'] for r in result}) != len(result) or len({r['frame'] for r in result}) != len(result):
        raise c.InputError('Duplicate kickoff half or frame')
    return sorted(result, key=lambda r: r['frame'])


def load_kickoffs(data_dir, meta):
    """Use only start-of-half tags containing the published exact kickoff anchor.

    Source availability is the kickoff frame: tags are an explicit offline input.
    Missing files produce no invented team. Conflicting tags are rejected.
    """
    from pathlib import Path
    from games import asset_paths
    from starter.load_smart_tagging import load, KICKOFF_VIDEO_FRAMES
    path = Path(asset_paths(data_dir=data_dir)['smart_tagging'])
    if not path.is_file():
        return [], None
    tags = load(data_dir=data_dir)
    records = []
    for half, name in ((1, '1st Half'), (2, '2nd Half')):
        frame = KICKOFF_VIDEO_FRAMES[name]
        if frame >= meta['n_frames']:
            continue
        hits = [e for e in tags if (e['code'] or '').strip().lower() == 'set pieces'
                and e['tags'].get('Type', '').strip().lower() == 'kick off (start)'
                and e['tags'].get('Half') == name and e['start_frame'] <= frame < e['end_frame']]
        teams = {t for e in hits for t, label in meta['teams'].items()
                 if e['tags'].get('Team', '').strip().casefold() == label.casefold()}
        if len(hits) == 1 and len(teams) <= 1:
            records.append(dict(half=half, frame=frame, team=next(iter(teams)) if teams else None, available_at_frame=frame,
                                source_id=f'smart_tagging:kickoff:{half}'))
    return validate_kickoffs(records, meta), path


def kickoff_setup(envelope, meta, radius_m=2., separation_m=3.):
    """Infer restarting team from an unambiguous center setup at a known kickoff.

    Does not invent a ball point. This is a formation hypothesis rather than
    observed individual control; only called at an explicit start-of-half anchor.
    """
    import math
    f=envelope['frame'];cx,cy=(v/2 for v in meta['pitch_m'])
    eligible=[p for p in envelope['players'] if p['source']=='observed'
              and not p.get('quality_flags') and p.get('available_at_frame',f)<=f]
    distances={t:min((math.hypot(p['x_m']-cx,p['y_m']-cy) for p in eligible if p['team']==t),default=None) for t in ('H','A')}
    candidates=[t for t in ('H','A') if distances[t] is not None and distances[t]<=radius_m]
    if len(candidates)!=1:
        return dict(team=None,reason='kickoff_center_ambiguous_or_missing',distances=distances)
    team=candidates[0];other='A' if team=='H' else 'H'
    if distances[other] is None or distances[other]-distances[team]<separation_m:
        return dict(team=None,reason='kickoff_center_opponent_not_separated',distances=distances)
    return dict(team=team,reason='kickoff_center_setup_inferred',distances=distances)
