"""Use automatic control events only on their actual supported frame interval."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.ball_events.artifacts import _cache_valid
from src.ball_events.events import validate_events

STRENGTH={'low':0,'medium':1,'high':2}


def supported_event_teams(events,n_frames,*,ball_valid=None,min_strength='medium'):
    if min_strength not in STRENGTH:raise ValueError('Unknown event strength floor')
    if ball_valid is not None:
        ball_valid=np.asarray(ball_valid)
        if ball_valid.shape!=(n_frames,) or ball_valid.dtype.kind!='b':
            raise ValueError('Explicit per-frame ball validity required')
    teams=np.full(n_frames,-1,np.int8);conflict=np.zeros(n_frames,bool)
    audit=dict(policy='confirmed-control-support-only-v1',accepted_events=0,
        rejected_quality_flags=0,rejected_strength=0,rejected_ball_support=0,
        unsupported_hold_frames=0)
    for event in events:
        if event['type']!='possible_control_acquisition':continue
        start,confirmed=event['event_frame'],event['confirmed_at_frame']
        left,right=event['support_start_frame'],event['support_end_frame']
        if not (type(start) is int and type(confirmed) is int and type(left) is int and type(right) is int
                and 0<=left<=start<=confirmed<right<=n_frames and event['team'] in ('H','A')):
            raise ValueError('Control support must be bounded by actual source frames')
        if event.get('quality_flags'):
            audit['rejected_quality_flags']+=1;continue
        if STRENGTH.get(event['strength'],-1)<STRENGTH[min_strength]:
            audit['rejected_strength']+=1;continue
        if ball_valid is not None and not ball_valid[left:right].all():
            audit['rejected_ball_support']+=1;continue
        code=0 if event['team']=='H' else 1
        indices=np.arange(start,confirmed+1)
        clashes=conflict[indices]|((teams[indices]>=0)&(teams[indices]!=code))
        conflict[indices[clashes]]=True
        teams[indices[~clashes]]=code;teams[indices[clashes]]=-1
        audit['accepted_events']+=1
    audit.update(conflicting_frames=int(conflict.sum()),supported_frames=int((teams>=0).sum()))
    return teams,audit


def load_automatic_bundle(folder,expected_meta,expected_inputs,expected_configuration):
    """Bind a Frente 5 run to raw inputs and the unchanged detector kernel."""
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if not _cache_valid(folder,manifest.get('cache_key')):
        raise ValueError('Complete hash-verified automatic event bundle required')
    provenance=manifest['provenance'];meta=provenance['input_metadata']
    if provenance.get('identity_version')!='ball-events-run-v2' or provenance.get('contract_version')!='v2':
        raise ValueError('Automatic event provenance version differs')
    for key in ('dataset_id','timeline','n_frames','fps','coordinates','pitch_m','teams'):
        if meta[key]!=expected_meta[key]:raise ValueError('Event timeline/source metadata differs')
    if provenance['input_sha256']!=expected_inputs or provenance['configuration']!=expected_configuration:
        raise ValueError('Event input hashes or frozen detector configuration differ')
    if provenance.get('hints'):
        raise ValueError('This integration accepts raw automatic tracking events without extra hints')
    root=Path(__file__).resolve().parents[2]
    kernel=[*sorted((root/'src/ball_events').glob('*.py')),root/'starter/load_atd.py',root/'games.py']
    for path in kernel:
        name=str(path.relative_to(root))
        with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if provenance['code_sha256'].get(name)!=digest:
            raise ValueError('Event detector kernel differs: '+name)
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines() if line.strip()]
    validate_events(events,meta)
    return events,dict(source_commit=provenance['commit'],cache_key=manifest['cache_key'],
        event_sha256=manifest['artifact_sha256']['events.jsonl'],source='raw ATD automatic detector',
        event_count=len(events),training=False,human_corrections=False)
