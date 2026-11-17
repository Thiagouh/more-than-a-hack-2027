"""Audit fixed ball filtering on three predeclared external SkillCorner matches.

No training or parameter selection. Vendor labels are used for scoring and for
constructing external fault tests, never by the filter or candidate classifier.
They do not establish a reference for dead-ball decisions.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from src.frente4.ball_quality import clean_ball_observations
from src.frente4.features import compute_features
from src.frente4.pipeline import candidates

MATCHES=('1874553','1886347','1899585')
SOURCE_COMMIT='4340d274572876239c154c90bc507a9b3250a656'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load_external(directory):
    directory=Path(directory)
    source=json.loads((directory/'source.json').read_text())
    if source['commit']!=SOURCE_COMMIT or source['match_id'] not in MATCHES:
        raise ValueError('Use the predeclared external source snapshot')
    for name,key in [('match.json','metadata_sha256'),('tracking.jsonl','tracking_sha256')]:
        if sha(directory/name)!=source[key]:raise ValueError('External source changed: '+name)
    match=json.loads((directory/'match.json').read_text())
    if str(match['id'])!=source['match_id']:raise ValueError('Match metadata differs')
    home,away=match['home_team']['id'],match['away_team']['id']
    mapping={p['id']:0 if p['team_id']==home else 1 for p in match['players']
             if p['team_id'] in (home,away)}
    slots=sorted(mapping);indices={pid:i for i,pid in enumerate(slots)}
    path=directory/'tracking.jsonl'
    with path.open('rb') as stream:
        first=json.loads(stream.readline())['frame']
        stream.seek(max(0,path.stat().st_size-65536))
        last=json.loads(stream.read().splitlines()[-1])['frame']
    n=last-first+1
    if n<=0:raise ValueError('Invalid external frame span')
    pitch=np.array([match['pitch_length'],match['pitch_width']],dtype=float)
    ball=np.full((n,2),np.nan,np.float32)
    players=np.full((n,len(slots),2),np.nan,np.float32)
    truth=np.full(n,-1,np.int8);period=np.zeros(n,np.int8)
    groups=Counter();previous=first-1
    with path.open() as stream:
        for line in stream:
            row=json.loads(line);frame=row['frame']
            if type(frame) is not int or not previous<frame<=last:
                raise ValueError('External frames must be unique and increasing')
            previous=frame;i=frame-first
            period[i]=row.get('period') or 0
            b=row.get('ball_data') or {}
            if b.get('is_detected') is True and b.get('x') is not None and b.get('y') is not None:
                ball[i]=np.array([b['x'],b['y']])+pitch/2
            for p in row.get('player_data') or []:
                j=indices.get(p['player_id'])
                if j is not None and p.get('x') is not None and p.get('y') is not None:
                    players[i,j]=np.array([p['x'],p['y']])+pitch/2
            group=(row.get('possession') or {}).get('group')
            groups[str(group)]+=1
            truth[i]={'home team':0,'away team':1}.get(group,-1)
    tr=dict(ball_xy=ball,players_xy=players,team_of_slot=np.array([mapping[p] for p in slots]),
            fps=10.,pitch=pitch,n_frames=n)
    return tr,truth,np.isin(period,[1,2]),source,dict(groups)


def metrics(prediction,truth,mask):
    known=mask&(truth>=0)
    accepted=known&(prediction>=0)
    correct=int((prediction[accepted]==truth[accepted]).sum())
    count=int(accepted.sum());total=int(known.sum())
    return dict(reference_frames=total,accepted=count,correct=correct,errors=count-correct,
                coverage=count/total if total else None,
                accepted_accuracy=correct/count if count else None)


def inject_wrong_object_faults(tr,truth,period,match_id):
    """Predeclared external stress test, never used on the competition match."""
    rng=np.random.default_rng(np.random.SeedSequence([20261117,int(match_id)]))
    fps=tr['fps'];n=tr['n_frames'];durations=(.5,1.,2.,4.)
    occupied=np.zeros(n,bool);fault=np.zeros(n,bool);windows=[]
    ball=tr['ball_xy'];players=tr['players_xy'];team=tr['team_of_slot']
    output=ball.copy();guard=round(5*fps);jump_distance=2*40/fps
    for k in range(20):
        duration=durations[k%len(durations)];length=round(duration*fps)
        if n<=length+6:break
        chosen=None
        for _ in range(5000):
            start=int(rng.integers(3,n-length-3));stop=start+length
            owner=int(truth[start])
            if (owner<0 or occupied[start:stop].any() or not period[start:stop].all()
                    or not (truth[start:stop]==owner).all()
                    or not np.isfinite(ball[start-3:start]).all()
                    or not np.isfinite(ball[stop:stop+3]).all()):continue
            eligible=[]
            for slot in np.flatnonzero(team!=owner):
                path=players[start:stop,slot]
                if (np.isfinite(path).all()
                        and np.linalg.norm(path[0]-ball[start-1])>=jump_distance
                        and np.linalg.norm(path[-1]-ball[stop])>=jump_distance):
                    eligible.append(int(slot))
            if eligible:
                chosen=(start,stop,int(rng.choice(eligible)));break
        if chosen is None:break
        start,stop,slot=chosen
        output[start:stop]=players[start:stop,slot]
        occupied[max(0,start-guard):min(n,stop+guard)]=True
        fault[start:stop]=True
        windows.append(dict(start_frame=start,end_frame_exclusive=stop,
                            duration_s=duration,opponent_slot=slot))
    return dict(tr,ball_xy=output),fault,dict(seed=20261117,requested_windows=20,
        created_windows=len(windows),durations_s=list(durations),windows=windows,
        synthetic_fault_frames=int(fault.sum()),references_used_for_test_construction=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fault-injection',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Use a new report path')
    config=json.loads(Path('configs/frente4_v1.json').read_text())
    code=[Path(__file__),Path('configs/frente4_v1.json'),*sorted(Path('src/frente4').glob('*.py'))]
    code_hashes={str(p):sha(p) for p in code}
    rows=[]
    for mid in MATCHES:
        tr,truth,period,source,groups=load_external(args.data/mid)
        pristine=candidates(compute_features(tr,config['control_radius_m'],config['near_radius_m']),config)
        fault=np.zeros(tr['n_frames'],bool);injection=None
        test_input=tr
        if args.fault_injection:test_input,fault,injection=inject_wrong_object_faults(tr,truth,period,mid)
        cleaned,_,audit=clean_ball_observations(test_input,config['out_margin_m'])
        raw_candidate=candidates(compute_features(test_input,config['control_radius_m'],config['near_radius_m']),config)
        new_candidate=candidates(compute_features(cleaned,config['control_radius_m'],config['near_radius_m']),config)
        kept=np.isfinite(cleaned['ball_xy']).all(axis=1)
        if not np.array_equal(cleaned['ball_xy'][kept],test_input['ball_xy'][kept]):
            raise ValueError('Filter changed a retained observation')
        row=dict(match_id=mid,source=source,vendor_groups=groups,quality=audit,
            pristine=metrics(pristine,truth,period),injection=injection,
            before=metrics(raw_candidate,truth,period),after=metrics(new_candidate,truth,period),
            candidates_removed=int(((raw_candidate>=0)&(new_candidate<0)).sum()),
            candidates_added=int(((raw_candidate<0)&(new_candidate>=0)).sum()),
            changed_team_on_retained_candidate=int(((raw_candidate>=0)&(new_candidate>=0)&(raw_candidate!=new_candidate)).sum()))
        if injection:
            row['synthetic_fault_observations_remaining']=int((fault&kept).sum())
            row['context_observations_removed']=int((~fault&np.isfinite(tr['ball_xy']).all(axis=1)&~kept).sum())
            row['fault_windows_by_duration']={str(duration):dict(
                windows=sum(w['duration_s']==duration for w in injection['windows']),
                frames=sum(w['end_frame_exclusive']-w['start_frame'] for w in injection['windows'] if w['duration_s']==duration),
                remaining=sum(int(kept[w['start_frame']:w['end_frame_exclusive']].sum()) for w in injection['windows'] if w['duration_s']==duration))
                for duration in injection['durations_s']}
        rows.append(row)
        print(json.dumps(row),flush=True)
    report=dict(source='SkillCorner Open Data',license='MIT',source_commit=SOURCE_COMMIT,
        protocol='docs/EXPERIMENTO-QUALIDADE-BOLA.md',matches=rows,training=False,
        fault_injection=args.fault_injection,code_sha256=code_hashes,
        limitations=['Vendor possession is not competition ground truth or a dead-ball reference.',
                     'Broadcast tracking at 10 fps; all supplied player positions, only detected ball positions.',
                     'Three predeclared matches; no parameter selection.'])
    if any(sha(p)!=digest for p,digest in code_hashes.items()):raise ValueError('Audit code changed during evaluation')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
