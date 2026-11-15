"""Freeze v4 predictions on reserved real windows without reading possession GT."""
import argparse
import json
from itertools import islice
from pathlib import Path
from collections import Counter
from src.ball_events.inputs import atd
from src.ball_events.artifacts import identity,write_json,sha
from src.team_possession.collective import CollectiveModel,configuration
from src.team_possession.restarts import load_kickoffs
from src.team_possession.recovery import load_evidence_package


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--evidence',type=Path);p.add_argument('--video',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text());windows=selection['reserved']+selection.get('boundary_challenges',[])
    meta,factory,hints,files,status=atd('data');end=max(w[1] for w in windows);meta['n_frames']=end
    files['selection']=a.selection
    game={}
    if a.evidence:
        game=load_evidence_package(a.evidence,meta,a.video,'game_evidence');files.update(evidence=a.evidence,video=a.video)
    a.out.mkdir(parents=True,exist_ok=False);kickoffs,path=load_kickoffs('data',meta)
    if path:files['kickoffs']=path
    cfg=configuration();detector_cfg=dict(persistence_frames=3,acquisition_persistence_frames=3,loss_persistence_frames=3,relative_speed_mps=5.)
    before=identity(files,meta,{'detector':detector_cfg,'collective':cfg},hints,status)
    write_json(a.out/'protocol.json',dict(selection=selection,provenance=before,ground_truth_opened=False,
        review_status='Reserved first v4 execution; manual outcome review follows frozen predictions.'))
    model=CollectiveModel(meta,detector_cfg,cfg,kickoffs=kickoffs);states=[];events=[]
    for row in islice(factory(),end):
        snap,_,batch,_=model.feed(row,game_evidence=game.get(row['frame'],[]))
        if any(lo<=row['frame']<hi for lo,hi in windows):states.append(snap);events.extend(batch)
    write_json(a.out/'predictions.json',dict(states=states,events=events))
    if identity(files,meta,{'detector':detector_cfg,'collective':cfg},hints,status)!=before:raise ValueError('Inputs/code changed during reserved execution')
    summary=[]
    for lo,hi in windows:
        rows=[r for r in states if lo<=r['frame']<hi]
        summary.append(dict(start_frame=lo,end_frame_exclusive=hi,frames=len(rows),
          labels=dict(Counter(r['label'] for r in rows)),reasons=dict(Counter(r['reason'] for r in rows)),
          observed_opponent_control_frames=sum(r['control_observed_team'] in ('H','A') and r['origin_team']!=r['control_observed_team'] for r in rows),
          rescued_against_observed_opponent_control=sum(r['reason'] in ('observed_receiver_continuity','observed_sender_outgoing_launch')
            and r['control_observed_team'] not in (None,r['label']) for r in rows),
          label_transitions=sum(a['label']!=b['label'] for a,b in zip(rows,rows[1:]))))
    write_json(a.out/'summary.json',dict(windows=summary,predictions_sha256=sha(a.out/'predictions.json'),
        possession_accuracy=None,reason='No independent possession truth supplied outside public window; no invented accuracy.',
        quality=model.finish()))


if __name__=='__main__':main()
