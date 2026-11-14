"""Paired 3/5 acquisition experiments; identical tracking, loss fixed at 3.

Synthetic expectations are declared before detection. Real input is optional and
never scored against possession ground truth. Outputs stay local under outputs/.
Run: python -m scripts.compare_persistence --out outputs/persistence-3-vs-5
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time

from src.ball_events import Detector
from src.ball_events.artifacts import canonical, identity, write_json
from src.ball_events.demo import metadata, point
from src.ball_events.events import validate_events
from src.ball_events.inputs import atd, standardized
from src.team_possession import possession

HYPOTHESIS = ('Five consecutive acquisition evidence frames may suppress brief false '
              'controls but miss genuine short controls; matched confirmations should '
              'arrive two frames later. Loss stays fixed at three; no default selection.')


def stationary(n, team='H'):
    return [dict(frame=f, ball=point(f,20), players=[
        point(f,20,track_id='near',team=team),
        point(f,26,track_id='far',team='A' if team=='H' else 'H')]) for f in range(n)]


def scenarios():
    cases=[]
    def add(name, frames, expected, rationale):
        cases.append(dict(name=name,frames=frames,expected_teams=expected,rationale=rationale))
    add('controle_H_sustentado',stationary(24),['H'],'Controle verdadeiro sustentado desde o primeiro movimento verificável.')
    add('controle_A_sustentado',stationary(24,'A'),['A'],'Mesmo caso, invertendo os times.')
    for team in ('H','A'):
        frames=stationary(24,team)
        for f in frames:
            f['players'][0]['extensions']={'role':'goalkeeper','synthetic':True}
        add(f'goleiro_{team}_controle',frames,[team],
            'Goleiro com time já resolvido; mesmas regras coletivas, sem cores reais presumidas.')
    for brief in (False,True):
        frames=stationary(65 if brief else 52)
        for f in frames:
            i=f['frame']
            if i<10:
                offset=0.
            elif i<25:
                offset=.4*(i-9)
            elif not brief or i<29:
                offset=6.
            elif i<44:
                offset=6.-.4*(i-28)
            else:
                offset=0.
            f['players'][0]['x_m']=20+offset
            f['players'][1]['x_m']=26-offset
        add('roubada_curta_4_evidencias' if brief else 'roubada_sustentada',frames,
            ['H','A','H'] if brief else ['H','A'],
            'Jogadores deslocam-se continuamente a 10 m/s na aproximação/afastamento. Controle A curto tem quatro amostras de movimento compatível (frames 25–28).')
    for count in (3,4,5):
        add(f'controle_real_H_{count}_evidencias',stationary(count+1),['H'],
            f'Controle real curto com {count} amostras elegíveis após aquecimento; o teste aceita que possa ser perdido.')
    add('controle_real_A_3_evidencias',stationary(4,'A'),['A'],'Controle curto legítimo do adversário.')
    add('proximidade_sem_controle_3_evidencias',stationary(4),[],
        'Jogador próximo sem controle real por hipótese: tracking idêntico ao controle curto. Demonstra falta de observabilidade, não superioridade validada.')
    frames=stationary(20)
    for f in frames:
        x=20+.15*f['frame']; f['ball']['x_m']=x;f['players'][0]['x_m']=x-.2
    add('conducao_compativel',frames,['H'],'Bola e jogador movem-se juntos a 3,75 m/s.')
    frames=stationary(16)
    for f in frames: f['ball']['x_m']=18+.4*f['frame']
    add('passe_sem_controle',frames,[],'Bola atravessa vizinhança dos dois times a 10 m/s; jogadores parados.')
    frames=stationary(12)
    for f in frames: f['players'][1]['x_m']=20.5
    add('disputa_sem_vencedor',frames,[],'Adversário também próximo; margem insuficiente para controle.')
    frames=stationary(12)
    for f in frames:
        f['players'].append(point(f['frame'],20.1,track_id='duplicate',team='H'))
    add('duplicatas',frames,[],'Rastros sobrepostos não devem fabricar controle.')
    frames=stationary(12)
    for f in frames: f['players'][0]['track_id']=f"new-{f['frame']}"
    add('identificador_novo_todo_frame',frames,[],'Não há velocidade individual verificável.')
    frames=stationary(16)
    for f in frames:
        f['ball']['x_m']=20+min(f['frame']*.1,.4)
        f['players'].append(point(f['frame'],20.4,track_id='teammate',team='H'))
    add('continuidade_entre_companheiros',frames,['H'],'Jogadores separados por 0,4 m; bola muda de vizinhança sem interromper evidência coletiva.')
    frames=stationary(20)
    frames[8]['ball']=point(8,None,source='missing')
    add('oclusao_e_reaquisicao',frames,['H','H'],'Lacuna não gera perda, mas exige nova sequência de evidência.')
    frames=stationary(24)
    for f in frames:
        if f['frame']>=8: f['ball']['x_m']=20+.4*(f['frame']-7)
    add('afastamento_sustentado',frames,['H'],'Aquisição H e posterior perda por afastamento; não inferir time adversário apenas pela proximidade.')
    frames=stationary(20)
    frames[2]['players'][0]['available_at_frame']=15
    add('evidencia_atrasada',frames,['H'],'Aquisição precisa esperar a evidência do frame 2 até o frame 15.')
    return cases


def config(persistence):
    return dict(persistence_frames=3, acquisition_persistence_frames=persistence,
                loss_persistence_frames=3, relative_speed_mps=5.)


def digest_frames(frames):
    return hashlib.sha256(canonical(frames).encode()).hexdigest()


def run_profile(meta, frames, hints, persistence):
    detector=Detector(meta,config(persistence),hints)
    events=[]; digest=hashlib.sha256(); begin=time.perf_counter()
    for frame in frames():
        digest.update((canonical(frame)+'\n').encode())
        _,batch=detector.feed(frame);events.extend(batch)
    quality=detector.finish()
    validate_events(events,meta)
    return dict(configuration=detector.cfg, input_stream_sha256=digest.hexdigest(),
                elapsed_s=time.perf_counter()-begin,events=events,quality=quality,
                possession=possession(events,meta))


def intervals(left,right):
    rows=[]
    for f,(a,b) in enumerate(zip(left,right)):
        if a==b: continue
        if rows and rows[-1]['end_frame_exclusive']==f and (rows[-1]['label_3'],rows[-1]['label_5'])==(a,b):
            rows[-1]['end_frame_exclusive']=f+1
        else: rows.append(dict(start_frame=f,end_frame_exclusive=f+1,label_3=a,label_5=b))
    return rows


def compare(meta, frames, hints, out, expected=None):
    results={str(p):run_profile(meta,frames,hints,p) for p in (3,5)}
    if results['3']['input_stream_sha256']!=results['5']['input_stream_sha256']:
        raise ValueError('Input tracking changed between paired runs')
    for p, result in results.items():
        folder=out/p;folder.mkdir(parents=True)
        write_json(folder/'result.json',result)
        with (folder/'possession.csv').open('w',newline='') as handle:
            writer=csv.writer(handle);writer.writerow(['frame','label'])
            writer.writerows(enumerate(result['possession']['labels']))
        from evaluation.possession_metrics import read_label_csv, validate_submission
        reason=validate_submission(read_label_csv(folder/'possession.csv'), meta['n_frames'])
        if reason:
            raise ValueError(f'Consumer CSV invalid: {reason}')
    summary={}
    for p,result in results.items():
        acquisitions=[e for e in result['events'] if e['type']=='possible_control_acquisition']
        summary[p]=dict(acquisitions=[{k:e[k] for k in ('team','event_frame','confirmed_at_frame','available_at_frame','track_ids')} for e in acquisitions],
                        event_counts=result['quality']['events'], elapsed_s=result['elapsed_s'],
                        label_counts={t:result['possession']['labels'].count(t) for t in ('H','A','D','X')})
        if expected is not None:
            summary[p]['expected_teams']=expected
            summary[p]['observed_teams']=[e['team'] for e in acquisitions]
            summary[p]['expected_acquisition_sequence_met']=summary[p]['observed_teams']==expected
    differences=intervals(results['3']['possession']['labels'],results['5']['possession']['labels'])
    report=dict(fps=meta['fps'],n_frames=meta['n_frames'], profiles=summary,
                identical_input_sha256=results['3']['input_stream_sha256'],
                different_label_frames=sum(r['end_frame_exclusive']-r['start_frame'] for r in differences),
                differences=differences, no_quality_claim_without_reference=True)
    write_json(out/'comparison.json',report)
    with (out/'differences.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['start_frame','end_frame_exclusive','label_3','label_5'])
        writer.writeheader();writer.writerows(differences)
    return report


def composite(cases):
    frames=[]; starts=[]
    for case in cases:
        start=len(frames);starts.append(start)
        for envelope in copy.deepcopy(case['frames']):
            envelope['frame']+=start
            for p in [envelope['ball'],*envelope['players']]:
                p['frame']+=start
                for key in ('support_start_frame','support_end_frame','available_at_frame'):
                    if key in p: p[key]+=start
                if 'track_id' in p: p['track_id']=case['name']+':'+p['track_id']
            frames.append(envelope)
    meta=metadata(len(frames));meta['dataset_id']='synthetic-paired-composite'
    meta['continuity_starts']=starts
    return meta,frames


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--input',type=Path,help='Standardized tracking manifest')
    source.add_argument('--data-dir',type=Path,help='Root with tactical ATD XML/TXT')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    # Declare hypotheses/expectations before any detector execution.
    cases=scenarios()
    write_json(args.out/'protocol.json',dict(hypothesis=HYPOTHESIS,profiles={str(p):config(p) for p in (3,5)},
        scenarios=[{k:v for k,v in case.items() if k!='frames'} for case in cases],
        real_tracking_requested=bool(args.input or args.data_dir), fps=25,
        synthetic_quality_is_not_match_quality=True))
    reports={}
    for case in cases:
        meta=metadata(len(case['frames'])); meta['dataset_id']=case['name']
        path=args.out/'scenarios'/case['name'];path.mkdir(parents=True)
        write_json(path/'input.json',dict(manifest=meta,frames=case['frames']))
        reports[case['name']]=compare(meta,lambda case=case:iter(case['frames']),[],path,case['expected_teams'])
    meta,frames=composite(cases)
    path=args.out/'same-tracking-synthetic';path.mkdir()
    write_json(path/'input.json',dict(manifest=meta,frames=frames))
    reports['same_tracking_synthetic']=compare(meta,lambda:iter(frames),[],path)
    provenance=identity({},meta,{'profiles':{str(p):config(p) for p in (3,5)}},[])
    write_json(args.out/'provenance.json',provenance)
    if args.input or args.data_dir:
        meta,factory,hints,files,status=standardized(args.input) if args.input else atd(args.data_dir)
        before=identity(files,meta,{},hints,status)
        path=args.out/'same-tracking-real';path.mkdir()
        reports['same_tracking_real']=compare(meta,factory,hints,path)
        if before!=identity(files,meta,{},hints,status): raise ValueError('Real input/code changed during paired experiment')
        write_json(path/'provenance.json',before)
    else:
        reports['same_tracking_real']=dict(status='not_run_assets_unavailable',
            required='BAR-ATH_tactical_atd.xml and BAR-ATH_tactical_atd.txt or standardized manifest')
    write_json(args.out/'summary.json',reports)
    from scripts.persistence_report import write_report
    write_report(args.out,reports,cases)
    print(args.out/'summary.json')


if __name__=='__main__': main()
