"""Render local tracking animation with GT/3/5 labels, not camera footage.

Optional dependency: Pillow; requires ffmpeg. Real frames and derivatives stay
under outputs/. Uses the already-frozen experiment window and predictions.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess

from evaluation import possession_metrics as pm
from src.ball_events.artifacts import sha
from starter.load_atd import iter_aligned_points
from src.team_possession.render_observations import render_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--ground-truth',type=Path,required=True)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--profiles',nargs='+',type=int)
    selection.add_argument('--models',nargs='+')
    args=parser.parse_args()
    from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
    protocol=json.loads((args.run/'protocol.json').read_text())
    start,end=protocol['start_frame'],protocol['end_frame_exclusive']
    source_provenance=protocol.get('detector_input_provenance',protocol.get('provenance'))
    meta=dict(source_provenance['input_metadata'])
    meta['n_frames']=protocol.get('original_n_frames',meta['n_frames'])
    profiles=args.models or [str(p) for p in (args.profiles or [3,5])]
    if not 1<=len(profiles)<=3 or len(set(profiles))!=len(profiles) or any(not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,40}',p) for p in profiles):
        raise ValueError('Choose one to three distinct model keys')
    xml=args.data_dir/'tactical/BAR-ATH_tactical_atd.xml'
    raw=args.data_dir/'tactical/BAR-ATH_tactical_atd.txt'
    expected=source_provenance['input_sha256']
    if sha(raw)!=expected['atd_raw'] or sha(xml)!=expected['atd_meta']:
        raise ValueError('Tracking differs from evaluated source')
    summary=json.loads((args.run/'summary.json').read_text())
    reference_hash=summary.get('ground_truth_sha256')
    if reference_hash is None:
        reference_hash=json.loads((args.run/'reference-sha256.json').read_text())['sha256']
    if sha(args.ground_truth)!=reference_hash:raise ValueError('Reference differs from evaluated source')
    gt=[pm.LABELS[int(v)] for v in pm.label_codes(pm.read_label_csv(args.ground_truth))]
    predictions={};state_rows={};model_configs={}
    for p in profiles:
        legacy=args.run/f'consumer-{p}.json'
        payload=json.loads(legacy.read_text()) if legacy.exists() else json.loads((args.run/p/'result.json').read_text())
        prediction=payload if legacy.exists() else payload['possession']
        predictions[p]=prediction['labels']
        state_rows[p]={s['frame']:s for s in prediction.get('states',[])}
        model_configs[p]=payload.get('configuration',{})
    def font(size):
        path=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
        return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default(size=size)
    title,body,big=font(24),font(17),font(27)
    colors={'H':'#2563eb','A':'#e85d32','D':'#64748b','X':'#a87e00'}
    names={'H':'Barcelona','A':'Athletic Club','D':'Bola morta','X':'Desconhecido'}
    telemetry_model=next((p for p in reversed(profiles) if state_rows[p]),None)
    width,height=960,1100 if telemetry_model else 820
    base=Image.new('RGB',(width,height),'#f3f6f4');draw=ImageDraw.Draw(base)
    heading='Tracking + gabarito publico | '+ ' / '.join(protocol.get('model_names',{}).get(p,p) for p in profiles)
    heading_size=24
    while draw.textlength(heading,font=font(heading_size))>width-48 and heading_size>14:heading_size-=1
    draw.text((24,16),heading,font=font(heading_size),fill='#18333e')
    status = protocol.get('integration_status')
    if status and status.get('team_identity_applied'):
        subtitle='Equipes visuais | identidades locais | cinza: time incerto | mesmas observacoes usadas na posse'
    elif status and status.get('sam3_executed'):
        mode='bola nas maos revisada' if status.get('visual_control_evidence_modes')==['reviewed_held_ball'] else 'controle experimental'
        subtitle=(f"SAM: {status.get('visual_observation_frames',0)} controles / {status.get('recovery_frames',0)} recuperacoes | revisao offline | base ATD"
                  if status.get('recovery_frames') else f"Piloto SAM 3: {status.get('visual_observation_frames',0)} frames | {mode} | demais equipes: ATD")
    elif status and status.get('goalkeeper_context_applied') is False:
        subtitle='Equipes do ATD | SAM 3 e regra visual dos goleiros: nao aplicados nesta execucao.'
    else:
        subtitle='Animacao gerada do ATD. Nao e filmagem nem o video demonstrativo original.'
    draw.text((24,49),subtitle,font=body,fill='#52636a')
    draw.text((24,170),f'Bola ausente: ate {protocol.get("missing_ball_grace_frames",5)} frames | Perda: {protocol.get("loss_frames",3)} frames.',font=body,fill='#18333e')
    left,top,pw,ph=70,218,820,530
    draw.rectangle((left,top,left+pw,top+ph),fill='#196346',outline='white',width=2)
    draw.line((left+pw/2,top,left+pw/2,top+ph),fill='white',width=2)
    radius=9.15/meta['pitch_m'][0]*pw
    draw.ellipse((left+pw/2-radius,top+ph/2-radius,left+pw/2+radius,top+ph/2+radius),outline='white',width=2)
    for x in (left,left+pw-16.5/meta['pitch_m'][0]*pw):
        draw.rectangle((x,top+ph/2-20.16/meta['pitch_m'][1]*ph,x+16.5/meta['pitch_m'][0]*pw,top+ph/2+20.16/meta['pitch_m'][1]*ph),outline='white',width=2)
    draw.text((24,785),'H azul | A laranja | cores dos pontos representam times, nao uniformes reais.',font=body,fill='#52636a')
    output=args.run/('tracking-gabarito-'+'-vs-'.join(profiles)+'.mp4')
    if output.exists():raise ValueError('Video output exists')
    command=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
        '-framerate',str(meta['fps']),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with (args.run/'video-render.log').open('w') as log:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        rendered=0
        try:
            for envelope in iter_aligned_points(xml,raw,meta['n_frames'],meta['pitch_m'],meta['fps'],meta['teams']):
                f=envelope['frame']
                if f<start:continue
                if f>=end:break
                image=base.copy();draw=ImageDraw.Draw(image)
                def card_title(p):
                    name=protocol.get('model_names',{}).get(p)
                    acquisition=model_configs[p].get('acquisition_persistence_frames',p)
                    return f'{name} ({acquisition} f)' if name else f'Aquisicao: {acquisition} '+('frame' if str(acquisition)=='1' else 'frames')
                cards=[(gt[f],'Gabarito oficial')]+[(predictions[p][f],card_title(p)) for p in profiles]
                card_step=912/len(cards)
                for index,(label,header) in enumerate(cards):
                    x=24+index*card_step
                    draw.rounded_rectangle((x,85,x+card_step-12,155),radius=8,fill=colors[label])
                    draw.text((x+12,91),header,font=body,fill='white')
                    draw.text((x+12,117),names[label],font=big,fill='white')
                draw.text((24,755),f'Trecho: {(f-start)/meta["fps"]:06.2f} s | frame tactical: {f} | tempo do video: {f/meta["fps"]:.2f} s',font=body,fill='#18333e')
                telemetry=state_rows[telemetry_model].get(f) if telemetry_model else None
                def project(position):
                    return (left+position[0]/meta['pitch_m'][0]*pw,top+position[1]/meta['pitch_m'][1]*ph)
                if telemetry and telemetry.get('metric_frame')==f:
                    forecast=telemetry.get('forecast_xy',[])
                    if len(forecast)>1:
                        points=[project(position) for position in forecast]
                        overlay=ImageDraw.Draw(image,'RGBA')
                        overlay.line(points,fill=(220,240,150,70),width=max(3,int(3/meta['pitch_m'][0]*pw)))
                        draw=ImageDraw.Draw(image)
                        draw.line(points,fill='#dcf096',width=2)
                review=(telemetry or {}).get('tracking_review',[])
                rejected={r['track_id'] for r in review if r['action']=='reject_observation'}
                assigned={r['track_id']:r for r in review if r['action']=='goalkeeper_assignment'}
                roles={r['track_id']:r for r in review if r['action']=='goalkeeper_role'}
                corrections={r['target_track_id']:r for r in (telemetry or {}).get('recovered_observations',[])
                             if r.get('status')=='accepted_team_correction'}
                identity_players=(telemetry or {}).get('team_identity_players')
                if identity_players is not None:
                    from src.team_identification.render import draw_identities
                    draw_identities(draw,identity_players,f,meta['pitch_m'],project,body)
                for raw_point in ([] if identity_players is not None else envelope['players']):
                    p=dict(raw_point)
                    if p['track_id'] in corrections:p['team']=corrections[p['track_id']]['team']
                    if render_status(p,f,meta['pitch_m']) not in ('drawn','drawn_with_quality_warning'):continue
                    x=left+p['x_m']/meta['pitch_m'][0]*pw;y=top+p['y_m']/meta['pitch_m'][1]*ph
                    if left-8<=x<=left+pw+8 and top-8<=y<=top+ph+8:
                        if p['track_id'] in assigned:
                            draw.ellipse((x-8,y-8,x+8,y+8),fill=colors['H'],outline='#9ef1bc',width=2)
                            draw.text((x+10,y-12),'G-H',font=body,fill='white')
                            continue
                        if p['track_id'] in rejected:
                            draw.line((x-5,y-5,x+5,y+5),fill='#bfc7c9',width=2)
                            draw.line((x-5,y+5,x+5,y-5),fill='#bfc7c9',width=2)
                            continue
                        if p['track_id'] in roles:
                            draw.ellipse((x-8,y-8,x+8,y+8),fill=colors[p['team']],outline='#9ef1bc',width=2)
                            draw.text((x+10 if x<left+pw/2 else x-45,y-12),'G-'+p['team'],font=body,fill='white')
                            continue
                        draw.ellipse((x-6,y-6,x+6,y+6),fill=colors[p['team']],outline='#ff00ff' if p['track_id'] in corrections else 'white',width=2 if p['track_id'] in corrections else 1)
                if telemetry and identity_players is None:
                    for recovered in telemetry.get('recovered_observations',[]):
                        if recovered.get('status')!='accepted':continue
                        x,y=project(recovered['pitch_xy'])
                        draw.rectangle((x-7,y-7,x+7,y+7),fill=colors[recovered['team']],outline='#ff00ff',width=2)
                b=envelope['ball']
                if render_status(b,f,meta['pitch_m']) in ('drawn','drawn_with_quality_warning'):
                    x=left+b['x_m']/meta['pitch_m'][0]*pw;y=top+b['y_m']/meta['pitch_m'][1]*ph
                    if left-8<=x<=left+pw+8 and top-8<=y<=top+ph+8:draw.ellipse((x-5,y-5,x+5,y+5),fill='white',outline='black',width=2)
                else:draw.text((left+12,top+12),'Bola ausente do tracking',font=body,fill='white')
                if telemetry:
                    game=telemetry.get('game_state',{})
                    recovery_count=sum(r.get('status')=='accepted' for r in telemetry.get('recovered_observations',[]))
                    annotation='Bola morta | evidencia visual revisada' if game.get('dead') else 'Recuperacoes visuais revisadas: '+str(recovery_count)
                    draw.text((left+12,top+ph-24),annotation,font=body,fill='white')
                    metrics=telemetry['metrics']
                    if telemetry.get('metric_frame')==f:
                        shown=set()
                        for candidate in telemetry.get('candidate_players',[]):
                            if candidate['team'] in shown:continue
                            x,y=project(candidate['position_xy'])
                            if not left<=x<=left+pw or not top<=y<=top+ph:continue
                            shown.add(candidate['team'])
                            draw.ellipse((x-11,y-11,x+11,y+11),outline='#dcf096',width=2)
                            label=f"{candidate['meeting_time_s']:.2f}s"
                            label_width=draw.textlength(label,font=body)
                            label_x=min(x+12,left+pw-label_width-4)
                            label_y=max(top+4,min(top+ph-22,y-28 if candidate['team']=='H' else y+12))
                            draw.text((label_x,label_y),label,font=body,fill='white')
                    draw.rounded_rectangle((24,815,936,1080),radius=10,fill='white')
                    def fmt(value,unit='',signed=False):
                        return 'indisponivel' if value is None else (f'{value:+.2f}' if signed else f'{value:.2f}')+unit
                    origin=names.get(telemetry.get('origin_team'),'indisponivel')
                    ctrl=names.get(telemetry.get('control_observed_team'),'nenhuma')
                    visual=telemetry.get('visual_control',{})
                    if ctrl=='nenhuma' and visual.get('accepted'):
                        ctrl=names.get(visual.get('team'),'incerto')+' (visual)'
                    draw.text((40,830),f"Estado: {telemetry['state'].upper()} | Origem: {origin} | Evid. controle: {ctrl}",font=body,fill='#18333e')
                    draw.text((40,865),'Encontro origem: '+fmt(metrics.get('origin_team_meeting_time_s'),' s')+' | adversario: '+fmt(metrics.get('opponent_meeting_time_s'),' s'),font=body,fill='#18333e')
                    draw.text((40,900),'Vantagem: '+fmt(metrics.get('meeting_time_advantage_s'),' s',True)+' | Erro trajetoria: '+fmt(metrics.get('trajectory_error_m'),' m'),font=body,fill='#18333e')
                    coverage=metrics.get('observed_players_by_team') or {}
                    draw.text((40,935),f"Rastros observados H/A: {coverage.get('H','-')}/{coverage.get('A','-')} | Bola indisponivel: {telemetry['ball_unavailable_run_frames']} frames",font=body,fill='#18333e')
                    explanations={
                        'causal_context_continuity':'Historico causal de 25 frames sustenta a equipe anterior',
                        'observed_receiver_continuity':'Recebedor observado se aproxima; hipotese de passe limitada',
                        'observed_sender_outgoing_launch':'Bola saindo do passador observado; aguarda recebedor',
                        'verified_restart':'Reinicio revisado; aguarda controle sustentado',
                        'verified_restart_tracking_confirmed':'Reinicio revisado e controle atual do mesmo time',
                        'dead_ball_evidence_expired':'Evidencia de interrupcao expirou; estado desconhecido',
                        'no_confirmed_origin':'Sem controle anterior confirmado',
                        'short_ball_gap_hold':'Ausencia curta: hipotese mantida',
                        'ball_unavailable':'Bola sem informacao suficiente',
                        'metrics_not_currently_available':'Evidencia ainda nao disponivel',
                        'sustained_probable_reception':'Recepcao provavel sustentada: recuperacao antecipada',
                        'kickoff_tracking_confirmed':'Posse inicial: kickoff conferido no tracking',
                        'kickoff_center_setup_inferred':'Kickoff: equipe inferida pela formacao no centro',
                        'kickoff_opening_prior':'Hipotese de abertura: aguarda controle; duracao limitada',
                        'visual_reviewed_goalkeeper_control':'Goleiro: SAM 3 + contexto + bola nas maos revisada',
                        'visual_control_corroborated':'Controle visual sustentado por observacoes SAM 3',
                        'current_control_evidence':'Controle do time sustentado pela observacao',
                        'origin_support_expired':'Origem antiga: aguarda novo controle',
                        'insufficient_team_motion_coverage':'Poucos rastros com movimento confiavel',
                        'trajectory_not_supported':'Trajetoria mudou ou ainda nao e confiavel',
                        'no_control_or_pass_motion':'Sem controle ou movimento de passe sustentado',
                        'no_supported_receiver':'Nenhum recebedor viavel no horizonte',
                        'plausible_same_team_pass':'Passe plausivel entre companheiros',
                        'opponent_interception_opportunity':'Adversario pode chegar primeiro; controle nao confirmado',
                        'ball_quality_or_pitch_boundary':'Bola fora do campo ou posicao incerta',
                    }
                    reason_text='Interrupcao verificada por revisao visual offline' if game.get('dead') else explanations.get(telemetry['reason'],'Evidencia insuficiente')
                    draw.text((40,970),'Motivo: '+reason_text,font=body,fill='#18333e')
                    context=telemetry.get('temporal_context',{})
                    if review:
                        legend=f"Contexto: {context.get('window_frames',0)} f | G-H/G-A: goleiros | cruz cinza: ATD rejeitado"
                    else:legend='Amarelo: previsao / candidatos | Magenta: observacao visual revisada.'
                    draw.text((40,1005),legend,font=body,fill='#52636a')
                    draw.text((40,1040),'Parametros experimentais | altura indisponivel | '+('D exige evidencia verificada' if 'game_state' in telemetry else 'sem modelo de bola morta'),font=body,fill='#52636a')
                if rendered==0:image.save(args.run/'preview.png')
                proc.stdin.write(image.tobytes());rendered+=1
        finally:
            proc.stdin.close();code=proc.wait()
    if code or rendered!=end-start:raise RuntimeError('Incomplete video render; inspect log')
    (args.run/'video-provenance.json').write_text(json.dumps(dict(script_sha256=sha(__file__),pillow=pillow_version,
        start_frame=start,end_frame_exclusive=end,frames=rendered,fps=meta['fps'],
        video_sha256=sha(output),camera_footage=False,synthetic_tracking=False),indent=2)+'\n')
    print(output)


if __name__=='__main__':main()
