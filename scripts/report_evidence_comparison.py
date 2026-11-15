"""Post-prediction v4 audit: reference labels are evaluation-only inputs."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from evaluation import possession_metrics as pm
from src.ball_events.artifacts import sha,write_json


def read(path):return json.loads(Path(path).read_text())


def gaps(rows):
    result=[];i=1
    while i<len(rows):
        if rows[i]['label']=='X' and rows[i-1]['label'] in ('H','A'):
            j=i
            while j<len(rows) and rows[j]['label']=='X':j+=1
            if j<len(rows) and rows[j]['label']==rows[i-1]['label'] and j-i<=25:
                result.append(dict(start=rows[i]['frame'],end=rows[j-1]['frame'],frames=j-i,team=rows[j]['label']))
            i=j
        else:i+=1
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--reserved',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    protocol=read(a.run/'protocol.json');start,end=protocol['start_frame'],protocol['end_frame_exclusive']
    old=read(a.run/'anterior/result.json');new=read(a.run/'coletivo/result.json')
    rows=new['possession']['states'];old_rows=old['possession']['states'];old_by={r['frame']:r for r in old_rows}
    # Load public labels only after both frozen results have been loaded and hashed.
    result_hashes={name:sha(a.run/name/'result.json') for name in ('anterior','coletivo')}
    gt_path=Path('evaluation/gt/BAR-ATH_possession_gt_public.csv')
    if sha(gt_path)!=read(a.run/'summary.json')['ground_truth_sha256']:raise ValueError('Reference changed')
    gt=pm.read_label_csv(gt_path);codes=pm.label_codes(gt);truth={f:pm.LABELS[int(codes[f])] for f in range(start,end)}
    changes=[];counts=Counter();confusions={}
    for r in rows:
        f=r['frame'];prior=old_by[f]
        if r['label']!=prior['label']:
            changes.append(dict(frame=f,before=prior['label'],after=r['label'],reference=truth[f],reason=r['reason'],
                correct=r['label']==truth[f],evidence_ids=';'.join(r['transition']['game_evidence_ids']),
                receiver_ids=';'.join(c['track_id'] for c in r['pass_continuity']['candidates'])))
            counts[(prior['label'],r['label'])]+=1
    for name,payload in [('before',old),('after',new)]:
        matrix={t:{p:0 for p in 'HADX'} for t in 'HADX'}
        for f in range(start,end):matrix[truth[f]][payload['possession']['labels'][f]]+=1
        confusions[name]=matrix
    acquisitions=[];previous=None
    for e in new['events']:
        if e['type']!='possible_control_acquisition':continue
        team=e['team'];f=e['confirmed_at_frame']
        if start<=f<end and previous is not None and previous!=team:
            acquisitions.append(dict(frame=f,team=team,before=old['possession']['labels'][f],after=new['possession']['labels'][f],event_id=e['event_id']))
        previous=team
    reserved=read(a.reserved/'summary.json');reserved_rows={r['frame']:r for r in read(a.reserved/'predictions.json')['states']}
    manual=read('outputs/possession-v4-selection/reserved-manual-observations.json')
    contacts=[dict(r,predicted_label=reserved_rows[r['frame']]['label'],tracking_control=reserved_rows[r['frame']]['control_observed_team']) for r in manual['frames']]
    audit=read('outputs/possession-v4-audit/audit.json')
    summary=dict(result_sha256=result_hashes,reference_sha256=sha(gt_path),
        status='exploratory public-window comparison; no parameter selection from scores; later launch development is explicitly recorded',
        metrics=read(a.run/'summary.json')['models'],confusion=confusions,
        changed_frames=len(changes),changes=[dict(before=b,after=c,frames=n) for (b,c),n in counts.items()],
        newly_assigned_correct=sum(r['correct'] for r in changes),newly_assigned_incorrect=sum(not r['correct'] for r in changes),
        short_same_team_gaps=dict(before=gaps(old_rows),after=gaps(rows)),
        detector_events_identical=old['events']==new['events'],adversary_acquisitions=acquisitions,
        reserved=reserved['windows'],manual_contacts=contacts,
        manual_contact_limitations='Single-frame RGB evidence is not a possession ground truth. No accuracy reported outside public window.',
        audit=dict(frames=len(audit['frames']),omitted_valid_in_pitch=sum(len(r['omitted_valid_in_pitch']) for r in audit['frames'])),
        sam_resources={str(f):read(f'outputs/possession-v4-sam-recovery-{f}/resources.json') for f in (11518,12860)},
        visual_scope=dict(previous_control_frames=[11689,11690,11691],new_recovery_frames=[11518],new_team_overlay_frames=[12860],
            dead_ball_frames=[12915,12969],restart_frame=12970,all_manual_review_is_offline=True,
            full_automatic_person_team_ball_recognition=False,line_references_approved=0),
        unresolved=['11513-11524 remains uncertain with trajectory warmup and opponent interception opportunity',
          'Early A46 point is mislocalized near home goalkeeper; no persistent identity repair or invented goalkeeper position',
          'Other missing players and team switches are not repaired globally',
          'Outside/elevated ball at 17456/18025 remains unverified, not automatically dead',
          'No reliable automatic ball-in-hands detector or approved line reference bundle',
          'Public spell F1 must be considered separately from frame accuracy'])
    write_json(a.out/'assessment.json',summary)
    with (a.out/'changes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    before,after=summary['metrics']['anterior'],summary['metrics']['coletivo']
    text=f'''# Comparação verificável — posse v4

Comparador: coletivo v3 com as três revisões SAM anteriores. Nova execução: `{a.run}`. Frames {start}–{end-1}, 25 fps; previsões salvas antes da leitura do gabarito. O recorte público já foi explorado: os números abaixo não são validação independente.

| Medida | Antes | Depois |
|---|---:|---:|
| Acurácia por frame | {before['metrics']['accuracy']:.2%} | {after['metrics']['accuracy']:.2%} |
| Macro F1 | {before['metrics']['macro_f1']:.4f} | {after['metrics']['macro_f1']:.4f} |
| F1 médio de segmentos oficial | {before['metrics']['spell_f1_mean']:.4f} | {after['metrics']['spell_f1_mean']:.4f} |
| X | {before['label_counts']['X']} | {after['label_counts']['X']} |
| D | {before['label_counts']['D']} | {after['label_counts']['D']} |
| Lacunas X de até 25 frames entre a mesma equipe | {len(gaps(old_rows))} | {len(gaps(rows))} |

Mudaram {len(changes)} frames: {summary['newly_assigned_correct']} concordam com a referência pública e {summary['newly_assigned_incorrect']} discordam. **A acurácia melhora, mas o F1 oficial de segmentos piora.** O detalhe por frame e as matrizes de confusão estão em `changes.csv` e `assessment.json`; não se deve apresentar a redução de X como melhoria universal.

## Exemplos e evidências

- **11873–11874:** saída de passe do jogador observado, limitada a 0,20 s desde o controle e 2 m do passador, com movimento para fora, cobertura e adversário afastado. Os dois frames deixam de ser X. Não há consulta ao recebedor futuro.
- **11915–11921:** o recebedor H permanece observado e se aproxima, embora o cálculo instantâneo deixe de encontrar encontro no horizonte. A hipótese mantém a idade do controle original, limitada a quatro segundos.
- **11513–11524:** continua X: há aquecimento após ausência e oportunidade adversária. Não foi preenchido para obter continuidade visual.
- **11518:** SAM propõe a pessoa de uniforme branco perto de (1124,413) pixels, sem observação ATD correspondente. Projeção revisada ≈(48,51;26,47) m; ponto ATD mais próximo a 4,05 m. A recuperação aparece somente nesse frame, com identificação local e sem controle individual artificial.
- **12860:** o ponto bruto A26 coincide com a pessoa de uniforme azul/vermelho. Uma sobreposição revisada corrige a equipe naquele frame; não altera o ATD nem atribui identidade persistente. A posição revisada difere 0,28 m do ponto bruto.
- **12915–12969:** D sustentado pela manipulação/colocação da bola para o reinício, revisão do gesto do árbitro e renovação explícita da evidência. Não se retroage ao início da ausência em 12831. Em **12970**, o chute observado encerra D; o modelo volta a X até sustentar controle pelo tracking.
- **17456 e 18025:** os pontos além do campo não receberam aprovação geométrica. O vídeo mostra bola elevada ou junto ao goleiro; uma homografia do solo não basta para decidir saída.

A auditoria de {len(audit['frames'])} frames encontrou **{summary['audit']['omitted_valid_in_pitch']} observações válidas dentro do campo omitidas** pela regra do renderizador. Há perdas e associações problemáticas no tracking. O ponto A46 junto ao goleiro H continua diagnosticado, sem posição substituta inventada. Não foi presumido offset temporal a partir do deslocamento da bola projetada.

## Avaliação reservada e limites

Foram processados {sum(w['frames'] for w in reserved['windows'])} frames em quatro janelas posteriores, com prefixo causal e política congelada: 14500–15100, 15400–16100, 17440–17480 e 18010–18150. Nenhuma recuperação nova de continuidade sobrepôs controle adversário observado nessas janelas. Isso é uma verificação de comportamento, **não uma estimativa de precisão**: não há gabarito independente de posse nesses trechos. As observações manuais de contato foram congeladas antes de ler as previsões reservadas e não foram tratadas como rótulos de posse. Caso se ajustem parâmetros depois, reservar novos lances.

Os eventos do detector são idênticos antes/depois. As recuperações adversárias confirmadas são listadas no relatório; as novas regras não reescrevem os eventos nem fabricam aquisições. A matriz pública e os desacordos mostram limites que permanecem.

## O que está integrado

O motor recebe evidências de interrupção/reinício, hipóteses de continuidade e recuperações revisadas. O renderizador mostra as recuperações/correções com marca magenta e preserva as ausências. Os três controles anteriores de goleiro continuam com confirmação manual de bola nas mãos. As novas observações recuperadas entram como evidência de recebedor e na representação; o detector original permanece intacto.

Tudo isso permanece **experimental**: dois frames novos de observação visual, três âncoras revisadas de jogo, sem classificador automático completo de equipes ou bola. As 22 referências reprovadas continuam reprovadas. O resultado negativo do limiar 0,7 não foi reaberto. A revisão é de Codex, sem certificação humana; disponibilidade declarada por frame serve ao replay offline, não comprova latência real.

As duas inferências SAM usaram CPU sequencialmente, baseline oficial, watchdog de 60 s e RSS amostrado máximo de 7 GB. Foram concluídas em 24,97 s / 6,79 GB e 25,80 s / 6,83 GB. Recursos e hashes estão no JSON de avaliação e nos pacotes originais. Não houve MPS, varredura de limiares ou inferência visual nos 3.000 frames completos.

## Reprodução

Os comandos de execução estão em `docs/POSSE-COLETIVA.md`. As fontes e máscaras ficam em `outputs/possession-v4-*`; o pacote integrado é `outputs/possession-v4-reviewed-evidence/evidence.json`. `protocol.json` registra configuração e hashes do código/dados efetivos. A animação é derivada das previsões salvas, sem recalcular métricas.
'''
    (a.out/'report.md').write_text(text)
    print(json.dumps(dict(changed=len(changes),correct=summary['newly_assigned_correct'],incorrect=summary['newly_assigned_incorrect'],
      gaps_before=len(gaps(old_rows)),gaps_after=len(gaps(rows)),events_identical=summary['detector_events_identical']),indent=2))


if __name__=='__main__':main()
