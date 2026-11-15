"""Post-prediction comparison and ablation for causal context and goalkeeper repair."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from src.ball_events.artifacts import sha,write_json


def read(p):return json.loads(Path(p).read_text())


def gaps(rows):
    n=0;i=1
    while i<len(rows):
        if rows[i]['label']=='X' and rows[i-1]['label'] in ('H','A'):
            j=i
            while j<len(rows) and rows[j]['label']=='X':j+=1
            n+=int(j<len(rows) and j-i<=25 and rows[j]['label']==rows[i-1]['label']);i=j
        else:i+=1
    return n


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--context-only',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    old=read(a.run/'anterior/result.json');new=read(a.run/'coletivo/result.json');context=read(a.context_only/'coletivo/result.json')
    rows=new['possession']['states'];old_rows=old['possession']['states'];by={r['frame']:r for r in old_rows}
    newsummary=read(a.run/'summary.json');contextsummary=read(a.context_only/'summary.json')
    metrics={'before':newsummary['models']['anterior'],'context_only':contextsummary['models']['coletivo'],'context_and_goalkeeper':newsummary['models']['coletivo']}
    stats=dict(metrics=metrics,short_same_team_unknown_gaps=dict(before=gaps(old_rows),after=gaps(rows)),
        source_hashes={str(p):sha(p) for p in (a.run/'anterior/result.json',a.run/'coletivo/result.json',a.context_only/'coletivo/result.json')},
        context_accepted=sum(r['temporal_context']['accepted'] for r in rows),
        context_blocks=dict(Counter(k for r in rows if r['label']=='X' for k,v in r['temporal_context']['checks'].items() if v is False)),
        goalkeeper_assignments=sum(any(t['action']=='goalkeeper_assignment' for t in r['tracking_review']) for r in rows),
        goalkeeper_source_ids=dict(Counter(t['track_id'] for r in rows for t in r['tracking_review'] if t['action']=='goalkeeper_assignment')),
        rejected_tracking_frames=sum(any(t['action']=='reject_observation' for t in r['tracking_review']) for r in rows),
        raw_detector_events_unchanged=old['events']==new['raw_detector_events'],
        corrected_detector_event_count=len(new['events']),previous_detector_event_count=len(old['events']),
        corrected_events_equal_previous=new['events']==old['events'],
        experimental=True,independent_accuracy_validation=False,
        interpretation='Fixed user requested 25 frames; public labels read only after frozen predictions. Goalkeeper assignments are heuristic source-point association with reviewed goal context, not automatic reference approval.')
    write_json(a.out/'assessment.json',stats)
    with (a.out/'changes.csv').open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['frame','before','after','reason','context_accepted','context_support_fraction'])
        for r in rows:
            if r['label']!=by[r['frame']]['label']:
                writer.writerow([r['frame'],by[r['frame']]['label'],r['label'],r['reason'],r['temporal_context']['accepted'],r['temporal_context']['origin_support_fraction']])
    lines=['# Contexto causal de 25 frames e correção do goleiro','',
        'Comparador: coletivo v4 anterior. Trecho tactical 11185–14184, 25 fps. As três configurações foram congeladas antes de consultar o gabarito; os resultados públicos continuam exploratórios.','',
        '| Medida | Antes (v4) | Só contexto 25 | Contexto + goleiro |','|---|---:|---:|---:|']
    for key,label in [('accuracy','Acurácia por frame'),('macro_f1','Macro F1'),('spell_f1_mean','F1 oficial de segmentos')]:
        vals=[metrics[k]['metrics'][key] for k in metrics];lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    for key in ('X','D'):
        vals=[metrics[k]['label_counts'][key] for k in metrics];lines.append('| Frames '+key+' | '+' | '.join(map(str,vals))+' |')
    lines+=['',f"As lacunas X de até 25 frames entre a mesma equipe passam de {gaps(old_rows)} para {gaps(rows)}. O contexto foi aceito em {stats['context_accepted']} frames. Reduzir X não garante acerto; comparar também os segmentos e os desacordos em `changes.csv`.",'',
        '## Contexto integrado','',
        'Janela causal de 25 frames, enfatizando os cinco mais recentes: peso linear por idade e fator 2 para esses cinco. Armazena evidência direta de controle e geometria de passe; uma atribuição contextual não se transforma em nova evidência. Exige pelo menos cinco frames de evidência independente e fração ponderada de suporte ≥0,75. Esses valores são heurísticas explícitas, não probabilidades calibradas nem limiares escolhidos pelo score público.',
        'A âncora continua sendo o último controle sustentado, com teto de quatro segundos. Bola observada válida, cobertura atual, ausência de controle/toque adversário e de interceptação iminente são requisitos. Movimento lento ou aquecimento de trajetória precisa de controle recente e companheiro observado perto da bola. D, reinício, quebra de continuidade, troca de origem e perda prolongada limpam a memória. X permanece quando esses critérios falham.','',
        '## Goleiro: causa e correção','',
        'O erro não era apenas uma cor: o ponto A46 alternava entre posição sem pessoa correspondente e o próprio goleiro verde do Barcelona, enquanto H24 às vezes ficava deslocado. Não se recoloriu A46 globalmente como Barcelona.',
        f"Em {stats['goalkeeper_assignments']} frames, um indício RGB conservador do uniforme verde, restrito à região do gol defendido e ao primeiro tempo revisado, foi associado a um ponto ATD já observado. Origens escolhidas: {stats['goalkeeper_source_ids']}. A equipe H vem do contexto de gol/período confirmado, não apenas da cor. A identificação local `reviewed-gk-H` evita mudanças indevidas de equipe na identidade persistente. Nenhuma coordenada é criada ou interpolada.",
        'Pontos conflitantes são marcados como rejeitados, ficam fora dos cálculos e aparecem como cruz cinza na animação. O goleiro H recebe G-H azul; o ponto do Athletic retomado no gol direito mantém equipe A e G-A. Quando a associação visual falha, não é inventada uma posição: permanece a abstenção da associação e o papel H previamente revisado não certifica a qualidade de toda coordenada H24.',
        'O piloto inicial de cor aceitou uma região curta do tronco como se fosse suficiente para associar os pés. Esse caso foi reprovado no desenvolvimento; a variante final exige componente de pelo menos 20 pixels de altura e preserva as abstenções. Há 935 associações propostas no pacote final; isso não equivale a 935 revisões humanas nem a reconhecimento automático validado.',
        'A correção é limitada à janela revisada. Não aprova referências de linha, não troca as equipes de todos os goleiros por sua posição no campo, não assume orientação de qualquer vídeo e não reutiliza as referências SAM reprovadas.','',
        '## Eventos e proveniência','',
        f"Detector bruto preservado: {stats['raw_detector_events_unchanged']}. O modelo corrigido usa observações com revisão de qualidade; seus eventos podem diferir dos brutos ({stats['previous_detector_event_count']} eventos anteriores, {stats['corrected_detector_event_count']} corrigidos). Os dois fluxos estão salvos separadamente. Portanto, a garantia anterior de eventos idênticos não é aplicada artificialmente a entradas agora corrigidas.",
        'As imagens, a associação por frame, os limites da revisão, os hashes do vídeo/ATD/homografia e o código efetivo constam em `outputs/context25-goalkeeper-association-reviewed/` e no `protocol.json` da execução. As posições ATD originais permanecem intactas. A disponibilidade das revisões é declarada para replay offline, não demonstra latência operacional real.','']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()
