"""Post-hoc diagnostics and local visual report; never changes model predictions."""
import argparse
import csv
from html import escape
from itertools import islice
import json
import math
from pathlib import Path

from src.ball_events.artifacts import sha,write_json
from src.team_possession.demo import cases
from src.team_possession.evaluation import forecast_error_scores
from scripts.persistence_report import timeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--simulation',type=Path,required=True)
    args=parser.parse_args();root=args.run
    protocol=json.loads((root/'protocol.json').read_text());summary=json.loads((root/'summary.json').read_text())
    synthetic=json.loads((root/'synthetic.json').read_text())
    results={key:json.loads((root/key/'result.json').read_text()) for key in ('anterior','coletivo')}
    start,end=protocol['start_frame'],protocol['end_frame_exclusive'];n=end-start
    states=results['coletivo']['possession']['states']
    raw=args.data_dir/'tactical/BAR-ATH_tactical_atd.txt'
    if sha(raw)!=protocol['provenance']['input_sha256']['atd_raw']:raise ValueError('Forecast reference differs from tracking')
    pitch=protocol['provenance']['input_metadata']['pitch_m'];positions={}
    with raw.open() as handle:
        for f,line in enumerate(islice(handle,end+50)):
            if f<start:continue
            xy=[float(v) for v in line.strip().split(':')[2].split(',')[:2]]
            if all(math.isfinite(v) for v in xy):positions[f]=[xy[i]*pitch[i] for i in range(2)]
    forecast=forecast_error_scores(states,positions,25)
    forecast['reference_kind']='future_observations_of_same_tracker_not_independent_physical_truth'
    diagnostics={'public_tracking_consistency':forecast,'synthetic':{}}
    for case in cases():
        xyref={r['frame']:[r['ball']['x_m'],r['ball']['y_m']] for r in case['frames'] if r['ball']['source']=='observed'}
        diagnostics['synthetic'][case['name']]=forecast_error_scores(synthetic['default'][case['name']]['states'],xyref,25)
    write_json(root/'forecast-diagnostics.json',diagnostics)
    with (root/'comparison-by-frame.csv').open() as handle:table=list(csv.DictReader(handle))
    gt=[r['reference'] for r in table]
    labels={key:[r[key] for r in table] for key in results}
    rows=[]
    for title,key,kind in [('Acerto por frame','accuracy','metric'),('Macro F1','macro_f1','metric'),('F1 temporal','spell_f1_mean','metric'),('Blocos H/A','n_pred_spells','metric')]:
        values=[summary['models'][model]['metrics'][key] for model in results]
        if key=='accuracy':values=[f'{100*v:.2f}%' for v in values]
        elif key!='n_pred_spells':values=[f'{v:.4f}' for v in values]
        rows.append((title,*map(str,values)))
    rows.append(('Frames X',*[f"{labels[key].count('X')} ({100*labels[key].count('X')/n:.2f}%)" for key in results]))
    scenario_rows=[]
    for name,case in synthetic['default'].items():
        before=case['evaluation']['baseline_pass_metrics']['overall']['retention']
        after=case['evaluation']['pass_metrics']['overall']['retention']
        fmt=lambda value:'sem passe de referência' if value is None else f'{100*value:.1f}%'
        scenario_rows.append((name,fmt(before),fmt(after)))
    def fmt_error(value):
        return 'indisponível' if value is None else f'{value:.2f}'
    forecast_rows=['| Horizonte | Referências válidas | RMSE (m) | P90 (m) |','|---|---:|---:|---:|']
    for horizon,diagnostic in forecast['by_horizon'].items():
        error=diagnostic['error_m']
        forecast_rows.append(f"| {horizon} s | {error['count']} | {fmt_error(error['rmse'])} | {fmt_error(error['p90'])} |")
    forecast_table='\n'.join(forecast_rows)
    counts=summary['states']
    md='''# Posse coletiva v1: implementação e comparação visual

Execução experimental em 14/11/2026. Mesmo recorte público de 120 segundos dos testes anteriores: tactical 11185–14184, 25 Hz. O prefixo causal 0–14184 foi processado para preservar o histórico; esta rodada não avaliou a partida inteira.

## O que mudou

A frente 5 continua emitindo os mesmos eventos v2 de controle. Agora expõe também a elegibilidade de controle corrente por time e seu suporte, sem mudar a regra de aquisição. A nova camada `CollectiveModel` mede origem, trajetória, oportunidades de recepção/interceptação, intervenção adversária, cobertura e incerteza. O contrato de métricas é `collective-metrics-v1`; a política é experimental e separada do consumidor anterior.

Perda de controle individual não apaga automaticamente a posse coletiva: um passe plausível mantém o time de origem. Ameaça de interceptação abre disputa; só aquisição adversária confirmada troca o time. A origem é atualizada com evidência corrente. Ausência prolongada e evidência insustentável levam a X; bola morta ainda não é inferida.

## Parâmetros congelados antes da referência

Aquisição=3, perda=3, tolerância de bola ausente=5. Novas hipóteses: ajuste causal até 5 pontos (mínimo 3), horizonte=2 s, reação=0,2 s, aceleração=3 m/s², velocidade máxima do modelo=8 m/s, raio=1,5 m, erro de trajetória máximo=0,75 m, pelo menos 2 rastros observados por time e 50% com movimento válido para sustentar passe. Sustentação sem nova evidência de controle limitada a 4 s; vantagem adversária até 0,12 s abre disputa. Movimento de passe pode ser indicado pela derivada instantânea ou ajustada, sem trocar a previsão ajustada pela observação futura.

São hipóteses heurísticas, não parâmetros fisiológicos medidos nem probabilidades calibradas. Foram verificadas variantes sintéticas de horizonte, reação e velocidade. Não se escolheu uma configuração pelo score público.

## Resultado no trecho público

| Medida | Anterior | Coletivo |
|---|---:|---:|
'''+ '\n'.join(f'| {a} | {b} | {c} |' for a,b,c in rows)+f'''

Gabarito: 8 blocos H/A. Estados do modelo novo: controle {counts.get('controle',0)} frames, passe {counts.get('passe',0)}, disputa {counts.get('disputa',0)}, incerteza {counts.get('incerteza',0)}. Incerteza pode manter o time durante uma ausência curta; por isso não equivale exatamente à contagem de X.

## Minha análise

- A cobertura e o acerto por frame aumentaram. O modelo deixou de abandonar automaticamente vários trechos entre controles, comportamento demonstrado também nos passes sintéticos.
- O F1 temporal ficou igual e a contagem de blocos continua 10 contra 8. O avaliador temporal já preenche X; tornar alguns desses frames explicitamente H/A pode melhorar o score por frame sem mudar sua sequência temporal preenchida.
- Foram 668 frames corretos adicionais e 74 frames errados adicionais em relação ao consumidor anterior. Responder mais também cria novos erros; não apresentar apenas a redução de X como melhora.
- A recuperação do adversário segue fraca: recall A de 11,90% neste recorte, contra 5,56% antes. Aos 60 s da animação, por exemplo, o gabarito é A, mas o modelo mantém H em disputa. A regra de esperar controle adversário precisa de avaliação independente, não apenas prolongar a origem.
- A hipótese de passe tem prazo e exige trajetória/recebedor/cobertura; não é repetição indefinida do último time. Ainda há abstenções após desvios, retorno de oclusão e falta de defensores observados.
- Nenhum dos dois modelos prevê D; os 138 frames de bola morta da referência continuam sem essa decisão. Não é uma solução final de posse.
- Não há referência pública de passes, recebedores ou oportunidades de interceptação. Essas métricas na partida são explicitamente indisponíveis; o CSV H/A/D não foi convertido em anotações inventadas de passe.

## Casos sintéticos com referência construída

Retenção = frames do passe com o time correto / frames de referência desse passe. São exemplos de comportamento, não uma amostra de futebol nem uma estimativa de precisão na partida.

| Caso | Retenção anterior | Retenção coletiva |
|---|---:|---:|
'''+ '\n'.join(f'| {a} | {b} | {c} |' for a,b,c in scenario_rows)+'''

A interceptação sintética teve uma recuperação de referência e foi reconhecida uma vez, sem antecipar troca apenas pela oportunidade prevista. Com defensor ausente, a nova versão prefere X; o modelo anterior podia acertar por conservar um time sem cobertura suficiente. Com oclusão curta, a nova versão exige reconstruir movimento antes de sustentar passe de novo.

A sensibilidade está em synthetic.json: horizonte de 1 s reduziu a retenção do passe longo sintético para 64,9%, contra 100% no horizonte padrão de 2 s. Reação de 0,3 s reduziu a retenção no exemplo de desvio. O limite de velocidade 6/10 m/s não diferenciou esses exemplos com jogadores estáticos, portanto esse parâmetro ainda não foi validado por eles. Não houve calibração física com dados reais.

## Diagnóstico de previsão

forecast-diagnostics.json mede o erro por horizonte contra futuras posições observadas **somente após a previsão estar pronta**. Na partida, a referência é o próprio tracker: é consistência de previsão, não verdade física independente. Referências ausentes são excluídas com contagens explícitas; não são posições interpoladas. Erros de encontro/recepção reais permanecem indisponíveis sem referência apropriada.

{forecast_table}

O erro cresce fortemente com o horizonte. O limite local de 0,75 m só verifica previsões recentes de um passo: não certifica o corredor completo de 2 s. Os tempos de encontro são hipóteses baseadas nessa extrapolação; é necessário calibrar previsões em dados independentes antes de usá-las como evidência forte.

## Visualização e arquivos

A animação compara gabarito, modelo anterior e coletivo, ambos com aquisição de 3 frames. O painel mostra estado, origem, evidência de controle, encontros dos dois times, vantagem temporal, erro, cobertura e motivo. A faixa amarela é o corredor previsto; círculos destacam os primeiros candidatos de cada time. Nada disso é probabilidade de conclusão de passe.

O vídeo foi gerado pela skill local a partir das previsões novas, com 120 s a 25 fps. É animação de tracking, não filmagem; as cores não são uniformes verificados. Arquivos reais e derivados permanecem em outputs/, ignorados pelo Git.

protocol.json registra os parâmetros e hashes antes da avaliação. measurements.jsonl contém as medições estimadas e suas disponibilidades; coletivo/result.json contém os estados que realmente estavam disponíveis a cada decisão. collective-assessment.json marca avaliações sem referência como indisponíveis. Os arquivos oficiais de avaliação foram preservados.
'''
    (root/'ANALISE.md').write_text(md)
    simulation=Path(args.simulation).resolve()
    video=simulation/'tracking-gabarito-anterior-vs-coletivo.mp4'
    import os
    relative_video=os.path.relpath(video,root.resolve())
    relative_preview=os.path.relpath(simulation/'qa-midpoint.png',root.resolve())
    html='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Posse coletiva e passes</title><style>body{font:16px/1.6 system-ui;color:#18333e;background:#f3f6f4;max-width:1100px;margin:auto;padding:25px}section{background:white;border-radius:12px;padding:25px;margin:20px 0}video{width:100%;max-height:850px;background:#14352d}svg{width:100%;height:30px}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}.note{background:#fff3d4;padding:20px;border-left:5px solid #d8ac28}</style><h1>Posse coletiva: anterior × novo</h1><p class="note">Mesmo trecho público de 120 s. Ambos usam aquisição=3, perda=3 e ausência=5. Parâmetros de trajetória e chegada experimentais, sem calibração. O gabarito não entra nas decisões.</p><section><video controls preload="metadata" poster="'''+escape(relative_preview)+'''" src="'''+escape(relative_video)+'''"></video></section><section><table><tr><th>Medida</th><th>Anterior</th><th>Coletivo</th></tr>'''+''.join(f'<tr><td>{escape(a)}</td><td>{escape(b)}</td><td>{escape(c)}</td></tr>' for a,b,c in rows)+'</table></section><section><h2>Linhas do tempo</h2>'+''.join(f'<p>{name}</p>'+timeline(values) for name,values in [('Gabarito',gt),('Anterior',labels['anterior']),('Coletivo',labels['coletivo'])])+'</section><section><pre>'+escape(md)+'</pre></section></html>'
    (root/'report.html').write_text(html)
    write_json(root/'analysis-provenance.json',dict(script_sha256=sha(__file__),evaluation_module_sha256=sha(Path(__file__).resolve().parents[1]/'src/team_possession/evaluation.py'),
        prediction_sha256={key:sha(root/key/'result.json') for key in results},tracking_sha256=sha(raw),changes_predictions=False))
    print(root/'report.html')


if __name__=='__main__':main()
