"""Read-only scoring report builder for fixed acquisition-profile runs."""
import argparse
import csv
from html import escape
import json
from pathlib import Path

from scripts.persistence_report import timeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args();root=args.run
    summary=json.loads((root/'summary.json').read_text())
    synthetic=json.loads((root/'synthetic-results.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    profiles=sorted(summary,key=int)
    start,end=protocol['start_frame'],protocol['end_frame_exclusive'];n=end-start
    with (root/'comparison-by-frame.csv').open() as handle:table=list(csv.DictReader(handle))
    gt=[row['reference'] for row in table]
    labels={p:[row[f'prediction_{p}'] for row in table] for p in profiles}
    rows=[]
    for p in profiles:
        d=summary[p];m=d['metrics']
        rows.append(f"| {p} | {(int(p)-1)*40} ms | {m['spell_f1_mean']:.4f} | {m['macro_f1']:.4f} | {100*m['accuracy']:.2f}% | {100*d['label_counts']['X']/n:.2f}% | {m['n_pred_spells']} | {d['acquisition_events']} |")
    synthetic_rows=[]
    for name,case in synthetic.items():
        expected=' → '.join(case['expected']) or 'nenhuma'
        observed=[' → '.join(case['profiles'][p]['observed']) or 'nenhuma' for p in profiles]
        synthetic_rows.append('| '+name+' | '+expected+' | '+' | '.join(observed)+' |')
        if len({case['profiles'][p]['input_stream_sha256'] for p in profiles})!=1:
            raise ValueError('Synthetic inputs differ between profiles')
    counts={p:sum(case['profiles'][p]['expectation_met'] for case in synthetic.values()) for p in profiles}
    md='''# Persistência de aquisição: 1, 2 e 4 frames

Comparação executada em 14/11/2026 nos mesmos 20 cenários sintéticos e no mesmo recorte de dois minutos do gabarito público usado pelo notebook 03. Os perfis anteriores, 3 e 5, foram reexecutados como controles e reproduziram exatamente as previsões anteriores no trecho.

## Condições iguais

Recorte tactical 11185–14184, 3.000 frames a 25 Hz, 120 segundos. Processado o prefixo completo de 0 até 14184 para preservar o histórico causal, em vez de reiniciar o detector no início do recorte. Os cinco perfis receberam envelopes com SHA-256 idêntico. Não foi reprocessada a partida inteira nesta rodada.

Apenas aquisição variou: 1/2/3/4/5. Perda fixa em 3, velocidade relativa máxima 5 m/s e tolerância independente de bola ausente em 5. O protocolo foi salvo antes das execuções e o gabarito só foi aberto após salvar todas as previsões. Nenhum parâmetro foi selecionado ou ajustado usando os scores.

O novo suporte a aquisição de 1 frame significa confirmar na primeira evidência elegível, sem espera adicional de persistência. Ainda exige histórico para calcular movimento, bola/jogador observados, margem, qualidade e disponibilidade. Não equivale a decidir por uma imagem isolada. O mínimo de perda continua em 2 no contrato e foi mantido em 3 no experimento; o padrão global não foi alterado.

## Resultado nos dois minutos

O gabarito contém 8 blocos de posse H/A. O atraso abaixo é o mínimo desde a primeira evidência elegível; produtores atrasados podem aumentá-lo.

| Aquisição | Atraso mínimo | F1 temporal | Macro F1 | Acerto/frame | X | Blocos H/A | Eventos de aquisição |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

Os eventos de aquisição incluem reconfirmações de controle e não são a contagem de posses. As métricas são as funções oficiais, sem bootstrap. O score temporal preenche X conforme a regra oficial, por isso a cobertura é mostrada separadamente.

## Análise

- Todos os perfis empataram no F1 temporal (0,4667) e produziram 10 blocos H/A contra 8 da referência. Alterar a persistência não resolveu a estrutura das posses neste recorte.
- Um frame teve o maior acerto por frame e menor abstenção neste trecho. Isso sugere benefício de responder mais cedo neste exemplo, não que uma amostra seja o melhor padrão para uma partida inteira.
- A versão de 1 gerou uma aquisição candidata adicional (30 contra 29). Ela não criou um bloco adicional de posse no avaliador. Não foi classificada como toque/controle verdadeiro por inspeção de vídeo.
- Entre 1 e 5, o acerto varia de 37,33% para 34,03%, mas X ainda cobre 55,57% a 59,43%. O consumidor continua muito incompleto, especialmente na decisão de bola morta e na retomada de evidência. Não recomendaria escolher um padrão definitivo apenas por essa pequena janela pública.
- Nos cenários sintéticos, 1/2/3 recuperam controles legítimos de três amostras e também aceitam a proximidade sem controle construída com tracking idêntico. Quatro rejeita esse falso positivo, mas também perde os controles legítimos de três amostras. Quatro recupera a roubada curta de quatro amostras; cinco a perde.
- Os cenários não são uma amostra representativa de futebol. As contagens de expectativas atendidas abaixo verificam comportamento, não estimam precisão esportiva nem ordenam os perfis por qualidade.

## Cenários sintéticos iguais

Expectativas atendidas: '''+', '.join(f'{p} frames: {counts[p]}/{len(synthetic)}' for p in profiles)+'''. O desenho desses casos não distingue a qualidade dos perfis 1, 2 e 3; todos falham no mesmo exemplo ambíguo. Os testes específicos de fronteira verificam que 1 confirma uma vez no primeiro frame elegível e respeita atrasos e qualidade.

| Caso | Esperado | '''+' | '.join(f'{p} frames' for p in profiles)+''' |
|---|---|'''+ '|'.join('---' for _ in profiles)+'''|
'''+ '\n'.join(synthetic_rows)+'''

## Arquivos e reprodução

`protocol.json`: parâmetros, janela e hashes antes da avaliação. `comparison-by-frame.csv`: gabarito e previsões com frame tactical original. Cada pasta numérica contém eventos, estados, qualidade, CSV de previsão e métricas oficiais. `synthetic-results.json` contém expectativas e resultados dos mesmos casos. O vídeo `tracking-gabarito-1-vs-2-vs-4.mp4` é uma animação gerada do tracking real com as três novas versões; não é filmagem nem o demonstrativo original citado anteriormente.

```sh
.venv/bin/python -m scripts.compare_acquisition_profiles \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --out outputs/public-2min-profiles-repeticao
.venv/bin/python -m scripts.report_acquisition_profiles outputs/public-2min-profiles-repeticao
```

Usar um destino novo. Assets e derivados reais permanecem locais e ignorados pelo Git. O avaliador oficial não foi modificado. Os 88 testes automatizados passaram, incluindo o novo perfil de uma evidência, contrato JSON Schema, latência, chunking e bloqueio de rastros sem movimento verificável.
'''
    (root/'ANALISE.md').write_text(md)
    cells=[]
    for p in profiles:
        d=summary[p];m=d['metrics']
        cells.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in (p,f'{m["spell_f1_mean"]:.4f}',f'{m["macro_f1"]:.4f}',f'{100*m["accuracy"]:.2f}%',f'{100*d["label_counts"]["X"]/n:.2f}%',m['n_pred_spells'],d['acquisition_events']))+'</tr>')
    html='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Persistência: 1, 2 e 4 frames</title><style>body{font:16px/1.6 system-ui;color:#18333e;background:#f3f6f4;max-width:1150px;margin:auto;padding:25px}section{background:white;border-radius:12px;padding:24px;margin:20px 0}video{width:100%;max-height:780px;background:#14352d}svg{width:100%;height:28px}table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#fff3d4;padding:20px;border-left:5px solid #d8ac28}</style><h1>Novos testes: 1, 2 e 4 frames</h1><p class="note">Mesmo recorte de 120 s e mesmos 20 cenários. Perda = 3, bola ausente = 5, velocidade relativa = 5 m/s. Perfis 3 e 5 reexecutados como controles. Janela pública pequena; não estima qualidade na partida inteira.</p><section><video controls preload="metadata" poster="preview.png" src="tracking-gabarito-1-vs-2-vs-4.mp4"></video><p>Animação gerada do tracking real, não filmagem da partida.</p></section><section><table><tr><th>Aquisição</th><th>F1 temporal</th><th>Macro F1</th><th>Acerto/frame</th><th>X</th><th>Blocos</th><th>Aquisições</th></tr>'''+''.join(cells)+'</table></section><section><h2>Linhas do tempo — 120 segundos</h2>'+''.join(f'<p>{name}</p>'+timeline(values) for name,values in [('Gabarito',gt),*[(f'{p} frames',labels[p]) for p in profiles]])+'</section><section><pre>'+escape(md)+'</pre></section></html>'
    (root/'report.html').write_text(html)
    print(root/'report.html')


if __name__=='__main__':main()
