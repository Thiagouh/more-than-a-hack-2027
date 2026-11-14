"""Postprocess saved paired results; never reads possession ground truth."""
import argparse
import hashlib
import math
import json
from collections import Counter
from pathlib import Path
from html import escape

parser=argparse.ArgumentParser(description='Analyze paired real tracking results without ground truth')
parser.add_argument('run',type=Path)
parser.add_argument('--tracking',type=Path,help='Optional raw ATD for visibility sensitivity')
args=parser.parse_args()
def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

root=args.run
base=root/'same-tracking-real'
r={p:json.loads((base/p/'result.json').read_text()) for p in ('3','5')}
comparison=json.loads((base/'comparison.json').read_text())
acq={p:[e for e in r[p]['events'] if e['type']=='possible_control_acquisition'] for p in r}
index={p:{(e['team'],e['event_frame']):e for e in acq[p]} for p in r}
common=sorted(index['3'].keys()&index['5'].keys())
only={p:[index[p][key] for key in sorted(index[p].keys()-index['5' if p=='3' else '3'].keys())] for p in r}
paired=[dict(team=key[0],event_frame=key[1],confirmation_extra_frames=index['5'][key]['confirmed_at_frame']-index['3'][key]['confirmed_at_frame'],availability_extra_frames=index['5'][key]['available_at_frame']-index['3'][key]['available_at_frame']) for key in common]
def timecode(frame):
 s=frame/25;return f'{int(s//3600):02}:{int(s%3600//60):02}:{s%60:06.3f}'
def runs(labels):
 result=[];start=0
 for end in range(1,len(labels)+1):
  if end<len(labels) and labels[end]==labels[start]:continue
  result.append((labels[start],start,end));start=end
 return result
cross=Counter(zip(r['3']['possession']['labels'],r['5']['possession']['labels']))
analysis=dict(identical_input_sha256=comparison['identical_input_sha256'],n_frames=comparison['n_frames'],fps=25,
 differences_frames=comparison['different_label_frames'],difference_intervals=len(comparison['differences']),
 differences_percentage=100*comparison['different_label_frames']/comparison['n_frames'],
 paired_acquisitions=len(common),acquisition_only_counts={p:len(only[p]) for p in r},
 paired_confirmation_extra_frames=dict(Counter(str(p['confirmation_extra_frames']) for p in paired)),
 paired_availability_extra_frames=dict(Counter(str(p['availability_extra_frames']) for p in paired)),
 label_cross_counts={a+'->'+b:n for (a,b),n in sorted(cross.items())},
 raw_label_run_counts={p:dict(Counter(label for label,a,b in runs(r[p]['possession']['labels']))) for p in r},
 exclusive_acquisitions={p:[{k:e[k] for k in ('team','event_frame','confirmed_at_frame','available_at_frame')} for e in only[p]] for p in r},
 warnings=['No ground truth read; differences are not errors.', 'No visual review of match or goalkeeper colours in this run.', 'Event-only possession holds last state between events, including visibility gaps.', 'Only acquisition changes (3 vs 5); loss fixed at 3.'])
visibility_note = ''
if args.tracking:
    expected_hash=json.loads((base/'provenance.json').read_text())['input_sha256'].get('atd_raw')
    if expected_hash is None or file_hash(args.tracking)!=expected_hash:
        raise ValueError('Visibility input must match the recorded ATD SHA-256')
    observed=[]
    with args.tracking.open() as handle:
        for line in handle:
            coordinates=line.strip().split(':')[2].split(',')[:2]
            observed.append(all(math.isfinite(float(v)) for v in coordinates))
    if len(observed)!=comparison['n_frames']:
        raise ValueError('Visibility input coverage differs from paired tracking')
    changed=[a!=b for a,b in zip(r['3']['possession']['labels'],r['5']['possession']['labels'])]
    visible_differences=sum(o and d for o,d in zip(observed,changed))
    absent_differences=sum(not o and d for o,d in zip(observed,changed))
    analysis['visibility_sensitivity']=dict(observed_ball_frames=sum(observed),missing_ball_frames=len(observed)-sum(observed),
        differences_with_observed_ball=visible_differences,differences_with_missing_ball=absent_differences,
        note='Post-hoc diagnostic on saved outputs, no new detector run and no validated possession policy')
    longest=max(comparison['differences'],key=lambda d:d['end_frame_exclusive']-d['start_frame'],default=None)
    if longest:
        start,end=longest['start_frame'],longest['end_frame_exclusive']
        analysis['longest_difference']=dict(longest,frames=end-start,seconds=(end-start)/25,
            ball_observed_frames=sum(observed[start:end]),ball_missing_frames=(end-start)-sum(observed[start:end]))
        visibility_note=f"A maior divergência cobre {end-start} frames ({(end-start)/25:.2f} s), mas em {(end-start)-sum(observed[start:end])} deles a bola não é observada. Ela representa {100*(end-start)/sum(changed):.2f}% dos frames divergentes."
    visibility_note=f"Dos {sum(changed)} frames divergentes, {absent_differences} ocorrem com bola ausente e {visible_differences} com bola observada. Entre os {sum(observed)} frames com bola observada, a divergência é {100*visible_differences/sum(observed):.2f}%. Isso é sensibilidade à visibilidade, não precisão; nem todo frame com bola observada tem tracking válido para controle. " + visibility_note + " O estado H mantido pelo protótipo durante a lacuna não prova posse. A maior divergência precisa ser interpretada como limitação do consumidor antes de escolher o limiar."

(base/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
rows=[]
for label,field,key in [('Aquisições','event_counts','possible_control_acquisition'),('Perdas','event_counts','possible_control_loss'),('Toques','event_counts','possible_touch'),('Disputas','event_counts','possible_dispute'),('Frames H','label_counts','H'),('Frames A','label_counts','A'),('Frames X','label_counts','X')]:
 rows.append(f"| {label} | {comparison['profiles']['3'][field].get(key,0)} | {comparison['profiles']['5'][field].get(key,0)} |")
longest=sorted(comparison['differences'],key=lambda d:d['end_frame_exclusive']-d['start_frame'],reverse=True)[:10]
clips=[]
for d in longest:
 clips.append(f"| {d['start_frame']}–{d['end_frame_exclusive']-1} | {timecode(d['start_frame'])}–{timecode(d['end_frame_exclusive'])} | {(d['end_frame_exclusive']-d['start_frame'])/25:.2f} s | {d['label_3']} | {d['label_5']} |")
md=f'''# Persistência no tracking real: 3 × 5

Tracking tactical BAR-ATH, {comparison['n_frames']:,} frames a 25 fps. Entrada idêntica nas duas execuções, conferida por SHA-256. Aquisição varia 3/5; perda fixa em 3; velocidade relativa máxima 5 m/s. Sem gabarito, seleção por score ou inspeção de vídeo nesta execução.

## Resultados observados

| Medida | 3 frames | 5 frames |
|---|---:|---:|
'''+ '\n'.join(rows)+f'''

Saídas H/A/X diferentes em **{analysis['differences_frames']:,} frames ({analysis['differences_percentage']:.2f}%)**, distribuídos por {analysis['difference_intervals']} intervalos. X é desconhecido; não é bola morta. D não é inferido.

Aquisições com mesmo time e mesmo instante estimado: **{len(common)}**. Exclusivas da versão de 3: **{len(only['3'])}**; exclusivas da de 5: **{len(only['5'])}**. Pareamento é por time e event_frame, não uma avaliação contra referência.

Diferenças de confirmação nas aquisições pareadas (frames adicionais de 5): `{analysis['paired_confirmation_extra_frames']}`. Diferenças de disponibilidade: `{analysis['paired_availability_extra_frames']}`.

## Interpretação

Cinco exige evidência mais longa e deixa de emitir parte dos candidatos. Não se sabe, sem referência independente, quantos eram falsos controles ou controles reais rápidos. Os cenários sintéticos mostram ambos os efeitos; menos eventos não demonstra melhora.

A saída H/A/X usa a mesma política experimental nas duas execuções: aquisição entregue define time, perda da aquisição ativa define X, toque/disputa não escolhem vencedor. Não reescreve frames passados. Mantém o último estado entre eventos, inclusive durante oclusões: isso pode prolongar um time sem observação atual e precisa de fontes adicionais antes de ser uma decisão final da frente 4.

A amostra tactical tem metadados com “Atletic Club”, aceito como variante explícita de “Athletic Club”. Os IDs e a ordem H/A são preservados. O campo declara 105×68 m, enquanto a configuração usa 105,3×68 m; a diferença já é aceita pelo diagnóstico existente (até 1 m) e fica registrada, sem alteração entre versões. A orientação não foi verificada visualmente.

Não escolheria o vencedor apenas com estas contagens. O próximo passo é revisar amostras de divergência e trechos escolhidos independentemente das versões, distinguindo controle, toque e posse coletiva. Não selecionar parâmetros no gabarito público.

## Efeito da ausência de bola

{visibility_note}

## Maiores divergências para inspeção

Tempos na timeline do vídeo tactical, não no relógio da partida nem no broadcast. A seleção abaixo é exploratória, não uma amostra para medir precisão.

| Frames inclusivos | Timestamp tactical | Duração | 3 | 5 |
|---|---|---:|---|---|
'''+ '\n'.join(clips)+'''

## Reprodução e arquivos

`comparison.json`: resumo e intervalos diferentes; `differences.csv`: intervalos com fim exclusivo; `analysis.json`: contagens e pareamento; `3/result.json` e `5/result.json`: eventos, diagnósticos e saídas completas; `3/possession.csv` e `5/possession.csv`: saídas estruturais validadas, não submissões recomendadas. `provenance.json` registra código, parâmetros e hashes dos arquivos reais.

Tudo permanece em outputs/, ignorado pelo Git. As cores reais dos goleiros ainda não foram verificadas.
'''
(base/'ANALISE.md').write_text(md)
colors={'H':'#2563eb','A':'#e85d32','X':'#e5b82e','D':'#64748b'}
def timeline(labels,start,end):
 parts=[]
 for label,a,b in runs(labels[start:end]):
  parts.append(f'<rect x="{a}" y="0" width="{b-a}" height="16" fill="{colors[label]}"><title>{label}: frames {start+a}–{start+b-1}</title></rect>')
 return f'<svg viewBox="0 0 {end-start} 16" preserveAspectRatio="none" role="img" aria-label="Linha do tempo de posse experimental">'+''.join(parts)+'</svg>'
sections=[]
for d in longest:
 start=max(0,d['start_frame']-50);end=min(comparison['n_frames'],d['end_frame_exclusive']+50)
 sections.append(f'<section><h2>{timecode(start)}–{timecode(end)}</h2><p>Contexto em frames {start}–{end-1}. Diferença central: {d["start_frame"]}–{d["end_frame_exclusive"]-1}.</p>'+''.join(f'<p>{p} frames</p>'+timeline(r[p]['possession']['labels'],start,end) for p in r)+'</section>')
html='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Tracking real: 3 contra 5</title><style>body{font:16px/1.55 system-ui;color:#18333e;background:#f4f6f4;max-width:1100px;margin:auto;padding:30px}section{background:white;padding:24px;margin:20px 0;border-radius:12px}svg{width:100%;height:30px;display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{background:#fff9db;padding:20px;border-left:5px solid #e5b82e}</style><h1>Tracking real: 3 contra 5</h1><p class="note">Resultados observados, sem referência de acerto. H azul · A laranja · X amarelo. A política de posse é provisória e mantém estado entre eventos.</p><section><pre>'''+escape(md)+'</pre></section>'+''.join(sections)+'</html>'
(base/'report.html').write_text(html)
print(json.dumps({k:v for k,v in analysis.items() if k!='exclusive_acquisitions'},indent=2))

provenance={'analysis_script_sha256':file_hash(__file__),
            'input_sha256':{str(base/p/'result.json'):file_hash(base/p/'result.json') for p in ('3','5')},
            'ground_truth_used':False}
if args.tracking: provenance['tracking_sha256']=file_hash(args.tracking)
(base/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
