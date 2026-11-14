"""Human-readable local report for the paired persistence experiment."""
from html import escape
from pathlib import Path


def timeline(labels, fps=25):
    colors={'H':'#2563eb','A':'#e85d32','D':'#64748b','X':'#e5b82e'}
    parts=[];start=0
    for end in range(1,len(labels)+1):
        if end<len(labels) and labels[end]==labels[start]: continue
        label=labels[start]
        parts.append(f'<rect x="{start}" y="0" width="{end-start}" height="14" fill="{colors[label]}"><title>{label}: frames {start}–{end-1}; {start/fps:.2f}–{end/fps:.2f}s</title></rect>')
        start=end
    return f'<svg role="img" aria-label="Linha de tempo H azul, A laranja, X amarelo" viewBox="0 0 {len(labels)} 14" preserveAspectRatio="none">'+''.join(parts)+'</svg>'


def write_report(out, reports, cases):
    import json
    out=Path(out)
    cards=[]
    for case in cases:
        name=case['name']; data=reports[name]
        rows=[]
        for p in ('3','5'):
            result=json.loads((out/'scenarios'/name/p/'result.json').read_text())
            summary=data['profiles'][p]
            observed=' → '.join(summary['observed_teams']) or 'nenhuma'
            expected=' → '.join(case['expected_teams']) or 'nenhuma'
            acquisitions='; '.join(f"{e['team']}: início {e['event_frame']}, confirmação {e['confirmed_at_frame']}, disponível {e['available_at_frame']}" for e in summary['acquisitions']) or 'Nenhuma aquisição'
            rows.append(f'<tr><td>{p} frames</td><td>{escape(observed)}</td><td>{"Sim" if summary["expected_acquisition_sequence_met"] else "Não"}</td><td>{escape(acquisitions)}</td></tr>')
            rows.append(f'<tr><td colspan="4">{timeline(result["possession"]["labels"])}<small>Frames 0–{data["n_frames"]-1}; {data["n_frames"]/25:.2f} s</small></td></tr>')
        cards.append(f'<section><h2>{escape(name.replace("_"," "))}</h2><p>{escape(case["rationale"])}</p><p>Sequência esperada: <b>{escape(expected)}</b>. Frames de saída diferentes: <b>{data["different_label_frames"]}</b>.</p><table><thead><tr><th>Persistência</th><th>Aquisições</th><th>Sequência esperada?</th><th>Tempos em frames</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>')
    combined=reports['same_tracking_synthetic']
    real=reports.get('same_tracking_real',{})
    real_text='Não executado: arquivos reais de tracking indisponíveis.' if 'profiles' not in real else 'Executado; consultar same-tracking-real/comparison.json. Sem referência independente, divergência não mede melhora.'
    html='''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Persistência: 3 contra 5 frames</title><style>
body{font:16px/1.55 system-ui;color:#18333e;background:#f4f6f4;max-width:1100px;margin:auto;padding:30px}h1,h2{line-height:1.2}section{background:white;padding:24px;margin:20px 0;border-radius:12px}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #ddd;padding:9px}svg{width:100%;height:28px;display:block}small{color:#58656a}.note{border-left:5px solid #e5b82e;padding:15px;background:#fff9db}li{margin:8px 0}</style>
<h1>Persistência: 3 contra 5 frames</h1><p>Experimento sintético pareado • 25 fps • aquisição 3/5 • perda fixa em 3 • velocidade relativa máxima 5 m/s.</p>
<div class="note"><b>Tracking real da partida:</b> '''+escape(real_text)+'''<br>As cores reais dos goleiros continuam sem verificação. Todos os cenários abaixo são construídos.</div>
<section><h2>O que os resultados permitem concluir</h2><ul>
<li>Controles sustentados continuam presentes; confirmação normalmente chega 2 frames (80 ms) depois com 5.</li>
<li>Controles reais de 3 ou 4 amostras elegíveis deixam de ser confirmados com 5.</li>
<li>Na roubada curta, 3 detecta H → A → H; 5 produz H → H e um intervalo X. Aumentar persistência pode esconder um controle real do adversário.</li>
<li>A proximidade breve sem controle foi filtrada por 5. Ela tem tracking idêntico ao exemplo de controle curto legítimo: só posição e movimento não resolvem essa ambiguidade.</li>
<li>Com evidência disponível apenas no frame 15, ambas as versões chegam ao consumidor no mesmo instante, apesar da confirmação diferente.</li>
<li>Não selecionar um vencedor pela quantidade de eventos ou X. Estes casos não representam frequências de jogo nem medem precisão esportiva.</li></ul></section>
<section><h2>Mesmo tracking sintético composto</h2>'''+f'<p>{combined["n_frames"]} frames, {combined["n_frames"]/25:.2f} s; 20 cenários com reinícios explícitos; entrada idêntica verificada por SHA-256.</p><table><tr><th>Medida</th><th>3</th><th>5</th></tr>'+''.join(f'<tr><td>{label}</td><td>{combined["profiles"]["3"][field].get(key,0)}</td><td>{combined["profiles"]["5"][field].get(key,0)}</td></tr>' for label,field,key in [('Aquisições','event_counts','possible_control_acquisition'),('Perdas','event_counts','possible_control_loss'),('Frames H','label_counts','H'),('Frames A','label_counts','A'),('Frames X','label_counts','X')])+f'</table><p>{combined["different_label_frames"]} frames de saída diferentes. A perda tem o mesmo limiar, mas seu número pode mudar quando não há aquisição predecessora.</p></section>'+'''
<section><h2>Como ler a frente 4 provisória</h2><p>O consumidor experimental só muda H/A/X quando o evento fica disponível, sem reescrever o passado. Aquisição define time, perda do controle ativo define X, toque/disputa não impõem vencedor e descontinuidade reinicia X. Não infere D. Entre eventos mantém o último estado, inclusive em oclusões: isso é uma limitação explícita, não evidência de posse contínua. Esta política não é a solução final da frente 4.</p><p>Legenda: H azul · A laranja · X amarelo. Os gráficos são estados causais do protótipo, não o gabarito.</p></section>'''+''.join(cards)+'</html>'
    (out/'report.html').write_text(html,encoding='utf-8')
