"""Streaming SVG and accessible event table; interval views retain source frame origins."""
import csv
from fractions import Fraction
from html import escape

from . import contract as c

COLORS = {'observed': '#2563eb', 'interpolated': '#8b5cf6', 'estimated': '#eab308', 'missing': '#94a3b8', 'unknown': '#ef4444'}
LANES = {'possible_touch': 160, 'possible_control_acquisition': 220, 'possible_control_loss': 280, 'possible_dispute': 340}


def render_svg(handle, events, runs, meta, start=0, end=None):
    end = meta['n_frames'] if end is None else end
    c.integer(start, 'view start'); c.integer(end, 'view end', 1)
    if not 0 <= start < end <= meta['n_frames']:
        raise c.InputError('Invalid half-open visualization interval')
    x = lambda f: 260 + (f-start)/(end-start)*980
    origin = meta.get('origin', {'first_frame': 0, 'fps': meta['fps'], 'timeline': meta['timeline']})
    absolute = lambda f: Fraction(origin['first_frame']) + Fraction(f)*Fraction(str(origin['fps']))/Fraction(str(meta['fps']))
    def write(s): handle.write(s+'\n')
    write('<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="530" viewBox="0 0 1280 530" role="img" aria-labelledby="title desc">')
    write('<title id="title">Eventos candidatos e disponibilidade temporal</title><desc id="desc">Consulte events-table.csv para a tabela acessível completa. Não há labels finais de posse.</desc>')
    write('<rect width="1280" height="530" fill="#f8fafc"/><g font-family="Arial" font-size="13" fill="#172033">')
    write('<text x="24" y="28" font-size="22">Eventos candidatos e qualidade da bola</text>')
    write(f'<text x="24" y="52">Frames locais [{start},{end}) · origem {escape(origin["timeline"])} · círculo: evento; traço: confirmação; triângulo: disponibilidade</text>')
    write('<text x="24" y="100">Origem da bola</text>')
    # At most one source mark per horizontal pixel for a full-match overview.
    # Mixed-source pixels are visibly hatched and are not relabeled as observations.
    bins = [set() for _ in range(980)]
    for source, a, b in runs:
        if b <= start or a >= end: continue
        lo = max(0, int((max(a,start)-start)/(end-start)*980))
        hi = min(980, max(lo+1, __import__('math').ceil((min(b,end)-start)/(end-start)*980)))
        for i in range(lo,hi): bins[i].add(source)
    write('<defs><pattern id="mixed" width="4" height="4" patternUnits="userSpaceOnUse"><rect width="4" height="4" fill="#e2e8f0"/><path d="M0,4 L4,0" stroke="#334155"/></pattern></defs>')
    for i, sources in enumerate(bins):
        if not sources: continue
        color = COLORS[next(iter(sources))] if len(sources)==1 else 'url(#mixed)'
        write(f'<rect x="{260+i}" y="84" width="1" height="20" fill="{color}"><title>{escape(",".join(sorted(sources)))}</title></rect>')
    for name, y in LANES.items():
        write(f'<text x="24" y="{y+5}">{name}</text><path d="M260,{y} H1240" stroke="#cbd5e1"/>')
    for e in events:
        if not start <= e['event_frame'] < end: continue
        y = LANES[e['type']]
        a, b, d = (x(min(end, max(start,e[k]))) for k in ('event_frame','confirmed_at_frame','available_at_frame'))
        color = '#2563eb' if e['team']=='H' else '#ea580c' if e['team']=='A' else '#6d28d9'
        title = escape(f"{e['event_id']} evento={e['event_frame']} confirmação={e['confirmed_at_frame']} disponível={e['available_at_frame']}")
        write(f'<g><title>{title}</title><path d="M{a:.2f},{y} H{d:.2f}" stroke="{color}"/><circle cx="{a:.2f}" cy="{y}" r="4" fill="{color}"/><path d="M{b:.2f},{y-6} V{y+6}" stroke="{color}"/><path d="M{d:.2f},{y-6} l-4,10 h8 Z" fill="{color}"/></g>')
        if e['available_at_frame'] >= end:
            write(f'<text x="1240" y="{y-10}" text-anchor="end" fill="{color}">→ fora do intervalo</text>')
    for i in range(6):
        f = start+(end-start)*i//5
        write(f'<text x="{x(f):.1f}" y="395" text-anchor="middle">{float(absolute(f)):.2f}</text>')
    write('<text x="260" y="419">Frame absoluto na timeline de origem</text>')
    for i,(source,color) in enumerate(COLORS.items()):
        write(f'<rect x="{24+i*245}" y="445" width="12" height="12" fill="{color}"/><text x="{42+i*245}" y="456">{source}</text>')
    write('<text x="24" y="486">H azul · A laranja · sem time roxo · pixels hachurados: múltiplas origens; use recorte para detalhe.</text>')
    write('<text x="24" y="512">Força heurística não é probabilidade. Tabela acessível: events-table.csv.</text></g></svg>')


def event_table(handle, events, meta=None):
    keys = ['event_id','type','team','event_frame','confirmed_at_frame','available_at_frame','parent_event_id','strength','availability_delay_frames','confirmation_delay_frames','absolute_event_frame']
    writer = csv.DictWriter(handle, fieldnames=keys, extrasaction='ignore', lineterminator='\n')
    writer.writeheader()
    for e in events:
        row=dict(e,confirmation_delay_frames=e['confirmed_at_frame']-e['event_frame'],absolute_event_frame=e['event_frame'])
        if meta and 'origin' in meta:
            o=meta['origin']
            row['absolute_event_frame']=float(Fraction(o['first_frame'])+Fraction(e['event_frame'])*Fraction(str(o['fps']))/Fraction(str(meta['fps'])))
        writer.writerow(row)
