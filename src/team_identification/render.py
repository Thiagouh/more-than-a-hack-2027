"""Draw precisely the identity observations supplied to possession."""
from collections import Counter
from src.team_possession.render_observations import render_status

COLORS = {'H': '#2563eb', 'A': '#e85d32', None: '#aeb6bf'}


def draw_identities(draw, players, frame, pitch, project, font):
    counts = Counter()
    for point in players:
        status = render_status(point, frame, pitch)
        counts[status] += 1
        if status not in ('drawn', 'drawn_with_quality_warning'):
            continue
        x, y = project([point['x_m'], point['y_m']])
        team = point.get('team')
        counts[team or 'unknown'] += 1
        color = COLORS[team]
        label = point.get('original_track_id') or point['track_id'].removeprefix('person-')
        if point.get('role') == 'goalkeeper':
            label = 'G-'+str(team)+' '+label
        elif team is None:
            label = '? '+label
        draw.ellipse((x-6, y-6, x+6, y+6), fill=color, outline='white')
        draw.text((x+8, y-9), label, font=font, fill=color, stroke_width=1, stroke_fill='#173d32')
    return dict(counts)
