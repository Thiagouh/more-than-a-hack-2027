"""Synthetic goalkeeper crops -> team resolution -> causal control events.

Run from the repository root: python -m scripts.demo_uniforms
"""
import json
from pathlib import Path

import numpy as np
from src.ball_events import detect
from src.ball_events.demo import metadata, point
from src.ball_events.uniforms import UniformClassifier, resolve_player


def main():
    def crop(color):
        return np.full((100, 40, 3), color, dtype=np.uint8)
    patterns = [dict(team=t, role=r, crops=[crop(color)]) for t, r, color in (
        ('H', 'player', (220, 20, 20)), ('A', 'player', (20, 20, 220)),
        ('H', 'goalkeeper', (20, 220, 20)), ('A', 'goalkeeper', (220, 220, 20)),
        (None, 'referee', (15, 15, 15)))]
    report = {'synthetic': True, 'real_goalkeeper_uniforms_verified': False, 'teams': {}}
    for team, color in [('H', (20, 220, 20)), ('A', (220, 220, 20))]:
        model = UniformClassifier('synthetic', patterns)
        frames, diagnostics = [], []
        for frame in range(10):
            visual = model.classify(crop(color), 'keeper', frame)
            player, diagnostic = resolve_player(point(frame, 20, track_id='keeper'), visual)
            diagnostics.append(diagnostic)
            players = [point(frame, 26, track_id='opponent', team='A' if team == 'H' else 'H')]
            if player is not None:
                players.append(player)
            frames.append(dict(frame=frame, ball=point(frame, 20), players=players))
        _, events, quality = detect(metadata(10), frames)
        report['teams'][team] = dict(diagnostics=diagnostics, events=events, quality=quality)
    out = Path('outputs/demo_uniforms')
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(out/'report.json')


if __name__ == '__main__':
    main()
