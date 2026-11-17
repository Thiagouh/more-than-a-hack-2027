"""Leitor do SkillCorner open data (tracking de transmissão, 10 fps, MIT).

Devolve o mesmo formato que a frente 4 usa (players_xy, ball_xy, team_of_slot, fps) mais a coluna
``possession`` do próprio tracking como rótulo: 0=casa, 1=visitante, -1=ninguém.
Coordenadas centradas no campo; os atributos usados depois são só distâncias relativas.
"""
import json

import numpy as np

FPS = 10.0


def load_match(match_json, tracking_jsonl):
    with open(match_json, encoding="utf-8") as handle:
        match = json.load(handle)
    team_of = {p["id"]: (0 if p["team_id"] == match["home_team"]["id"] else 1) for p in match["players"]}
    slots = sorted(team_of)
    index = {pid: i for i, pid in enumerate(slots)}
    rows = []
    with open(tracking_jsonl, encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    first = min(r["frame"] for r in rows)
    n = max(r["frame"] for r in rows) - first + 1
    players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    poss = np.full(n, -1, dtype=np.int8)
    period = np.zeros(n, dtype=np.int8)
    code = {"home team": 0, "away team": 1}
    for r in rows:
        i = r["frame"] - first
        period[i] = r.get("period") or 0
        b = r.get("ball_data") or {}
        if b.get("x") is not None and b.get("is_detected", True):
            ball[i] = (b["x"], b["y"])
        pos = r.get("possession") or {}
        poss[i] = code.get(pos.get("group"), -1)
        for p in r.get("player_data") or []:
            j = index.get(p["player_id"])
            if j is not None:
                players[i, j] = (p["x"], p["y"])
    blocks = []
    for per in (1, 2):
        idx = np.flatnonzero(period == per)
        if len(idx):
            blocks.append((int(idx[0]), int(idx[-1])))
    return dict(players_xy=players, ball_xy=ball, team_of_slot=np.array([team_of[s] for s in slots]),
                fps=FPS, possession=poss, period=period, blocks=blocks, match_id=match["id"])


def hold_labels(poss, blocks):
    """Posse sem lacunas: onde ninguém tem a bola vale o último time (o início recebe o primeiro).
    Fora dos blocos: -1. É o que o desafio chama de 'bola solta mantém a posse'."""
    out = np.full(len(poss), -1, dtype=np.int8)
    for first, last in blocks:
        seg = poss[first:last + 1].copy()
        known = np.flatnonzero(seg >= 0)
        if not len(known):
            continue
        idx = np.maximum.accumulate(np.where(seg >= 0, np.arange(len(seg)), -1))
        seg = np.where(idx >= 0, seg[np.maximum(idx, 0)], seg[known[0]])
        out[first:last + 1] = seg
    return out
