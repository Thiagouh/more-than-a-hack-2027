"""Leitura do ATD tático uma única vez, com cache NumPy.

Formato bruto (uma linha por frame): ``<frame_id>:<x,y,v>;<x,y,v>;...:<x,y,v>`` com 46 slots de
jogador e a bola no fim; coordenadas normalizadas 0-1, ``NaN`` quando ausente. ``frame_id`` começa
em 1 e a submissão em 0, logo ``video_frame = frame_id - 1``.

Devolve arrays em metros (campo 105,3 x 68,0 de ``games.py``), com NaN onde não há observação.
Esta é a entrega provisória da frente 1; se a frente 1 publicar o cache dela, basta trocar
``load_tracking`` mantendo as mesmas chaves.
"""
import hashlib
import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

HOME, AWAY = 0, 1
CACHE_VERSION = "frente4-atd-v1"


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def team_of_each_slot(meta_path):
    """int8[slots]: 0 para o primeiro time do XML (mandante), 1 para o segundo."""
    root = ET.parse(meta_path).getroot()
    teams = [e for e in root.iter() if _local(e.tag) == "Team"]
    if len(teams) != 2:
        raise ValueError("{}: esperado 2 times, achei {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [e.get("teamId") for e in root.iter() if _local(e.tag) == "Player"]
    if not slots:
        raise ValueError("{}: nenhum Player no XML".format(meta_path))
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    out = []
    for team in (e for e in root.iter() if _local(e.tag) == "Team"):
        name = next((c.text for c in team if _local(c.tag) == "Name"), None)
        out.append((name or "").strip())
    return tuple(out)


def _cache_key(paths):
    h = hashlib.sha1(CACHE_VERSION.encode())
    for p in paths:
        st = os.stat(p)
        h.update("{}|{}|{}".format(os.path.basename(p), st.st_size, int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def parse_raw(raw_path, n_slots, n_frames, pitch):
    """Lê o TXT inteiro de uma vez. Dezenas de segundos na primeira vez; depois vem do cache."""
    import pandas as pd

    with open(raw_path, encoding="utf-8") as handle:
        text = handle.read()
    text = text.replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64).values
    expected_cols = 1 + 3 * n_slots + 3
    if table.shape[1] != expected_cols:
        raise ValueError("ATD: {} colunas, esperava {} (slots={})".format(
            table.shape[1], expected_cols, n_slots))
    frame_ids = table[:, 0].astype(np.int64)
    if len(frame_ids) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
        raise ValueError("ATD: frame_id precisa ser exatamente 1..{} em ordem".format(n_frames))
    length, width = pitch
    players = table[:, 1:1 + 3 * n_slots].reshape(n_frames, n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    players_xy = players[:, :, :2].astype(np.float32)
    players_xy[:, :, 0] *= length
    players_xy[:, :, 1] *= width
    ball_xy = ball[:, :2].astype(np.float32)
    ball_xy[:, 0] *= length
    ball_xy[:, 1] *= width
    return dict(players_xy=players_xy,
                players_speed_atd=players[:, :, 2].astype(np.float32),
                ball_xy=ball_xy,
                ball_speed_atd=ball[:, 2].astype(np.float32))


def load_tracking(data_dir=None, cache_dir=None, game=GAME):
    """dict com players_xy [n,46,2], ball_xy [n,2] (metros, NaN=ausente), team_of_slot [46],
    players_speed_atd, ball_speed_atd, n_frames, fps, pitch, teams, cache_path."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    cache_dir = cache_dir or os.path.join(ROOT, "outputs", "cache")
    cache_path = os.path.join(cache_dir, "atd_tactical_{}.npz".format(_cache_key([meta_path, raw_path])))

    team_of_slot = team_of_each_slot(meta_path)
    if os.path.isfile(cache_path):
        with np.load(cache_path) as z:
            arrays = {k: z[k] for k in z.files}
    else:
        arrays = parse_raw(raw_path, len(team_of_slot), n_frames, pitch)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache_path, **arrays)
    arrays.update(team_of_slot=team_of_slot, n_frames=n_frames, fps=GAMES[game]["fps"],
                  pitch=pitch, teams=team_names(meta_path), cache_path=cache_path)
    return arrays
