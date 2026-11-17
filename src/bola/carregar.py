"""Lê o ATD bruto (txt + xml) e guarda um cache NumPy.

Formato de cada linha do txt: ``frame_id:slot1;slot2;...;slot46:bola``, cada item ``x,y,s``
(x, y normalizados 0-1). frame_id começa em 1; o índice do array é o frame do vídeo (0..N-1).

Este carregador é provisório para a Frente 2 andar sem esperar a Frente 1. Quando a função
oficial de dados existir, ``carregar()`` deve passar a usá-la mantendo a mesma saída.

    python -m src.bola.carregar        # gera outputs/cache/atd_tactical.npz e imprime resumo
"""

import io
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402

N_SLOTS = 46
CACHE_DIR = os.path.join(ROOT, "outputs", "cache")


def _ler_txt(caminho):
    with open(caminho) as f:
        texto = f.read().replace(":", ",").replace(";", ",")
    tabela = pd.read_csv(io.StringIO(texto), header=None, dtype=np.float32,
                         na_values=["NaN"], engine="c").to_numpy()
    assert tabela.shape[1] == 1 + 3 * N_SLOTS + 3, tabela.shape
    frame_id = tabela[:, 0].astype(np.int64)
    assert (frame_id == np.arange(1, len(frame_id) + 1)).all(), "frames não contíguos a partir de 1"
    jogadores = tabela[:, 1:1 + 3 * N_SLOTS].reshape(-1, N_SLOTS, 3)
    bola = tabela[:, 1 + 3 * N_SLOTS:]
    return jogadores, bola


def _ler_times(caminho_xml):
    """'H' ou 'A' por slot, a partir do teamId de cada <Player> e do nome do time."""
    xml = open(caminho_xml).read()
    nomes = dict(re.findall(r'<Team id="([^"]+)">\s*<Name>([^<]+)</Name>', xml))
    lado = {}
    for team_id, nome in nomes.items():
        lado[team_id] = "H" if "barcelona" in nome.lower() else "A"
    slots = re.findall(r'<Player id="(\d+)" teamId="([^"]+)"', xml)
    times = np.empty(N_SLOTS, dtype="<U1")
    for pid, team_id in slots:
        times[int(pid) - 1] = lado[team_id]
    return times


def carregar(source="tactical", usar_cache=True):
    """dict com:
    jogadores (N, 46, 3) float32  x, y normalizados e velocidade do slot
    bola      (N, 3)     float32  x, y normalizados e velocidade
    times     (46,)      'H'/'A' por slot
    pitch     (comprimento, largura) em metros
    """
    cache = os.path.join(CACHE_DIR, "atd_{}.npz".format(source))
    if usar_cache and os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files} | {"pitch": GAMES[GAME]["pitch"]}
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    jogadores, bola = _ler_txt(raw)
    times = _ler_times(meta)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, jogadores=jogadores, bola=bola, times=times)
    return {"jogadores": jogadores, "bola": bola, "times": times, "pitch": GAMES[GAME]["pitch"]}


def em_metros(xy, pitch=GAMES[GAME]["pitch"]):
    """Converte (..., 2) normalizado para metros."""
    return xy[..., :2] * np.asarray(pitch, dtype=np.float32)


if __name__ == "__main__":
    d = carregar(usar_cache=False)
    b = d["bola"]
    print("frames:", len(b))
    print("slots por time:", dict(zip(*np.unique(d["times"], return_counts=True))))
    print("bola com x,y: {:.1%}".format(np.isfinite(b[:, 0]).mean()))
    print("bola com velocidade: {:.1%}".format(np.isfinite(b[:, 2]).mean()))
