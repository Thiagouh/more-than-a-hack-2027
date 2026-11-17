"""Trilha "limpa" da bola: observado / preenchido / estimado / ausente, por frame.

Etapas (todas automáticas, nenhuma correção manual):
  1. Falhas -> ausente: posição muito fora do campo, ou trecho com velocidade fisicamente
     impossível (o tracker "desliza" ~50 m em <1 s quando pula para outro objeto).
  2. Buracos curtos (<= MAX_BURACO_CURTO frames) com as duas pontas observadas: interpolação
     linear -> preenchido.
  3. Buracos longos: estimativa pelo conjunto de jogadores -> estimado (ver estimar.py).

    python -m src.bola.limpar          # gera outputs/bola/bola_limpa.csv e imprime resumo
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.bola.carregar import carregar, em_metros  # noqa: E402
from games import GAME, GAMES  # noqa: E402

# Params
V_MAX = 40.0            # m/s. Chute forte ~30-35 m/s; acima disso entre frames é falha.
DILATAR_FALHA = 2       # frames removidos em volta de cada salto impossível
MARGEM_FORA = 5.0       # m fora da linha ainda aceitos (bola saindo é informação, não erro)
MAX_BURACO_CURTO = 12   # frames (~0,5 s)
FPS = GAMES[GAME]["fps"]

AUSENTE, OBSERVADO, PREENCHIDO, ESTIMADO = 0, 1, 2, 3
NOMES_FONTE = np.array(["ausente", "observado", "preenchido", "estimado"])


def trechos(mascara):
    """(inícios, fins exclusivos) dos trechos True."""
    m = np.r_[False, mascara, False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def marcar_falhas(xy, pitch=GAMES[GAME]["pitch"]):
    """Máscara de frames observados que devem virar ausentes."""
    obs = np.isfinite(xy[:, 0])
    comp, larg = pitch
    fora = obs & ((xy[:, 0] < -MARGEM_FORA) | (xy[:, 0] > comp + MARGEM_FORA)
                  | (xy[:, 1] < -MARGEM_FORA) | (xy[:, 1] > larg + MARGEM_FORA))
    v = np.linalg.norm(np.diff(xy, axis=0), axis=1) * FPS
    salto = np.zeros(len(xy), bool)
    idx = np.flatnonzero(v > V_MAX)
    salto[idx] = True
    salto[idx + 1] = True
    if DILATAR_FALHA:
        salto = np.convolve(salto, np.ones(2 * DILATAR_FALHA + 1), "same") > 0
    return (fora | salto) & obs


def preencher_curtos(xy, fonte):
    xy = xy.copy()
    fonte = fonte.copy()
    ini, fim = trechos(fonte == AUSENTE)
    for a, z in zip(ini, fim):
        if z - a > MAX_BURACO_CURTO or a == 0 or z >= len(xy):
            continue
        t = np.arange(1, z - a + 1) / (z - a + 1)
        xy[a:z] = xy[a - 1] + t[:, None] * (xy[z] - xy[a - 1])
        fonte[a:z] = PREENCHIDO
    return xy, fonte


def tabela_buracos(xy, fonte, pitch=GAMES[GAME]["pitch"]):
    """Uma linha por buraco restante: duração, pontas e se a bola sumiu fora do campo."""
    comp, larg = pitch
    linhas = []
    for a, z in zip(*trechos(fonte == AUSENTE)):
        antes = xy[a - 1] if a > 0 else np.array([np.nan, np.nan])
        depois = xy[z] if z < len(xy) else np.array([np.nan, np.nan])
        saiu = bool(np.isfinite(antes[0]) and (antes[0] < 0 or antes[0] > comp
                                               or antes[1] < 0 or antes[1] > larg))
        linhas.append({"inicio": a, "fim": z, "frames": z - a, "segundos": (z - a) / FPS,
                       "x_antes": antes[0], "y_antes": antes[1],
                       "x_depois": depois[0], "y_depois": depois[1],
                       "salto_m": float(np.linalg.norm(depois - antes)),
                       "sumiu_fora_do_campo": saiu})
    return pd.DataFrame(linhas)


def limpar(dados=None, estimar_longos=True):
    """(por_frame, buracos, falha).

    por_frame: frame, x, y (metros), fonte, buraco_id (-1 fora de buraco; indexa `buracos`).
    buracos:   uma linha por buraco que sobrou após a limpeza (ver tabela_buracos).
    falha:     máscara dos frames observados que foram descartados.
    """
    dados = dados or carregar()
    xy = em_metros(dados["bola"]).astype(np.float64)
    fonte = np.where(np.isfinite(xy[:, 0]), OBSERVADO, AUSENTE).astype(np.int8)

    falha = marcar_falhas(xy)
    xy[falha] = np.nan
    fonte[falha] = AUSENTE

    xy, fonte = preencher_curtos(xy, fonte)

    buraco_id = np.full(len(xy), -1, np.int32)
    for k, (a, z) in enumerate(zip(*trechos(fonte == AUSENTE))):
        buraco_id[a:z] = k
    buracos = tabela_buracos(xy, fonte)

    if estimar_longos:
        from src.bola.estimar import estimar_buracos
        xy, fonte = estimar_buracos(xy, fonte, dados)

    return pd.DataFrame({"frame": np.arange(len(xy)), "x": xy[:, 0], "y": xy[:, 1],
                         "fonte": NOMES_FONTE[fonte], "buraco_id": buraco_id}), buracos, falha


def _resumo(df, falha):
    g = np.zeros(len(df), bool)
    g[11185:90652] = True
    g[110808:190198] = True
    print("falhas removidas (todos os frames): {}".format(int(falha.sum())))
    print("falhas removidas (avaliados):       {}".format(int(falha[g].sum())))
    print("fonte nos frames avaliados:")
    print(df.loc[g, "fonte"].value_counts(normalize=True).map("{:.1%}".format).to_string())


if __name__ == "__main__":
    df, buracos, falha = limpar()
    saida = os.path.join(ROOT, "outputs", "bola")
    os.makedirs(saida, exist_ok=True)
    df.to_csv(os.path.join(saida, "bola_limpa.csv"), index=False, float_format="%.3f")
    buracos.to_csv(os.path.join(saida, "buracos.csv"), index_label="buraco_id",
                   float_format="%.2f")
    _resumo(df, falha)
    print("salvo em outputs/bola/bola_limpa.csv e outputs/bola/buracos.csv")
