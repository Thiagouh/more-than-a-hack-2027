"""Onde colocar a troca de posse dentro de um buraco em que o time antes != time depois?

Estratégias comparadas:
  - fim:      mantém o time de antes até a bola reaparecer (= "herdar posse");
  - início:   o time de depois assume desde o começo do buraco (troca retroativa);
  - meio:     troca no meio do buraco;
  - modelo:   ponto de troca que melhor concorda com a probabilidade do modelo A
              (maximiza soma de P(antes) antes do corte + P(depois) depois do corte).

Avaliação sem gabarito: buracos sintéticos no jogo inteiro (validação cruzada por tempo). A
referência é uma aproximação de posse: time do jogador mais próximo da bola REAL, suavizado por
voto da maioria em ±2 s, como no baseline. É uma referência imperfeita, mas igual para todas as
estratégias.

    python -m src.bola.troca
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.bola.modelo import (SEMENTE, amostras, features, preparar, sortear,  # noqa: E402
                             time_mais_proximo, treinar)

# Params
JANELA_REF = 50     # ±frames do voto da maioria na referência de posse
JANELA_PONTA = 25   # frames observados usados para decidir o time antes/depois do buraco
N_BURACOS = 4000


def maioria(c):
    """1 = H, 0 = A, -1 sem informação."""
    c = c[c >= 0]
    return -1 if len(c) == 0 else int(c.mean() >= 0.5)


def suavizar(c, w):
    h = np.r_[0, np.cumsum(c == 1)]
    a = np.r_[0, np.cumsum(c == 0)]
    i = np.arange(len(c))
    lo, hi = np.maximum(0, i - w), np.minimum(len(c), i + w + 1)
    nh, na = h[hi] - h[lo], a[hi] - a[lo]
    return np.where(nh + na == 0, -1, (nh >= na).astype(int))


def ponto_de_troca(p_depois):
    """Índice k que maximiza sum(1 - p[:k]) + sum(p[k:]). p = P(time de depois) por frame."""
    ganho = np.r_[0, np.cumsum(1 - p_depois)] + np.r_[np.cumsum(p_depois[::-1])[::-1], 0]
    return int(np.argmax(ganho))


def rotular(antes, depois, L, estrategia, p_h=None):
    """Códigos (1 H / 0 A) para os L frames de um buraco."""
    if antes < 0 and depois < 0:
        return np.full(L, -1)
    if antes < 0 or depois < 0 or antes == depois:
        return np.full(L, antes if antes >= 0 else depois)
    k = {"fim": L, "início": 0, "meio": L // 2}.get(estrategia)
    if k is None:
        p_dep = p_h if depois == 1 else 1 - p_h
        k = ponto_de_troca(p_dep)
    return np.r_[np.full(k, antes), np.full(L - k, depois)]


def main():
    _, bola, jog_m, jog_v, eh_h = preparar()
    n = len(bola)
    cod_real = np.full(n, -1)
    ok = np.isfinite(bola[:, 0])
    cod_real[ok] = time_mais_proximo(bola[ok], jog_m[ok], eh_h)
    ref = suavizar(cod_real, JANELA_REF)

    rng = np.random.default_rng(SEMENTE)
    buracos = sortear(bola, rng, N_BURACOS)
    X, y, meta = amostras(bola, jog_m, jog_v, eh_h, buracos)

    estrategias = ["fim", "início", "meio", "modelo"]
    acertos = {e: [] for e in estrategias}
    pesos, trocou = [], []
    for teste in (1, 2):
        modelo = treinar(X[meta.tempo != teste], y[meta.tempo != teste])
        for a, z, tempo in buracos:
            if tempo != teste:
                continue
            L = z - a
            # o buraco está escondido: só o que está fora dele é visível
            antes = maioria(cod_real[max(0, a - JANELA_PONTA):a])
            depois = maioria(cod_real[z:z + JANELA_PONTA])
            verdade = ref[a:z]
            valido = verdade >= 0
            if valido.sum() == 0:
                continue
            frames = np.arange(a, z)
            p_h = modelo.prob_h(features(bola, a, z, frames, jog_m, jog_v, eh_h))
            for e in estrategias:
                pred = rotular(antes, depois, L, e, p_h)
                acertos[e].append((pred[valido] == verdade[valido]).mean())
            pesos.append(valido.sum())
            trocou.append(antes >= 0 and depois >= 0 and antes != depois)

    pesos, trocou = np.array(pesos), np.array(trocou)
    print("buracos sintéticos avaliados: {} ({:.0%} com time antes != depois)\n".format(
        len(pesos), trocou.mean()))
    linhas = []
    for e in estrategias:
        acc = np.array(acertos[e])
        linhas.append({"estratégia": e,
                       "acerto (todos os buracos)": np.average(acc, weights=pesos),
                       "acerto (só buracos com troca)": np.average(acc[trocou], weights=pesos[trocou])})
    print(pd.DataFrame(linhas).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
