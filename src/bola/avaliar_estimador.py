"""Mede o erro do estimador de bola sem gabarito: apaga trechos onde a bola FOI observada,
reconstrói e compara com a posição real.

Os buracos sintéticos têm as durações dos buracos reais (5-30 s) e só caem em frames avaliados
cujas pontas e interior estão observados. Semente fixa, para que resultados sejam comparáveis.

    python -m src.bola.avaliar_estimador
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.estimar import mean_shift, palpite_temporal  # noqa: E402
from src.bola.limpar import marcar_falhas  # noqa: E402

# Params
SEMENTE = 2027
N_BURACOS = 400
DURACOES = (125, 250, 500, 750)   # frames: 5, 10, 20, 30 s
RAIO_ACERTO = (5.0, 10.0)         # m


def sortear_buracos(obs, avaliado, duracao, n, rng):
    ok = obs & avaliado
    # janela [a-1, a+duracao] toda observada
    cs = np.r_[0, np.cumsum(ok)]
    larg = duracao + 2
    inicios = np.flatnonzero(cs[larg:] - cs[:-larg] == larg) + 1
    if len(inicios) == 0:
        return []
    return rng.choice(inicios, size=min(n, len(inicios)), replace=False)


def metricas(erro):
    return "med {:5.1f} m  p90 {:5.1f} m  <5m {:4.0%}  <10m {:4.0%}".format(
        np.median(erro), np.percentile(erro, 90),
        (erro < RAIO_ACERTO[0]).mean(), (erro < RAIO_ACERTO[1]).mean())


def main():
    dados = carregar()
    xy = em_metros(dados["bola"]).astype(np.float64)
    xy[marcar_falhas(xy)] = np.nan
    obs = np.isfinite(xy[:, 0])
    jog = em_metros(dados["jogadores"][..., :2]).astype(np.float64)
    avaliado = np.zeros(len(xy), bool)
    avaliado[11185:90652] = True
    avaliado[110808:190198] = True
    rng = np.random.default_rng(SEMENTE)

    configs = {
        "interpolação": None,
        "jogadores σ=5 p=0": dict(sigma=5, peso=0.0),
        "jogadores σ=8 p=0": dict(sigma=8, peso=0.0),
        "jogadores σ=8 p=.3": dict(sigma=8, peso=0.3),
        "jogadores σ=12 p=.3": dict(sigma=12, peso=0.3),
        "jogadores σ=8 p=.5": dict(sigma=8, peso=0.5),
    }
    for dur in DURACOES:
        inicios = sortear_buracos(obs, avaliado, dur, N_BURACOS // len(DURACOES), rng)
        print("\nburacos de {} frames ({:.0f} s), n={}".format(dur, dur / 25, len(inicios)))
        for nome, params in configs.items():
            erros = []
            for a in inicios:
                z = a + dur
                palpite = palpite_temporal(xy, a, z)
                est = palpite if params is None else mean_shift(palpite, jog[a:z], **params)
                erros.append(np.linalg.norm(est - xy[a:z], axis=1))
            print("  {:<22} {}".format(nome, metricas(np.concatenate(erros))))


if __name__ == "__main__":
    main()
