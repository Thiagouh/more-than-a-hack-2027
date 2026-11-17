"""Estimativa da bola nos buracos longos a partir do conjunto de jogadores.

Ideia (v1): começa de um "palpite" temporal (interpolação linear entre as pontas do buraco, ou a
última posição conhecida se só houver uma ponta) e desloca o palpite para onde os jogadores estão
concentrados ao redor dele (mean-shift com núcleo gaussiano). Jogadores convergem para a bola;
o palpite temporal evita que a estimativa pule para outro aglomerado.

Os parâmetros são escolhidos por src/bola/avaliar_estimador.py, que apaga trechos onde a bola foi
observada e mede o erro da reconstrução. Nenhum gabarito é usado.
"""

import numpy as np

from src.bola.carregar import em_metros

# Params
SIGMA_M = 8.0        # raio do núcleo gaussiano em metros
ITERACOES = 3        # passos de mean-shift
PESO_PALPITE = 0.5   # 0 = só jogadores, 1 = só interpolação (escolhido em avaliar_estimador)

AUSENTE, OBSERVADO, PREENCHIDO, ESTIMADO = 0, 1, 2, 3


def palpite_temporal(xy, a, z):
    """Interpolação linear entre xy[a-1] e xy[z]; mantém a ponta disponível se só houver uma."""
    n = len(xy)
    antes = xy[a - 1] if a > 0 and np.isfinite(xy[a - 1, 0]) else None
    depois = xy[z] if z < n and np.isfinite(xy[z, 0]) else None
    if antes is None and depois is None:
        return None
    if antes is None:
        return np.repeat(depois[None], z - a, 0)
    if depois is None:
        return np.repeat(antes[None], z - a, 0)
    t = np.arange(1, z - a + 1) / (z - a + 1)
    return antes + t[:, None] * (depois - antes)


def mean_shift(palpite, jogadores_xy, sigma=SIGMA_M, iteracoes=ITERACOES, peso=PESO_PALPITE):
    """palpite (L, 2); jogadores_xy (L, S, 2) em metros com NaN onde o slot não existe."""
    valido = np.isfinite(jogadores_xy[..., 0])
    P = np.where(valido[..., None], jogadores_xy, 0.0)
    p = palpite.copy()
    for _ in range(iteracoes):
        d2 = ((P - p[:, None, :]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * sigma ** 2)) * valido
        s = w.sum(1)
        centro = (w[..., None] * P).sum(1) / np.maximum(s, 1e-9)[:, None]
        centro[s < 1e-6] = p[s < 1e-6]
        p = centro
    return peso * palpite + (1 - peso) * p


def estimar_trecho(xy, a, z, jogadores_m, **params):
    palpite = palpite_temporal(xy, a, z)
    if palpite is None:
        return None
    return mean_shift(palpite, jogadores_m[a:z], **params)


def estimar_buracos(xy, fonte, dados, **params):
    """Preenche todo buraco ainda AUSENTE com a estimativa pelos jogadores."""
    from src.bola.limpar import trechos
    xy = xy.copy()
    fonte = fonte.copy()
    jogadores_m = em_metros(dados["jogadores"][..., :2]).astype(np.float64)
    for a, z in zip(*trechos(fonte == AUSENTE)):
        est = estimar_trecho(xy, a, z, jogadores_m, **params)
        if est is not None:
            xy[a:z] = est
            fonte[a:z] = ESTIMADO
    return xy, fonte
