"""Baseline da organização refeito com código nosso, vetorizado sobre o cache do ATD.

Mesmo algoritmo de starter/baseline.py (jogador mais próximo da bola, D onde a bola falta, voto
majoritário centrado com empate para o mandante) para que os quatro CSVs saiam byte a byte iguais
aos de evaluation/results/. Isso confere o cache, a conversão de coordenadas e o mapa de times
antes de qualquer regra nova. Nota pública esperada de baseline_smooth50: spellF1 0,3256.
"""
import filecmp
import os

import numpy as np

from .features import HOME, AWAY, NOBODY

VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25),
            ("baseline_smooth50", 50), ("baseline_smooth125", 125))


def nearest_player_codes(tr):
    """int8[n]: time do jogador mais próximo da bola (0 H, 1 A), -1 sem bola ou sem jogador."""
    from .features import compute_features
    return compute_features(tr, control_radius_m=np.inf)["nearest_team"]


def smooth(codes, window):
    """Voto majoritário em janela centrada de +/- window frames; D onde ninguém tem a bola.
    Reimplementação com somas acumuladas; empate -> H, igual ao starter."""
    codes = np.asarray(codes)
    if window <= 0:
        return np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    lo = np.maximum(0, idx - window)
    hi = np.minimum(len(codes), idx + window + 1)
    home = home_c[hi] - home_c[lo]
    away = away_c[hi] - away_c[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def write_submission(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def reproduce(tr, out_dir, reference_dir=None):
    """Escreve as quatro variantes em out_dir e compara com reference_dir (se existir).
    Devolve {nome: (caminho, idêntico ou None)}."""
    codes = nearest_player_codes(tr)
    result = {}
    for name, window in VARIANTS:
        path = write_submission(os.path.join(out_dir, name + ".csv"), smooth(codes, window))
        identical = None
        if reference_dir:
            ref = os.path.join(reference_dir, name + ".csv")
            if os.path.isfile(ref):
                identical = filecmp.cmp(path, ref, shallow=False)
        result[name] = (path, identical)
    return result
