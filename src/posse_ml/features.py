"""Features por frame para prever o time em posse. Invariantes ao fps (offsets em segundos),
ao lado do campo (distâncias relativas) e ao uso de tracking de transmissão ou tático.

Todas offline: usam frames passados e futuros, como a máquina de estados da frente 4.
"""
import numpy as np
import pandas as pd

from src.frente4.features import compute_features

OFFSETS_S = (-3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0)
CONTEXT_S = 12.0          # até quando procurar a última/próxima observação


def _shift(a, k):
    out = np.full_like(a, np.nan, dtype=np.float32)
    if k > 0:
        out[k:] = a[:-k]
    elif k < 0:
        out[:k] = a[-k:]
    else:
        out[:] = a
    return out


def _age_and_value(signed, limit):
    """Último valor observado antes (e sua idade em frames), e o primeiro depois."""
    s = pd.Series(signed)
    seen = s.notna().to_numpy()
    idx = np.where(seen, np.arange(len(s)), np.nan)
    last_idx = pd.Series(idx).ffill().to_numpy()
    next_idx = pd.Series(idx).bfill().to_numpy()
    pos = np.arange(len(s))
    prev_val = s.ffill(limit=limit).to_numpy()
    next_val = s.bfill(limit=limit).to_numpy()
    return prev_val, pos - last_idx, next_val, next_idx - pos


def build(tr):
    """Matriz [n, d] float32 e lista de nomes."""
    fps = float(tr["fps"])
    f = compute_features(tr)
    obs = f["ball_observed"]
    dh = np.where(np.isfinite(f["dist_h"]), f["dist_h"], np.nan)
    da = np.where(np.isfinite(f["dist_a"]), f["dist_a"], np.nan)
    signed = np.clip(da - dh, -25, 25).astype(np.float32)          # >0: casa mais perto
    signed[~obs] = np.nan
    near = np.clip(np.fmin(dh, da), 0, 40).astype(np.float32)
    near[~obs] = np.nan
    speed = np.clip(f["ball_speed"], 0, 40).astype(np.float32)
    crowd = (f["n_h_near"].astype(np.float32) - f["n_a_near"].astype(np.float32))
    crowd[~obs] = np.nan
    base = {"obs": obs.astype(np.float32), "signed": signed, "near": near, "speed": speed, "crowd": crowd}

    cols, names = [], []
    for off in OFFSETS_S:
        k = int(round(off * fps))
        for name, arr in base.items():
            cols.append(_shift(arr, k)); names.append("{}@{:+.1f}s".format(name, off))
    limit = int(CONTEXT_S * fps)
    pv, pa, nv, na = _age_and_value(signed, limit)
    cols += [pv, (pa / fps).astype(np.float32), nv, (na / fps).astype(np.float32)]
    names += ["signed_prev", "age_prev_s", "signed_next", "age_next_s"]
    for w_s in (1.0, 3.0):
        w = max(1, int(round(w_s * fps)))
        m = pd.Series(signed).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
        o = pd.Series(obs.astype(np.float32)).rolling(2 * w + 1, center=True, min_periods=1).mean().to_numpy()
        cols += [m.astype(np.float32), o.astype(np.float32)]
        names += ["signed_mean_{:g}s".format(w_s), "obs_frac_{:g}s".format(w_s)]
    return np.column_stack(cols).astype(np.float32), names
