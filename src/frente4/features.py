"""Features por frame a partir dos arrays do tracking. Tudo vetorizado em NumPy.

Nada aqui decide posse; só mede. Distâncias em metros, velocidades em m/s.
"""
import warnings

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1


def speed_from_positions(xy, fps):
    """|dxy| * fps entre frames consecutivos observados; NaN quando um dos dois falta."""
    v = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    v[1:] = np.linalg.norm(xy[1:] - xy[:-1], axis=-1) * fps
    return v


def compute_features(tr, control_radius_m=3.0, near_radius_m=5.0):
    """Devolve dict de arrays com uma entrada por frame.

    ball_observed  bool      bola vista pelo tracking
    ball_x/ball_y  float     metros (NaN se ausente)
    ball_speed     float     m/s calculada das posições (NaN sem par de observações)
    dist_h/dist_a  float     distância do jogador mais próximo de cada time (inf se nenhum)
    nearest_team   int8      0=H, 1=A, -1 ninguém (bola ausente ou sem jogador observado)
    nearest_dist   float     distância desse jogador (inf se -1)
    margin         float     |dist_h - dist_a| (quanto maior, mais clara a posse)
    cand           int8      nearest_team se nearest_dist <= control_radius_m, senão -1
    n_h_near/n_a_near int16  jogadores de cada time a <= near_radius_m da bola
    players_observed int16   jogadores observados no frame
    players_mean_speed float m/s média dos jogadores observados (NaN se nenhum)
    """
    players = tr["players_xy"]          # [n, slots, 2]
    ball = tr["ball_xy"]                # [n, 2]
    team = np.asarray(tr["team_of_slot"])
    fps = float(tr["fps"])
    n = ball.shape[0]

    ball_observed = ~np.isnan(ball[:, 0])
    d = np.linalg.norm(players - ball[:, None, :], axis=-1)   # NaN onde falta alguém
    d = np.where(np.isnan(d), np.inf, d).astype(np.float32)

    is_h = team == HOME
    dist_h = d[:, is_h].min(axis=1) if is_h.any() else np.full(n, np.inf, np.float32)
    dist_a = d[:, ~is_h].min(axis=1) if (~is_h).any() else np.full(n, np.inf, np.float32)

    nearest_team = np.full(n, NOBODY, dtype=np.int8)
    nearest_dist = np.minimum(dist_h, dist_a)
    seen = np.isfinite(nearest_dist)
    # Empate vai para o mandante, como no baseline da organização.
    nearest_team[seen] = np.where(dist_h[seen] <= dist_a[seen], HOME, AWAY)
    margin = np.where(seen, np.abs(dist_h - dist_a), np.nan).astype(np.float32)

    cand = np.where(seen & (nearest_dist <= control_radius_m), nearest_team, NOBODY).astype(np.int8)

    near = d <= near_radius_m
    n_h_near = near[:, is_h].sum(axis=1).astype(np.int16)
    n_a_near = near[:, ~is_h].sum(axis=1).astype(np.int16)

    pspeed = speed_from_positions(players, fps)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        players_mean_speed = np.nanmean(pspeed, axis=1).astype(np.float32)
    players_observed = (~np.isnan(players[:, :, 0])).sum(axis=1).astype(np.int16)

    return dict(
        ball_observed=ball_observed,
        ball_x=ball[:, 0], ball_y=ball[:, 1],
        ball_speed=speed_from_positions(ball, fps),
        dist_h=dist_h, dist_a=dist_a,
        nearest_team=nearest_team, nearest_dist=nearest_dist, margin=margin,
        cand=cand, n_h_near=n_h_near, n_a_near=n_a_near,
        players_observed=players_observed, players_mean_speed=players_mean_speed,
    )
