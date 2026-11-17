"""Cola das partes: tracking -> features -> bola morta -> máquina de estados -> CSV.

``run()`` é o comando único que regenera a submissão. ``from_csv()`` roda a mesma máquina de
estados sobre um CSV de candidatos (por exemplo o baseline_raw da organização), útil sem os dados
da partida ou para testar só a lógica de posse.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks  # noqa: E402

from . import deadball
from .baseline import write_submission
from .features import compute_features, NOBODY
from .fsm import Params, possession_labels, codes_to_labels, labels_to_codes, summarize, CODE_H, CODE_A

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "frente4_v1.json")
KICKOFFS = {1: 11185, 2: 113742}   # docs/DATA.md e docs/FAQ.md; starter/load_smart_tagging.py


def load_config(path=None):
    cfg = {}
    if path is None and os.path.isfile(DEFAULT_CONFIG):
        path = DEFAULT_CONFIG
    if path:
        with open(path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    return cfg


def graded_blocks(game=GAME):
    return [(first, last) for _, first, last in half_video_blocks(game)]


STRENGTH = {"low": 0, "medium": 1, "high": 2}


def events_to_team_array(events, n_frames, hold_frames=12, min_strength="medium"):
    """Eventos v2 da frente 5 -> int8[n] com o time de cada aquisição de controle confirmada,
    mantido por hold_frames a partir de event_frame. -1 onde não há evento.
    Só entram aquisições com força >= min_strength (low/medium/high, ordinal)."""
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    code = {"H": 0, "A": 1}
    floor = STRENGTH.get(min_strength, 0)
    for e in events:
        if e.get("type") != "possible_control_acquisition" or e.get("team") not in code:
            continue
        if STRENGTH.get(e.get("strength", "high"), 2) < floor:
            continue
        start = int(e["event_frame"])
        out[max(0, start):min(n_frames, start + hold_frames)] = code[e["team"]]
    return out


def read_events_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidates(feat, cfg):
    """Candidato geométrico: mais próximo dentro do raio, mas ninguém quando

    - a disputa é apertada (margem menor que contested_margin_m): bola disputada fica com o
      último time, como manda docs/CHALLENGE.md;
    - a bola corre mais que max_control_speed_mps: uma bola em voo ou num passe longo não está
      sob controle de quem por acaso está mais perto dela.
    """
    cand = feat["cand"].copy()
    margin = cfg.get("contested_margin_m", 0.0)
    if margin > 0:
        contested = np.nan_to_num(feat["margin"], nan=np.inf) < margin
        cand[contested] = NOBODY
    max_speed = cfg.get("max_control_speed_mps", 0.0)
    if max_speed > 0:
        flying = np.nan_to_num(feat["ball_speed"], nan=0.0) > max_speed
        cand[flying] = NOBODY
    return cand


def aplicar_sinal_bola(tr, sinal):
    """Integra o sinal da frente 2 (src/bola/entrega.py).

    - ball_xy passa a conter só as posições ``observado`` do sinal: o tracker "desliza" 30-50 m
      em menos de 1 s e a frente 2 já zerou esses frames; assim distâncias e velocidade não veem
      o deslize. Posições ``preenchido``/``estimado`` NÃO entram: estimativa não é observação.
    - devolve bool[n] com o primeiro frame de cada buraco que começou com a bola fora das
      linhas (abre D). Não troca posse: o estado é mantido nos buracos pela máquina de estados.
    """
    n = tr["n_frames"]
    if len(sinal) != n:
        raise ValueError("sinal_bola tem {} linhas; o tracking tem {}".format(len(sinal), n))
    if not np.array_equal(sinal["frame"].to_numpy(), np.arange(n)):
        raise ValueError("sinal_bola: coluna frame deve ser 0..n-1")
    tr = dict(tr)
    obs = (sinal["fonte"].to_numpy() == "observado")
    ball = np.full((n, 2), np.nan, dtype=tr["ball_xy"].dtype)
    ball[obs, 0] = sinal["x"].to_numpy()[obs]
    ball[obs, 1] = sinal["y"].to_numpy()[obs]
    tr["ball_xy"] = ball
    saiu = sinal["saiu_campo"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
    bid = sinal["buraco_id"].to_numpy()
    starts = np.zeros(n, dtype=bool)
    first = np.ones(n, dtype=bool)
    first[1:] = bid[1:] != bid[:-1]
    starts[(bid >= 0) & first & saiu] = True
    return tr, starts


def ler_deadball_externo(path, n):
    """CSV do detector de bola morta (dead_ball_features.csv) -> bool[n] com o início (retroativo)
    de cada D confirmado. Só evidência; quem decide é a máquina de estados."""
    import pandas as pd
    t = pd.read_csv(path, usecols=["frame", "dead_ball_transition", "dead_ball_start_frame"])
    if len(t) != n or not np.array_equal(t["frame"].to_numpy(), np.arange(n)):
        raise ValueError("deadball externo: precisa ter uma linha por frame 0..{}".format(n - 1))
    ok = t["dead_ball_transition"].fillna("") == "dead_ball_confirmed"
    starts = np.zeros(n, dtype=bool)
    idx = t.loc[ok, "dead_ball_start_frame"].dropna().astype(int).to_numpy()
    starts[idx[(idx >= 0) & (idx < n)]] = True
    return starts


def build_labels(tr, tags=(), cfg=None, events=None, game=GAME, sinal_bola=None, ext_starts=None):
    cfg = cfg or {}
    hole_out = None
    if sinal_bola is not None:
        tr, hole_out = aplicar_sinal_bola(tr, sinal_bola)
    feat = compute_features(tr, control_radius_m=cfg.get("control_radius_m", 3.0),
                            near_radius_m=cfg.get("near_radius_m", 5.0))
    blocks = graded_blocks(game)
    dead = deadball.dead_ball_evidence(feat, blocks, KICKOFFS, GAMES[game]["pitch"], tags,
                                       {k: v for k, v in cfg.items() if k in deadball.DEFAULTS},
                                       hole_out_starts=hole_out, ext_starts=ext_starts)
    event_team = None
    if events:
        event_team = events_to_team_array(events, tr["n_frames"], cfg.get("event_hold_frames", 12),
                                          cfg.get("event_min_strength", "medium"))
    codes = possession_labels(candidates(feat, cfg), dead["dead"], blocks, Params.from_dict(cfg),
                              hint_team=dead["hint_team"], event_team=event_team)
    diag = dict(summary=summarize(codes, blocks, tr["fps"]),
                dead_reasons={deadball.REASON_NAMES[k]: int(np.sum(dead["reason"] == k))
                              for k in deadball.REASON_NAMES},
                restarts=len(dead["restarts"]),
                ball_observed_fraction=float(feat["ball_observed"].mean()),
                candidate_fraction=float(np.mean(candidates(feat, cfg) >= 0)),
                params=Params.from_dict(cfg).to_dict(), dead_config=dead["config"])
    return codes_to_labels(codes), diag


def _ler_sinal(path):
    import pandas as pd
    return pd.read_csv(path, keep_default_na=False, na_values=[""],
                       dtype={"time_proximo": str, "time_antes": str, "time_depois": str,
                              "posse_buraco": str, "posse_buraco_modelo": str, "saiu_campo": str})


def run(out, data_dir=None, config_path=None, events_path=None, cache_dir=None, game=GAME, bola_path=None,
        deadball_path=None):
    from .atd import load_tracking
    from starter.load_smart_tagging import load as load_tags

    cfg = load_config(config_path)
    tr = load_tracking(data_dir, cache_dir, game)
    tags = load_tags(game, data_dir) if cfg.get("use_set_piece_tags", True) else []
    events = read_events_jsonl(events_path) if events_path else None
    sinal = None
    if bola_path:
        from src.bola.entrega import carregar_sinal
        sinal = carregar_sinal() if bola_path == "auto" else _ler_sinal(bola_path)
    ext = ler_deadball_externo(deadball_path, tr["n_frames"]) if deadball_path else None
    labels, diag = build_labels(tr, tags, cfg, events, game, sinal_bola=sinal, ext_starts=ext)
    path = write_submission(out, labels)
    diag.update(teams=dict(H=tr["teams"][0], A=tr["teams"][1]), cache=tr["cache_path"],
                config=config_path or DEFAULT_CONFIG, events=events_path, bola=bola_path, deadball=deadball_path)
    with open(os.path.splitext(out)[0] + ".diagnostico.json", "w", encoding="utf-8") as handle:
        json.dump(diag, handle, indent=2, ensure_ascii=False)
    return path, diag


def from_csv(csv_path, out, config_path=None, game=GAME):
    """Candidatos lidos de um CSV frame,label (H/A = candidato, D/X = ninguém); sem tracking só
    existe a regra de D antes do kickoff. Serve para exercitar a máquina de estados."""
    import pandas as pd

    cfg = load_config(config_path)
    table = pd.read_csv(csv_path, dtype={"label": "object"})
    codes_in = labels_to_codes(table["label"].values)
    cand = np.where((codes_in == CODE_H) | (codes_in == CODE_A), codes_in, NOBODY).astype(np.int8)
    n = len(cand)
    blocks = graded_blocks(game)
    dead = np.zeros(n, dtype=bool)
    for half, (first, last) in enumerate(blocks, start=1):
        kickoff = KICKOFFS.get(half, first)
        dead[first:kickoff] = True
    codes = possession_labels(cand, dead, blocks, Params.from_dict(cfg))
    labels = codes_to_labels(codes)
    path = write_submission(out, labels)
    return path, dict(summary=summarize(codes, blocks), params=Params.from_dict(cfg).to_dict())
