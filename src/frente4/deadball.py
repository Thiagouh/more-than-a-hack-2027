"""Evidência de bola morta por frame (regras de D) e pistas de time para reposições.

Fontes, em ordem de confiança:

1. **Antes do pontapé inicial de cada tempo**: do primeiro frame avaliado até o kickoff
   publicado (docs/DATA.md e docs/FAQ.md). No 2º tempo são os 2.934 frames 110808-113741.
2. **Bola fora do campo**: a bola observada além da linha (com margem) por alguns frames
   consecutivos. Fica D até a reposição: primeiro frame em que a bola é vista dentro do campo
   e em movimento, depois de um tempo mínimo de D.
3. **Set pieces do Smart Tagging** (escanteio, tiro de meta, falta, lateral, kick off após gol):
   se ainda não estiver D no início da marcação, abre um D ali e procura a reposição do mesmo
   jeito. A marcação é uma janela de ~5 s em torno da reposição, não o instante.
4. **Bola parada por muito tempo** (opcional): bola observada quase parada por
   ``still_frames`` abre um D retroativo ao início da parada. Hipótese ainda não validada:
   goleiro segurando a bola pode cair aqui.

Nenhuma destas regras usa o gabarito. Um D nunca ultrapassa ``max_dead_frames``.
"""
import numpy as np

from .fsm import runs, NOBODY, H, A

REASON_NONE, REASON_PRE_KICKOFF, REASON_BALL_OUT, REASON_SET_PIECE, REASON_STILL, REASON_LOST_OUTSIDE = 0, 1, 2, 3, 4, 5
REASON_NAMES = {REASON_NONE: "alive", REASON_PRE_KICKOFF: "pre_kickoff", REASON_BALL_OUT: "ball_out",
                REASON_SET_PIECE: "set_piece_tag", REASON_STILL: "ball_still", REASON_LOST_OUTSIDE: "ball_out_then_lost"}

DEFAULTS = dict(
    out_margin_m=0.5,                 # bola além da linha por mais que isso conta como fora
    out_min_frames=3,                 # frames consecutivos fora para disparar
    min_dead_after_trigger_frames=25, # 1 s mínimo de D depois do gatilho
    max_dead_frames=2250,             # 90 s: teto de segurança para um D
    restart_speed_mps=3.0,            # bola dentro do campo e a esta velocidade = reposição
    use_set_piece_tags=True,
    use_stillness=True,
    still_speed_mps=0.5,
    still_frames=100,                 # 4 s
    restart_hint_frames=50,           # pista de time vale por 2 s depois da reposição
)

SET_PIECE_CODE = "set pieces"
KICKOFF_START = "kick off (start)"


def team_code(name):
    """'Barcelona' -> H, 'Atletic Club'/'Athletic Club' -> A, outros (N/A) -> -1."""
    s = (name or "").strip().lower()
    if s.startswith("barc"):
        return H
    if s.startswith("atl") or s.startswith("ath"):
        return A
    return NOBODY


def set_piece_windows(tags):
    """[(start_frame, end_frame, type, team_code)] para SET PIECES, exceto kick off inicial."""
    out = []
    for tag in tags:
        if (tag.get("code") or "").strip().lower() != SET_PIECE_CODE:
            continue
        kind = (tag["tags"].get("Type") or "").strip().lower()
        if kind == KICKOFF_START:
            continue
        out.append((int(tag["start_frame"]), int(tag["end_frame"]), kind,
                    team_code(tag["tags"].get("Team"))))
    out.sort()
    return out


def kickoff_hints(tags):
    """[(start_frame, end_frame, team_code)] dos dois kick off iniciais, para a pista de time."""
    out = []
    for tag in tags:
        if (tag.get("code") or "").strip().lower() != SET_PIECE_CODE:
            continue
        if (tag["tags"].get("Type") or "").strip().lower() == KICKOFF_START:
            out.append((int(tag["start_frame"]), int(tag["end_frame"]), team_code(tag["tags"].get("Team"))))
    return out


def dead_ball_evidence(feat, blocks, kickoffs, pitch, tags=(), cfg=None, lost_outside_starts=None):
    """Devolve dict(dead bool[n], reason int8[n], hint_team int8[n], restarts list).

    feat      saída de features.compute_features
    blocks    [(first, last)] trechos avaliados, inclusivos
    kickoffs  {1: frame, 2: frame} frames publicados do pontapé inicial de cada tempo
    pitch     (comprimento, largura) em metros
    tags      lista de starter.load_smart_tagging.load (pode ser vazia)
    """
    cfg = {**DEFAULTS, **(cfg or {})}
    n = len(feat["ball_observed"])
    dead = np.zeros(n, dtype=bool)
    reason = np.zeros(n, dtype=np.int8)
    hint_team = np.full(n, NOBODY, dtype=np.int8)
    restarts = []

    observed = feat["ball_observed"]
    x, y = feat["ball_x"], feat["ball_y"]
    length, width = pitch
    m = cfg["out_margin_m"]
    with np.errstate(invalid="ignore"):
        outside = observed & ((x < -m) | (x > length + m) | (y < -m) | (y > width + m))
        inside = observed & ~outside
        speed = feat["ball_speed"]
        moving = inside & (np.nan_to_num(speed, nan=0.0) >= cfg["restart_speed_mps"])
        still = observed & (np.nan_to_num(speed, nan=np.inf) < cfg["still_speed_mps"])

    out_starts = np.zeros(n, dtype=bool)
    for start, stop in runs(outside):
        if stop - start >= cfg["out_min_frames"]:
            out_starts[start] = True

    still_starts = np.zeros(n, dtype=bool)
    if cfg["use_stillness"]:
        for start, stop in runs(still):
            if stop - start >= cfg["still_frames"]:
                still_starts[start] = True

    lost_outside=np.zeros(n,dtype=bool) if lost_outside_starts is None else np.asarray(lost_outside_starts,dtype=bool)
    if lost_outside.shape!=(n,):raise ValueError('Lost-outside evidence must align with all frames')
    tag_starts = {}
    if cfg["use_set_piece_tags"]:
        for start, stop, kind, team in set_piece_windows(tags):
            tag_starts.setdefault(start, (stop, kind, team))
    tag_team_by_frame = np.full(n, NOBODY, dtype=np.int8)
    for start, stop, kind, team in (set_piece_windows(tags) if cfg["use_set_piece_tags"] else []):
        if team >= 0:
            tag_team_by_frame[max(0, start):min(n, stop + cfg["restart_hint_frames"])] = team
    for start, stop, team in kickoff_hints(tags):
        if team >= 0:
            tag_team_by_frame[max(0, start):min(n, stop + cfg["restart_hint_frames"])] = team

    for half_index, (first, last) in enumerate(blocks, start=1):
        first, last = int(first), int(last)
        kickoff = int(kickoffs.get(half_index, first))
        if kickoff > first:
            dead[first:kickoff] = True
            reason[first:kickoff] = REASON_PRE_KICKOFF
        if kickoff <= last:
            hint_team[kickoff:min(last + 1, kickoff + cfg["restart_hint_frames"])] = \
                np.where(tag_team_by_frame[kickoff:min(last + 1, kickoff + cfg["restart_hint_frames"])] >= 0,
                         tag_team_by_frame[kickoff:min(last + 1, kickoff + cfg["restart_hint_frames"])], NOBODY)

        is_dead = False
        dead_start = -1
        why = REASON_NONE
        for f in range(max(first, kickoff), last + 1):
            if not is_dead:
                if out_starts[f]:
                    is_dead, dead_start, why = True, f, REASON_BALL_OUT
                elif lost_outside[f]:
                    is_dead, dead_start, why = True, f, REASON_LOST_OUTSIDE
                elif f in tag_starts:
                    is_dead, dead_start, why = True, f, REASON_SET_PIECE
                elif still_starts[f]:
                    is_dead, dead_start, why = True, f, REASON_STILL
            if is_dead:
                age = f - dead_start
                if age >= cfg["min_dead_after_trigger_frames"] and moving[f]:
                    is_dead = False
                    restarts.append((f, REASON_NAMES[why]))
                    stop = min(last + 1, f + cfg["restart_hint_frames"])
                    hint_team[f:stop] = tag_team_by_frame[f:stop]
                    continue
                if age >= cfg["max_dead_frames"]:
                    is_dead = False
                    restarts.append((f, REASON_NAMES[why] + "_timeout"))
                    continue
                dead[f] = True
                reason[f] = why

    return dict(dead=dead, reason=reason, hint_team=hint_team, restarts=restarts, config=cfg)
