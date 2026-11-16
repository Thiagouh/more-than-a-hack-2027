"""Máquina de estados de posse: candidato por frame -> H/A/D/X com memória.

Ideia central (ver docs/EVALUATION.md): o ranking é por blocos de posse, então trocar de time a
cada frame ruidoso custa muito mais do que atrasar uma troca real em alguns frames. Regras:

1. **Tempo mínimo para trocar.** O adversário só assume quando é o candidato (jogador mais
   próximo dentro do raio de controle) em ``min_switch_frames`` frames, tolerando até
   ``max_gap_frames`` frames sem candidato (bola ausente) no meio. Um toque isolado ou um desvio
   não troca a posse, como pede docs/CHALLENGE.md.
2. **Troca retroativa.** Confirmada a troca, o novo time recebe desde o primeiro frame da
   sequência que a confirmou, não desde a confirmação. O gabarito marca a troca no primeiro
   contato controlado, e o scorer cobra atraso de 1 s em ~35 pontos de spellF1.
3. **Bola some: manter o estado.** Sem candidato, o último time continua. A bola ausente não é
   bola morta (docs/FAQ.md). Só a evidência explícita em ``dead`` produz D.
4. **Bola morta reinicia.** Depois de um trecho D, o primeiro time que controla a bola
   (``restart_confirm_frames``) recebe retroativamente desde o fim do D, porque após uma
   reposição há sempre um possuidor claro.
5. **Fora do jogo é X.** Frames fora dos blocos avaliados (antes do apito, intervalo, depois do
   fim) recebem X: custa zero.

Tudo é offline: o CSV é gerado de uma vez, por isso olhar para frente é legítimo.
"""
from dataclasses import dataclass, asdict

import numpy as np

H, A, NOBODY = 0, 1, -1
CODE_H, CODE_A, CODE_D, CODE_X = 0, 1, 2, 3          # mesma codificação de possession_metrics
LABELS = np.array(["H", "A", "D", "X"])


@dataclass
class Params:
    min_switch_frames: int = 12       # 0,48 s de candidato adversário para confirmar a troca
    max_gap_frames: int = 5           # buracos sem candidato tolerados dentro dessa sequência
    restart_confirm_frames: int = 3   # frames do primeiro candidato depois de D ou do início
    min_dead_frames: int = 5          # evidência de bola morta mais curta que isso é ignorada

    @classmethod
    def from_dict(cls, d):
        keys = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in keys})

    def to_dict(self):
        return asdict(self)


def runs(mask):
    """[(start, stop)] semiabertos dos trechos True."""
    m = np.asarray(mask, dtype=bool)
    if not len(m):
        return []
    edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def drop_short_runs(mask, min_len):
    """Copia de mask com os trechos True menores que min_len apagados."""
    out = np.asarray(mask, dtype=bool).copy()
    for start, stop in runs(out):
        if stop - start < min_len:
            out[start:stop] = False
    return out


def possession_labels(cand, dead, blocks, params=None, hint_team=None, event_team=None):
    """Códigos int8 por frame (0=H, 1=A, 2=D, 3=X).

    cand        int8[n]  candidato geométrico: 0=H, 1=A, -1 ninguém
    dead        bool[n]  evidência de bola morta (frente 3 / deadball.py)
    blocks      [(first, last)] intervalos avaliados, inclusivos (os dois tempos)
    hint_team   int8[n]  opcional: time sugerido (p.ex. quem cobra a bola parada); só é usado
                         enquanto se espera o primeiro controlador depois de um D
    event_team  int8[n]  opcional: time de eventos de controle da frente 5; quando >= 0
                         substitui o candidato geométrico naquele frame
    """
    p = params or Params()
    cand = np.asarray(cand, dtype=np.int8)
    n = len(cand)
    dead = np.asarray(dead, dtype=bool)
    codes = np.full(n, CODE_X, dtype=np.int8)

    merged = cand.copy()
    if event_team is not None:
        event_team = np.asarray(event_team, dtype=np.int8)
        merged = np.where(event_team >= 0, event_team, merged).astype(np.int8)

    for first, last in blocks:
        first, last = int(first), int(last)
        if first > last:
            continue
        stop = last + 1
        dead_block = drop_short_runs(dead[first:stop], p.min_dead_frames)

        owner = NOBODY
        wait_start = first
        streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0

        for f in range(first, stop):
            if dead_block[f - first]:
                codes[f] = CODE_D
                owner = NOBODY
                wait_start = f + 1
                streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
                continue

            c = int(merged[f])

            if owner == NOBODY:
                # Esperando o primeiro controlador depois de um D ou do início do tempo.
                if c == NOBODY and hint_team is not None and hint_team[f] >= 0:
                    c = int(hint_team[f])
                if c == NOBODY:
                    if streak_count:
                        gap += 1
                        if gap > p.max_gap_frames:
                            streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
                    continue  # fica X provisoriamente; é preenchido na confirmação
                if c == streak_team:
                    streak_count += 1
                    gap = 0
                else:
                    streak_team, streak_count, streak_start, gap = c, 1, f, 0
                if streak_count >= p.restart_confirm_frames:
                    owner = c
                    codes[wait_start:f + 1] = owner
                    streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
                continue

            if c == owner:
                codes[f] = owner
                streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
                continue

            if c == NOBODY:
                codes[f] = owner
                if streak_count:
                    gap += 1
                    if gap > p.max_gap_frames:
                        streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
                continue

            # Candidato adversário.
            if c == streak_team:
                streak_count += 1
                gap = 0
            else:
                streak_team, streak_count, streak_start, gap = c, 1, f, 0
            if streak_count >= p.min_switch_frames:
                owner = c
                codes[streak_start:f + 1] = owner       # troca retroativa
                streak_team, streak_count, streak_start, gap = NOBODY, 0, -1, 0
            else:
                codes[f] = owner

    return codes


def codes_to_labels(codes):
    return LABELS[np.asarray(codes, dtype=np.int64)]


def labels_to_codes(labels):
    labels = np.asarray(labels).astype("<U1")
    codes = np.full(len(labels), CODE_X, dtype=np.int8)
    for code, name in enumerate(LABELS):
        codes[labels == name] = code
    return codes


def summarize(codes, blocks, fps=25):
    """Contagem de posses (blocos H/A contínuos dentro dos trechos avaliados), duração média,
    percentuais. Diagnóstico sem gabarito."""
    codes = np.asarray(codes)
    n_spells, total = 0, 0
    for first, last in blocks:
        window = codes[first:last + 1]
        alive = (window == CODE_H) | (window == CODE_A)
        for start, stop in runs(alive):
            seg = window[start:stop]
            breaks = np.flatnonzero(seg[1:] != seg[:-1])
            n_spells += 1 + len(breaks)
            total += stop - start
    counts = {name: int(np.sum(codes == code)) for code, name in enumerate(LABELS)}
    return dict(n_spells=n_spells,
                mean_duration_s=(total / n_spells / fps) if n_spells else 0.0,
                counts=counts)
