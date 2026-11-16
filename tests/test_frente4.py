"""Frente 4: casos sintéticos da máquina de estados, das regras de D e do baseline refeito.

Nenhum teste lê tracking, vídeo ou gabarito. O único arquivo da partida usado é o
baseline_raw.csv público da organização, que só contém o time do jogador mais próximo.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.frente4 import deadball  # noqa: E402
from src.frente4.baseline import smooth  # noqa: E402
from src.frente4.features import compute_features  # noqa: E402
from src.frente4.fsm import (Params, possession_labels, drop_short_runs, summarize,  # noqa: E402
                             CODE_H, CODE_A, CODE_D, CODE_X)
from src.frente4.pipeline import events_to_team_array, candidates  # noqa: E402

H, A, N = 0, 1, -1


def seq(spec):
    """'H*5 A*3 N*2' -> int8 array de candidatos."""
    out = []
    for item in spec.split():
        team, count = item.split("*")
        out += [{"H": H, "A": A, "N": N}[team]] * int(count)
    return np.array(out, dtype=np.int8)


def labels(codes):
    return "".join("HADX"[c] for c in codes)


class StateMachineTests(unittest.TestCase):
    P = Params(min_switch_frames=4, max_gap_frames=2, restart_confirm_frames=2, min_dead_frames=2)

    def run_fsm(self, cand, dead=None, **kw):
        dead = np.zeros(len(cand), bool) if dead is None else dead
        return possession_labels(cand, dead, [(0, len(cand) - 1)], self.P, **kw)

    def test_flicker_does_not_switch(self):
        cand = seq("H*6 A*3 H*1 A*2 H*6")
        self.assertEqual(labels(self.run_fsm(cand)), "H" * 18)

    def test_sustained_opponent_switches_retroactively(self):
        cand = seq("H*6 A*4 A*4")
        out = labels(self.run_fsm(cand))
        self.assertEqual(out, "H" * 6 + "A" * 8)

    def test_gap_inside_opponent_streak_is_tolerated(self):
        cand = seq("H*6 A*2 N*2 A*2 A*2")
        self.assertEqual(labels(self.run_fsm(cand)), "H" * 6 + "A" * 8)

    def test_long_gap_breaks_the_streak(self):
        cand = seq("H*6 A*3 N*3 A*3 H*3")
        self.assertEqual(labels(self.run_fsm(cand)), "H" * 18)

    def test_missing_ball_holds_state(self):
        cand = seq("H*4 N*20 A*4")
        self.assertEqual(labels(self.run_fsm(cand)), "H" * 24 + "A" * 4)

    def test_dead_ball_splits_and_restart_backfills(self):
        cand = seq("H*6 N*4 N*3 A*2 A*4")
        dead = np.zeros(len(cand), bool)
        dead[6:10] = True
        out = labels(self.run_fsm(cand, dead))
        # Depois do D, 3 frames sem candidato são preenchidos retroativamente com quem reiniciou.
        self.assertEqual(out, "H" * 6 + "D" * 4 + "A" * 9)

    def test_short_dead_evidence_is_ignored(self):
        cand = seq("H*10")
        dead = np.zeros(10, bool)
        dead[4] = True
        self.assertEqual(labels(self.run_fsm(cand, dead)), "H" * 10)
        self.assertEqual(drop_short_runs(dead, 2).sum(), 0)

    def test_ungraded_frames_are_x(self):
        cand = seq("H*10")
        out = possession_labels(cand, np.zeros(10, bool), [(3, 6)], self.P)
        self.assertEqual(labels(out), "XXXHHHHXXX")

    def test_waiting_without_confirmation_stays_x(self):
        cand = seq("N*5 H*1 N*5")
        self.assertEqual(labels(self.run_fsm(cand)), "X" * 11)

    def test_hint_used_only_while_waiting(self):
        cand = seq("N*4 H*6 N*4")
        hint = np.full(14, N, np.int8)
        hint[:4] = A           # quem cobra a reposição
        hint[10:] = A          # pista tardia não derruba posse corrente
        out = labels(self.run_fsm(cand, hint_team=hint))
        self.assertEqual(out, "A" * 4 + "H" * 10)

    def test_events_override_geometry(self):
        cand = seq("H*6 N*8")
        events = [dict(type="possible_control_acquisition", team="A", event_frame=6)]
        event_team = events_to_team_array(events, len(cand), hold_frames=4)
        self.assertEqual(labels(self.run_fsm(cand, event_team=event_team)), "H" * 6 + "A" * 8)

    def test_summary_counts_spells_inside_blocks(self):
        codes = np.array([CODE_X, CODE_H, CODE_H, CODE_A, CODE_D, CODE_A, CODE_X], np.int8)
        s = summarize(codes, [(1, 5)], fps=1)
        self.assertEqual(s["n_spells"], 3)
        self.assertEqual(s["counts"], dict(H=2, A=2, D=1, X=2))


def synthetic_tracking(n=400, fps=25, pitch=(105.3, 68.0)):
    """Dois jogadores (um H, um A) e uma bola; tudo observado, a menos que o teste apague."""
    players = np.full((n, 2, 2), np.nan, np.float32)
    players[:, 0] = (50.0, 34.0)
    players[:, 1] = (60.0, 34.0)
    ball = np.full((n, 2), np.nan, np.float32)
    ball[:] = (50.5, 34.0)
    return dict(players_xy=players, ball_xy=ball, team_of_slot=np.array([H, A], np.int8),
                fps=fps, pitch=pitch, n_frames=n)


class FeatureAndDeadBallTests(unittest.TestCase):
    def test_features_nearest_and_candidate(self):
        tr = synthetic_tracking(10)
        tr["ball_xy"][5] = np.nan
        feat = compute_features(tr, control_radius_m=3.0)
        self.assertEqual(feat["nearest_team"][0], H)
        self.assertAlmostEqual(float(feat["dist_h"][0]), 0.5, places=5)
        self.assertEqual(feat["cand"][5], N)
        self.assertFalse(feat["ball_observed"][5])
        tr["ball_xy"][:, 0] = 55.0      # a 5 m dos dois: ninguém dentro do raio
        feat = compute_features(tr, control_radius_m=3.0)
        self.assertTrue((feat["cand"] == N).all())
        self.assertTrue((feat["nearest_team"][[0, 1]] == H).all())  # empate -> mandante

    def test_candidates_gated_by_margin_and_ball_speed(self):
        tr = synthetic_tracking(10)
        tr["ball_xy"][5:, 0] = 50.5 + 0.6 * np.arange(5)      # 15 m/s a partir do frame 6
        feat = compute_features(tr, control_radius_m=3.0)
        cand = candidates(feat, dict(contested_margin_m=0.0, max_control_speed_mps=10.0))
        self.assertTrue((cand[:6] == H).all())
        self.assertTrue((cand[6:] == N).all())
        tr["ball_xy"][:, 0] = 55.1                              # 5,1 m de H e 4,9 m de A
        feat = compute_features(tr, control_radius_m=6.0)
        self.assertTrue((feat["cand"] == A).all())
        cand = candidates(feat, dict(contested_margin_m=0.5, max_control_speed_mps=0.0))
        self.assertTrue((cand == N).all())

    def test_ball_out_then_restart(self):
        tr = synthetic_tracking(400)
        tr["ball_xy"][100:160, 0] = -1.0           # fora pela linha de fundo
        tr["ball_xy"][160:, 0] = 1.0 + 0.3 * np.arange(240)   # volta andando a 7,5 m/s
        feat = compute_features(tr)
        cfg = dict(use_stillness=False, min_dead_after_trigger_frames=10)
        ev = deadball.dead_ball_evidence(feat, [(0, 399)], {1: 0}, tr["pitch"], (), cfg)
        self.assertTrue(ev["dead"][100:160].all())
        self.assertFalse(ev["dead"][:100].any())
        self.assertFalse(ev["dead"][161:].any())
        self.assertEqual(ev["restarts"][0][1], "ball_out")

    def test_pre_kickoff_is_dead(self):
        tr = synthetic_tracking(100)
        feat = compute_features(tr)
        ev = deadball.dead_ball_evidence(feat, [(10, 99)], {1: 40}, tr["pitch"], (),
                                         dict(use_stillness=False))
        self.assertTrue(ev["dead"][10:40].all())
        self.assertFalse(ev["dead"][40:].any())
        self.assertFalse(ev["dead"][:10].any())

    def test_set_piece_tag_opens_dead_until_ball_moves(self):
        tr = synthetic_tracking(300)
        tr["ball_xy"][200:, 0] = 50.5 + 0.3 * np.arange(100)
        feat = compute_features(tr)
        tags = [dict(code="SET PIECES", start_frame=120, end_frame=240,
                     tags={"Type": "Corner Kick", "Team": "Barcelona", "Half": "1st Half"})]
        ev = deadball.dead_ball_evidence(feat, [(0, 299)], {1: 0}, tr["pitch"], tags,
                                         dict(use_stillness=False))
        # A velocidade só aparece no 2º frame do movimento, por isso a reposição é em 201.
        self.assertTrue(ev["dead"][120:201].all())
        self.assertFalse(ev["dead"][201:].any())
        self.assertEqual(ev["hint_team"][201], H)
        self.assertEqual(deadball.team_code("Atletic Club"), A)
        self.assertEqual(deadball.team_code("N/A"), N)

    def test_stillness_rule(self):
        tr = synthetic_tracking(300)
        tr["ball_xy"][200:, 0] = 50.5 + 0.3 * np.arange(100)
        feat = compute_features(tr)
        ev = deadball.dead_ball_evidence(feat, [(0, 299)], {1: 0}, tr["pitch"], (),
                                         dict(use_stillness=True, still_frames=50))
        self.assertTrue(ev["dead"][1:201].all())
        self.assertFalse(ev["dead"][201:].any())


class BaselineTests(unittest.TestCase):
    def test_smooth_matches_starter(self):
        from starter.baseline import smooth as reference
        rng = np.random.RandomState(0)
        codes = rng.choice([H, A, N], size=5000, p=[0.4, 0.35, 0.25]).astype(np.int8)
        for window in (0, 1, 25, 50, 125):
            np.testing.assert_array_equal(smooth(codes, window), reference(codes, window))

    def test_smooth_reproduces_shipped_baselines(self):
        import pandas as pd
        folder = os.path.join(ROOT, "evaluation", "results")
        raw = pd.read_csv(os.path.join(folder, "baseline_raw.csv"))["label"].values
        codes = np.where(raw == "H", H, np.where(raw == "A", A, N)).astype(np.int8)
        for name, window in (("baseline_smooth25", 25), ("baseline_smooth50", 50),
                             ("baseline_smooth125", 125)):
            ref = pd.read_csv(os.path.join(folder, name + ".csv"))["label"].values
            np.testing.assert_array_equal(smooth(codes, window), ref, err_msg=name)


if __name__ == "__main__":
    unittest.main()
