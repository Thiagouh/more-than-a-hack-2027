"""Integração frente 2 -> frente 4, só com dados sintéticos."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.frente4 import deadball  # noqa: E402
from src.frente4.pipeline import aplicar_sinal_bola  # noqa: E402


def tracking(n):
    ball = np.full((n, 2), 50.0, dtype=np.float32)
    return dict(n_frames=n, fps=25.0, ball_xy=ball, players_xy=np.zeros((n, 2, 2), np.float32),
                team_of_slot=np.array([0, 1]))


def sinal(n, fonte, buraco, saiu):
    return pd.DataFrame(dict(frame=np.arange(n), x=np.full(n, 10.0), y=np.full(n, 20.0),
                             fonte=fonte, buraco_id=buraco, saiu_campo=saiu))


class AplicarSinalBola(unittest.TestCase):
    def test_so_observado_entra_e_inicio_do_buraco_com_saida(self):
        n = 10
        fonte = ["observado"] * 3 + ["estimado"] * 4 + ["observado"] * 3
        buraco = [-1] * 3 + [7] * 4 + [-1] * 3
        saiu = [False] * 3 + [True] * 4 + [False] * 3
        tr, starts = aplicar_sinal_bola(tracking(n), sinal(n, fonte, buraco, saiu))
        self.assertTrue(np.isnan(tr["ball_xy"][3:7]).all())       # estimativa não é observação
        self.assertTrue((tr["ball_xy"][[0, 1, 2, 7, 8, 9]] == [10.0, 20.0]).all())
        self.assertEqual(list(np.flatnonzero(starts)), [3])

    def test_buraco_sem_saida_nao_abre_d(self):
        n = 6
        _, starts = aplicar_sinal_bola(
            tracking(n), sinal(n, ["observado"] * 2 + ["estimado"] * 4, [-1] * 2 + [1] * 4, [False] * 6))
        self.assertFalse(starts.any())

    def test_saiu_campo_lido_como_texto(self):
        n = 4
        _, starts = aplicar_sinal_bola(
            tracking(n), sinal(n, ["observado", "estimado", "estimado", "observado"],
                               [-1, 0, 0, -1], ["False", "True", "True", "False"]))
        self.assertEqual(list(np.flatnonzero(starts)), [1])

    def test_tamanho_incompativel_falha(self):
        with self.assertRaises(ValueError):
            aplicar_sinal_bola(tracking(5), sinal(4, ["observado"] * 4, [-1] * 4, [False] * 4))


class DeadComBuracoFora(unittest.TestCase):
    def test_buraco_com_saida_abre_d_ate_reposicao(self):
        n = 120
        nan = np.full(n, np.nan, dtype=np.float32)
        x = np.full(n, 50.0, dtype=np.float32); y = np.full(n, 30.0, dtype=np.float32)
        x[10:60] = np.nan; y[10:60] = np.nan
        speed = np.zeros(n, dtype=np.float32); speed[80:] = 8.0     # reposição em movimento
        feat = dict(ball_observed=~np.isnan(x), ball_x=x, ball_y=y, ball_speed=speed)
        starts = np.zeros(n, dtype=bool); starts[10] = True
        ev = deadball.dead_ball_evidence(feat, [(0, n - 1)], {1: 0}, (105.0, 68.0), (), None, starts)
        self.assertTrue(ev["dead"][10:80].all())
        self.assertFalse(ev["dead"][80:].any())
        self.assertEqual(ev["reason"][10], deadball.REASON_HOLE_OUT)
        sem = deadball.dead_ball_evidence(feat, [(0, n - 1)], {1: 0}, (105.0, 68.0), (), {"use_stillness": False})
        self.assertFalse(sem["dead"].any())


class DeadExterno(unittest.TestCase):
    def test_inicio_externo_abre_d_ate_reposicao(self):
        n = 100
        x = np.full(n, 50.0, dtype=np.float32); y = np.full(n, 30.0, dtype=np.float32)
        speed = np.zeros(n, dtype=np.float32); speed[70:] = 8.0
        feat = dict(ball_observed=np.ones(n, bool), ball_x=x, ball_y=y, ball_speed=speed)
        ext = np.zeros(n, dtype=bool); ext[20] = True
        ev = deadball.dead_ball_evidence(feat, [(0, n - 1)], {1: 0}, (105.0, 68.0), (),
                                         {"use_stillness": False}, ext_starts=ext)
        self.assertTrue(ev["dead"][20:70].all())
        self.assertFalse(ev["dead"][:20].any() or ev["dead"][70:].any())
        self.assertEqual(ev["reason"][20], deadball.REASON_EXT)


if __name__ == "__main__":
    unittest.main()
