"""Casos sintéticos: nenhum arquivo da partida é carregado."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import possession_metrics as pm
from comparar_csv import compare, load


def codes(labels):
    return pm.label_codes(list(labels))


class EvaluationTests(unittest.TestCase):
    def test_difference_intervals_include_edges(self):
        result = compare(codes('HHHAAA'), codes('AHHADD'))
        self.assertEqual(result['frames_diferentes'], 3)
        self.assertEqual([(r['inicio'], r['fim_inclusivo']) for r in result['intervalos']],
                         [(0, 0), (4, 5)])
        self.assertEqual(compare(codes('HA'), codes('HA'))['intervalos'], [])

    def test_coverage_mismatch(self):
        with self.assertRaises(ValueError):
            compare(codes('H'), codes('HH'))

    def test_invalid_csvs(self):
        for content in ('frame,label\n0,H\n0,A\n', 'frame,label\n0,\n',
                        'frame,label\n0,Z\n', 'frame,label\n1,H\n'):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'test.csv'
                path.write_text(content)
                with self.assertRaises(ValueError):
                    load(path)

    def test_x_and_d_have_different_temporal_effects(self):
        gt = codes('HHHHAAAA')
        with_x = pm.evaluate(gt, codes('HHXHAAAA'))
        with_d = pm.evaluate(gt, codes('HHDHAAAA'))
        self.assertAlmostEqual(with_x['spell_f1_mean'], 1)
        self.assertLess(with_x['macro_f1'], pm.evaluate(gt, gt)['macro_f1'])
        self.assertEqual(with_x['n_pred_spells'], 2)
        self.assertEqual(with_d['n_pred_spells'], 3)

    def test_x_fill_never_crosses_ungraded_gap(self):
        gt = codes('HHXXAA')
        pred = codes('HHXXXX')
        filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
        self.assertEqual(filled.tolist(), pred.tolist())


if __name__ == '__main__':
    unittest.main()
