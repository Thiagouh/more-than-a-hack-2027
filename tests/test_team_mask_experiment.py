import tempfile
import types
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.experiment_team_masks import select_probability, metrics, passes, bounded_worker


class MaskExperimentTests(unittest.TestCase):
    def test_selection_locked_to_baseline_not_variant_or_score(self):
        p = np.zeros((2,12,12),np.float32)
        p[0,2:10,2:10] = .6
        p[1,3:9,3:9] = .9
        chosen,index = select_probability(p,[.6,.99],[2,2,10,10])
        self.assertEqual(index,0)
        self.assertEqual(int((chosen > .7).sum()),0)
        self.assertEqual(int((chosen > .5).sum()),64)

    def test_empty_and_invalid_probabilities(self):
        p,index = select_probability(np.zeros((0,12,12)),[],[0,0,12,12])
        self.assertIsNone(index)
        self.assertFalse(p.any())
        for value in (float('nan'),-1,1.1):
            with self.assertRaises(ValueError):
                select_probability(np.full((1,12,12),value),[1],[0,0,12,12])

    def test_losses_and_gate_boundaries(self):
        target = np.ones((10,10),bool)
        self.assertEqual(metrics(np.zeros_like(target),target)['coverage'],0)
        b=dict(extracted=100,precision=.8,coverage=.8)
        v=dict(extracted=90,precision=.85,coverage=.75)
        self.assertTrue(passes(b,v))
        self.assertFalse(passes(b,dict(v,coverage=.749)))
        self.assertFalse(passes(b,dict(v,precision=.849)))
        self.assertFalse(passes(b,dict(v,extracted=0)))

    def test_watchdog_kills_over_memory(self):
        class Process:
            pid=1
            returncode=0
            def poll(self):return None
            def kill(self):self.returncode=-9
            def wait(self):return self.returncode
        class Monitored:
            def memory_info(self):
                class Info:rss=7_000_000_001
                return Info()
        import types
        fake=types.SimpleNamespace(Process=lambda pid:Monitored(),NoSuchProcess=ProcessLookupError)
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules',{'psutil':fake}),patch(
                'scripts.experiment_team_masks.subprocess.Popen',return_value=Process()):
            result=bounded_worker(Path('truth'),'H01',Path('weights'),'0'*64,Path(tmp)/'case')
            self.assertEqual(result['status'],'memory_limit')
            self.assertEqual(result['returncode'],-9)

    def test_watchdog_kills_over_time(self):
        class Process:
            pid=1
            returncode=0
            def poll(self):return None
            def kill(self):self.returncode=-9
            def wait(self):pass
        import types
        fake=types.SimpleNamespace(Process=lambda pid:types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss=1)),NoSuchProcess=ProcessLookupError)
        with tempfile.TemporaryDirectory() as tmp,patch.dict('sys.modules',{'psutil':fake}),patch(
                'scripts.experiment_team_masks.subprocess.Popen',return_value=Process()),patch(
                'scripts.experiment_team_masks.time.monotonic',side_effect=[0,61,61,61]):
            self.assertEqual(bounded_worker(Path('t'),'H01',Path('w'),'0'*64,Path(tmp)/'case')['status'],'time_limit')

    def test_negative_h01_never_expands(self):
        from scripts.experiment_team_masks import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);truth=root/'truth.json';truth.write_text('{}')
            args=types.SimpleNamespace(truth=truth,out=root/'run',resume=False,
                checkpoint=Path('weights'),checkpoint_sha256='0'*64,h01_review=None)
            def fake_worker(truth,label,checkpoint,digest,out):
                out.mkdir()
                (out/'result.json').write_text(json.dumps(dict(numeric_gate=False)))
                return dict(status='completed')
            with patch('scripts.experiment_team_masks.load_truth'),patch(
                    'scripts.experiment_team_masks.runtime_provenance',return_value={}),patch(
                    'scripts.experiment_team_masks.bounded_worker',side_effect=fake_worker) as worker:
                run(args)
            self.assertEqual(worker.call_count,1)
            report=json.loads((args.out/'experiment.json').read_text())
            self.assertEqual(report['status'],'negative_H01_numeric_gate')
            self.assertFalse(report['expanded'])

    def test_positive_numeric_requires_visual_review(self):
        from scripts.experiment_team_masks import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);truth=root/'truth.json';truth.write_text('{}')
            args=types.SimpleNamespace(truth=truth,out=root/'run',resume=False,
                checkpoint=Path('weights'),checkpoint_sha256='0'*64,h01_review=None)
            def fake_worker(truth,label,checkpoint,digest,out):
                out.mkdir()
                (out/'result.json').write_text(json.dumps(dict(numeric_gate=True)))
                return dict(status='completed')
            with patch('scripts.experiment_team_masks.load_truth'),patch(
                    'scripts.experiment_team_masks.runtime_provenance',return_value={}),patch(
                    'scripts.experiment_team_masks.bounded_worker',side_effect=fake_worker) as worker:
                run(args)
            self.assertEqual(worker.call_count,1)
            self.assertEqual(json.loads((args.out/'experiment.json').read_text())['status'],
                             'awaiting_H01_visual_review')


if __name__ == '__main__':unittest.main()
