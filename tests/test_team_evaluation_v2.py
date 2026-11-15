import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from src.team_identification.evaluation import evaluate
from src.team_identification.protocol import (validate_truth,load_truth,freeze_truth,
    validate_result_binding,validate_splits,in_region)
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256
from tests.team_evaluation_fixtures import person,truth_for,prediction
from scripts.demo_team_evaluation import create_inputs


class EvaluationV2Tests(unittest.TestCase):
    def score(self, people, detections, **changes):
        truth = truth_for(people)
        truth.update(changes)
        return evaluate(truth,[dict(frame=0,detections=detections)])

    def test_goalkeeper_team_success_is_not_joint_success(self):
        report = self.score([person(role='goalkeeper')],[prediction()])
        self.assertEqual(report['precision_accepted'],1)
        self.assertEqual(report['precision_team_and_role'],0)
        self.assertEqual(report['strata']['role:goalkeeper_H']['coverage'],1)
        self.assertEqual(report['role_confusion']['goalkeeper'],{'player':1})
        self.assertIsNone(report['strata']['role:goalkeeper_A']['precision_accepted'])

    def test_goalkeeper_missed_uncertain_and_partial(self):
        report = self.score([person('g1',role='goalkeeper',partial=True),
            person('g2',x=30,role='goalkeeper',classification='A')],
            [prediction(classification='uncertain',role='unknown')])
        self.assertEqual(report['strata']['role:goalkeeper']['evaluable_athletes'],2)
        self.assertEqual(report['strata']['role:goalkeeper']['coverage'],0)
        self.assertEqual(report['missed_people'],1)
        self.assertEqual(report['role_confusion']['goalkeeper'],{'unknown':1,'missed':1})

    def test_other_person_is_not_a_success_and_outside_is_diagnostic(self):
        report = self.score([person(role='other',classification='uncertain')],
            [prediction(),prediction(box=[150,0,160,40])])
        self.assertEqual(report['accepted'],1)
        self.assertEqual(report['precision_accepted'],0)
        self.assertEqual(report['outside_region_detections'],1)
        self.assertEqual(report['evaluable_athletes'],0)
        self.assertIsNone(report['coverage'])
        uncertain = self.score([person(role='other',classification='uncertain')],
                               [prediction(classification='uncertain',role='unknown')])
        self.assertEqual(uncertain['uncertain_detections'],1)
        self.assertIsNone(uncertain['precision_accepted'])

    def test_empty_roles_and_global_extra_penalty(self):
        report = self.score([person()],[prediction(),prediction(box=[30,0,40,40])])
        self.assertEqual(report['precision_accepted'],.5)
        self.assertEqual(report['strata']['role:player']['precision_accepted'],1)
        self.assertEqual(report['strata']['role:goalkeeper']['evaluable_athletes'],0)
        self.assertFalse(report['sampling']['sufficient_for_exploratory_analysis'])

    def test_matching_is_class_blind(self):
        report = self.score([person('h'),person('a',x=30,classification='A')],
            [prediction(classification='A'),prediction(box=[30,0,40,40],classification='H')])
        self.assertEqual(report['correct_accepted'],0)
        self.assertEqual(report['covered_athletes'],2)

    def test_schema_requires_explicit_roles_and_compatible_labels(self):
        for changes in ({'role':None},{'role':'referee'},{'role':'other'},
                        {'classification':'uncertain','evaluable':True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_truth(truth_for([person(**changes)]))
        with self.assertRaisesRegex(ValueError,'v2'):
            validate_truth(dict(reviewed=True,reviewer='legacy',frames=[]))
        with self.assertRaises(ValueError):
            self.score([person()],[prediction(role='referee')])

    def test_polygon_boundary_and_multiple_regions(self):
        regions = [[[0,0],[10,0],[10,10],[0,10]],[[20,0],[30,0],[30,10],[20,10]]]
        self.assertTrue(in_region([9,1,11,3],regions))
        self.assertTrue(in_region([21,1,23,3],regions))
        self.assertFalse(in_region([14,1,16,3],regions))
        truth = truth_for([person()])
        truth['frames'][0]['people'][0]['box'] = [150,0,160,40]
        with self.assertRaisesRegex(ValueError,'outside'):
            validate_truth(truth)

    def test_sampling_groups_replays_and_unknown_time(self):
        truth = truth_for([person()],count=10)
        truth['frames'][1]['group_id'] = truth['frames'][0]['group_id']
        with self.assertRaisesRegex(ValueError,'one frame'):
            validate_truth(truth)
        truth['sample_kind'] = 'challenge'
        report = evaluate(truth,[dict(frame=i,detections=[prediction()]) for i in range(10)])
        self.assertEqual(report['sampling']['group_count'],9)
        self.assertIsNone(report['uncertainty']['intervals']['coverage'])
        truth = truth_for([person()],count=10)
        truth['frames'][0].update(temporal_origin_known=False,source_time_s=None)
        report = evaluate(truth,[dict(frame=i,detections=[prediction()]) for i in range(10)])
        self.assertEqual(report['uncertainty']['warning'],'unknown_temporal_origin')

    def test_bootstrap_is_repeatable_and_not_person_based(self):
        truth = truth_for([person()],count=10)
        predictions = [dict(frame=i,detections=[prediction(classification='A' if i<2 else 'H')])
                       for i in range(10)]
        first = evaluate(truth,predictions)
        second = evaluate(truth,predictions)
        self.assertEqual(first['uncertainty'],second['uncertainty'])
        lo,hi = first['uncertainty']['intervals']['precision_accepted']
        self.assertLessEqual(lo,.8)
        self.assertGreaterEqual(hi,.8)
        self.assertEqual(first['uncertainty']['group_count'],10)
        self.assertIsNone(first['strata_uncertainty']['role:goalkeeper']['intervals']['coverage'])

    def test_sparse_denominators_do_not_claim_a_confidence_interval(self):
        truth = truth_for([person()],count=10)
        predictions = [dict(frame=i,detections=[prediction()] if i==0 else []) for i in range(10)]
        report = evaluate(truth,predictions)
        self.assertIsNone(report['uncertainty']['intervals']['precision_accepted'])
        self.assertIsNotNone(report['uncertainty']['intervals']['coverage'])
        self.assertEqual(report['uncertainty']['denominator_groups']['precision_accepted'],1)

    def test_split_leakage_and_duplicate_pixels(self):
        truth = truth_for([person()])
        f = truth['frames'][0]
        truth['excluded_segments'] = [dict(source_sha256=f['source_sha256'],
            source_segment_id=f['source_segment_id'],split='development')]
        with self.assertRaisesRegex(ValueError,'segment'):
            validate_truth(truth)
        truth = truth_for([person()],count=2)
        truth['frames'][1]['pixel_sha256'] = truth['frames'][0]['pixel_sha256']
        with self.assertRaisesRegex(ValueError,'Repeated'):
            validate_truth(truth)
        with self.assertRaisesRegex(ValueError,'source segments'):
            validate_splits(truth_for([person()]),dict(references=[{}]))

    def test_full_sample_can_meet_operational_minima_without_validation(self):
        people = [person(f'p{i}',x=i*8,role='goalkeeper' if i==0 else 'player',
                         classification='H' if i%2==0 else 'A',
                         lighting='sun' if i%2 else 'shade',
                         occlusion='isolated' if i%2 else 'overlapped') for i in range(10)]
        truth = truth_for(people,count=30)
        # Alternate goalkeeper teams across independent groups.
        for i,f in enumerate(truth['frames']):
            f['people'] = copy.deepcopy(people)
            f['people'][0]['classification'] = 'H' if i%2 else 'A'
        preds = [dict(frame=f['frame'],detections=[prediction(box=p['box'],
            classification=p['classification'],role=p['role']) for p in f['people']]) for f in truth['frames']]
        report = evaluate(truth,preds)
        self.assertTrue(report['point_targets_observed'])
        self.assertTrue(report['sampling']['sufficient_for_exploratory_analysis'])
        self.assertFalse(report['possession_integration_authorized'])


class FrozenProtocolTests(unittest.TestCase):
    def test_annotation_draft_preparation_does_not_require_references(self):
        from scripts.prepare_team_evaluation import main as prepare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root/'frames'
            frames.mkdir()
            Image.new('RGB',(40,80),'red').save(frames/'1.png')
            recording = root/'synthetic-source'
            recording.write_bytes(b'synthetic source hash fixture, not video')
            with patch('sys.argv',['prepare','--frames-dir',str(frames),'--recording',str(recording),
                '--seconds','1','--out',str(root/'draft')]):
                prepare()
            path = root/'draft/visual-truth-draft.json'
            truth = read_json(path)
            self.assertFalse(truth['reviewed'])
            self.assertIsNone(truth['frames'][0]['group_id'])
            self.assertEqual(truth['frames'][0]['evaluation_region'],[])
            self.assertFalse((root/'draft/run-manifest.json').exists())
            with patch('sys.argv',['prepare','--freeze',str(path),'--out',str(root/'frozen')]):
                with self.assertRaisesRegex(ValueError,'Complete reviewed'):
                    prepare()

    def test_freeze_and_loading_detect_asset_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'verified_regions')
            path = root/'truth/frozen-truth.json'
            truth = load_truth(path)
            with self.assertRaises(FileExistsError):
                freeze_truth(root/'raw/truth-draft.json',root/'truth')
            mask = path.parent/truth['frames'][0]['people'][0]['mask']
            mask.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'mask hash'):
                load_truth(path)

    def test_image_replacement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'full_frame')
            path = root/'truth/frozen-truth.json'
            truth = load_truth(path)
            Image.new('RGB',(240,120),'white').save(path.parent/truth['frames'][0]['image'])
            with self.assertRaisesRegex(ValueError,'image hash'):
                load_truth(path)

    def test_verified_runner_uses_only_masks_and_pixels_and_never_sam(self):
        from scripts.run_team_pilot import main as run
        from scripts.evaluate_team_pilot import main as score
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'verified_regions')
            with patch('scripts.run_team_pilot.Sam3') as sam, patch('sys.argv',[
                'run','--manifest',str(root/'run/run-manifest.json'),'--out',str(root/'results')]):
                run()
                sam.assert_not_called()
            with patch('sys.argv',['score','--truth',str(root/'truth/frozen-truth.json'),
                '--detections',str(root/'results/detections.jsonl'),'--out',str(root/'report')]):
                score()
            report = read_json(root/'report/evaluation.json')
            self.assertEqual(report['evaluation_mode'],'verified_regions')
            self.assertEqual(report['precision_team_and_role'],1)
            self.assertFalse(report['model_executed'])
            self.assertTrue(report['synthetic'])
            self.assertFalse(report['sampling']['sufficient_for_exploratory_analysis'])
            truth = load_truth(root/'truth/frozen-truth.json')
            rows = [json.loads(line) for line in (root/'results/detections.jsonl').read_text().splitlines()]
            provenance = read_json(root/'results/provenance.json')
            validate_result_binding(truth,root/'truth/frozen-truth.json',rows,provenance)
            bad_provenance = dict(provenance,started_at_utc='2000-01-01T00:00:00+00:00')
            with self.assertRaisesRegex(ValueError,'before predictions'):
                validate_result_binding(truth,root/'truth/frozen-truth.json',rows,bad_provenance)
            with self.assertRaisesRegex(ValueError,'binding mismatch'):
                validate_result_binding(truth,root/'truth/frozen-truth.json',rows,
                                        dict(provenance,evaluation_mode='full_frame'))
            for key in ('truth_sha256','image_sha256','pixel_sha256','evaluation_mode'):
                bad = copy.deepcopy(rows)
                bad[0][key] = 'bad'
                with self.subTest(key=key), self.assertRaises(ValueError):
                    validate_result_binding(truth,root/'truth/frozen-truth.json',bad,provenance)
            with (root/'results/detections.jsonl').open('a') as stream:
                stream.write('\n')
            with patch('sys.argv',['score','--truth',str(root/'truth/frozen-truth.json'),
                '--detections',str(root/'results/detections.jsonl'),'--out',str(root/'bad-report')]):
                with self.assertRaisesRegex(ValueError,'unchanged'):
                    score()

    def test_runner_rejects_truth_edit_before_loading_model(self):
        from scripts.run_team_pilot import main as run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'full_frame')
            path = root/'truth/frozen-truth.json'
            truth = read_json(path)
            truth['frames'][0]['people'][0]['classification'] = 'A'
            write_json(path,truth)
            with patch('scripts.run_team_pilot.Sam3') as sam, patch('sys.argv',[
                'run','--manifest',str(root/'run/run-manifest.json'),
                '--checkpoint','unused','--checkpoint-sha256','0'*64,'--out',str(root/'results')]):
                with self.assertRaisesRegex(ValueError,'binding mismatch'):
                    run()
                sam.assert_not_called()
            self.assertFalse((root/'results').exists())

    def test_future_freeze_time_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'full_frame')
            path = root/'truth/frozen-truth.json'
            truth = read_json(path)
            truth['frozen_at_utc'] = '9999-01-01T00:00:00+00:00'
            write_json(path,truth)
            with self.assertRaisesRegex(ValueError,'precede execution'):
                load_truth(path)

    def test_region_mode_rejects_missing_or_repeated_regions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'verified_regions')
            truth = load_truth(root/'truth/frozen-truth.json')
            with self.assertRaisesRegex(ValueError,'every frozen region'):
                evaluate(truth,[dict(frame=1,detections=[])])

    def test_freeze_requires_individual_verified_mask_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'inputs'
            create_inputs(root,'verified_regions')
            path = root/'raw/truth-draft.json'
            truth = read_json(path)
            truth['frames'][0]['people'][0]['mask_reviewed'] = False
            write_json(path,truth)
            with self.assertRaisesRegex(ValueError,'reviewed individual masks'):
                freeze_truth(path,root/'bad-freeze')
            self.assertFalse((root/'bad-freeze').exists())

    def test_referee_precision_does_not_add_athlete_coverage(self):
        truth = truth_for([person(role='referee',classification='referee')])
        report = evaluate(truth,[dict(frame=0,detections=[prediction(classification='referee',role='referee')])])
        self.assertEqual(report['precision_team_and_role'],1)
        self.assertIsNone(report['coverage'])

    def test_synthetic_demo_exercises_both_modes_without_weights(self):
        from scripts.demo_team_evaluation import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'demo'
            with patch('sys.argv',['demo','--out',str(root)]), patch('builtins.print'):
                main()
            for mode in ('full_frame','verified_regions'):
                report = read_json(root/mode/'report/evaluation.json')
                self.assertEqual(report['precision_accepted'],1)
                self.assertEqual(report['evaluation_mode'],mode)
                self.assertFalse(report['model_executed'])
                self.assertTrue(report['synthetic'])


if __name__=='__main__':
    unittest.main()
