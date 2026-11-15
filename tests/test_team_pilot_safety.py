import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from src.ball_events.uniforms import pixel_histogram
from src.team_identification.pilot import (descriptor, torso_evidence, TeamClassifier,
                                           Sam3, sha256, SAM3_REVISION, verify_installation,
                                           reviewed_box_quality)
from src.team_identification.manifests import (freeze, load_frozen, write_json,
                                              validate_initial, validate_run)
from src.team_identification.evaluation import evaluate


class PilotSafetyTests(unittest.TestCase):
    def test_boundary_pixels_removed_without_reframing(self):
        image=np.full((100,60,3),(0,255,0),np.uint8)
        mask=np.zeros((100,60),bool)
        mask[10:90,10:50]=True
        mask[26:54,21:24]=False
        core,q,reason=torso_evidence(image,mask,return_mask=True)
        self.assertIsNone(reason)
        self.assertGreater(q['boundary_torso_pixels'],0)
        self.assertFalse(core[30,24])
        self.assertTrue(core[30,30])
        self.assertFalse(core[:26].any())
        excluded=np.zeros_like(mask);excluded[30:35,30:35]=True
        other,_,_=torso_evidence(image,mask,excluded,return_mask=True)
        self.assertFalse(other[excluded].any())
        self.assertTrue(np.all(other <= core))

    def test_box_leakage_rejects_instead_of_clipping(self):
        mask=np.ones((80,40),bool)
        before=mask.copy()
        q,reason=reviewed_box_quality(mask,[10,0,30,80])
        self.assertEqual(reason,'mask_escapes_reviewed_box')
        self.assertEqual(q['outside_box_fraction'],.5)
        np.testing.assert_array_equal(mask,before)
        with self.assertRaises(ValueError):
            reviewed_box_quality(mask,[-1,0,40,80])

    def test_offline_audit_preserves_candidates_and_detects_tampering(self):
        from scripts.audit_team_masks import audit
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            manifest,spec=self.make_candidate(root)
            for row in spec['references']:
                row['box']=[0,0,40,80]
            write_json(manifest,spec)
            before=sha256(manifest)
            audit(manifest,root/'audit')
            report=json.loads((root/'audit/audit.json').read_text())
            self.assertFalse(report['model_executed'])
            self.assertTrue(all(not r['approved'] for r in report['findings']))
            self.assertEqual(sha256(manifest),before)
            with self.assertRaises(FileExistsError):
                audit(manifest,root/'audit')
            (root/spec['references'][0]['mask']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                audit(manifest,root/'bad')

    def setUp(self):
        self.red=np.full((80,40,3),(255,0,0),np.uint8)
        self.blue=np.full((80,40,3),(0,0,255),np.uint8)
        self.mask=np.ones((80,40),bool)

    def test_all_pixels_and_external_invariance(self):
        image=self.red.copy()
        image[16:44,8:20]=(0,0,255)
        expected=pixel_histogram(image[16:44,8:32].reshape(-1,3))
        np.testing.assert_allclose(descriptor(image,self.mask),expected)
        image[:16]=(0,255,0)
        image[44:]=(0,255,0)
        np.testing.assert_allclose(descriptor(image,self.mask),expected)

    def test_overlap_does_not_reframe_torso(self):
        excluded=np.zeros_like(self.mask)
        excluded[:16]=True
        d,q,reason=torso_evidence(self.red,self.mask,excluded)
        self.assertIsNone(reason)
        self.assertEqual(q['torso_pixels'],28*24)
        self.assertEqual(q['removed_torso_pixels'],0)
        np.testing.assert_allclose(d,descriptor(self.red,self.mask))
        excluded[:35]=True
        self.assertEqual(torso_evidence(self.red,self.mask,excluded)[2],'insufficient_torso')

    def test_invalid_masks_and_thresholds(self):
        for mask in (np.full((80,40),.5),np.ones((10,10)),np.full((80,40),np.nan)):
            with self.assertRaises(ValueError):
                descriptor(self.red,mask)
        with self.assertRaises(ValueError):
            descriptor(self.red.astype(float)*np.nan,self.mask)
        with self.assertRaises(ValueError):
            TeamClassifier([],max_distance=float('nan'))
        with self.assertRaises(ValueError):
            TeamClassifier([(self.red,self.mask,'H','goalkeeper'),(self.blue,self.mask,'A','player')])

    def test_duplicate_containment_and_wide_abstain(self):
        classifier=TeamClassifier([(self.red,self.mask,'H','player'),(self.blue,self.mask,'A','player')])
        rows=classifier.classify(self.red,[self.mask,self.mask])
        self.assertTrue(all(r['reason']=='duplicate_mask' and r['team'] is None for r in rows))
        small=self.mask.copy();small[:10]=False
        self.assertTrue(all(r['team'] is None for r in classifier.classify(self.red,[self.mask,small])))
        self.assertEqual(torso_evidence(np.zeros((40,80,3),np.uint8),np.ones((40,80),bool))[2],
                         'merged_or_non_upright_mask')

    def test_checkpoint_fails_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'weights';path.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                Sam3(path,'0'*64)
            with self.assertRaises(ValueError):
                Sam3(path,'bad')

    def test_revision_and_dirty_checkout_rejected(self):
        for responses in (['wrong',''],[SAM3_REVISION,' M sam3/model_builder.py']):
            with patch('src.team_identification.pilot.subprocess.check_output',side_effect=responses):
                with self.assertRaises(RuntimeError):verify_installation('/tmp/sam3/__init__.py')
        with patch('src.team_identification.pilot.subprocess.check_output',side_effect=[SAM3_REVISION,'']):
            self.assertEqual(verify_installation('/tmp/sam3/__init__.py'),SAM3_REVISION)

    def test_missing_box_mask_preserves_slot(self):
        sam=Sam3.__new__(Sam3)
        sam.segment=lambda image,box:(np.zeros((0,80,40),bool),np.zeros(0))
        masks,scores=sam.segment_boxes(Image.fromarray(self.red),[[0,0,40,80]])
        self.assertEqual(masks.shape,(1,80,40))
        self.assertFalse(masks.any())
        self.assertEqual(scores.tolist(),[0.])

    def test_context_crop_restores_original_coordinates(self):
        sam=Sam3.__new__(Sam3)
        calls=[]
        def segment(image,box):
            calls.append((image.size,box))
            return np.ones((1,image.height,image.width),bool),np.array([.9])
        sam.segment=segment
        masks,scores=sam.segment_boxes(Image.fromarray(self.red),[[10,20,20,40]],context_crop=True)
        self.assertEqual(masks.shape,(1,80,40))
        self.assertEqual(calls,[((40,60),[10,20,20,40])])
        self.assertTrue(masks[0,:60].all())
        self.assertFalse(masks[0,60:].any())

    def make_candidate(self,root):
        refs=[]
        for index,(team,role,lighting,color) in enumerate([
            ('H','player','sun',(255,0,0)),('H','player','shade',(150,0,0)),
            ('A','player','sun',(0,0,255)),('A','player','shade',(0,0,150)),
            (None,'referee','sun',(255,255,0))]):
            image=root/f'image{index}.png';mask=root/f'mask{index}.png'
            Image.new('RGB',(40,80),color).save(image)
            Image.fromarray(self.mask.astype('uint8')*255).save(mask)
            refs.append(dict(label=f'P{index}',team=team,role=role,lighting=lighting,
                             approved=True,origin='pre-kickoff',image=image.name,mask=mask.name,
                             image_sha256=sha256(image),mask_sha256=sha256(mask)))
        spec=dict(stage='candidate',sam3_revision=SAM3_REVISION,checkpoint_sha256='0'*64,mask_reviewed=True,reviewer='test',
                  references=refs)
        path=root/'candidate.json';write_json(path,spec)
        return path,spec

    def test_freeze_approval_lighting_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path,spec=self.make_candidate(root)
            bad=copy.deepcopy(spec);bad['mask_reviewed']=False;write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'approve'):
                freeze(path,root/'bad')
            bad=copy.deepcopy(spec);bad['references'][1]['approved']=False;write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'sun and shade'):
                freeze(path,root/'bad')
            write_json(path,spec);freeze(path,root/'frozen')
            bundle=root/'frozen/frozen-references.json'
            classifier,frozen=load_frozen(bundle)
            self.assertEqual(classifier.classify(self.red,[self.mask])[0]['team'],'H')
            with self.assertRaises(FileExistsError):
                freeze(path,root/'frozen')
            (bundle.parent/frozen['references'][0]['mask']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                load_frozen(bundle)

    def test_run_manifest_split_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path,_=self.make_candidate(root);freeze(path,root/'frozen')
            bundle=root/'frozen/frozen-references.json'
            _,frozen=load_frozen(bundle)
            image=root/'eval.png';Image.new('RGB',(40,80),(254,0,0)).save(image)
            spec=dict(source='broadcast',frozen_references=str(bundle),frozen_sha256=sha256(bundle),
                      frames=[dict(frame=1,timestamp_s=1,image=image.name,image_sha256=sha256(image),split='evaluation')])
            validate_run(spec,root,frozen)
            spec['frames'][0]['split']='reference'
            with self.assertRaises(ValueError):validate_run(spec,root,frozen)

    def test_initial_box_validation(self):
        spec=dict(source='broadcast',reviewed=False,initial_annotations=[
            dict(label='H',team='H',role='player',box=[0,0,40,80]),
            dict(label='A',team='A',role='player',box=[0,0,40,80])])
        validate_initial(spec,Image.fromarray(self.red))
        spec['initial_annotations'][0]['box'][0]=-1
        with self.assertRaises(ValueError):validate_initial(spec,Image.fromarray(self.red))

    def test_runner_mocked_without_pixel_mutation(self):
        from scripts.run_team_pilot import main
        from scripts.demo_team_evaluation import create_inputs
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'demo'
            masks=create_inputs(root,'full_frame')
            image=root/'raw/evaluation.png'
            manifest=root/'run/run-manifest.json'
            before=sha256(image)
            with patch('scripts.run_team_pilot.Sam3') as sam, patch('sys.argv',[
                'pilot','--manifest',str(manifest),'--checkpoint','unused',
                '--checkpoint-sha256','0'*64,'--out',str(root/'out')]):
                sam.return_value.segment.return_value=(masks,np.ones(len(masks)))
                sam.return_value.environment={'mock':True}
                main()
            self.assertEqual(sha256(image),before)
            row=json.loads((root/'out/detections.jsonl').read_text())
            self.assertEqual(row['detections'][0]['classification'],'H')
            self.assertTrue((root/'out/provenance.json').exists())

    def test_evaluation_duplicates_misses_and_uncertainty(self):
        from tests.team_evaluation_fixtures import person,truth_for,prediction
        truth=truth_for([person('p1',0),person('p2',30)])
        predictions=[dict(frame=0,detections=[prediction(),prediction(classification='uncertain',role='unknown')])]
        report=evaluate(truth,predictions)
        self.assertEqual(report['precision_accepted'],1)
        self.assertEqual(report['coverage'],.5)
        self.assertEqual(report['duplicates'],1)
        self.assertEqual(report['missed_people'],1)
        self.assertFalse(report['targets_observed'])
