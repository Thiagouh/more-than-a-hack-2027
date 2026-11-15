import tempfile
import unittest
from pathlib import Path
from PIL import Image
from scripts.review_sam3_observation import review
from src.team_identification.manifests import write_json,read_json
from src.team_identification.pilot import sha256
from tests.test_visual_collective_bridge import observation


class SAMObservationReviewTests(unittest.TestCase):
    def fixture(self,root):
        image=root/'image.png';Image.new('RGB',(20,40),'green').save(image)
        mask=root/'mask.png';Image.new('L',(20,40),255).save(mask)
        spec=dict(schema='sam3-visual-control-v1',sam3_executed=True,synthetic=True,
            image=str(image),image_sha256=sha256(image),mask_file=str(mask),mask_sha256=sha256(mask),
            mask_reviewed=False,quality_reason=None,observations=[observation(0)])
        package=root/'package.json';write_json(package,spec)
        verdict=dict(package_sha256=sha256(package),image_sha256=sha256(image),mask_sha256=sha256(mask),
            approved=True,reviewer='synthetic test reviewer',notes='Synthetic mask fixture only')
        path=root/'review.json';write_json(path,verdict)
        return package,path,spec,verdict

    def test_approval_preserves_candidates_and_binds_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package,path,spec,_=self.fixture(root);before=sha256(package)
            result=review(package,path,root/'approved')
            self.assertTrue(result['mask_reviewed']);self.assertEqual(len(result['observations']),1)
            self.assertEqual(result['candidate_sha256'],before);self.assertEqual(sha256(package),before)

    def test_rejection_does_not_produce_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package,path,spec,verdict=self.fixture(root)
            verdict['approved']=False;write_json(path,verdict)
            result=review(package,path,root/'rejected')
            self.assertFalse(result['mask_reviewed']);self.assertEqual(result['observations'],[])
            self.assertTrue(Path(spec['mask_file']).exists())

    def test_changed_package_or_asset_cannot_reuse_review(self):
        for kind in ('package','mask','image-binding'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);package,path,spec,verdict=self.fixture(root)
                if kind=='package':spec['quality_reason']='small_mask';write_json(package,spec)
                elif kind=='mask':Path(spec['mask_file']).write_bytes(b'changed')
                else:verdict['image_sha256']='0'*64;write_json(path,verdict)
                with self.assertRaises(ValueError):review(package,path,root/'bad')
                self.assertFalse((root/'bad').exists())

    def test_held_ball_review_is_explicitly_recorded_not_automatic_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package,path,spec,verdict=self.fixture(root)
            spec['observations'][0].update(control_evidence='none',ball_control_observed=False)
            write_json(package,spec);verdict.update(package_sha256=sha256(package),held_ball_verified=True)
            write_json(path,verdict);result=review(package,path,root/'approved')
            self.assertTrue(result['held_ball_manual_review'])
            self.assertEqual(result['observations'][0]['control_evidence'],'reviewed_held_ball')
            self.assertEqual(result['observations'][0]['reviewer'],verdict['reviewer'])

    def test_review_cannot_override_geometric_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);package,path,spec,verdict=self.fixture(root)
            spec['quality_reason']='mask_escapes_reviewed_box';write_json(package,spec)
            verdict['package_sha256']=sha256(package);write_json(path,verdict)
            with self.assertRaisesRegex(ValueError,'automatic mask blockers'):review(package,path,root/'bad')
