import tempfile
import unittest
from pathlib import Path
from src.ball_events.demo import metadata
from src.ball_events.contract import InputError
from src.team_identification.manifests import write_json
from src.team_identification.pilot import sha256
from src.team_possession.visual_control import load_visual_package
from tests.test_visual_collective_bridge import observation


class VisualPackageTests(unittest.TestCase):
    def package(self,root):
        image=root/'image.png';image.write_bytes(b'synthetic-image-fixture')
        o=observation(0);o['image_sha256']=sha256(image)
        spec=dict(schema='sam3-visual-control-v1',timeline='synthetic',sam3_executed=True,
            image=str(image),image_sha256=sha256(image),observations=[o])
        path=root/'package.json';write_json(path,spec)
        return path,spec,image

    def test_image_hash_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,spec,image=self.package(Path(tmp))
            self.assertEqual(load_visual_package(path,metadata(5))[0][0]['team'],'H')
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(InputError,'image changed'):load_visual_package(path,metadata(5))

    def test_external_observations_cannot_be_relabelled_tactical_by_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,spec,_=self.package(Path(tmp));spec['timeline']='external';write_json(path,spec)
            meta=metadata(5);meta['timeline']='tactical'
            with self.assertRaisesRegex(InputError,'timeline'):load_visual_package(path,meta)

    def test_tactical_requires_actual_matching_source_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec,_=self.package(root)
            video=root/'source.mp4';video.write_bytes(b'synthetic-video-fixture')
            mask=root/'mask.png';mask.write_bytes(b'synthetic-mask-fixture')
            spec.update(frame=0,timeline='tactical',same_source_video_frame=True,source_video_sha256=sha256(video),
                mask_reviewed=True,mask_reviewer='synthetic-test',mask_file=str(mask),mask_sha256=sha256(mask))
            spec['observations'][0]['timeline']='tactical';write_json(path,spec)
            meta=metadata(5);meta['timeline']='tactical'
            self.assertTrue(load_visual_package(path,meta,video))
            with self.assertRaisesRegex(InputError,'binding required'):load_visual_package(path,meta)
            video.write_bytes(b'other-video')
            with self.assertRaisesRegex(InputError,'video hash mismatch'):load_visual_package(path,meta,video)

    def test_binding_per_observation_and_executed_model_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,spec,_=self.package(Path(tmp));spec['sam3_executed']=False;write_json(path,spec)
            with self.assertRaises(InputError):load_visual_package(path,metadata(5))
            spec['sam3_executed']=True;spec['observations'][0]['image_sha256']='2'*64;write_json(path,spec)
            with self.assertRaisesRegex(InputError,'binding mismatch'):load_visual_package(path,metadata(5))
