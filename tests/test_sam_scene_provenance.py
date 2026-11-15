import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from scripts.run_sam3_goalkeeper import worker
from src.team_identification.pilot import sha256
from src.team_identification.manifests import read_json,write_json


class SAMSceneProvenanceTests(unittest.TestCase):
    def run_fixture(self,root,change=False):
        image=root/'image.png';Image.new('RGB',(64,128),(10,200,30)).save(image)
        policy=root/'policy.json';write_json(policy,read_json(Path('configs/team-pilot-goalkeeper-context.json')))
        manifest=root/'scene.json';write_json(manifest,dict(image=str(image),image_sha256=sha256(image),
            frame=0,timeline='external',box=[10,10,40,100],context=dict(half=1,pitch_end='left',
                goalkeeper_candidate=True,distinctive_kit=True,near_defended_goal=True,period_verified=True,
                pitch_end_verified=True,orientation_verified=True,play_origin_verified=True,view='wide_pitch',
                kit_family='green',notes='Synthetic context fixture')))
        out=root/'out';out.mkdir()
        class FakeSAM:
            environment={'synthetic':True,'model_executed':False}
            def __init__(self,*args,**kwargs):pass
            def segment(self,crop,box):
                if change:
                    scene=read_json(manifest);scene['changed_during_inference']=True;write_json(manifest,scene)
                masks=np.zeros((1,crop.height,crop.width),bool);masks[0,10:100,10:40]=True
                return masks,np.array([.8])
        args=argparse.Namespace(manifest=manifest,policy=policy,out=out,checkpoint=root/'unused',checkpoint_sha256='0'*64)
        with patch('scripts.run_sam3_goalkeeper.Sam3',FakeSAM):worker(args)
        return out,manifest

    def test_freezes_actual_input_and_code_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,manifest=self.run_fixture(Path(tmp))
            result=read_json(out/'visual-observations.json')
            self.assertEqual(result['manifest_sha256'],sha256(manifest))
            self.assertEqual(len(result['code_sha256']),4)
            self.assertFalse(result['mask_reviewed'])

    def test_changed_input_is_not_published_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'changed during SAM'):self.run_fixture(root,True)
            self.assertFalse((root/'out/visual-observations.json').exists())
