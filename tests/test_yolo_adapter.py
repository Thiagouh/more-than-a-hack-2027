"""Coordinate and source contracts without downloading neural weights."""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from src.team_identification.yolo import YoloPeople


class YoloAdapterTests(unittest.TestCase):
    def test_original_frame_masks_and_person_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights = Path(tmp)/'weights.pt'
            weights.write_bytes(b'test weights')
            result = types.SimpleNamespace(
                masks=types.SimpleNamespace(xy=[np.array([[10,20],[20,20],[20,40],[10,40]])]),
                boxes=types.SimpleNamespace(conf=types.SimpleNamespace(cpu=lambda: np.array([.8]))))
            class Model:
                task='segment'
                names={0:'person'}
                ckpt_path=weights
                def predict(self, image, **options):
                    self.options=options
                    return [result]
            model=Model()
            fake=types.SimpleNamespace(YOLO=lambda _:model,__version__='test')
            with patch.dict(sys.modules, ultralytics=fake):
                adapter=YoloPeople(weights,device='cpu')
                masks, metadata=adapter.segment(Image.new('RGB',(100,80)))
            self.assertEqual(masks[0].shape,(80,100))
            self.assertTrue(masks[0][30,15])
            self.assertFalse(masks[0][15,30])
            self.assertEqual(model.options['classes'],[0])
            self.assertEqual(metadata['confidences'],[.8])
            self.assertEqual(len(adapter.environment['checkpoint_sha256']),64)

    def test_boxes_cannot_masquerade_as_silhouettes(self):
        fake=types.SimpleNamespace(YOLO=lambda _:types.SimpleNamespace(task='detect'))
        with patch.dict(sys.modules, ultralytics=fake):
            with self.assertRaisesRegex(ValueError,'segmentation'):
                YoloPeople('unused.pt')
