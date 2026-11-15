import unittest
from unittest.mock import patch
from types import SimpleNamespace
from src.team_identification.mac_compat import transform, enable, Overlay
from src.team_identification.pilot import Sam3


class MacCompatibilityTests(unittest.TestCase):
    def test_device_validation_precedes_checkpoint(self):
        with self.assertRaisesRegex(ValueError,'Device'):
            Sam3('missing','0'*64,device='metal')

    def test_edt_fails_instead_of_faking_output(self):
        namespace={}
        exec(transform('sam3.model.edt','', 'mps'),namespace)
        with self.assertRaisesRegex(RuntimeError,'Video EDT'):
            namespace['edt_triton'](None)

    def test_pinned_anchor_guard(self):
        for name in ('sam3.model.position_encoding','sam3.model.decoder',
                     'sam3.model_builder','sam3.perflib.fused'):
            with self.assertRaises(RuntimeError):transform(name,'changed source','mps')

    def test_cache_device_replacement(self):
        self.assertIn("device='mps'",transform('sam3.model.decoder','f(device="cuda")','mps'))

    def test_cuda_cannot_enable_overlay(self):
        with self.assertRaises(ValueError):enable('/tmp/sam3/__init__.py','cuda')

    def test_backend_change_requires_fresh_process(self):
        with patch('sys.meta_path',[Overlay('/tmp/sam3','mps')]):
            with self.assertRaises(RuntimeError):enable('/tmp/sam3/__init__.py','cpu')

    def test_synchronous_validation_preserves_failure(self):
        namespace={}
        source=('def validate(condition):\n    torch._assert_async(condition)\n'
                'class Encoder:\n'
                '    def points(self, points, points_mask, img_feats):\n'
                '        n_points, bs = points.shape[:2]\n'
                '    def boxes(self, boxes, boxes_mask, img_feats):\n'
                '        n_boxes, bs = boxes.shape[:2]\n')
        exec(transform('sam3.model.geometry_encoders',source,'mps'),namespace)
        namespace['validate'](SimpleNamespace(item=lambda:True))
        with self.assertRaises(RuntimeError):
            namespace['validate'](SimpleNamespace(item=lambda:False))
        encoder=namespace['Encoder']()
        encoder.d_model=256
        features=SimpleNamespace(new_empty=lambda shape:shape)
        prompt=SimpleNamespace(shape=(0,1,2))
        self.assertEqual(encoder.points(prompt,'mask',features),((0,1,256),'mask'))
        self.assertEqual(encoder.boxes(prompt,'mask',features),((0,1,256),'mask'))

    def test_checkpoint_rotary_components_are_preserved(self):
        source=('def device_setup(model, device):\n    if device == "cuda":\n        model = model.cuda()\n'
                'def _create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=False):\n    pass\n'
                'def load(model, sam3_image_ckpt):\n    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)\n')
        namespace={}
        exec(transform('sam3.model_builder',source,'mps'),namespace)
        key='block.freqs_cis'
        ckpt={key:SimpleNamespace(real=3,imag=4)}
        model=SimpleNamespace(state_dict=lambda:{key+'_real':None},load_state_dict=lambda weights,strict:([],[]))
        namespace['load'](model,ckpt)
        self.assertEqual((ckpt[key],ckpt[key+'_real'],ckpt[key+'_imag']),(3,3,4))
