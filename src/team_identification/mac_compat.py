"""Explicit runtime overlays for pinned SAM3 image-only inference on Apple/CPU.

Never used for CUDA. The official checkout remains unchanged. Unsupported video
EDT fails loudly rather than producing substitute masks.
"""
import importlib.abc
import importlib.util
from pathlib import Path
import sys

VERSION = 'sam3-image-mac-v5'
MODULES = {'sam3.model.edt', 'sam3.model.position_encoding', 'sam3.model.decoder',
           'sam3.model_builder', 'sam3.perflib.fused', 'sam3.model.vitdet',
           'sam3.model.geometry_encoders'}


def transform(name, source, device):
    if name == 'sam3.model.geometry_encoders':
        if 'torch._assert_async(' not in source:
            raise RuntimeError('Pinned geometry validation anchor changed')
        source=source.replace('torch._assert_async(', '_mac_assert_async(')
        anchor='return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()'
        if anchor in source:
            source=source.replace(anchor,'return True if mask.numel() == 0 else bool((mask.cpu().long() == torch.sort(mask.cpu().long(), dim=-1)[0]).all())')
        source=source.replace('scale.pin_memory().to(', 'scale.to(')
        for kind in ('points','boxes'):
            anchor=f'        n_{kind}, bs = {kind}.shape[:2]'
            if source.count(anchor)!=1:
                raise RuntimeError('Pinned empty prompt anchor changed')
            source=source.replace(anchor,anchor+f'\n        if n_{kind} == 0:\n            return img_feats.new_empty((0, bs, self.d_model)), {kind}_mask')
        source += '\n\ndef _mac_assert_async(condition):\n    valid = condition if isinstance(condition, bool) else condition.item()\n    if not valid:\n        raise RuntimeError("Geometry sequence must be right-padded")\n'
    if name == 'sam3.model.edt':
        return ("def edt_triton(*args, **kwargs):\n"
                "    raise RuntimeError('Video EDT is unavailable in the image-only Mac adapter')\n")
    if name in ('sam3.model.position_encoding', 'sam3.model.decoder'):
        if 'device="cuda"' not in source:
            raise RuntimeError('Pinned cache device anchor changed')
        source = source.replace('device="cuda"', 'device='+repr(device))
    if name == 'sam3.model_builder':
        anchor = 'if device == "cuda":\n        model = model.cuda()'
        if source.count(anchor) != 1:
            raise RuntimeError('Pinned device setup anchor changed')
        source = source.replace(anchor, 'model = model.to(device)')
        anchor = 'def _create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=False):'
        if source.count(anchor) != 1:
            raise RuntimeError('Pinned ViT anchor changed')
        source = source.replace(anchor, anchor.replace('use_rope_real=False','use_rope_real=True'))
        anchor='    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)'
        if source.count(anchor)!=1:
            raise RuntimeError('Pinned checkpoint loader anchor changed')
        source=source.replace(anchor,
            '    expected = model.state_dict()\n'
            '    for key, value in list(sam3_image_ckpt.items()):\n'
            '        if key.endswith(".freqs_cis") and key + "_real" in expected:\n'
            '            sam3_image_ckpt[key + "_real"] = value.real\n'
            '            sam3_image_ckpt[key + "_imag"] = value.imag\n'
            '            sam3_image_ckpt[key] = value.real\n'+anchor)
    if name == 'sam3.perflib.fused':
        anchor = '    if torch.is_grad_enabled():'
        if source.count(anchor) != 1:
            raise RuntimeError('Pinned fused MLP anchor changed')
        source = source.replace(anchor,
            '    if mat1.device.type != "cuda":\n'
            '        act = activation() if isinstance(activation, type) else activation\n'
            '        return act(torch.nn.functional.linear(mat1, linear.weight, linear.bias))\n'+anchor)
    if name == 'sam3.model.vitdet':
        anchor='self.register_buffer("freqs_cis", freqs_cis)'
        if source.count(anchor)!=1:
            raise RuntimeError('Pinned rotary buffer anchor changed')
        source=source.replace(anchor,'self.register_buffer("freqs_cis", freqs_cis.real if self.use_rope_real else freqs_cis)')
    return source


class Overlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, package, device):
        self.package, self.device = Path(package), device

    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULES:
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = self.package.parent.joinpath(*module.__name__.split('.')).with_suffix('.py')
        module.__file__ = str(path)
        source = transform(module.__name__, path.read_text(), self.device)
        exec(compile(source, str(path), 'exec'), module.__dict__)


def enable(package_file, device):
    if device not in ('mps','cpu'):
        raise ValueError('Mac overlay is restricted to mps/cpu')
    for finder in sys.meta_path:
        if isinstance(finder,Overlay):
            if finder.device != device:
                raise RuntimeError('Use a new process when changing image backend')
            return
    if any(name in sys.modules for name in MODULES):
        raise RuntimeError('Start a fresh process before importing SAM3 with the Mac adapter')
    sys.meta_path.insert(0,Overlay(Path(package_file).parent,device))
