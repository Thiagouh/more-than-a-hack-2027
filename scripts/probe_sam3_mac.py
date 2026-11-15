"""Probe Apple image compatibility; optionally benchmark one authorized SAM3 box."""
import argparse
import faulthandler
import importlib.util
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from src.team_identification.pilot import Sam3, verify_installation, sha256
from src.team_identification.mac_compat import enable, VERSION
from src.team_identification.manifests import read_json, write_json, load_image, validate_initial


def mask_review(image, mask, box):
    overlay=np.asarray(image).copy()
    overlay[mask]=(.55*overlay[mask]+.45*np.array([255,0,200])).astype(np.uint8)
    x0,y0,x1,y1=box
    pad=max(x1-x0,y1-y0)
    region=(max(0,int(x0-pad)),max(0,int(y0-pad)),min(image.width,int(x1+pad)),min(image.height,int(y1+pad)))
    review=Image.fromarray(overlay).crop(region)
    return review.resize((review.width*3,review.height*3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--checkpoint-sha256')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--diagnostics',action='store_true',help='Dump Python stack every 60 seconds during a stalled probe')
    parser.add_argument('--box-context-crop',action='store_true',help='Use a 3x box-height context crop and restore mask to original coordinates')
    args=parser.parse_args()
    if args.diagnostics:faulthandler.dump_traceback_later(60,repeat=True)
    if bool(args.checkpoint) != bool(args.checkpoint_sha256):
        parser.error('Checkpoint and its authorized SHA256 must be supplied together')
    if args.checkpoint and not args.manifest:
        parser.error('A reviewed initial manifest is required for inference')
    if args.out.exists():
        raise FileExistsError(args.out)
    import torch
    import psutil
    if args.device=='mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable')
    package=importlib.util.find_spec('sam3')
    if package is None:raise RuntimeError('SAM3 is not installed')
    verify_installation(package.origin)
    enable(package.origin,args.device)
    from sam3.model_builder import _create_sam3_transformer
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.vitdet import Mlp, Attention
    from sam3.model.sam3_image_processor import Sam3Processor
    started=time.perf_counter()
    with torch.inference_mode():
        x=torch.ones((1,4,8),device=args.device)
        mlp=Mlp(8,16).to(args.device).eval()
        y=mlp(x)
        if not torch.isfinite(y).all().item():raise RuntimeError('Nonfinite MLP output')
        pos=PositionEmbeddingSine(256,normalize=True,precompute_resolution=28)
        encoding=pos(torch.ones((1,1,2,2),device=args.device))
        if encoding.device.type != args.device:raise RuntimeError('Position cache on wrong device')
        transformer=_create_sam3_transformer().to(args.device).eval()
        reference=Attention(32,num_heads=4,input_size=(2,2),use_rope=True).eval()
        portable=Attention(32,num_heads=4,input_size=(2,2),use_rope=True,use_rope_real=True).eval()
        portable.load_state_dict({k:v for k,v in reference.state_dict().items() if 'freqs_cis' not in k},strict=False)
        sample=torch.randn((1,2,2,32))
        expected=reference(sample)
        actual=portable.to(args.device)(sample.to(args.device)).cpu()
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        # Constructor exercises the previously hard-coded CUDA coordinate cache.
        del transformer,mlp,pos
    if args.device=='mps':torch.mps.synchronize()
    report=dict(platform=platform.platform(),torch=torch.__version__,device=args.device,
                mps_available=torch.backends.mps.is_available(),compatibility=VERSION,
                component_checks=['SAM3 image API import',args.device+' tensor arithmetic','MLP forward',
                                  'position encoding cache','decoder construction','real rotary attention vs CPU complex reference'],
                component_seconds=time.perf_counter()-started,
                segmentation_executed=False,
                process_rss_bytes=psutil.Process().memory_info().rss)
    if args.checkpoint:
        spec=read_json(args.manifest)
        image=load_image(args.manifest.parent/spec['initial_image'])
        validate_initial(spec,image)
        if spec.get('reviewed') is not True:raise ValueError('Reviewed boxes required')
        box=spec['initial_annotations'][0]['box']
        started=time.perf_counter()
        print('Loading authorized checkpoint...',flush=True)
        sam=Sam3(args.checkpoint,args.checkpoint_sha256,device=args.device)
        load_seconds=time.perf_counter()-started
        print('Model loaded; segmenting first reviewed box...',flush=True)
        started=time.perf_counter()
        masks,scores=sam.segment_boxes(image,[box],context_crop=args.box_context_crop)
        if args.device=='mps':torch.mps.synchronize()
        report.update(segmentation_executed=True,model_load_seconds=load_seconds,
                      inference_seconds=time.perf_counter()-started,
                      mask_pixels=int(masks[0].sum()),score=float(scores[0]),
                      mask_available=bool(masks[0].any()),
                      checkpoint_sha256=args.checkpoint_sha256.lower(),
                      manifest_sha256=sha256(args.manifest),environment=sam.environment,
                      box_context_crop=args.box_context_crop,
                      process_rss_bytes=psutil.Process().memory_info().rss)
        args.out.mkdir(parents=True,exist_ok=False)
        Image.fromarray(masks[0].astype(np.uint8)*255).save(args.out/'single-box-mask.png')
        mask_review(image,masks[0],box).save(args.out/'single-box-review.png')
    else:
        args.out.mkdir(parents=True,exist_ok=False)
    if args.device=='mps':
        report.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    report['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    write_json(args.out/'mac-probe.json',report)
    if args.diagnostics:faulthandler.cancel_dump_traceback_later()
    print(report)


if __name__=='__main__':main()
