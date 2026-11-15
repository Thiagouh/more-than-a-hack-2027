"""Bounded development experiment: same SAM probabilities at 0.5 and 0.7.

Isolated from production executors. Freeze manual truth before running H01;
expansion requires a hash-bound visual review of H01 and passing metrics.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.team_identification.manifests import load_image, read_json, write_json, validate_initial
from src.team_identification.pilot import Sam3, sha256, torso_evidence, reviewed_box_quality, DESCRIPTOR_VERSION
from src.team_identification.protocol import runtime_provenance

CASES = ('H01', 'H08', 'H09', 'A05', 'R01')
SECONDS = 60
RSS_BYTES = 7_000_000_000


def freeze(manifest, manual, out, context_candidates=None):
    spec, review = read_json(manifest), read_json(manual)
    image_path = (manifest.parent/spec['initial_image']).resolve()
    image = load_image(image_path)
    validate_initial(spec, image)
    if spec.get('reviewed') is not True or review.get('reviewed') is not True or not review.get('reviewer'):
        raise ValueError('Explicit box and manual torso review required')
    if review.get('image_sha256') != sha256(image_path):
        raise ValueError('Manual review image mismatch')
    if set(review['cases']) != set(CASES):
        raise ValueError('Exactly the five fixed cases required')
    rows = []
    for label in CASES:
        row = dict(next(r for r in spec['initial_annotations'] if r['label'] == label))
        entry = review['cases'][label]
        if label == 'H09':
            if entry.get('verifiable') is not False:
                raise ValueError('H09 must remain non-verifiable and abstain')
        else:
            if entry.get('verifiable') is not True or len(entry.get('polygon', [])) < 3:
                raise ValueError('Reviewed manual torso polygon required')
            points = np.asarray(entry['polygon'], dtype=float)
            x0,y0,x1,y1 = row['box']
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or (points[:,0] < x0).any() or (points[:,0] >= x1).any()
                    or (points[:,1] < y0).any() or (points[:,1] >= y1).any()):
                raise ValueError('Manual polygon outside reviewed box')
        rows.append(dict(row, manual=entry))
    out.mkdir(parents=True, exist_ok=False)
    image.save(out/'image.png')
    for row in rows:
        mask = Image.new('L', image.size)
        if row['manual']['verifiable']:
            ImageDraw.Draw(mask).polygon([tuple(p) for p in row['manual']['polygon']], fill=255)
        mask.save(out/f"{row['label']}-manual.png")
        row['manual_sha256'] = sha256(out/f"{row['label']}-manual.png")
    context = []
    if context_candidates is not None:
        candidates = read_json(context_candidates)
        if candidates.get('stage') != 'candidate':
            raise ValueError('Candidate context required')
        for other in candidates['references']:
            if other['image_sha256'] != sha256(image_path):
                raise ValueError('Context must use the same original RGB image')
            source = context_candidates.parent/other['mask']
            if sha256(source) != other['mask_sha256']:
                raise ValueError('Context mask changed')
            target = out/f"context-{other['label']}.png"
            with Image.open(source) as m:
                from src.team_identification.pilot import binary_mask
                binary_mask(m,(image.height,image.width))
                m.save(target)
            context.append(dict(label=other['label'],mask=target.name,sha256=sha256(target)))
    write_json(out/'frozen-development.json', dict(context=context,stage='frozen-development', split='development',
        reviewer=review['reviewer'], review_sha256=sha256(manual), manifest_sha256=sha256(manifest),
        original_image_sha256=sha256(image_path), image_sha256=sha256(out/'image.png'),
        excluded_from_reserved_evaluation=True, descriptor_version=DESCRIPTOR_VERSION,
        cases=rows, provenance=runtime_provenance()))


def load_truth(path):
    spec = read_json(path)
    if (spec.get('stage') != 'frozen-development' or spec.get('split') != 'development'
            or spec.get('descriptor_version') != DESCRIPTOR_VERSION
            or tuple(r['label'] for r in spec['cases']) != CASES):
        raise ValueError('Frozen fixed development set required')
    if sha256(path.parent/'image.png') != spec['image_sha256']:
        raise ValueError('Development image changed')
    for row in spec['cases']:
        if sha256(path.parent/f"{row['label']}-manual.png") != row['manual_sha256']:
            raise ValueError('Manual truth changed')
    for row in spec.get('context', []):
        if sha256(path.parent/row['mask']) != row['sha256']:
            raise ValueError('Frozen overlap context changed')
    return spec


def select_probability(probabilities, scores, box):
    p = np.asarray(probabilities)
    scores = np.asarray(scores)
    if (p.ndim != 3 or len(p) != len(scores) or not np.isfinite(p).all()
            or (p < 0).any() or (p > 1).any() or not np.isfinite(scores).all()):
        raise ValueError('Invalid SAM probabilities')
    if not len(p):
        return np.zeros(p.shape[1:], np.float32), None
    baseline = p > .5
    region = np.zeros(p.shape[1:], bool)
    x0,y0,x1,y1 = map(int, box)
    region[y0:y1,x0:x1] = True
    alignment = [int((m & region).sum())/max(1,int((m | region).sum())) for m in baseline]
    best = max(range(len(p)), key=lambda i:(alignment[i],float(scores[i])))
    return p[best], best


def metrics(core, manual):
    n, target = int(core.sum()), int(manual.sum())
    correct = int((core & manual).sum())
    return dict(extracted=n, correct=correct, manual=target,
                precision=correct/n if n else 0., coverage=correct/target if target else None,
                background_pixels=n-correct)


def passes(baseline, variant):
    return (variant['extracted'] > 0 and baseline['coverage'] is not None
            and variant['precision']-baseline['precision'] >= .05-1e-12
            and variant['coverage'] >= baseline['coverage']-.05-1e-12)


def panel(image, masks, cores, box, dest):
    x,y,X,Y = map(int, box)
    crop = (max(0,x-8),max(0,y-8),min(image.width,X+8),min(image.height,Y+8))
    tiles = [image]
    labels = ['original']
    pixels = np.asarray(image)
    for threshold,mask,core in zip((.5,.7),masks,cores):
        overlay = pixels.copy()
        overlay[mask] = (.5*overlay[mask]+.5*np.array([255,0,255])).astype(np.uint8)
        tiles += [Image.fromarray(overlay), Image.fromarray(np.where(core[:,:,None],pixels,220).astype('uint8'))]
        labels += [f'mask {threshold}',f'torso {threshold}']
    w,h = (crop[2]-crop[0])*5,(crop[3]-crop[1])*5
    sheet = Image.new('RGB',(w*5,h+25),'white')
    draw = ImageDraw.Draw(sheet)
    for i,(tile,label) in enumerate(zip(tiles,labels)):
        sheet.paste(tile.crop(crop).resize((w,h),Image.Resampling.NEAREST),(i*w,25))
        draw.text((i*w+4,5),label,fill='black')
    sheet.save(dest)


def worker(truth, label, checkpoint, digest, out):
    # Parent watchdog includes startup, hash verification, model load, inference and output.
    spec = load_truth(truth)
    row = next(r for r in spec['cases'] if r['label'] == label)
    image = load_image(truth.parent/'image.png')
    started = time.monotonic()
    sam = Sam3(checkpoint,digest,device='cpu')
    loaded = time.monotonic()
    x0,y0,x1,y1 = row['box']
    side = 3*max(x1-x0,y1-y0)
    cx,cy = (x0+x1)/2,(y0+y1)/2
    left,top = max(0,int(cx-side/2)),max(0,int(cy-side/2))
    right,bottom = min(image.width,int(cx+side/2)),min(image.height,int(cy+side/2))
    crop = image.crop((left,top,right,bottom))
    import torch
    with torch.inference_mode():
        state = sam.processor.set_image(crop)
        sam.processor.set_text_prompt(prompt='person',state=state)
        result = sam.processor.add_geometric_prompt(box=[(cx-left)/crop.width,(cy-top)/crop.height,
            (x1-x0)/crop.width,(y1-y0)/crop.height],label=True,state=state)
    # Despite its name, the pinned official masks_logits field is already sigmoid probabilities.
    probabilities = result['masks_logits'].detach().cpu().numpy()[:,0]
    scores = result['scores'].detach().cpu().numpy()
    selected,index = select_probability(probabilities,scores,[x0-left,y0-top,x1-left,y1-top])
    full = np.zeros((image.height,image.width),np.float32)
    full[top:bottom,left:right] = selected
    np.save(out/'probabilities.npy',selected)
    manual = np.asarray(Image.open(truth.parent/f'{label}-manual.png')) != 0
    excluded = np.zeros(manual.shape,bool)
    context = spec.get('context', [])
    if label != 'H01' and not context:
        raise ValueError('Expansion requires frozen same-source masks for overlap exclusion')
    for other in context:
        if other['label'] != label:
            excluded |= np.asarray(Image.open(truth.parent/other['mask'])) != 0
    findings, masks, cores = [], [], []
    for threshold in (.5,.7):
        mask = full > threshold
        core,quality,reason = torso_evidence(image,mask,excluded,return_mask=True)
        box_quality,box_reason = reviewed_box_quality(mask,row['box'])
        # Frozen context is identical for both thresholds; no extra model inference.
        # H01 can run alone because its reviewed box is spatially isolated.
        if label == 'H09':
            core,reason = None,'manual_non_verifiable_posture'
        if box_reason:
            core,reason = None,box_reason
        core = np.zeros_like(mask) if core is None else core
        masks.append(mask);cores.append(core)
        Image.fromarray(mask.astype('uint8')*255).save(out/f'mask-{threshold}.png')
        Image.fromarray(core.astype('uint8')*255).save(out/f'torso-{threshold}.png')
        findings.append(dict(threshold=threshold,quality=quality,box_quality=box_quality,
            abstention_reason=reason,**metrics(core,manual)))
    panel(image,masks,cores,row['box'],out/'comparison.png')
    write_json(out/'result.json',dict(label=label,device='cpu',truth_sha256=sha256(truth),
        checkpoint_sha256=digest,candidate_index=index,score=float(scores[index]) if index is not None else None,
        probability_sha256=sha256(out/'probabilities.npy'),model_load_seconds=loaded-started,
        inference_and_output_seconds=time.monotonic()-loaded,findings=findings,
        numeric_gate=passes(*findings) if label != 'H09' else not cores[1].any(),
        approved=False,references_frozen=False,environment=sam.environment))


def bounded_worker(truth,label,checkpoint,digest,out):
    import psutil
    out.mkdir()
    cmd = [sys.executable,'-m','scripts.experiment_team_masks','worker','--truth',str(truth),
           '--label',label,'--checkpoint',str(checkpoint),'--checkpoint-sha256',digest,'--out',str(out)]
    started,peak,status = time.monotonic(),0,'completed'
    with (out/'worker.log').open('w') as log:
        process = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak,monitored.memory_info().rss)
            except psutil.NoSuchProcess:
                break
            if time.monotonic()-started >= SECONDS or peak > RSS_BYTES:
                status = 'time_limit' if time.monotonic()-started >= SECONDS else 'memory_limit'
                process.kill();break
            time.sleep(.05)
        process.wait()
    elapsed = time.monotonic()-started
    if status == 'completed' and process.returncode:
        status = 'worker_failed'
    report = dict(status=status,wall_seconds=elapsed,peak_sampled_rss_bytes=peak,
                  limit_seconds=SECONDS,limit_rss_bytes=RSS_BYTES,returncode=process.returncode)
    write_json(out/'resources.json',report)
    return report


def run(args):
    load_truth(args.truth)
    if args.resume:
        previous = read_json(args.out/'experiment.json')
        if (previous['truth_sha256'] != sha256(args.truth) or previous['status'] != 'awaiting_H01_visual_review'
                or previous['checkpoint_sha256'] != args.checkpoint_sha256):
            raise ValueError('Resume requires unchanged pending H01 experiment')
    else:
        args.out.mkdir(parents=True,exist_ok=False)
    provenance = runtime_provenance()
    provenance['experiment_script_sha256'] = sha256(__file__)
    report = dict(truth_sha256=sha256(args.truth),checkpoint_sha256=args.checkpoint_sha256,provenance=provenance,cases=[],
                  expanded=False,approved=False,status='running')
    write_json(args.out/'experiment.json',report)
    for label in CASES:
        if label != 'H01' and not report['expanded']:
            break
        resources = (read_json(args.out/label/'resources.json') if args.resume and label == 'H01'
                     else bounded_worker(args.truth,label,args.checkpoint,args.checkpoint_sha256,args.out/label))
        report['cases'].append(dict(label=label,**resources))
        if resources['status'] != 'completed':
            report['status']='stopped_resource_or_worker_failure';break
        result = read_json(args.out/label/'result.json')
        if label == 'H01':
            if not result['numeric_gate']:
                report['status']='negative_H01_numeric_gate';break
            review = read_json(args.h01_review) if args.h01_review else {}
            if not (review.get('result_sha256') == sha256(args.out/label/'result.json')
                    and review.get('reviewer') and review.get('halo_visible') is False):
                report['status']='awaiting_H01_visual_review';break
            report['expanded']=True
    if len(report['cases']) == 5 and report['cases'][-1]['status'] == 'completed':
        results = [read_json(args.out/l/'result.json') for l in CASES if l != 'H09']
        pairs = [r['findings'] for r in results]
        gain = np.mean([v['precision']-b['precision'] for b,v in pairs])
        good = all(v['extracted'] and v['precision'] >= b['precision'] and
            v['coverage'] >= b['coverage']-.05-1e-12 for b,v in pairs)
        report.update(mean_precision_gain=float(gain),status='favorable_requires_human_review'
                      if good and gain >= .05-1e-12 else 'negative_aggregate')
    write_json(args.out/'experiment.json',report)
    print(report['status'],flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--manifest',type=Path,required=True);f.add_argument('--manual',type=Path,required=True)
    f.add_argument('--context-candidates',type=Path)
    for name in ('run','worker'):
        p = sub.add_parser(name)
        p.add_argument('--truth',type=Path,required=True)
        p.add_argument('--checkpoint',type=Path,required=True)
        p.add_argument('--checkpoint-sha256',required=True)
        if name == 'worker':p.add_argument('--label',choices=CASES,required=True)
        else:
            p.add_argument('--h01-review',type=Path)
            p.add_argument('--resume',action='store_true')
    for p in (f,sub.choices['run'],sub.choices['worker']):p.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'freeze':freeze(args.manifest,args.manual,args.out,args.context_candidates)
    elif args.command == 'worker':worker(args.truth,args.label,args.checkpoint,args.checkpoint_sha256,args.out)
    else:run(args)


if __name__ == '__main__':main()
