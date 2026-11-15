"""Generate and run both v2 examples using synthetic pixels and a simulated SAM."""
import argparse
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from src.team_identification.manifests import freeze, write_json, read_json, pixel_sha256
from src.team_identification.pilot import SAM3_REVISION, sha256
from src.team_identification.protocol import freeze_truth


def create_inputs(out, mode):
    """Small reusable fixture: reference/development/evaluation are distinct segments."""
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    raw = out/'raw'
    raw.mkdir()
    reference_rows = []
    palette = [('H','player','sun',(255,0,0)),('H','player','shade',(160,0,0)),
        ('A','player','sun',(0,0,255)),('A','player','shade',(0,0,160)),
        (None,'referee','sun',(255,255,0)),('H','goalkeeper','sun',(0,255,0)),
        ('A','goalkeeper','sun',(0,255,255))]
    for i,(team,role,lighting,color) in enumerate(palette):
        image,mask = raw/f'ref-{i}.png',raw/f'ref-mask-{i}.png'
        Image.new('RGB',(40,80),color).save(image)
        Image.new('L',(40,80),255).save(mask)
        reference_rows.append(dict(label=f'ref{i}',team=team,role=role,lighting=lighting,
            approved=True,origin='pre-kickoff',goalkeeper_verified=role=='goalkeeper',
            image=image.name,mask=mask.name,image_sha256=sha256(image),mask_sha256=sha256(mask),
            source_sha256='1'*64,source_segment_id='pre-kickoff'))
    candidate = raw/'candidate.json'
    write_json(candidate,dict(stage='candidate',sam3_revision=SAM3_REVISION,
        checkpoint_sha256='0'*64,mask_reviewed=True,reviewer='synthetic-demo',references=reference_rows))
    freeze(candidate,out/'references')
    image = np.full((120,240,3),90,np.uint8)
    people,masks = [],[]
    for i,(team,role,color) in enumerate([('H','player',(255,0,0)),('A','player',(0,0,255)),
                                       ('H','goalkeeper',(0,255,0)),('referee','referee',(255,255,0))]):
        x = 10+i*55
        box = [x,20,x+30,100]
        mask = np.zeros(image.shape[:2],bool)
        mask[20:100,x:x+30] = True
        masks.append(mask)
        image[mask] = color
        path = raw/f'person-{i}.png'
        Image.fromarray(mask.astype('uint8')*255).save(path)
        people.append(dict(label=f'p{i}',box=box,role=role,classification=team,evaluable=True,
            partial=False,lighting='sun' if i%2 else 'shade',occlusion='isolated',
            mask=path.name,mask_sha256=sha256(path),mask_reviewed=True))
    image_path = raw/'evaluation.png'
    Image.fromarray(image).save(image_path)
    truth = dict(schema_version=2,stage='draft',reviewed=True,reviewer='synthetic-demo',
        annotations_complete=True,synthetic=True,evaluation_mode=mode,sample_kind='main',
        excluded_segments=[dict(source_sha256='1'*64,source_segment_id='pre-kickoff',split='reference'),
                           dict(source_sha256='1'*64,source_segment_id='development',split='development')],
        frames=[dict(frame=1,timestamp_s=1,image=image_path.name,image_sha256=sha256(image_path),
            pixel_sha256=pixel_sha256(image),width=240,height=120,source_sha256='1'*64,
            source_segment_id='held-out',group_id='synthetic-lance-1',temporal_origin_known=True,
            source_time_s=60,preparation='Generated flat-color synthetic rectangles; no video/model',
            evaluation_region=[[[0,0],[240,0],[240,120],[0,120]]],people=people)])
    draft = raw/'truth-draft.json'
    write_json(draft,truth)
    freeze_truth(draft,out/'truth')
    from scripts.prepare_team_evaluation import main as prepare
    with patch('sys.argv',['prepare','--truth',str(out/'truth/frozen-truth.json'),
               '--references',str(out/'references/frozen-references.json'),'--out',str(out/'run')]):
        prepare()
    spec = read_json(out/'run/run-manifest.json')
    spec['synthetic'] = True
    write_json(out/'run/run-manifest.json',spec)
    return np.asarray(masks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    from scripts.run_team_pilot import main as run
    from scripts.evaluate_team_pilot import main as score
    for mode in ('full_frame','verified_regions'):
        out = args.out/mode
        masks = create_inputs(out,mode)
        argv = ['run','--manifest',str(out/'run/run-manifest.json'),'--out',str(out/'results')]
        if mode=='full_frame':
            argv += ['--checkpoint','SIMULATED-NO-WEIGHTS','--checkpoint-sha256','0'*64]
        with patch('scripts.run_team_pilot.Sam3') as sam, patch('sys.argv',argv):
            sam.return_value.segment.return_value = (masks,np.ones(len(masks)))
            sam.return_value.environment = {'simulated':True}
            run()
        # The full-frame path exercised a simulated adapter, never a real model.
        provenance_path = out/'results/provenance.json'
        provenance = read_json(provenance_path)
        provenance.update(model_executed=False,synthetic=True,sam_simulated=mode=='full_frame')
        write_json(provenance_path,provenance)
        with patch('sys.argv',['score','--truth',str(out/'truth/frozen-truth.json'),
            '--detections',str(out/'results/detections.jsonl'),'--out',str(out/'report')]):
            score()
    print(f'Synthetic examples only; no model executed. Reports: {args.out.resolve()}')


if __name__=='__main__':
    main()
