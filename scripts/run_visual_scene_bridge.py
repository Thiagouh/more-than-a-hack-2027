"""Exercise real SAM observation in the possession engine on one external scene.

No source alignment, ball coordinates, player positions or team truth are invented.
"""
import argparse
from pathlib import Path
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256
from src.team_possession.visual_control import load_visual_package
from src.team_possession.collective import CollectiveModel


def run(package,out):
    spec=read_json(package)
    if spec.get('timeline')!='external' or any(o['frame']!=0 for o in spec['observations']):
        raise ValueError('Single external scene at local frame zero required')
    meta=dict(contract_version='v1',dataset_id='BAR-ATH-reviewed-screen-scene',timeline='external',
        n_frames=1,fps=25,coordinates='metres_corner_fixed',pitch_m=[105.3,68.],
        teams={'H':'Barcelona','A':'Athletic Club'},producers={'unlocated_visual_scene':'SAM3 real RGB'},continuity_starts=[])
    observations=load_visual_package(package,meta)
    # Missing locations remain missing; the actual ball-in-hands review is separate.
    envelope=dict(frame=0,ball=dict(frame=0,x_m=None,y_m=None,source='missing',
        producer='unlocated_visual_scene',quality_flags=[]),players=[])
    baseline=CollectiveModel(meta).feed(envelope)[0]
    state,_,events,_=CollectiveModel(meta).feed(envelope,observations.get(0,[]))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'scene-bridge.json',dict(synthetic=False,sam3_executed=True,
        package_sha256=sha256(package),image_sha256=spec['image_sha256'],
        timeline='external',tactical_alignment_known=False,ball_coordinates_fabricated=False,
        held_ball_manual_review=spec['held_ball_manual_review'],baseline_without_visual=baseline,
        with_sam3_visual=state,detector_events=events,model_reexecuted=True,
        collective_code_sha256=sha256(Path(__file__).resolve().parents[1]/'src/team_possession/collective.py'),
        visual_bridge_sha256=sha256(Path(__file__).resolve().parents[1]/'src/team_possession/visual_control.py')))
    print('Sem visual:',baseline['label'],'| SAM3 + contexto:',state['label'],state['reason'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.package,a.out)


if __name__=='__main__':main()
