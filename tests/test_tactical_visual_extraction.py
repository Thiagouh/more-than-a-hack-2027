import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.prepare_tactical_visual import extract_exact,homographies,project_observed
from src.team_identification.visual_geometry import projected_ball_contact


class TacticalExtractionTests(unittest.TestCase):
    def test_exact_decoded_frames_not_screen_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i,c in enumerate(('red','green','blue','white')):Image.new('RGB',(64,48),c).save(root/f'input-{i:02d}.png')
            video=root/'test.mp4'
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-framerate','25','-i',str(root/'input-%02d.png'),
                '-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            rows=extract_exact(video,[3,1],root/'frames',expected_frames=4)
            self.assertEqual([r['frame'] for r in rows],[1,3])
            first=np.asarray(Image.open(root/'frames/00000001.png')).mean(axis=(0,1))
            self.assertGreater(first[1],first[0]+60)
            last=np.asarray(Image.open(root/'frames/00000003.png')).mean()
            self.assertGreater(last,240)
            with self.assertRaisesRegex(ValueError,'timeline'):extract_exact(video,[0],root/'wrong',expected_frames=5)
            self.assertFalse((root/'wrong').exists())

    def test_projection_uses_normalized_pitch_and_observed_current_points(self):
        h=np.array([[100,0,10],[0,200,20],[0,0,1]],float)
        point=dict(source='observed',x_m=50,y_m=25,quality_flags=[],available_at_frame=3)
        self.assertEqual(project_observed(point,h,[100,50],3),[60.,120.])
        self.assertIsNone(project_observed(point,h,[100,50],2))
        self.assertIsNone(project_observed(dict(point,source='missing'),h,[100,50],3))
        self.assertIsNone(project_observed(dict(point,quality_flags=['jump']),h,[100,50],3))

    def test_homography_missing_or_invalid_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'h.csv'
            p.write_text('frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n0,1,0,0,0,1,0,0,0,1,0,1\n1,nan,nan,nan,nan,nan,nan,nan,nan,nan,nan,0\n')
            hs=homographies(p,[0,1]);self.assertTrue(hs[0]['valid']);self.assertFalse(hs[1]['valid'])
            with self.assertRaisesRegex(ValueError,'lacks'):homographies(p,[2])

    def test_complete_draft_assembly_binds_image_and_source_frame(self):
        from unittest.mock import patch
        from scripts.prepare_tactical_visual import prepare
        from src.team_identification.pilot import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);h=root/'h.csv';h.write_text('frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n0,100,0,0,0,100,0,0,0,1,0,1\n')
            raw=root/'raw.txt';raw.write_text('synthetic fixture')
            video=root/'video.mp4';video.write_bytes(b'synthetic fixture')
            row=dict(frame=0,players=[],ball=dict(source='missing',quality_flags=[]))
            def extract(v,frames,out,**kwargs):
                out.mkdir();p=out/'00000000.png';Image.new('RGB',(100,100),'green').save(p)
                return [dict(frame=0,image=p.name,image_sha256=sha256(p),source_video_sha256=sha256(video),timeline='tactical',same_source_video_frame=True)]
            with patch('games.asset_paths',return_value={'homography':h,'atd_raw':raw}),patch(
                'src.ball_events.inputs.atd',return_value=({'n_frames':1,'fps':25,'pitch_m':[100,50]},lambda:iter([row]))),patch(
                'scripts.prepare_tactical_visual.extract_exact',side_effect=extract):
                prepare(video,[0],root/'out',root)
            draft=json.loads((root/'out/00000000-scene-draft.json').read_text())
            self.assertEqual(draft['frame'],0);self.assertTrue(Path(draft['image']).is_file())
            self.assertFalse(draft['reviewed']);self.assertFalse(draft['ball_observation']['projection_reviewed'])
            self.assertEqual(draft['ball_observation']['source'],'missing')


class ProjectedBallTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((100,60),bool);self.mask[10:90,20:40]=True
        self.scene=dict(frame=10,timeline='tactical',same_source_video_frame=True,image_sha256='1'*64,
            ball_observation=dict(frame=10,available_at_frame=10,source='observed',xy=[30,40],
                image_sha256='1'*64,projection_reviewed=True,homography_valid=True,reprojection_error=.3))

    def test_contact_requires_same_frame_verified_projection(self):
        self.assertTrue(projected_ball_contact(self.mask,self.scene,(0,0))['accepted'])
        import copy
        for field,value in [('frame',9),('available_at_frame',9),('available_at_frame',11),('source','missing'),
                            ('projection_reviewed',False),('homography_valid',False),
                            ('image_sha256','2'*64),('reprojection_error',3),('xy',[float('nan'),40])]:
            scene=copy.deepcopy(self.scene);scene['ball_observation'][field]=value
            self.assertFalse(projected_ball_contact(self.mask,scene,(0,0))['accepted'],field)
        self.assertFalse(projected_ball_contact(self.mask,dict(self.scene,timeline='external'),(0,0))['accepted'])

    def test_ball_below_body_or_in_background_is_not_control(self):
        import copy
        for xy in ([30,95],[5,40],[30,11]):
            s=copy.deepcopy(self.scene);s['ball_observation']['xy']=xy
            self.assertFalse(projected_ball_contact(self.mask,s,(0,0))['accepted'])

    def test_crop_coordinates_restore_without_shifting_source(self):
        import copy
        s=copy.deepcopy(self.scene);s['ball_observation']['xy']=[130,240]
        self.assertTrue(projected_ball_contact(self.mask,s,(100,200))['accepted'])
        self.assertEqual(s['ball_observation']['xy'],[130,240])


if __name__=='__main__':unittest.main()


class SAMBallContactTests(unittest.TestCase):
    def test_ball_candidate_must_be_small_and_near_upper_body(self):
        from src.team_identification.visual_geometry import sam_ball_contact
        person=np.zeros((100,60),bool);person[10:90,20:40]=True
        balls=np.zeros((2,100,60),bool);balls[0,38:41,28:31]=True;balls[1,80:83,28:31]=True
        result=sam_ball_contact(person,balls,np.array([.7,.99]))
        self.assertTrue(result['accepted']);self.assertEqual(result['candidate_index'],0)
        self.assertTrue(result['independent_ball_detector'])
        balls[0]=True
        self.assertFalse(sam_ball_contact(person,balls,np.array([.7,.99]))['accepted'])

    def test_invalid_ball_dimensions_rejected(self):
        from src.team_identification.visual_geometry import sam_ball_contact
        with self.assertRaises(ValueError):sam_ball_contact(np.ones((20,20),bool),np.zeros((1,10,10),bool),np.array([.8]))
