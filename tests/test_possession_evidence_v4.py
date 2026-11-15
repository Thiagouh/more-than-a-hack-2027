import copy
import tempfile
import json
import unittest
from pathlib import Path
from src.ball_events.demo import metadata,point
from src.ball_events.artifacts import sha
from src.team_possession.game_state import GameState
from src.team_possession.collective import CollectiveModel,DEFAULTS
from src.team_possession.pass_continuity import PassContinuity
from src.team_possession.recovery import apply_recoveries,load_evidence_package
from src.team_possession.render_observations import render_status


def evidence(f,kind='interruption',**kw):
    return dict(evidence_id=f'e-{f}',frame=f,available_at_frame=f,valid_until_frame=f+3,
                kind=kind,verified=True,source_id='rgb-review',reviewer='test',reason='explicit referee signal',**kw)


def recovery(f=0,**kw):
    return dict(observation_id='p',frame=f,available_at_frame=f,reviewer='test',reason='visible person',
        image_sha256='a'*64,mask_sha256='b'*64,sam3_executed=True,mask_reviewed=True,feet_verified=True,
        homography_verified=True,projection_reviewed=True,team='H',pitch_xy=[40.,20.],**kw)


class GameEvidenceTests(unittest.TestCase):
    def test_stationary_and_missing_do_not_prove_dead(self):
        for missing in (False,True):
            m=GameState(metadata(30))
            for f in range(20):
                b=point(f,None if missing else 20,source='missing' if missing else 'observed')
                self.assertFalse(m.step(f,b,not missing)['dead'])

    def test_stop_restart_expiry_and_conflict(self):
        m=GameState(metadata(30));b=lambda f:point(f,20)
        self.assertTrue(m.step(0,b(0),True,[evidence(0)])['dead'])
        for f in (1,2,3):self.assertTrue(m.step(f,b(f),True)['dead'])
        r=m.step(4,b(4),True);self.assertFalse(r['dead']);self.assertEqual(r['transition'],'dead_ball_evidence_expired')
        m.step(5,b(5),True,[evidence(5)])
        self.assertFalse(m.step(6,b(6),True,[evidence(6,'restart',team='H')])['dead'])
        r=m.step(7,b(7),True,[evidence(7),dict(evidence(7,'restart'),evidence_id='other')])
        self.assertFalse(r['dead']);self.assertTrue(r['diagnostics'])

    def test_crossing_requires_review_consecutive_inside_and_plausible_motion(self):
        for x,ok,review,expected in ((-.3,True,True,True),(-5,True,True,False),(-.3,False,True,False),(-.3,True,False,False),(-.05,True,True,False)):
            m=GameState(metadata(10));m.step(0,point(0,.2),True)
            e=evidence(1,'geometry',boundary_uncertainty_m=.1);e['verified']=review
            self.assertEqual(m.step(1,point(1,x),ok,[e])['dead'],expected)
        m=GameState(metadata(10));m.step(0,point(0,.2),True)
        m.step(1,point(1,None,source='missing'),False)
        self.assertFalse(m.step(2,point(2,-.3),True,[evidence(2,'geometry',boundary_uncertainty_m=.1)])['dead'])

    def test_late_and_future_evidence_never_rewrite(self):
        m=GameState(metadata(10));e=evidence(0);e['available_at_frame']=3
        self.assertFalse(m.step(0,point(0,20),True,[e])['dead'])
        for f in range(1,4):self.assertFalse(m.step(f,point(f,20),True,[e] if f==3 else [])['dead'])

    def test_dead_clears_pass_and_does_not_change_detector_events(self):
        a=CollectiveModel(metadata(15));b=CollectiveModel(metadata(15))
        for f in range(15):
            row=dict(frame=f,ball=point(f,20),players=[point(f,20,track_id='h',team='H'),point(f,40,track_id='a',team='A')])
            r=a.feed(row,game_evidence=[evidence(5)] if f==5 else [])
            q=b.feed(row)
            self.assertEqual(r[2],q[2])
            if 5<=f<=8:self.assertEqual(r[0]['label'],'D');self.assertIsNone(a.origin)
            if f==9:self.assertEqual(r[0]['label'],'X')


class RestartTrackingTests(unittest.TestCase):
    def test_verified_restart_with_current_matching_control_resumes_without_fake_event(self):
        for restart_team in ('H','A',None):
            model=CollectiveModel(metadata(15));states=[]
            for f in range(15):
                row=dict(frame=f,ball=point(f,20),players=[point(f,20,track_id='h',team='H'),point(f,40,track_id='a',team='A')])
                evidence_rows=[evidence(5)] if f==5 else [evidence(7,'restart',team=restart_team)] if f==7 else []
                states.append(model.feed(row,game_evidence=evidence_rows)[0])
            self.assertEqual(states[6]['label'],'D')
            self.assertEqual(states[7]['label'],'H' if restart_team=='H' else 'X')
            if restart_team=='H':
                self.assertEqual(states[7]['reason'],'verified_restart_tracking_confirmed')
                self.assertIn('e-7',states[7]['support']['hint_ids'])
                self.assertEqual(states[8]['label'],'H')


class PassContinuityTests(unittest.TestCase):
    def step(self,m,f,ball_x,players=None,**checks):
        defaults=dict(observed=True,coherent=True,no_opponent_control=True,within_age=True)
        defaults.update(checks)
        return m.step(f,'H',players or [point(f,30,track_id='h',team='H')],point(f,ball_x),
                      (10.,0.),DEFAULTS,defaults,[])

    def test_seed_and_approaching_receiver_survive_arrival_loss(self):
        m=PassContinuity();self.assertTrue(self.step(m,0,20)['accepted'])
        # Retain an observed approaching receiver after the fitted ray turns away.
        r=m.step(1,'H',[point(1,30,track_id='h',team='H')],point(1,21),
                 (8.,3.),DEFAULTS,{'coherent':True},[])
        self.assertTrue(r['accepted']);self.assertTrue(r['candidates'][0]['approaching'])

    def test_each_blocker_abandons_and_does_not_fill_arbitrary_gap(self):
        for blocker in ('observed','coherent','no_opponent_control','within_age'):
            m=PassContinuity();self.step(m,0,20)
            self.assertFalse(self.step(m,1,21,**{blocker:False})['accepted']);self.assertFalse(m.candidates)
        m=PassContinuity();self.step(m,0,20)
        r=self.step(m,1,21,players=[point(1,30,40,track_id='h',team='H')])
        self.assertFalse(r['accepted'])

    def test_candidate_switch_and_team_switch(self):
        m=PassContinuity();self.step(m,0,20)
        r=self.step(m,1,21,players=[point(1,30,track_id='new',team='H')])
        self.assertEqual(r['abandoned'],['h']);self.assertEqual(r['candidates'][0]['started_frame'],1)
        r=m.step(2,'A',[point(2,30,track_id='new',team='H')],point(2,22),(10.,0.),DEFAULTS,{'ok':True},[])
        self.assertFalse(r['accepted'])


class LaunchTests(unittest.TestCase):
    def test_outgoing_launch_is_bounded_and_requires_separation_from_opponent(self):
        for opponent in (80.,22.):
            model=CollectiveModel(metadata(25));states=[]
            for f in range(25):
                x=20. if f<8 else 20.+(f-7)*.4
                row=dict(frame=f,ball=point(f,x),players=[point(f,20.,track_id='h',team='H'),
                    point(f,opponent,track_id='a',team='A'),point(f,100,50,track_id='h2',team='H'),
                    point(f,90,50,track_id='a2',team='A')])
                states.append(model.feed(row)[0])
            launches=[r for r in states if r['reason']=='observed_sender_outgoing_launch']
            if opponent==80.:
                self.assertTrue(launches)
                self.assertTrue(all(r['pass_continuity']['launch']['age_s']<=.2 for r in launches))
            else:self.assertFalse(launches)


class RecoveryAndRenderTests(unittest.TestCase):
    def test_recovery_dedup_and_abstentions(self):
        row=dict(frame=0,ball=point(0,20),players=[]);meta=metadata(10)
        ps,ds=apply_recoveries([recovery(),recovery()],row,meta)
        self.assertEqual(len(ps),1);self.assertEqual(ds[1]['status'],'duplicate_observation')
        for key,val in (('team',None),('feet_verified',False),('homography_verified',False),('projection_reviewed',False),('available_at_frame',1)):
            r=recovery();r[key]=val
            self.assertFalse(apply_recoveries([r],row,meta)[0])
        row['players']=[point(0,40,track_id='atd',team='A')]
        self.assertFalse(apply_recoveries([recovery()],row,meta)[0])

    def test_correction_is_local_and_keeps_raw_input(self):
        row=dict(frame=0,ball=point(0,20),players=[point(0,40,track_id='a',team='A')])
        original=copy.deepcopy(row);r=recovery();r.update(kind='team_correction',target_track_id='a')
        ps,ds=apply_recoveries([r],row,metadata(10))
        self.assertEqual(ps[0]['team'],'H');self.assertEqual(ps[0]['track_id'],'synthetic:a')
        self.assertEqual(ds[0]['status'],'accepted_team_correction');self.assertEqual(row,original)
        row['players'][0]['available_at_frame']=1
        self.assertFalse(apply_recoveries([r],row,metadata(10))[0])

    def test_collective_correction_replaces_canonical_candidate_without_duplicate(self):
        from unittest.mock import patch
        row=dict(frame=0,ball=point(0,20),players=[point(0,40,track_id='a',team='A')])
        r=recovery();r.update(kind='team_correction',target_track_id='a')
        model=CollectiveModel(metadata(10))
        with patch.object(model.pass_continuity,'step',wraps=model.pass_continuity.step) as call:
            model.feed(row,recovered_observations=[r])
        candidates=call.call_args.args[2]
        self.assertEqual(len(candidates),1)
        self.assertEqual(candidates[0]['track_id'],'synthetic:a')
        self.assertEqual(candidates[0]['team'],'H')

    def test_package_hash_and_source_frame_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'v';video.write_bytes(b'video');image=root/'i';image.write_bytes(b'image')
            row=evidence(0);row.update(image='i',image_sha256=sha(image),source_frame=0)
            package=dict(schema='tactical-reviewed-evidence-v1',timeline='tactical',same_source_video_frame=True,
                         source_video_sha256=sha(video),game_evidence=[row]);path=root/'p.json'
            path.write_text(json.dumps(package));self.assertIn(0,load_evidence_package(path,metadata(10),video,'game_evidence'))
            row['source_frame']=1;path.write_text(json.dumps(package))
            with self.assertRaises(ValueError):load_evidence_package(path,metadata(10),video,'game_evidence')
            row['source_frame']=0;path.write_text(json.dumps(package));image.write_bytes(b'changed')
            with self.assertRaises(ValueError):load_evidence_package(path,metadata(10),video,'game_evidence')

    def test_recovery_projection_and_actual_sam_proposal_are_bound(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'v';video.write_bytes(b'video')
            Image.new('RGB',(100,100),'green').save(root/'image.png')
            mask=Image.new('L',(100,100),0)
            from PIL import ImageDraw
            ImageDraw.Draw(mask).rectangle((35,10,45,21),fill=255);mask.save(root/'mask.png')
            h=root/'h.csv';h.write_text('frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n0,100,0,0,0,100,0,0,0,1,0,1\n')
            r=recovery();r.update(image='image.png',mask='mask.png',image_sha256=sha(root/'image.png'),
                mask_sha256=sha(root/'mask.png'),source_frame=0,foot_pixel=[40.,20.],box=[35,10,45,21],
                pitch_xy=[42.12,13.6],proposal_package='proposal.json')
            proposal=root/'proposal.json'
            proposal.write_text(json.dumps(dict(source_video_sha256=sha(video),recoveries=[r.copy()])))
            r['proposal_package_sha256']=sha(proposal)
            package=dict(schema='tactical-reviewed-evidence-v1',timeline='tactical',same_source_video_frame=True,
                source_video_sha256=sha(video),homography_file='h.csv',homography_sha256=sha(h),recoveries=[r])
            path=root/'p.json';path.write_text(json.dumps(package))
            self.assertIn(0,load_evidence_package(path,metadata(10),video,'recoveries'))
            r['pitch_xy']=[43.,13.6];path.write_text(json.dumps(package))
            with self.assertRaisesRegex(ValueError,'coordinates'):load_evidence_package(path,metadata(10),video,'recoveries')
            r['pitch_xy']=[42.12,13.6];path.write_text(json.dumps(package));proposal.write_text('{}')
            with self.assertRaisesRegex(ValueError,'proposal'):load_evidence_package(path,metadata(10),video,'recoveries')

    def test_valid_observation_drawn_and_future_hidden(self):
        self.assertEqual(render_status(point(0,20),0,[105.3,68]),'drawn')
        self.assertEqual(render_status(point(0,20,available_at_frame=1),0,[105.3,68]),'not_yet_available')
        self.assertEqual(render_status(point(0,None,source='missing'),0,[105.3,68]),'missing')
