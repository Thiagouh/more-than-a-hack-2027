import copy
import itertools
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.ball_events.demo import metadata, point
from src.team_identification.association import associate, feet_to_pitch, LocalIdentities, ImageIdentities, _assignment
from src.team_identification.temporal import TeamMemory
from src.team_identification.overlay import apply_overlay, apply_team_corrections
from src.team_identification.pipeline import IdentityPipeline
from src.team_identification.render import draw_identities
from src.team_possession.collective import CollectiveModel
from src.team_identification.identity_evaluation import evaluate_identity, evaluate_atd_correspondences, promotion_gate
from team_evaluation_fixtures import truth_for, person, prediction


def observation(f, team='A', target='slot', identity='person-1', **kwargs):
    return dict(frame=f, players=[dict(frame=f, identity=identity, target_track_id=target,
        raw_xy=[20.,20.] if target else None, pitch_xy=[20.,20.], team=team,
        role='player' if team else 'unknown', association_status='matched' if target else 'recovered',
        available_at_frame=f, support_start_frame=f, reason='synthetic_test', **kwargs)])


def envelope(f=0):
    return dict(frame=f, ball=point(f,20.), players=[point(f,20.,track_id='slot',team='H')])


class AssociationTests(unittest.TestCase):
    def test_image_identity_survives_small_box_motion_and_retires_ambiguous_crossing(self):
        tracker=ImageIdentities()
        ids=tracker.step(0,[[10,10,20,50],[40,10,50,50]])
        moved=tracker.step(1,[[41,10,51,50],[11,10,21,50]])
        self.assertEqual(ids,moved[::-1])
        crossed=tracker.step(2,[[25,10,40,50],[25,10,40,50]])
        self.assertTrue(set(ids).isdisjoint(crossed))
        after_gap=tracker.step(4,[[25,10,40,50]])
        self.assertTrue(set(crossed).isdisjoint(after_gap))

    def test_image_ambiguity_cannot_pick_one_old_person_by_detection_order(self):
        tracker=ImageIdentities()
        old=tracker.step(0,[[10,10,30,50],[12,10,32,50]])
        new=tracker.step(1,[[11,10,31,50],[11,10,31,50]])
        self.assertTrue(set(old).isdisjoint(new))

    def test_global_assignment_matches_exhaustive_optimum(self):
        rng=np.random.default_rng(4)
        for n,m in [(1,3),(3,3),(3,5)]:
            for _ in range(10):
                cost=rng.uniform(0,10,(n,m))
                chosen,total=_assignment(cost)
                expected=min(sum(cost[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(m),n))
                self.assertAlmostEqual(total,expected)
                self.assertEqual(len(set(chosen)),n)

    def test_one_to_one_and_class_blind(self):
        rows=associate([[10.,10.],[20.,20.]],[[20.1,20.],[10.1,10.]])
        self.assertEqual([r['point_index'] for r in rows],[1,0])

    def test_ambiguous_swap_abstains(self):
        rows=associate([[10.,10.],[10.,10.]],[[9.9,10.],[10.1,10.]])
        self.assertEqual([r['status'] for r in rows],['ambiguous','ambiguous'])

    def test_outside_gate_missing_and_invalid(self):
        rows=associate([[0.,0.],None],[[20.,20.]])
        self.assertEqual([r['status'] for r in rows],['unmatched','invalid_geometry'])
        self.assertEqual(associate([],[]),[])
        self.assertEqual(associate([[0.,0.]],[])[0]['status'],'unmatched')

    def test_delivered_homography_direction_and_quality(self):
        h=np.array([[1000.,0,100.],[0,500.,50.],[0,0,1.]])
        self.assertEqual(feet_to_pitch([600.,300.],h,[100.,50.],1.),[50.,25.])
        self.assertIsNone(feet_to_pitch([600.,300.],h,[100.,50.],3.))
        self.assertIsNone(feet_to_pitch([600.,300.],np.zeros((3,3)),[100.,50.],1.))
        self.assertIsNone(feet_to_pitch([-100.,300.],h,[100.,50.],1.))

    def test_identity_continuity_ignores_slot_order_and_resets(self):
        tracker=LocalIdentities()
        a=tracker.step(0,[[10.,10.],[20.,20.]])
        b=tracker.step(1,[[20.1,20.],[10.1,10.]])
        self.assertEqual(a,b[::-1])
        c=tracker.step(2,[[20.2,20.],[10.2,10.]],reset=True)
        self.assertTrue(set(a).isdisjoint(c))

    def test_crossing_ambiguity_and_slot_reuse_get_new_ids(self):
        tracker=LocalIdentities()
        ids=tracker.step(0,[[10.,10.],[10.4,10.]])
        new=tracker.step(1,[[10.2,10.],[10.2,10.]])
        self.assertTrue(set(ids).isdisjoint(new))
        far=tracker.step(2,[[70.,50.]])
        self.assertTrue(set(new).isdisjoint(far))


class MemoryTests(unittest.TestCase):
    def test_minimum_and_bad_image_hold_and_expire(self):
        memory=TeamMemory()
        for f in range(4):self.assertIsNone(memory.step('a',f,'A','player')['team'])
        self.assertEqual(memory.step('a',4,'A','player')['team'],'A')
        self.assertEqual(memory.step('a',5)['team'],'A')
        self.assertIsNone(memory.step('a',30)['team'])

    def test_switch_requires_repeated_evidence(self):
        memory=TeamMemory(window_frames=8,min_observations=3,switch_observations=3)
        for f in range(3):memory.step('a',f,'A','player')
        self.assertEqual(memory.step('a',3,'H','player')['team'],'A')
        self.assertEqual(memory.step('a',4,'A','player')['team'],'A')
        for f in range(5,13):last=memory.step('a',f,'H','player')
        self.assertEqual(last['team'],'H')

    def test_delayed_evidence_and_new_identity_do_not_inherit(self):
        memory=TeamMemory(min_observations=2)
        for f in range(3):self.assertIsNone(memory.step('a',f,'A','player',f+1)['team'])
        for f in range(3,5):memory.step('a',f,'A','player')
        self.assertIsNone(memory.step('b',5)['team'])
        memory.reset()
        self.assertIsNone(memory.step('a',6)['team'])

    def test_referee_and_goalkeeper_roles_are_distinct(self):
        memory=TeamMemory(min_observations=2)
        for f in range(2):r=memory.step('r',f,None,'referee');g=memory.step('g',f,'H','goalkeeper')
        self.assertEqual(r['role'],'referee')
        self.assertEqual(g['role'],'goalkeeper')


class OverlayTests(unittest.TestCase):
    def test_explicit_corrections_preserve_raw_slots_coordinates_and_unknowns(self):
        raw=envelope();before=copy.deepcopy(raw)
        corrected=apply_team_corrections(raw,observation(0),[105.3,68.])
        self.assertEqual(raw,before)
        self.assertEqual(corrected['players'][0]['team'],'A')
        self.assertEqual(corrected['players'][0]['track_id'],'team-corrected:slot:A')
        self.assertEqual(corrected['players'][0]['x_m'],raw['players'][0]['x_m'])
        self.assertEqual(corrected['players'][0]['extensions']['team_source']['method'],'accepted_visual_correction')
        uncertain=apply_team_corrections(raw,observation(0,team=None),[105.3,68.])
        self.assertEqual(uncertain['players'][0]['team'],'H')
        self.assertEqual(uncertain['players'][0]['extensions']['team_source']['method'],'provided_atd_metadata_unverified')

    def test_corrections_do_not_recover_people_or_replay_late_evidence(self):
        raw=envelope()
        recovered=apply_team_corrections(raw,observation(0,target=None),[105.3,68.])
        self.assertEqual(len(recovered['players']),1)
        self.assertEqual(recovered['players'][0]['team'],'H')
        late=observation(0);late['players'][0]['available_at_frame']=1
        corrected=apply_team_corrections(raw,late,[105.3,68.])
        self.assertEqual(corrected['players'][0]['team'],'H')

    def test_corrects_metadata_without_mutating_coordinates_or_raw(self):
        raw=envelope();before=copy.deepcopy(raw)
        corrected,display=apply_overlay(raw,observation(0),[105.3,68.])
        self.assertEqual(raw,before)
        self.assertEqual(corrected['players'][0]['team'],'A')
        self.assertEqual(corrected['players'][0]['x_m'],20.)
        self.assertEqual(display[0]['team'],'A')

    def test_unknown_is_visible_and_not_used_for_possession(self):
        corrected,display=apply_overlay(envelope(),observation(0,None),[105.3,68.])
        self.assertEqual(corrected['players'],[])
        self.assertEqual(len(display),1)
        image=Image.new('RGB',(100,100),'black')
        counts=draw_identities(ImageDraw.Draw(image),display,0,[105.3,68.],lambda xy:xy,ImageFont.load_default())
        self.assertEqual(counts['unknown'],1)
        self.assertEqual(image.getpixel((20,20)),(174,182,191))

    def test_future_evidence_never_used_or_drawn_as_confirmed(self):
        obs=observation(0);obs['players'][0]['available_at_frame']=2
        corrected,display=apply_overlay(envelope(),obs,[105.3,68.])
        self.assertEqual(corrected['players'],[])
        self.assertIsNone(display[0]['team'])

    def test_invalid_binding_duplicate_and_unknown_role_rejected(self):
        for change in ({'raw_xy':[1.,1.]},{'role':'referee'},{'frame':1}):
            obs=observation(0);obs['players'][0].update(change)
            with self.assertRaises(ValueError):apply_overlay(envelope(),obs,[105.3,68.])
        obs=observation(0);obs['players']*=2
        with self.assertRaises(ValueError):apply_overlay(envelope(),obs,[105.3,68.])

    def test_recovery_registers_producer_and_reaches_collective(self):
        raw=envelope();raw['players']=[]
        model=CollectiveModel(metadata(5),team_identity={0:observation(0,target=None)})
        state,_,_,_=model.feed(raw)
        self.assertEqual(state['team_identity_players'][0]['team'],'A')
        self.assertEqual(model.detector.history[-1]['players'][0]['team'],'A')

    def test_uncertain_metadata_cannot_seed_control(self):
        overlays={f:observation(f,None) for f in range(5)}
        model=CollectiveModel(metadata(5),team_identity=overlays)
        for f in range(5):
            state,_,_,_=model.feed(envelope(f))
            self.assertEqual(state['label'],'X')
            self.assertEqual(model.detector.history[-1]['players'],[])

    def test_absent_overlay_is_legacy_compatible(self):
        a=CollectiveModel(metadata(5));b=CollectiveModel(metadata(5),team_identity={})
        for f in range(5):self.assertEqual(a.feed(envelope(f)),b.feed(envelope(f)))

    def test_entering_overlay_clears_old_metadata_control(self):
        model=CollectiveModel(metadata(10),team_identity={5:observation(5,None)})
        for f in range(5):model.feed(envelope(f))
        state,_,_,_=model.feed(envelope(5))
        self.assertEqual(state['label'],'X')
        self.assertEqual(model.detector.history[-1]['players'],[])
        self.assertIn(5,model.meta['continuity_starts'])
        self.assertIn(6,model.meta['continuity_starts'])

    def test_matched_and_recovered_positions_share_identity_namespace(self):
        a,_=apply_overlay(envelope(),observation(0),[105.3,68.])
        raw=envelope();raw['players']=[]
        b,_=apply_overlay(raw,observation(0,target=None),[105.3,68.])
        self.assertEqual(a['players'][0]['track_id'],b['players'][0]['track_id'])
        self.assertEqual(a['players'][0]['producer'],b['players'][0]['producer'])

    def test_ambiguous_proposals_do_not_duplicate_raw_points(self):
        obs=observation(0,target=None);obs['players'][0]['association_status']='ambiguous'
        corrected,display=apply_overlay(envelope(),obs,[105.3,68.])
        self.assertEqual(corrected['players'],[])
        self.assertEqual(len(display),1)


class PipelineTests(unittest.TestCase):
    class Classifier:
        def classify(self,image,masks):
            return [dict(team='A',role='player',reason='uniform_match',classification='A') for _ in masks]

    def test_box_geometry_is_marked_as_approximation_without_fake_masks(self):
        from src.team_identification.supervised_uniforms import SupervisedUniforms
        model=SupervisedUniforms(dict(version='user-supervised-torso-nearest-v1',status='frozen',
            max_distance=.35,min_margin=.1,examples=[dict(label=label,example_id=label,
            descriptor=[.005]*200) for label in ('Barcelona','Athletic')]))
        pipeline=IdentityPipeline(metadata(5),model,identity_method='image-overlap')
        h=np.array([[105.3,0.,27.],[0.,68.,49.],[0.,0.,1.]])
        row=pipeline.step_boxes(envelope(0),Image.new('RGB',(150,150),'red'),[[40,20,55,70]],
            dict(valid=True,matrix=h,reprojection_error=1.))
        detection=row['detections'][0]
        self.assertEqual(detection['feet_method'],'estimated_box_bottom')
        self.assertEqual(detection['geometry_confidence'],'uncalibrated_box_approximation')
        self.assertNotIn('mask',detection)
        self.assertEqual(row['players'][0]['geometry_method'],'automatic_box_bottom_and_homography')
        with self.assertRaises(ValueError):
            pipeline.step_boxes(envelope(1),Image.new('RGB',(150,150)),[[float('nan'),0,5,5]],dict(valid=False))

    def test_image_continuity_does_not_turn_invalid_geometry_into_a_player(self):
        pipeline=IdentityPipeline(metadata(5),self.Classifier(),identity_method='image-overlap')
        mask=np.zeros((150,150),bool);mask[20:70,40:55]=True
        image=Image.new('RGB',(150,150))
        first=pipeline.step(envelope(0),image,[mask],dict(valid=False))
        second=pipeline.step(envelope(1),image,[mask],dict(valid=False))
        self.assertEqual(first['players'],[])
        self.assertEqual(second['players'],[])
        self.assertEqual(first['detections'][0]['identity'],second['detections'][0]['identity'])
        self.assertFalse(second['detections'][0]['geometry_valid'])

    def test_mask_to_team_to_corrected_model(self):
        meta=metadata(10)
        pipeline=IdentityPipeline(meta,self.Classifier(),dict(memory_min_observations=2))
        mask=np.zeros((150,150),bool);mask[20:70,40:55]=True
        # Feet (47,69) project onto (20,20), the raw H slot.
        h=np.array([[105.3,0.,27.],[0.,68.,49.],[0.,0.,1.]])
        homography=dict(valid=True,matrix=h,reprojection_error=1.)
        image=Image.new('RGB',(150,150),'white')
        first=pipeline.step(envelope(0),image,[mask],homography)
        second=pipeline.step(envelope(1),image,[mask],homography)
        self.assertIsNone(first['players'][0]['team'])
        self.assertEqual(second['players'][0]['team'],'A')
        self.assertEqual(second['players'][0]['target_track_id'],'slot')
        model=CollectiveModel(meta,team_identity={0:first,1:second})
        model.feed(envelope(0));state,_,_,_=model.feed(envelope(1))
        self.assertEqual(state['team_identity_players'][0]['team'],'A')
        self.assertEqual(model.detector.history[-1]['players'][0]['team'],'A')

    def test_invalid_homography_keeps_raw_display_uncertain(self):
        pipeline=IdentityPipeline(metadata(5),self.Classifier())
        mask=np.zeros((150,150),bool);mask[20:70,40:55]=True
        row=pipeline.step(envelope(),Image.new('RGB',(150,150)),[mask],dict(valid=False))
        corrected,display=apply_overlay(envelope(),row,[105.3,68.])
        self.assertEqual(corrected['players'],[])
        self.assertIsNone(display[0]['team'])


class IdentityEvaluationTests(unittest.TestCase):
    def test_errors_coverage_and_unknown_are_separate(self):
        truth=truth_for([person(),person('away',x=20,classification='A')])
        detections=[prediction(classification='A'),prediction(box=[20,0,30,40],classification='uncertain',role='unknown')]
        report=evaluate_identity(truth,[dict(frame=0,detections=detections)])
        self.assertEqual(report['identity']['team_errors'],1)
        self.assertEqual(report['identity']['team_coverage'],.5)
        self.assertEqual(report['identity']['uncertain'],1)
        self.assertFalse(report['association_validation_complete'])

    def test_association_errors_and_identity_switches_need_review(self):
        truth=truth_for([person(person_id='athlete',association_reviewed=True,expected_track_id='1')],count=2)
        truth['sample_kind']='challenge'
        for f in truth['frames']:f['group_id']='sequence-1'
        predictions=[dict(frame=f,detections=[prediction(identity=f'id-{f}',association_status='matched',target_track_id='2')]) for f in range(2)]
        report=evaluate_identity(truth,predictions)
        self.assertEqual(report['identity']['association_errors'],2)
        self.assertEqual(report['identity']['identity_switches'],1)
        self.assertTrue(report['association_validation_complete'])

    def test_gate_cannot_reward_lower_coverage(self):
        def result(errors,classified,complete=True):
            return dict(association_validation_complete=complete,identity_tracking_review_complete=True,identity=dict(athletes=10,team_errors=errors,
                classified=classified,association_errors=0,associated=classified,identity_switches=0))
        self.assertFalse(promotion_gate(result(0,5),result(2,10))['accepted'])
        self.assertTrue(promotion_gate(result(0,10),result(2,10))['accepted'])
        self.assertFalse(promotion_gate(result(0,10,False),result(2,10))['accepted'])

    def test_atd_baseline_uses_reviewed_correspondence_and_reports_missing(self):
        truth=truth_for([person(association_reviewed=True,expected_track_id='slot'),
                         person('missing',x=20,classification='A',association_reviewed=True,expected_track_id=None)])
        report=evaluate_atd_correspondences(truth,[envelope()])
        self.assertEqual(report['identity']['classified'],1)
        self.assertEqual(report['identity']['missed'],1)
        self.assertEqual(report['identity']['team_errors'],0)
        self.assertIn('oracle',report['baseline_method'])


if __name__=='__main__':unittest.main()
