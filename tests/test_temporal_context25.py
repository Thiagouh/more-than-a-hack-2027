import copy
import unittest
from unittest.mock import patch
from src.ball_events.demo import metadata,point
from src.ball_events.dependencies import combine
from src.team_possession.temporal_context import TemporalContext
from src.team_possession.collective import CollectiveModel,configuration
from src.team_possession.tracking_review import apply_tracking_review


def current(f,own=1.):
    return dict(frame=f,support=combine(frame=f).as_dict(),distances={'H':own,'A':20.})


class ContextTests(unittest.TestCase):
    def feed(self,m,f,label='H',reason='current_control_evidence',control='H',**kwargs):
        values=dict(origin='H',base_label=label,base_reason=reason,control=control,current=current(f),
            checks={'observed':True,'no_opponent':True},age_s=.5,max_age_s=4.)
        values.update(kwargs);return m.step(f,**values)

    def test_25_frames_recent_emphasis_and_exact_expiry_without_self_reinforcement(self):
        m=TemporalContext(25,5)
        for f in range(10):self.feed(m,f)
        r=self.feed(m,10,'X','no_supported_receiver',None)
        self.assertTrue(r['accepted']);self.assertEqual(r['window_frames'],25)
        self.assertEqual(r['support']['support_start_frame'],0)
        # Inferred outputs supply no new facts, so support eventually expires.
        for f in range(11,40):r=self.feed(m,f,'X','no_supported_receiver',None)
        self.assertFalse(r['accepted']);self.assertEqual(r['independent_evidence_frames'],0)
        self.assertEqual(r['buffered_frames'],25)

    def test_recent_frames_have_more_weight(self):
        a=TemporalContext();b=TemporalContext()
        for f in range(25):
            ra=self.feed(a,f,label='X',reason='no_supported_receiver',control='H' if f<5 else None)
            rb=self.feed(b,f,label='X',reason='no_supported_receiver',control='H' if f>=20 else None)
        self.assertGreater(rb['weighted_support']['H'],ra['weighted_support']['H'])

    def test_slow_ball_needs_recent_control_and_observed_nearby_own_player(self):
        for distance,expected in ((1.,True),(5.,False)):
            m=TemporalContext()
            for f in range(8):self.feed(m,f)
            r=self.feed(m,8,'X','no_control_or_pass_motion',None,current=current(8,distance))
            self.assertEqual(r['accepted'],expected)

    def test_opponent_control_barrier_dead_and_quality_abstain(self):
        for change in ({'checks':{'observed':False}}, {'control':'A'}, {'dead':True}, {'barrier':True}, {'age_s':4.01}):
            m=TemporalContext()
            for f in range(8):self.feed(m,f)
            r=self.feed(m,8,'X','no_supported_receiver',None,**change) if 'control' not in change else self.feed(m,8,'X','no_supported_receiver','A')
            self.assertFalse(r['accepted'])
        m=TemporalContext()
        for f in range(8):self.feed(m,f)
        self.feed(m,8,'X','no_supported_receiver','A')
        self.assertFalse(self.feed(m,9,'X','no_supported_receiver',None)['accepted'])

    def test_cannot_initialize_from_nothing_or_override_expired_origin(self):
        m=TemporalContext()
        for f in range(30):self.assertFalse(self.feed(m,f,'X','no_confirmed_origin',None,origin=None)['accepted'])
        for f in range(30,40):self.feed(m,f)
        self.assertFalse(self.feed(m,40,'X','origin_support_expired',None)['accepted'])

    def test_streaming_chunks_and_prefix_are_identical(self):
        a=TemporalContext();b=TemporalContext();expected=[];actual=[]
        for f in range(40):expected.append(self.feed(a,f,'H' if f<10 else 'X','current_control_evidence' if f<10 else 'no_supported_receiver','H' if f<10 else None))
        for chunk in (range(7),range(7,19),range(19,40)):
            for f in chunk:actual.append(self.feed(b,f,'H' if f<10 else 'X','current_control_evidence' if f<10 else 'no_supported_receiver','H' if f<10 else None))
        self.assertEqual(expected,actual)

    def test_window_configuration_is_recorded_and_validated(self):
        self.assertEqual(configuration()['context_frames'],25)
        for cfg in ({'context_frames':0},{'context_recent_frames':26},{'context_frames':101}):
            with self.assertRaises(ValueError):configuration(cfg)


class GoalkeeperTrackingTests(unittest.TestCase):
    def policy(self):
        return dict(rules=[dict(rule_id='bad',track_id='46',action='reject_observation',start_frame=0,end_frame_exclusive=5,
            available_at_frame=0,x_max_m=52.65,reason='reviewed conflicting home goalkeeper point')],assignments_by_frame={})

    def row(self,f=0):
        return dict(frame=f,ball=point(f,20),players=[point(f,5,track_id='24',team='H'),point(f,7,track_id='46',team='A')])

    def test_reject_is_visible_quality_flag_preserving_raw_and_scope(self):
        original=self.row();snapshot=copy.deepcopy(original);row,diagnostics=apply_tracking_review(original,self.policy())
        self.assertEqual(original,snapshot);self.assertTrue(row['players'][1]['quality_flags'])
        self.assertEqual(row['players'][1]['team'],'A');self.assertEqual(row['players'][1]['x_m'],7.)
        self.assertEqual(diagnostics[0]['action'],'reject_observation')
        row,diagnostics=apply_tracking_review(self.row(5),self.policy());self.assertFalse(diagnostics)
        row=self.row();row['players'][1]['x_m']=95.;self.assertFalse(apply_tracking_review(row,self.policy())[1])

    def test_source_assignment_relabels_without_duplicate_or_future_point(self):
        policy=self.policy();policy['assignments_by_frame'][0]=dict(frame=0,available_at_frame=0,team='H',track_id='46',
            raw_xy=[7.,20.],image_rgb_sha256='a'*64,box=[1,2,3,30],cost_px=2.,quality='heuristic')
        row,diagnostics=apply_tracking_review(self.row(),policy)
        usable=[p for p in row['players'] if not p['quality_flags']]
        self.assertEqual(len(usable),1);self.assertEqual(usable[0]['team'],'H')
        self.assertEqual(usable[0]['track_id'],'reviewed-gk-H');self.assertEqual(usable[0]['x_m'],7.)
        self.assertTrue(any(d['action']=='goalkeeper_assignment' for d in diagnostics))
        source=self.row();source['players'][1]['available_at_frame']=1
        self.assertFalse(any(p['track_id']=='reviewed-gk-H' for p in apply_tracking_review(source,policy)[0]['players']))

    def test_current_raw_detector_excludes_rejected_point(self):
        model=CollectiveModel(metadata(8),tracking_review=self.policy())
        for f in range(8):
            state,_,_,measurement=model.feed(self.row(f))
            if f<5:self.assertEqual(measurement['values']['observed_players_by_team']['A'],0)

    def test_rgb_cue_can_choose_mislabeled_source_or_abstain(self):
        import numpy as np
        from scripts.prepare_goalkeeper_track_review import associate
        image=np.zeros((100,100,3),dtype=np.uint8);image[20:45,6:14]=[10,220,30]
        points=[point(0,20,46,track_id='24',team='H'),point(0,10,46,track_id='46',team='A')]
        r=associate(image,np.diag([100.,100.,1.]),[100.,100.],points)
        self.assertEqual(r['selected']['track_id'],'46')
        image[:]=0;image[30:40,6:14]=[10,220,30]
        self.assertIsNone(associate(image,np.diag([100.,100.,1.]),[100.,100.],points)['selected'])
