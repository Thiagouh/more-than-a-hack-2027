"""Visibility timeout is distinct from acquisition persistence."""
import copy
import unittest
from src.ball_events import detect
from src.ball_events.contract import InputError
from src.ball_events.demo import metadata
from src.team_possession import possession
from scripts.compare_persistence import stationary, config


class MissingBallTests(unittest.TestCase):
    def setup_case(self, n=24, acquisition=3):
        meta=metadata(n)
        _,events,_=detect(meta,stationary(n),config(acquisition))
        return meta,events

    def test_five_missing_frames_tolerated_sixth_expires(self):
        meta,events=self.setup_case()
        visibility=[True]*8+[False]*8+[True]*8
        r=possession(events,meta,ball_observed=visibility,missing_ball_grace_frames=5)
        self.assertEqual(r['labels'][8:13],['H']*5)
        self.assertEqual(r['labels'][13:],['X']*11)
        self.assertEqual(r['missing_ball_timeouts'],1)
        self.assertEqual([d['frame'] for d in r['decisions'] if d['reason']=='missing_ball_timeout'],[13])

    def test_short_gap_and_zero_grace(self):
        meta,events=self.setup_case()
        visibility=[True]*8+[False]*3+[True]*13
        r=possession(events,meta,ball_observed=visibility,missing_ball_grace_frames=5)
        self.assertEqual(r['labels'][3:],['H']*21)
        zero=possession(events,meta,ball_observed=visibility,missing_ball_grace_frames=0)
        self.assertEqual(zero['labels'][8:],['X']*16)

    def test_independent_from_acquisition_threshold(self):
        visibility=[True]*8+[False]*16
        for acquisition in (3,5):
            meta,events=self.setup_case(acquisition=acquisition)
            r=possession(events,meta,ball_observed=visibility,missing_ball_grace_frames=5)
            self.assertEqual(r['labels'][13:],['X']*11)
            self.assertEqual(r['labels'][12],'H')

    def test_delayed_old_acquisition_cannot_revive_after_gap(self):
        meta,events=self.setup_case()
        events=copy.deepcopy(events)
        events[0]['available_at_frame']=18
        events[0]['availability_delay_frames']=17
        events[0]['dependencies']['available_at_frame']=18
        r=possession(events,meta,ball_observed=[True]*8+[False]*8+[True]*8)
        self.assertEqual(r['labels'],['X']*24)
        self.assertEqual(r['decisions'][-1]['reason'],'acquisition_before_expired_gap')

    def test_delayed_acquisition_during_expired_gap_is_blocked(self):
        meta,events=self.setup_case()
        events=copy.deepcopy(events)
        events[0]['available_at_frame']=14
        events[0]['availability_delay_frames']=13
        events[0]['dependencies']['available_at_frame']=14
        r=possession(events,meta,ball_observed=[True]*8+[False]*16)
        self.assertEqual(r['labels'],['X']*24)
        self.assertEqual(r['decisions'][-1]['reason'],'ball_unobserved_after_grace')

    def test_reacquisition_after_visibility_returns(self):
        meta,events=self.setup_case()
        # Generate a later genuine acquisition using the detector's gap reset.
        frames=stationary(24)
        from src.ball_events.demo import point
        for f in range(8,16):frames[f]['ball']=point(f,None,source='missing')
        _,events,_=detect(meta,frames,config(3))
        r=possession(events,meta,ball_observed=[True]*8+[False]*8+[True]*8)
        self.assertEqual(r['labels'][16:19],['X']*3)
        self.assertEqual(r['labels'][19:24],['H']*5)

    def test_reject_bad_visibility_and_grace(self):
        meta,events=self.setup_case()
        for visibility in ([True], [1]*24):
            with self.assertRaises(InputError):possession(events,meta,ball_observed=visibility)
        for grace in (-1,True,1.5):
            with self.assertRaises(InputError):possession(events,meta,ball_observed=[True]*24,missing_ball_grace_frames=grace)
