"""Validate experiment integrity and provisional consumer, not sporting accuracy."""
import copy
import unittest

from scripts.compare_persistence import scenarios, config, run_profile, composite, intervals
from src.ball_events import detect
from src.ball_events import contract as c
from src.ball_events.demo import metadata
from src.team_possession import possession
from src.team_possession.consumer import POLICY_VERSION


class PersistenceComparisonTests(unittest.TestCase):
    def paired(self,name):
        case=next(case for case in scenarios() if case['name']==name)
        meta=metadata(len(case['frames']))
        results=[run_profile(meta,lambda:iter(case['frames']),[],p) for p in (3,5)]
        self.assertEqual(results[0]['input_stream_sha256'],results[1]['input_stream_sha256'])
        return results

    def test_long_control_delay_is_two_frames(self):
        a,b=self.paired('controle_H_sustentado')
        self.assertEqual(a['events'][0]['event_frame'],b['events'][0]['event_frame'])
        self.assertEqual(b['events'][0]['confirmed_at_frame']-a['events'][0]['confirmed_at_frame'],2)
        self.assertEqual(a['possession']['labels'][:3],['X']*3)
        self.assertEqual(b['possession']['labels'][:5],['X']*5)

    def test_short_legitimate_control_is_missed_by_five(self):
        a,b=self.paired('controle_real_H_3_evidencias')
        self.assertEqual([e['team'] for e in a['events']],['H'])
        self.assertEqual(b['events'],[])

    def test_delayed_input_can_mask_confirmation_difference(self):
        a,b=self.paired('evidencia_atrasada')
        self.assertEqual([a['events'][0]['available_at_frame'],b['events'][0]['available_at_frame']],[15,15])
        self.assertEqual(a['possession']['labels'],b['possession']['labels'])

    def test_loss_stays_fixed(self):
        a,b=self.paired('afastamento_sustentado')
        losses=[[e for e in r['events'] if e['type']=='possible_control_loss'][0] for r in (a,b)]
        self.assertEqual(losses[0]['event_frame'],losses[1]['event_frame'])
        self.assertEqual(losses[0]['confirmed_at_frame'],losses[1]['confirmed_at_frame'])
        self.assertEqual(losses[0]['confirmed_at_frame']-losses[0]['event_frame'],2)

    def test_independent_configuration_validation(self):
        cfg=c.config({'persistence_frames':5,'loss_persistence_frames':3},25)
        self.assertEqual((cfg['acquisition_persistence_frames'],cfg['loss_persistence_frames']),(5,3))
        for bad in (True,1,2.5):
            with self.assertRaises(c.InputError): c.config({'loss_persistence_frames':bad},25)

    def test_composite_resets_and_interval_boundaries(self):
        meta,frames=composite(scenarios()[:2])
        r=run_profile(meta,lambda:iter(frames),[],3)
        self.assertEqual(r['possession']['labels'][24:27],['X']*3)
        self.assertEqual(intervals(['H','H','A'],['X','X','A']),[
            dict(start_frame=0,end_frame_exclusive=2,label_3='H',label_5='X')])

    def test_stale_acquisition_cannot_overwrite_newer_team(self):
        case=next(x for x in scenarios() if x['name']=='roubada_sustentada')
        meta=metadata(len(case['frames']))
        _,events,_=detect(meta,case['frames'],config(3))
        events=copy.deepcopy([e for e in events if e['type']=='possible_control_acquisition'])
        self.assertEqual([e['team'] for e in events],['H','A'])
        # Emulate an earlier acquisition delayed past a newer one by a producer.
        events[0]['available_at_frame']=40
        events[0]['availability_delay_frames']=40-events[0]['event_frame']
        events[0]['dependencies']['available_at_frame']=40
        r=possession(events,meta)
        self.assertEqual(r['labels'][40],'A')
        self.assertEqual(r['decisions'][-1]['reason'],'stale_acquisition')
        self.assertEqual(r['policy_version'],POLICY_VERSION)
        self.assertNotIn('D',r['labels'])

    def test_no_label_before_event_availability(self):
        a,_=self.paired('evidencia_atrasada')
        self.assertEqual(a['possession']['labels'][:15],['X']*15)
        self.assertEqual(a['possession']['labels'][15],'H')
