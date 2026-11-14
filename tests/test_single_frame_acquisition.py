"""One eligible frame still requires valid historical motion evidence."""
import unittest
from pathlib import Path
import json
from src.ball_events import Detector, detect
from src.ball_events import contract as c
from src.ball_events.demo import metadata
from src.ball_events.events import validate_events
from scripts.compare_persistence import config, stationary


class SingleFrameAcquisitionTests(unittest.TestCase):
    def test_first_eligible_frame_confirms_once(self):
        meta=metadata(20)
        frames=stationary(20)
        _,events,_=detect(meta,frames,config(1))
        acq=[e for e in events if e['type']=='possible_control_acquisition']
        self.assertEqual(len(acq),1)
        e=acq[0]
        self.assertEqual((e['event_frame'],e['confirmed_at_frame'],e['available_at_frame']),(1,1,1))
        self.assertEqual(e['lookahead_frames'],0)
        self.assertLess(e['support_start_frame'],e['event_frame'])
        validate_events(events,meta)
        from jsonschema import Draft202012Validator
        Draft202012Validator(json.loads(Path('schemas/event-v2.schema.json').read_text())).validate(e)

    def test_delayed_evidence_still_delays_delivery(self):
        frames=stationary(8);frames[0]['players'][0]['available_at_frame']=15
        _,events,_=detect(metadata(8),frames,config(1))
        e=events[0]
        self.assertEqual(e['confirmed_at_frame'],1)
        self.assertEqual(e['available_at_frame'],15)

    def test_no_motion_no_control_even_at_one(self):
        frames=stationary(10)
        for f in frames:f['players'][0]['track_id']=str(f['frame'])
        _,events,_=detect(metadata(10),frames,config(1))
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in events))

    def test_chunk_equivalence_and_loss_still_three(self):
        frames=stationary(16)
        for f in frames[6:]:f['ball']['x_m']=20+.4*(f['frame']-5)
        meta=metadata(16)
        expected=detect(meta,frames,config(1))
        d=Detector(meta,config(1));features=[];events=[]
        for chunk in (frames[:1],frames[1:7],frames[7:]):
            for row in chunk:
                feature,batch=d.feed(row);features.append(feature);events.extend(batch)
        self.assertEqual((features,events,d.finish()),expected)
        losses=[e for e in events if e['type']=='possible_control_loss']
        self.assertEqual(len(losses),1)
        self.assertEqual(losses[0]['confirmed_at_frame']-losses[0]['event_frame'],2)

    def test_one_is_only_allowed_for_acquisition_override(self):
        self.assertEqual(c.config(config(1),25)['acquisition_persistence_frames'],1)
        for bad in (0,True,1.5):
            with self.assertRaises(c.InputError):c.config({'acquisition_persistence_frames':bad},25)
        with self.assertRaises(c.InputError):c.config({'loss_persistence_frames':1},25)
