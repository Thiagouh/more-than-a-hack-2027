"""Collective pass cases are synthetic, independent of match reference labels."""
import copy
import json
import math
import unittest

from src.ball_events.demo import metadata, point
from src.ball_events.dependencies import combine
from src.team_possession.collective import CollectiveModel, configuration
from src.team_possession.kinematics import MotionWindow, time_to_reach


def passing_frames(n=45, receiver=28., intercept=False, two_receivers=False):
    for f in range(n):
        x=20. if f<8 else min(28.,20.+.4*(f-7))
        players=[point(f,20,track_id='sender',team='H'),point(f,receiver,track_id='receiver',team='H'),
                 point(f,28 if intercept else 40,y=20 if intercept else 40,track_id='a',team='A'),
                 point(f,70,y=50,track_id='a2',team='A')]
        if two_receivers:
            players[1]['y_m']=19
            players.append(point(f,receiver,y=21,track_id='receiver2',team='H'))
        yield dict(frame=f,ball=point(f,x),players=players)


def run(frames,cfg=None,meta=None):
    meta=meta or metadata(len(frames));model=CollectiveModel(meta,collective_config=cfg)
    snapshots=[];events=[];measurements=[]
    for row in frames:
        snapshot,_,batch,measurement=model.feed(row)
        snapshots.append(snapshot);events.extend(batch);measurements.append(measurement)
    return snapshots,events,measurements,model.finish()


class MotionTests(unittest.TestCase):
    def test_linear_fit_and_prequential_error(self):
        w=MotionWindow(25)
        for f in range(5):
            p=point(f,20+f*.2)
            fit,error=w.update(p,combine(frame=f),True)
        self.assertAlmostEqual(fit['velocity'][0],5)
        self.assertAlmostEqual(error[0],0)
        p=point(5,30)
        _,error=w.update(p,combine(frame=5),True)
        self.assertAlmostEqual(error[0],9)
        self.assertEqual(w.errors[-1][1].end,6)

    def test_signed_motion_away_takes_longer(self):
        cfg=configuration()
        toward=time_to_reach((0,0),(4,0),(10,0),cfg)
        away=time_to_reach((0,0),(-4,0),(10,0),cfg)
        self.assertGreater(away,toward)
        self.assertEqual(time_to_reach((0,0),(0,0),(.5,0),cfg),0)
        self.assertIsNone(time_to_reach((0,0),(20,0),(10,0),cfg))

    def test_motion_gap_resets_history(self):
        w=MotionWindow(25)
        for f in (0,1,2,4):
            p=point(f,20.)
            fit,_=w.update(p,combine(frame=f),True)
        self.assertIsNone(fit)
        self.assertEqual(len(w.samples),1)


class CollectiveTests(unittest.TestCase):
    def test_short_pass_preserves_team_and_refreshes_support(self):
        snapshots,_,_,_=run(list(passing_frames()))
        flight=snapshots[13:25]
        self.assertTrue(any(s['state']=='passe' for s in flight))
        self.assertEqual({s['label'] for s in flight},{'H'})
        self.assertEqual(snapshots[-1]['state'],'controle')
        self.assertEqual(snapshots[-1]['metrics']['last_supported_control_age_s'],0)
        self.assertEqual(snapshots[-1]['metrics']['origin_support_fraction'],1)

    def test_sustained_reception_precedes_rigid_control(self):
        snapshots,events,_,_=run(list(passing_frames(receiver=40,intercept=True)))
        acq=next(e for e in events if e['type']=='possible_control_acquisition' and e['team']=='A')
        early=next(s for s in snapshots if s['label']=='A')
        self.assertLess(early['frame'],acq['available_at_frame'])
        self.assertEqual(early['reason'],'sustained_probable_reception')
        self.assertTrue(all(early['reception']['checks'].values()))
        self.assertIsNone(early['origin_event_id'])
        self.assertFalse(early['released_acquisitions'])
        self.assertEqual(snapshots[acq['available_at_frame']]['label'],'A')

    def test_two_same_team_receivers_are_not_ambiguity(self):
        snapshots,_,_,_=run(list(passing_frames(two_receivers=True)))
        row=next(s for s in snapshots[14:25] if s['metrics']['receiver_candidates_count']>=2)
        self.assertEqual(row['label'],'H')

    def test_missing_opponent_coverage_does_not_make_safe_pass(self):
        frames=list(passing_frames())
        for row in frames[12:26]:row['players']=[p for p in row['players'] if p['team']=='H']
        snapshots,_,_,_=run(frames)
        self.assertEqual(snapshots[18]['label'],'X')
        self.assertEqual(snapshots[18]['reason'],'insufficient_team_motion_coverage')
        self.assertIsNone(snapshots[18]['metrics']['meeting_time_advantage_s'])

    def test_long_ball_gap_expires_and_does_not_bridge(self):
        frames=list(passing_frames(55))
        for f in range(12,25):frames[f]['ball']=point(f,None,source='missing')
        snapshots,_,_,_=run(frames)
        self.assertEqual(snapshots[17]['label'],'X')
        self.assertIsNone(snapshots[17]['origin_team'])
        self.assertFalse(snapshots[17]['forecast_xy'])
        self.assertEqual(snapshots[25]['label'],'X')

    def test_delayed_points_do_not_leak_into_prefix(self):
        frames=list(passing_frames())
        changed=copy.deepcopy(frames)
        for f in range(8,14):
            frames[f]['ball']['available_at_frame']=30
            changed[f]['ball']['available_at_frame']=30
            changed[f]['ball']['x_m']+=20.
        a,_,measurements,_=run(frames);b,_,_,_=run(changed)
        self.assertEqual([(s['label'],s['state']) for s in a[:30]],[(s['label'],s['state']) for s in b[:30]])
        self.assertIsNone(measurements[10]['values']['ball_speed_fitted_mps'])
        self.assertTrue(measurements[10]['unavailable_inputs'])
        self.assertIn('input_not_yet_available',measurements[10]['values']['quality_flags'])
        for s in a:
            if s['support']:self.assertLessEqual(s['support']['available_at_frame'],s['frame'])

    def test_segment_resets_origin_and_chunk_equivalence(self):
        frames=list(passing_frames());meta=metadata(len(frames));meta['continuity_starts']=[15]
        expected=run(frames,meta=meta)
        self.assertEqual(expected[0][15]['label'],'X')
        model=CollectiveModel(meta);actual=[]
        for chunk in (frames[:2],frames[2:17],frames[17:]):
            for row in chunk:actual.append(model.feed(row)[0])
        self.assertEqual(actual,expected[0])
        self.assertEqual(model.finish(),expected[3])

    def test_pass_support_has_a_separate_finite_lifetime(self):
        snapshots,_,_,_=run(list(passing_frames()),cfg={'max_pass_without_control_s':.4})
        row=snapshots[19]
        self.assertEqual(row['label'],'X')
        self.assertEqual(row['reason'],'origin_support_expired')
        self.assertGreater(row['metrics']['last_supported_control_age_s'],.4)

    def test_unavailable_receiver_count_is_not_zero(self):
        frames=list(passing_frames())
        for row in frames[12:20]:
            row['players']=[p for p in row['players'] if p['team']=='A']
        snapshots,_,_,_=run(frames)
        row=snapshots[16]
        self.assertIsNone(row['metrics']['receiver_candidates_count'])
        self.assertFalse(row['metric_validity']['receiver_candidates_count']['valid'])

    def test_metrics_json_has_no_infinity_and_censor_reason(self):
        snapshots,_,measurements,_=run(list(passing_frames(receiver=80)))
        json.dumps(snapshots,allow_nan=False)
        row=measurements[18]
        self.assertTrue(row['arrivals']['censored_candidates'])
        self.assertNotEqual(row['arrivals']['teams']['H']['meeting_time_s'],math.inf)
        self.assertEqual(set(snapshots[18]['metrics']),set(snapshots[18]['metric_validity']))
