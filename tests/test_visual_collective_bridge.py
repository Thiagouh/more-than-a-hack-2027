import copy
import unittest
from unittest.mock import patch
from src.ball_events.demo import metadata,point
from src.ball_events.contract import InputError
from src.team_possession.collective import CollectiveModel
from src.team_possession.restarts import kickoff_setup,load_kickoffs


def observation(f,team='H',manual=True):
    return dict(observation_id=f'sam3:{f}',frame=f,available_at_frame=f,timeline='synthetic',
        image_sha256='1'*64,team=team,role='goalkeeper',classification_method='sam3_goalkeeper_context',
        sam3_executed=True,mask_quality_ok=True,context_valid=True,
        control_evidence='reviewed_held_ball' if manual else 'ball_body_proximity',
        ball_control_observed=True,reviewer='synthetic test')


def empty(f):return dict(frame=f,ball=point(f,None,source='missing'),players=[])


class VisualBridgeTests(unittest.TestCase):
    def test_reviewed_ball_control_bridges_without_fabricating_ball_or_events(self):
        model=CollectiveModel(metadata(5))
        state,feature,events,row=model.feed(empty(0),[observation(0)])
        self.assertEqual(state['label'],'H')
        self.assertEqual(state['reason'],'visual_reviewed_goalkeeper_control')
        self.assertEqual(events,[])
        self.assertFalse(row['ball_valid'])
        self.assertEqual(state['ball_unavailable_run_frames'],1)
        self.assertIn('sam3:0',state['support']['hint_ids'])

    def test_team_identification_without_control_cannot_seed_possession(self):
        o=observation(0);o.update(control_evidence='none',ball_control_observed=False)
        self.assertEqual(CollectiveModel(metadata(5)).feed(empty(0),[o])[0]['label'],'X')

    def test_failed_masks_and_context_abstain(self):
        for field in ('sam3_executed','mask_quality_ok','context_valid','ball_control_observed'):
            o=observation(0);o[field]=False
            self.assertEqual(CollectiveModel(metadata(5)).feed(empty(0),[o])[0]['label'],'X',field)

    def test_auto_proximity_requires_three_adjacent_frames_and_sources(self):
        model=CollectiveModel(metadata(5))
        states=[model.feed(empty(f),[observation(f,manual=False)])[0] for f in range(3)]
        self.assertEqual([s['label'] for s in states],['X','X','H'])
        self.assertEqual(states[-1]['reason'],'visual_control_corroborated')
        self.assertEqual(states[-1]['support']['hint_ids'],['sam3:0','sam3:1','sam3:2'])
        self.assertEqual(states[-1]['support']['support_start_frame'],0)

    def test_visual_support_stays_bounded_during_long_control(self):
        model=CollectiveModel(metadata(12))
        for f in range(10):state=model.feed(empty(f),[observation(f,manual=False)])[0]
        self.assertEqual(state['support']['hint_ids'],['sam3:7','sam3:8','sam3:9'])
        self.assertEqual(state['support']['support_start_frame'],7)
        self.assertEqual(state['visual_control']['streak_frames'],3)

    def test_gaps_and_conflicts_break_sequence(self):
        model=CollectiveModel(metadata(5))
        model.feed(empty(0),[observation(0,manual=False)])
        model.feed(empty(1))
        self.assertEqual(model.feed(empty(2),[observation(2,manual=False)])[0]['label'],'X')
        other=observation(3,'A');other['observation_id']='other'
        state=model.feed(empty(3),[observation(3),other])[0]
        self.assertEqual(state['label'],'X')
        self.assertEqual(state['visual_control']['reason'],'conflicting_visual_teams')

    def test_late_observation_never_rewrites_prefix(self):
        model=CollectiveModel(metadata(5));o=observation(0);o['available_at_frame']=2
        self.assertEqual(model.feed(empty(0),[o])[0]['label'],'X')
        model.feed(empty(1))
        self.assertEqual(model.feed(empty(2),[o])[0]['label'],'X')

    def test_wrong_timeline_and_future_frames_rejected_before_advancing(self):
        for change in ({'timeline':'tactical'},{'frame':1},{'available_at_frame':-1}):
            model=CollectiveModel(metadata(5));o=dict(observation(0),**change)
            with self.assertRaises(InputError):model.feed(empty(0),[o])
            self.assertEqual(model.last_frame,-1)

    def test_tracking_control_conflict_is_not_overwritten(self):
        model=CollectiveModel(metadata(5))
        row=dict(frame=0,ball=point(0,20),players=[point(0,20,track_id='a',team='A'),point(0,40,track_id='h',team='H')])
        model.feed(row)
        row['frame']=1;row['ball']['frame']=1
        for p in row['players']:p['frame']=1
        state=model.feed(row,[observation(1)])[0]
        self.assertFalse(state['visual_control']['accepted'])
        self.assertEqual(state['visual_control']['reason'],'visual_tracking_control_conflict')


class KickoffSetupTests(unittest.TestCase):
    def test_unknown_tag_team_uses_center_setup_when_ball_missing(self):
        meta=metadata(5);cx,cy=(v/2 for v in meta['pitch_m'])
        row=dict(frame=0,ball=point(0,None,source='missing'),players=[
            dict(point(0,cx,track_id='a',team='A'),y_m=cy),
            dict(point(0,cx-10,track_id='h',team='H'),y_m=cy)])
        k=dict(frame=0,team=None,half=1,available_at_frame=0,source_id='known-kickoff')
        state,_,events,_=CollectiveModel(meta,kickoffs=[k]).feed(row)
        self.assertEqual(state['label'],'A');self.assertEqual(state['reason'],'kickoff_center_setup_inferred')
        self.assertEqual(events,[])
        self.assertEqual(row['ball']['source'],'missing')
        self.assertEqual(state['support']['hint_ids'],['known-kickoff'])

    def test_ambiguous_center_and_missing_opponent_do_not_guess(self):
        meta=metadata(5);cx,cy=(v/2 for v in meta['pitch_m'])
        row=dict(frame=0,ball=point(0,None,source='missing'),players=[
            dict(point(0,cx,track_id='a',team='A'),y_m=cy)])
        self.assertIsNone(kickoff_setup(row,meta)['team'])
        row['players'].append(dict(point(0,cx,track_id='h',team='H'),y_m=cy))
        self.assertIsNone(kickoff_setup(row,meta)['team'])

    def test_opening_prior_survives_motion_warmup_but_has_bounded_expiry(self):
        meta=metadata(15);cx,cy=(v/2 for v in meta['pitch_m'])
        k=dict(frame=0,team=None,half=1,available_at_frame=0,source_id='known-kickoff')
        model=CollectiveModel(meta,collective_config={'max_pass_without_control_s':.24},kickoffs=[k])
        states=[]
        for f in range(9):
            row=dict(frame=f,ball=dict(point(f,None,source='missing'),y_m=None) if f<2 else dict(point(f,cx+5),y_m=cy),
                players=[dict(point(f,cx,track_id='a',team='A'),y_m=cy),
                         dict(point(f,cx+20,track_id='h',team='H'),y_m=cy)])
            states.append(model.feed(row)[0])
        self.assertIsNotNone(states[1]['kickoff_opening_prior'])
        self.assertEqual(states[2]['reason'],'kickoff_opening_prior')
        self.assertTrue(all(s['label']=='A' for s in states[:7]))
        self.assertIsNone(states[7]['kickoff_opening_prior'])
        self.assertEqual(states[7]['label'],'X')
        self.assertEqual(states[1]['metrics']['last_supported_control_age_s'],.04)

    def test_opening_prior_stops_when_opponent_is_near_ball(self):
        meta=metadata(5);cx,cy=(v/2 for v in meta['pitch_m'])
        k=dict(frame=0,team=None,half=1,available_at_frame=0,source_id='known-kickoff')
        model=CollectiveModel(meta,kickoffs=[k])
        def row(f,ball):
            return dict(frame=f,ball=ball,players=[dict(point(f,cx,track_id='a',team='A'),y_m=cy),
                dict(point(f,cx+10,track_id='h',team='H'),y_m=cy)])
        model.feed(row(0,point(0,None,source='missing')))
        state=model.feed(row(1,dict(point(1,cx+10),y_m=cy)))[0]
        self.assertIsNone(state['kickoff_opening_prior'])
        self.assertNotEqual(state['reason'],'kickoff_opening_prior')

    def test_center_inference_resets_independently_for_both_halves(self):
        meta=metadata(10);meta['continuity_starts']=[5]
        cx,cy=(v/2 for v in meta['pitch_m'])
        records=[dict(frame=f,team=None,half=h,available_at_frame=f,source_id=f'kickoff:{h}') for f,h in ((0,1),(5,2))]
        model=CollectiveModel(meta,kickoffs=records);states=[]
        for f in range(10):
            players=[dict(point(f,cx+(0 if ((team=='A')==(f<5)) else 20),track_id=team,team=team),y_m=cy) for team in ('H','A')]
            states.append(model.feed(dict(frame=f,ball=point(f,None,source='missing'),players=players))[0])
        self.assertEqual(states[0]['label'],'A');self.assertEqual(states[5]['label'],'H')
        self.assertEqual(states[5]['kickoff_opening_prior']['frame'],5)
        self.assertEqual(states[5]['support']['support_start_frame'],5)

    def test_loader_preserves_start_event_with_na_team(self):
        tags=[dict(code='SET PIECES',start_frame=11180,end_frame=11190,
            tags={'Half':'1st Half','Type':'Kick Off (Start)','Team':'N/A'})]
        with patch('pathlib.Path.is_file',return_value=True),patch('starter.load_smart_tagging.load',return_value=tags):
            rows,_=load_kickoffs(None,metadata(120000))
        self.assertEqual(len(rows),1);self.assertIsNone(rows[0]['team'])
