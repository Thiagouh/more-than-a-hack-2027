import unittest
from unittest.mock import patch

from src.ball_events.demo import metadata, point
from src.ball_events.dependencies import combine
from src.ball_events.contract import InputError
from src.team_possession.collective import CollectiveModel
from src.team_possession.restarts import load_kickoffs


def kickoff(frame=0, team='H', half=1, available=None):
    return dict(frame=frame,team=team,half=half,available_at_frame=frame if available is None else available,
                source_id=f'test:{half}')


def frame(f, team='H'):
    other='A' if team=='H' else 'H'
    return dict(frame=f,ball=point(f,20),players=[
        point(f,20,track_id='h' if team=='H' else 'a',team=team),
        point(f,45,track_id='a' if other=='A' else 'h',team=other)])


class KickoffTests(unittest.TestCase):
    def test_both_halves_reset_and_assign_immediately_without_events(self):
        meta=metadata(10);meta['continuity_starts']=[5]
        model=CollectiveModel(meta,kickoffs=[kickoff(),kickoff(5,'A',2)])
        states=[]
        for f in range(10):states.append(model.feed(frame(f,'H' if f<5 else 'A'))[0])
        for f,t in ((0,'H'),(5,'A')):
            self.assertEqual(states[f]['label'],t)
            self.assertEqual(states[f]['reason'],'kickoff_tracking_confirmed')
            self.assertEqual(states[f]['support']['support_start_frame'],f)
            self.assertIsNone(states[f]['origin_event_id'])
            self.assertEqual(states[f]['support']['hint_ids'],[f'test:{1 if f==0 else 2}'])

    def test_conflict_missing_and_absent_use_normal_inference(self):
        for records,row in (([kickoff(team='A')],frame(0)),([],frame(0)),
                            ([kickoff()],dict(frame(0),ball=point(0,None,source='missing')))):
            state=CollectiveModel(metadata(5),kickoffs=records).feed(row)[0]
            self.assertEqual(state['label'],'X')
            self.assertNotEqual(state['reason'],'kickoff_tracking_confirmed')

    def test_late_records_never_change_prefix_or_apply_late(self):
        records=[kickoff(team='A',available=4)]
        models=[CollectiveModel(metadata(8),kickoffs=records),CollectiveModel(metadata(8))]
        for f in range(8):
            states=[m.feed(frame(f))[0] for m in models]
            self.assertEqual(states[0]['label'],states[1]['label'])
            if f==4:self.assertEqual(states[0]['kickoff']['reason'],'late_kickoff_ignored')

    def test_chunk_equivalence_and_detector_unchanged(self):
        rows=[frame(f,'H' if f<5 else 'A') for f in range(10)]
        records=[kickoff(),kickoff(5,'A',2)]
        a=CollectiveModel(metadata(10),kickoffs=records)
        b=CollectiveModel(metadata(10),kickoffs=records)
        baseline=CollectiveModel(metadata(10))
        expected=[a.feed(r) for r in rows]
        actual=[]
        for chunk in (rows[:3],rows[3:6],rows[6:]):
            actual.extend(b.feed(r) for r in chunk)
        self.assertEqual(expected,actual)
        self.assertEqual([r[2] for r in expected],[baseline.feed(r)[2] for r in rows])

    def test_invalid_records(self):
        for change in ({'team':'Z'},{'half':True},{'frame':99},{'available_at_frame':-1}):
            with self.assertRaises(InputError):
                CollectiveModel(metadata(10),kickoffs=[dict(kickoff(),**change)])
        with self.assertRaises(InputError):
            CollectiveModel(metadata(10),kickoffs=[kickoff(),kickoff(5)])

    def test_tag_loader_maps_only_start_of_half(self):
        tags=[dict(code='SET PIECES',start_frame=11180,end_frame=11190,
                   tags={'Half':'1st Half','Type':'Kick Off (Start)','Team':'Synthetic Home'})]
        with patch('pathlib.Path.is_file',return_value=True),patch('starter.load_smart_tagging.load',return_value=tags):
            records,path=load_kickoffs(None,metadata(120000))
        self.assertEqual(records[0]['frame'],11185)
        self.assertEqual(records[0]['team'],'H')
        self.assertEqual(len(records),1)


class ReceptionTests(unittest.TestCase):
    def row(self,f,**changes):
        candidate=dict(team='A',track_id='receiver',distance_m=5-f*.1,
            meeting_time_s=.4,approach_distance_m=1.,relative_speed_mps=4.)
        candidate.update(changes.pop('candidate',{}))
        values=dict(observed_players_by_team={'H':2,'A':2},valid_motion_fraction_by_team={'H':1.,'A':1.},
            trajectory_error_m=0.,trajectory_error_rms_m=0.,ball_speed_fitted_mps=4.)
        values.update(changes.pop('values',{}))
        row=dict(frame=f,ball_valid=True,values=values,support=combine(frame=f).as_dict(),
            arrivals=dict(candidates=[candidate],teams={'H':{'meeting_time_s':.8}}))
        row.update(changes)
        return row

    def model(self):
        m=CollectiveModel(metadata(100));m.origin='H'
        return m

    def test_exact_duration_and_cumulative_support(self):
        m=self.model()
        results=[m._reception_step(self.row(f),f) for f in range(6)]
        self.assertFalse(any(r['accepted'] for r in results[:5]))
        self.assertTrue(results[5]['accepted'])
        self.assertEqual(results[5]['duration_s'],.2)
        self.assertEqual(m.reception_support['support_start_frame'],1)
        self.assertEqual(m.reception_support['support_end_frame'],6)

    def test_isolated_advantage_and_each_broken_requirement_reset(self):
        variants=[{'candidate':{'approach_distance_m':3.}}, {'candidate':{'meeting_time_s':.7}},
            {'values':{'trajectory_error_m':2.}}, {'values':{'trajectory_error_rms_m':2.}},
            {'values':{'ball_speed_fitted_mps':0.}}, {'ball_valid':False},
            {'values':{'observed_players_by_team':{'H':1,'A':2}}},
            {'arrivals':{'candidates':[dict(team='A',track_id='receiver',distance_m=4.,
                meeting_time_s=.4,approach_distance_m=1.)], 'teams':{'H':{'meeting_time_s':.5}}}}]
        for variant in variants:
            m=self.model()
            for f in range(5):m._reception_step(self.row(f),f)
            result=m._reception_step(self.row(5,**variant),5)
            self.assertFalse(result['accepted'])
            self.assertIsNone(m.reception)

    def test_same_candidate_closing_gap_and_confirmed_priority(self):
        for variant in ('different_player','not_closing','missing','confirmed'):
            m=self.model()
            for f in range(5):m._reception_step(self.row(f),f)
            row=self.row(5)
            if variant=='different_player':row['arrivals']['candidates'][0]['track_id']='another'
            if variant=='not_closing':row['arrivals']['candidates'][0]['distance_m']=8.
            result=m._reception_step(None if variant=='missing' else row,5,variant=='confirmed')
            self.assertFalse(result['accepted'])
            self.assertIsNone(m.reception)

    def test_kickoff_resets_reception_sequence(self):
        m=CollectiveModel(metadata(10),kickoffs=[kickoff(5)])
        for f in range(5):m.feed(frame(f))
        m.origin='H'
        for f in range(5):m._reception_step(self.row(f),f)
        m.feed(frame(5))
        self.assertIsNone(m.reception)
