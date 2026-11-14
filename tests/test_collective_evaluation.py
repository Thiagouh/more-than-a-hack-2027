import unittest
from src.team_possession.evaluation import evaluate_collective,pass_scores,meeting_error_scores
from src.team_possession.collective import CollectiveModel
from src.team_possession.demo import cases


def predict(case,cfg=None):
    model=CollectiveModel(case['manifest'],collective_config=cfg)
    snapshots=[];events=[]
    for row in case['frames']:
        snapshot,_,batch,_=model.feed(row);snapshots.append(snapshot);events.extend(batch)
    return snapshots,events,model.finish()


class CollectiveEvaluationTests(unittest.TestCase):
    def test_complete_pass_reference_and_denominators(self):
        case=cases()[0];states,events,_=predict(case)
        r=evaluate_collective(states,events,25,passes=case['passes'],recoveries=[])
        self.assertEqual(r['pass_metrics']['overall']['reference_passes'],1)
        self.assertEqual(r['pass_metrics']['overall']['retention'],1)
        self.assertEqual(r['pass_metrics']['overall']['false_turnovers_per_100_passes'],0)
        self.assertFalse(r['possession_metrics']['available'])
        self.assertFalse(r['meeting_error']['available'])

    def test_long_pass_preserves_team_in_rolling_horizon(self):
        case=cases()[1];states,events,_=predict(case)
        r=pass_scores(states,case['passes'])
        self.assertEqual(r['overall']['retention'],1)
        self.assertTrue(any(s['state']=='passe' for s in states))

    def test_interception_precision_and_separate_timing_errors(self):
        case=cases()[3];states,events,_=predict(case)
        r=evaluate_collective(states,events,25,recoveries=case['recoveries'])['recovery_metrics']
        self.assertTrue(r['available'])
        self.assertEqual(r['matched'],1)
        self.assertEqual(r['precision'],1)
        self.assertEqual(r['recall'],1)
        self.assertGreater(r['confirmation_delay_s']['mean'],r['event_time_error_s']['mean'])

    def test_reference_not_provided_is_unavailable_not_zero(self):
        case=cases()[0];states,events,_=predict(case)
        r=evaluate_collective(states,events,25)
        self.assertFalse(r['pass_metrics']['available'])
        self.assertFalse(r['recovery_metrics']['available'])
        self.assertFalse(r['meeting_error']['available'])

    def test_abstention_is_not_a_false_team_turnover(self):
        states=[dict(frame=f,label='X') for f in range(5)]
        r=pass_scores(states,[dict(start_frame=0,end_frame_exclusive=5,team='H')])['overall']
        self.assertEqual(r['abstention'],1)
        self.assertEqual(r['false_turnovers_per_100_passes'],0)
        self.assertEqual(r['retention'],0)

    def test_forecast_errors_use_future_reference_only_for_scoring(self):
        from src.team_possession.evaluation import forecast_error_scores
        states=[dict(frame=10,metric_frame=10,forecast_xy=[[1.,0.],[2.,0.]])]
        result=forecast_error_scores(states,{11:[1.,0.],12:[3.,0.]},25,horizons_s=(.04,.08))
        self.assertEqual(result['by_horizon']['0.04']['error_m']['mae'],0)
        self.assertEqual(result['by_horizon']['0.08']['error_m']['mae'],1)
        missing=forecast_error_scores(states,{},25,horizons_s=(.04,))
        self.assertIsNone(missing['by_horizon']['0.04']['error_m']['mae'])
        self.assertEqual(missing['by_horizon']['0.04']['missing_reference'],1)

    def test_meeting_reference_scored_separately(self):
        case=cases()[0];states,_,_=predict(case)
        state=next(s for s in states if s['metrics']['origin_team_meeting_time_s'] is not None)
        reference=[dict(prediction_frame=state['frame'],team='H',reference_time_s=state['frame']/25+state['metrics']['origin_team_meeting_time_s'])]
        result=meeting_error_scores(states,reference,25)
        self.assertEqual(result['errors_s']['mae'],0)
