"""All inputs are synthetic. Never access network, match assets or ground truth."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch

from src.ball_events import Detector, detect
from src.ball_events import contract as c
from src.ball_events.artifacts import OFFICIAL, REPO, consume, identity, run, sha
from src.ball_events.demo import metadata, point, write_inputs
from src.ball_events.evaluation import evaluate_events, report_csv
from src.ball_events.inputs import standardized
from starter.load_atd import iter_aligned_points


def sequence(xs, player_x=20., away_x=26.):
    return [dict(frame=f, ball=point(f, x, source='missing' if x is None else 'observed'),
                 players=[point(f, player_x, track_id='h', team='H'),
                          point(f, away_x, track_id='a', team='A')]) for f, x in enumerate(xs)]


def events(frames, cfg=None, meta=None, hints=()):
    return detect(meta or metadata(len(frames)), frames, cfg, hints)[1]


class DetectorTests(unittest.TestCase):
    def test_pass_no_touch_or_control_for_near_opponent(self):
        frames = sequence([19.6+i*.4 for i in range(15)], player_x=18, away_x=22)
        out = events(frames)
        self.assertFalse(any(e['type'] in ('possible_touch', 'possible_control_acquisition') for e in out))

    def test_deflection_is_not_control(self):
        out = events(sequence([19.4,19.8,20.2,19.8,19.4,19.0]))
        self.assertTrue(any(e['type']=='possible_touch' and e['event_frame']==2 and e['confirmed_at_frame']==3 for e in out))
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in out))

    def test_persistent_control_once_and_two_frame_experiment(self):
        frames = sequence([20.]*10)
        out = [e for e in events(frames) if e['type']=='possible_control_acquisition']
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]['event_frame'],out[0]['confirmed_at_frame']), (1,3))
        exp = [e for e in events(frames, {'persistence_frames':2}) if e['type']=='possible_control_acquisition']
        self.assertEqual((exp[0]['event_frame'],exp[0]['confirmed_at_frame']), (1,2))
        self.assertEqual(out[0]['lookahead_frames'], 2)
        self.assertNotIn('label', out[0])

    def test_persistent_separation_loss(self):
        out = events(sequence([20.]*6+[20.4+i*.4 for i in range(10)]))
        losses = [e for e in out if e['type']=='possible_control_loss']
        self.assertEqual(len(losses),1)
        self.assertEqual(losses[0]['confirmed_at_frame']-losses[0]['event_frame'],2)

    def test_dispute_no_winner_one_per_episode(self):
        frames = sequence([20.]*8, player_x=19.5, away_x=20.5)
        out = events(frames)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['type'],'possible_dispute')
        self.assertIsNone(out[0]['team'])

    def test_occlusion_no_loss_or_bridge(self):
        frames = sequence([20.]*6+[None]+[20.]*2)
        features, out, quality = detect(metadata(len(frames)),frames)
        self.assertFalse(any(e['type']=='possible_control_loss' for e in out))
        self.assertEqual(quality['max_missing_ball_frames'],1)
        self.assertIsNone(features[7]['ball_speed_mps'])
        self.assertFalse(any(e['event_frame'] <= 6 < e['support_end_frame'] for e in out))

    def test_estimated_interpolated_unknown_never_confirm_control_touch_loss(self):
        for source in ('estimated','interpolated','unknown'):
            frames = sequence([20.,20.,20.1,19.9,20.,20.])
            for f in frames:
                f['ball']['source']=source
            self.assertFalse(any(e['type'] in ('possible_touch','possible_control_acquisition','possible_control_loss') for e in events(frames)))

    def test_stationary_ball_and_hint_never_emit_dead_label(self):
        frames = sequence([20.]*6)
        hint = dict(hint_id='h',type='restart',start_frame=0,end_frame=5,available_at_frame=12,
                    team=None,strength='low',producer='synthetic')
        out = events(frames,hints=[hint])
        self.assertGreaterEqual(out[0]['available_at_frame'],12)
        self.assertEqual(out[0]['context_hints'][0]['hint_id'],'h')
        self.assertTrue(all('label' not in e for e in out))

    def test_jump_suppresses_touch(self):
        features, out, _ = detect(metadata(6),sequence([20.,20.,40.,40.,40.,40.]))
        self.assertIn('ball_jump',features[2]['quality_flags'])
        self.assertFalse(any(e['type']=='possible_touch' for e in out))

    def test_duplicates_and_track_changes_do_not_manufacture_control(self):
        frames = sequence([20.]*8)
        for f in frames:
            f['players'].append(point(f['frame'],20.1,track_id='duplicate',team='H'))
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in events(frames)))
        frames = sequence([20.]*8)
        for f in frames:
            f['players'][0]['track_id']=f"h{f['frame']}"
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in events(frames)))

    def test_missing_opponent_has_no_margin(self):
        frames = sequence([20.]*8)
        for f in frames:
            f['players']=f['players'][:1]
        features,out,_=detect(metadata(8),frames)
        self.assertIsNone(features[-1]['H_margin_m'])
        self.assertEqual(out,[])

    def test_delayed_opponent_delays_margin_and_event(self):
        frames=sequence([20.]*6)
        frames[2]['players'][1]['available_at_frame']=20
        out=events(frames)
        self.assertEqual(out[0]['available_at_frame'],20)

    def test_boundaries_edges_and_chunk_equivalence(self):
        frames=sequence([20.]*10)
        meta=metadata(10)
        meta['continuity_starts']=[3]
        expected=detect(meta,frames)
        d=Detector(meta)
        features,out=[],[]
        for chunk in (frames[:2],frames[2:7],frames[7:]):
            for frame in chunk:
                f,e=d.feed(frame)
                features.append(f)
                out.extend(e)
        self.assertEqual((features,out,d.finish()),expected)
        self.assertFalse(any(e['support_start_frame']<3<e['support_end_frame'] for e in out))
        self.assertEqual(events(sequence([20.,20.,20.])),[])

    def test_contract_errors(self):
        for key,value in [('contract_version','v2'),('coordinates','pixels'),('fps',float('nan')),('timeline','broadcast')]:
            meta=metadata(2)
            meta[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                Detector(meta)
        for mutate in ('duplicate','nonfinite','source','support','wrongteam'):
            frames=sequence([20.,20.])
            if mutate=='duplicate': frames[0]['players']*=2
            if mutate=='nonfinite': frames[0]['ball']['x_m']=float('inf')
            if mutate=='source': frames[0]['ball']['source']='magic'
            if mutate=='support': frames[0]['ball']['support_end_frame']=9
            if mutate=='wrongteam': frames[1]['players'][0]['team']='A'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                events(frames)
        with self.assertRaises(ValueError):
            detect(metadata(3),sequence([20.,20.]))
        with self.assertRaises(ValueError):
            events(list(reversed(sequence([20.,20.]))))


class AdapterAndArtifactTests(unittest.TestCase):
    def test_atd_strict_alignment_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta=Path(tmp)/'meta.xml'; raw=Path(tmp)/'raw.txt'
            meta.write_text('<Root><Team id="h"/><Team id="a"/><Player id="slot1" teamId="h"/><Player id="slot2" teamId="a"/></Root>')
            valid='1:0.2,0.3,0;NaN,NaN,NaN:0.2,0.3,0\n2:0.2,0.3,0;0.5,0.5,0:NaN,NaN,NaN\n'
            raw.write_text(valid)
            result=list(iter_aligned_points(meta,raw,2,[105.3,68]))
            self.assertEqual(result[0]['frame'],0)
            self.assertAlmostEqual(result[0]['ball']['x_m'],21.06)
            self.assertEqual(result[1]['ball']['source'],'missing')
            for bad in [valid.replace('2:','3:'),valid.splitlines()[0]+'\n', valid.replace(';NaN,NaN,NaN',''),valid.replace('0.2,0.3,0','inf,0.3,0')]:
                raw.write_text(bad)
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    list(iter_aligned_points(meta,raw,2,[105.3,68]))
            meta.write_text(meta.read_text().replace('teamId="a"','teamId="unknown"'))
            raw.write_text(valid)
            with self.assertRaises(ValueError):
                list(iter_aligned_points(meta,raw,2,[105.3,68]))

    def test_standard_files_optional_hint_and_missing_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=write_inputs(Path(tmp)/'inputs')
            (path.parent/'hints.jsonl').unlink()
            m,f,h,paths,status=standardized(path)
            self.assertEqual(status,'optional_file_missing')
            self.assertEqual(len(list(f())),30)
            (path.parent/'ball.csv').unlink()
            with self.assertRaises(ValueError): standardized(path)

    def test_run_reproducibility_cache_invalidation_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            path=write_inputs(tmp/'inputs')
            m,f,h,inputs,status=standardized(path)
            def execute(name,cfg=None):
                return run(m,f,h,inputs,tmp/name,cfg,'synthetic hypothesis',status,tmp/'cache',['test'])
            a=execute('a'); b=execute('b')
            self.assertFalse(a['cache_hit']); self.assertTrue(b['cache_hit'])
            for name in ('features.csv','events.jsonl','quality.json','manifest.json','events.svg'):
                self.assertEqual(sha(tmp/'a'/name),sha(tmp/'b'/name))
            ET.parse(tmp/'a/events.svg')
            self.assertFalse(execute('c',{'persistence_frames':2})['cache_hit'])
            original_key=a['cache_key']
            # Even a byte-only change is conservatively a new input identity.
            with (path.parent/'ball.csv').open('a') as handle: handle.write('\n')
            self.assertNotEqual(execute('d')['cache_key'],original_key)
            (tmp/'cache'/original_key/'features.csv').write_text('corrupt')
            # Original bytes restored: corrupt cache must be rejected.
            p=path.parent/'ball.csv'; p.write_text(p.read_text()[:-1])
            self.assertFalse(execute('e')['cache_hit'])
            def broken():
                yield next(iter(sequence([20.])))
                raise ValueError('bad row')
            with self.assertRaises(ValueError):
                run(m,broken,h,inputs,tmp/'failed',hypothesis='x')
            self.assertFalse((tmp/'failed').exists())
            attempts = [json.loads(p.read_text()) for p in (tmp/'_experiments').glob('*.json')]
            self.assertTrue(any(a['status'] == 'failed_processing' for a in attempts))
            with self.assertRaises(ValueError):
                run(m,f,h,inputs,tmp/'nohypothesis')
            before=identity(inputs,m,c.config(),h)
            with patch('src.ball_events.artifacts.sha',side_effect=lambda p: 'changed' if str(p).endswith('demo.py') else sha(p)):
                after=identity(inputs,m,c.config(),h)
            self.assertNotEqual(before,after)

    def test_consumer_checks_order_and_availability(self):
        out=events(sequence([20.]*8))
        self.assertEqual(consume(out,metadata(8))['received_events'],1)
        invalid=copy.deepcopy(out); invalid[0]['available_at_frame']=0
        with self.assertRaises(ValueError): consume(invalid,metadata(8))
        with self.assertRaises(ValueError): consume(out+out,metadata(8))

    def test_report_csv_uses_official_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); csv=tmp/'x.csv'
            csv.write_text('frame,label\n0,H\n1,X\n2,H\n3,D\n')
            result=report_csv(csv,4,25,tmp/'report')
            self.assertEqual(result['posses_brutas'],2)
            self.assertFalse(result['quality_measured'])
            csv.write_text('frame,label\n1,H\n')
            with self.assertRaises(ValueError): report_csv(csv,1,25,tmp/'bad')
            self.assertFalse((tmp/'bad').exists())

    def test_partial_reference_matching_is_one_to_one(self):
        ref={'metadata':{'provenance':'synthetic','independence':'constructed','timing':'original'},
             'events':[{'reference_id':'1','type':'pass','team':'H','time_s':1.}]}
        from test_ball_events_v2 import touch
        pred={'fps':25,'manifest':metadata(50),'events':[touch('p1',25),touch('p2',26)]}
        result=evaluate_events(pred,ref)
        self.assertFalse(result['precision_defined'])
        self.assertEqual(result['tolerances_s']['0.2']['matched'],1)
        self.assertEqual(result['tolerances_s']['0.2']['unreferenced_predictions'],1)

    def test_official_evaluator_is_unchanged(self):
        for path,digest in OFFICIAL.items():
            self.assertEqual(sha(REPO/path),digest)


class AdditionalBoundaryTests(unittest.TestCase):
    def test_relative_derivative_dependencies_are_in_support_and_availability(self):
        frames=sequence([20.]*6)
        frames[0]['ball']['available_at_frame']=50
        feature,out,_=detect(metadata(6),frames)
        acquired=next(e for e in out if e['type']=='possible_control_acquisition')
        self.assertEqual(acquired['support_start_frame'],0)
        self.assertEqual(acquired['available_at_frame'],50)
        self.assertEqual(feature[1]['available_at_frame'],50)

    def test_player_jump_and_quality_flags_suspend_individual_confirmation(self):
        frames=sequence([20.]*6)
        frames[2]['players'][0]['x_m']=35.
        feature,out,_=detect(metadata(6),frames)
        self.assertIn('player_jump', feature[2]['quality_flags'])
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in out))
        frames=sequence([20.]*6)
        for f in frames: f['players'][0]['quality_flags']=['identity_uncertain']
        self.assertFalse(any(e['type']=='possible_control_acquisition' for e in events(frames)))

    def test_invalid_configuration_and_csv_order_are_rejected(self):
        for cfg in ([], {'persistence_frames':1}, {'persistence_frames':True}, {'control_m':float('inf')}, {'unknown':1}):
            with self.subTest(cfg=cfg),self.assertRaises(ValueError): c.config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path=write_inputs(Path(tmp)/'inputs')
            ball=path.parent/'ball.csv'
            lines=ball.read_text().splitlines()
            lines[1],lines[2]=lines[2],lines[1]
            ball.write_text('\n'.join(lines)+'\n')
            m,f,_,_,_=standardized(path)
            with self.assertRaises(ValueError): list(f())

    def test_cli_hides_malformed_records_and_leaves_no_final_output(self):
        import contextlib
        import io
        from src.ball_events.__main__ import main
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); path=write_inputs(tmp/'inputs')
            ball=path.parent/'ball.csv'
            ball.write_text(ball.read_text().replace('20.0','PRIVATE_INVALID_RECORD',1))
            stderr=io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result=main(['detect','--input',str(path),'--hypothesis','contract test','--out',str(tmp/'failed')])
            self.assertEqual(result,1)
            self.assertNotIn('PRIVATE_INVALID_RECORD',stderr.getvalue())
            self.assertFalse((tmp/'failed').exists())


if __name__ == '__main__':
    unittest.main()
