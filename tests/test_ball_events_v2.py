"""Independent regression cases from review A1–A9 and properties of the replacement contracts."""
import copy
import errno
import io
import itertools
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.ball_events import detect, Detector
from src.ball_events import contract as c
from src.ball_events.artifacts import ARTIFACTS, _cache_valid, canonical, identity, run, sha
from src.ball_events.demo import metadata, point, write_inputs
from src.ball_events.events import consume, replay, validate_events
from src.ball_events.evaluation import evaluate_events, matching
from src.ball_events.inputs import standardized


def touch(identifier='p1', frame=25, available=None, team='H'):
    available=frame+1 if available is None else available
    return dict(contract_version='v2',event_id=identifier,type='possible_touch',team=team,
                track_ids=['synthetic:h'],event_frame=frame,confirmed_at_frame=frame+1,
                support_start_frame=max(0,frame-1),support_end_frame=frame+2,
                available_at_frame=available,lookahead_frames=1,availability_delay_frames=available-frame,
                strength='medium',components={'delta_speed_mps':3.},reasons=['synthetic'],quality_flags=[],context_hints=[],
                parent_event_id=None,segment_start_frame=0,
                dependencies=dict(support_start_frame=max(0,frame-1),support_end_frame=frame+2,
                                  available_at_frame=available,point_refs=['synthetic:ball@'+str(frame)],hint_ids=[],event_ids=[]))


def sequence():
    xs=[20.]*8+[20.4,20.8,21.2,21.6,22.,22.4,22.8,23.2]
    return [dict(frame=f,ball=point(f,x),players=[point(f,20.,track_id='h',team='H'),point(f,26.,track_id='a',team='A')]) for f,x in enumerate(xs)]


def reference(times):
    return dict(metadata=dict(provenance='constructed',independence='synthetic',timing='original'),
                events=[dict(reference_id=f'r{i}',type='pass',team='H',time_s=t) for i,t in enumerate(times)])


def cache_worker(path, out, cache, queue):
    try:
        m,f,h,inputs,status=standardized(path)
        result=run(m,f,h,inputs,out,hypothesis='concurrent synthetic run',hints_status=status,cache_dir=cache)
        queue.put(('ok',result['cache_key']))
    except Exception as error: queue.put(('error',type(error).__name__))


def cache_reader(path,cache,queue):
    import hashlib,time
    from src.ball_events.artifacts import cache_lock
    try:
        m,_,h,files,status=standardized(path)
        key=hashlib.sha256(canonical(identity(files,m,c.config(None,m['fps']),h,status)).encode()).hexdigest()
        for _ in range(1000):
            if cache.is_dir():
                with cache_lock(cache,key):
                    if _cache_valid(cache/key,key):
                        queue.put(('ok',key));return
            time.sleep(.01)
        queue.put(('error','reader_timeout'))
    except Exception as error:queue.put(('error',type(error).__name__))


class ReviewRegressions(unittest.TestCase):
    def test_A1_parent_point_and_hint_delays_are_transitive(self):
        for delayed_by in ('point','hint'):
            frames=sequence(); meta=metadata(len(frames)); hints=[]
            if delayed_by=='point': frames[1]['players'][1]['available_at_frame']=100
            else:
                hints=[dict(hint_id='delayed',type='opaque',producer='synthetic',start_frame=0,end_frame=4,
                            available_at_frame=100,team=None,strength='low')]
            _,ev,_=detect(meta,frames,hints=hints)
            acquisition=next(e for e in ev if e['type']=='possible_control_acquisition')
            loss=next(e for e in ev if e['type']=='possible_control_loss')
            self.assertEqual(loss['parent_event_id'],acquisition['event_id'])
            self.assertEqual(loss['available_at_frame'],100)
            self.assertEqual(loss['support_start_frame'],acquisition['support_start_frame'])
            check=replay(ev,meta,drain=True)
            ids=[row['event_id'] for row in check['deliveries']]
            self.assertLess(ids.index(acquisition['event_id']),ids.index(loss['event_id']))
            self.assertTrue(replay(ev,meta)['pending_at_end'])

    def test_A2_cache_destination_rejected_and_postpublication_fault_is_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); path=write_inputs(tmp/'input'); m,f,h,inputs,status=standardized(path)
            occupied=tmp/'file'; occupied.write_text('occupied')
            with self.assertRaises(c.InputError):
                run(m,f,h,inputs,tmp/'invalid',hypothesis='x',cache_dir=occupied)
            self.assertFalse((tmp/'invalid').exists())
            for code in (errno.ENOSPC,errno.EACCES):
                with patch('src.ball_events.artifacts.cache_write',side_effect=OSError(code,'simulated')):
                    result=run(m,f,h,inputs,tmp/str(code),hypothesis='x',cache_dir=tmp/'cache')
                self.assertEqual(result['cache_status'],'write_failed')
                self.assertTrue((tmp/str(code)/'manifest.json').exists())
            attempts=[json.loads(p.read_text()) for p in (tmp/'_experiments').glob('*.json')]
            self.assertTrue(any(a['status']=='rejected' for a in attempts))
            self.assertEqual(sum(a['status']=='completed' for a in attempts),2)

    def test_A3_structural_corruption_is_always_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry=Path(tmp)
            for value in (None,[],{},dict(cache_key='key',artifact_sha256=list(ARTIFACTS)),
                          dict(cache_key='key',artifact_sha256=None),
                          dict(cache_key='key',artifact_sha256={k:3 for k in ARTIFACTS})):
                (entry/'manifest.json').write_text(json.dumps(value))
                self.assertFalse(_cache_valid(entry,'key'))

    def test_A4_all_malformed_collection_boundaries_are_safe(self):
        goodpred={'fps':25,'manifest':metadata(50),'events':[touch()]}; goodref=reference([1.])
        for value in (None,[],True,'x',1):
            for pred,ref in ((value,goodref),(goodpred,value),(goodpred,{'metadata':value,'events':[]})):
                with self.subTest(value=value),self.assertRaises(c.InputError): evaluate_events(pred,ref)
            for key in ('events','manifest'):
                if key=='events' and value==[]: continue
                bad=dict(goodpred); bad[key]=value
                with self.assertRaises(c.InputError): evaluate_events(bad,goodref)
        for baditem in (None,[],True):
            with self.assertRaises(c.InputError): evaluate_events(goodpred,dict(goodref,events=[baditem]))
        with tempfile.TemporaryDirectory() as tmp:
            path=write_inputs(Path(tmp)/'input'); meta=json.loads(path.read_text()); meta['files']=['players','ball']; path.write_text(json.dumps(meta))
            with self.assertRaises(c.InputError): standardized(path)

    def test_A5_duplicate_ids_and_bool_frame_rejected(self):
        pred={'fps':25,'manifest':metadata(50),'events':[touch(),touch()]}
        with self.assertRaises(c.InputError): evaluate_events(pred,reference([1.,1.1]))
        pred['events']=[touch()]; pred['events'][0]['event_frame']=True
        with self.assertRaises(c.InputError): evaluate_events(pred,reference([1.]))

    def test_A6_contract_and_causal_graph_are_checked(self):
        for key,value in [('contract_version','v99'),('type','invalid'),('team','INVALID'),('components',[]),
                          ('available_at_frame',True),('lookahead_frames',0),('event_frame',100)]:
            e=touch(); e[key]=value
            with self.subTest(key=key),self.assertRaises(c.InputError): consume([e],metadata(50))
        _,ev,_=detect(metadata(16),sequence())
        loss=next(e for e in ev if e['type']=='possible_control_loss')
        with self.assertRaises(c.InputError): validate_events([loss],metadata(16))
        invalid=copy.deepcopy(ev)
        child=next(e for e in invalid if e['type']=='possible_control_loss')
        child['parent_event_id']=child['event_id']; child['dependencies']['event_ids']=[child['event_id']]
        with self.assertRaises(c.InputError): validate_events(invalid,metadata(16))
        invalid=copy.deepcopy(ev); next(e for e in invalid if e['type']=='possible_control_loss')['team']='A'
        with self.assertRaises(c.InputError): validate_events(invalid,metadata(16))

    def test_A7_omitted_flags_rejected_in_api_and_blank_csv(self):
        frames=sequence(); del frames[0]['ball']['quality_flags']
        with self.assertRaises(c.InputError): detect(metadata(16),frames)
        with tempfile.TemporaryDirectory() as tmp:
            path=write_inputs(Path(tmp)/'input'); ball=path.parent/'ball.csv'
            ball.write_text(ball.read_text().replace('[]','',1))
            m,f,_,_,_=standardized(path)
            with self.assertRaises(c.InputError): list(f())

    def test_A8_rejection_is_audited_before_config_and_cli_read(self):
        from src.ball_events.__main__ import main
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); path=write_inputs(tmp/'input'); m,f,h,inputs,_=standardized(path)
            with self.assertRaises(c.InputError): run(m,f,h,inputs,tmp/'bad',{'persistence_frames':1},'x')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['detect','--input',str(tmp/'missing.json'),'--hypothesis','x','--out',str(tmp/'cli')]),1)
            attempts=[json.loads(p.read_text()) for p in (tmp/'_experiments').glob('*.json')]
            self.assertEqual(len(attempts),2)
            self.assertTrue(all(a['status']=='rejected' for a in attempts))

    def test_A9_timing_error_is_distinct_from_confirmation_and_delivery(self):
        pred={'fps':25,'manifest':metadata(50),'events':[touch(frame=25,available=30)]}
        row=evaluate_events(pred,reference([1.]))['tolerances_s']['0.2']['pairs'][0]
        self.assertEqual(row['event_time_error_s'],0.)
        self.assertEqual(row['confirmation_delay_s'],.04)
        self.assertEqual(row['availability_delay_s'],.2)
        self.assertEqual(row['delivery_time_error_s'],.2)


class MatchingProperties(unittest.TestCase):
    def test_maximum_cardinality_and_boundary(self):
        ps=[touch('a',30),touch('b',2)]
        refs=reference([.2,.5])['events']
        self.assertEqual(len(matching(ps,refs,100,.2)),2)
        pair=matching([touch('a',30)],reference([.5])['events'],100,.2)
        self.assertEqual(len(pair),1)
        self.assertEqual(matching(list(reversed(ps)),list(reversed(refs)),100,.2),matching(ps,refs,100,.2))

    def test_compare_against_exhaustive_small_assignments(self):
        from fractions import Fraction
        for ptimes in itertools.combinations(range(0,7),3):
            ps=[touch(str(i),t) for i,t in enumerate(ptimes)]
            refs=reference([0.,.2,.5])['events']
            chosen=matching(ps,refs,10,.2)
            cost=lambda pairs:sum(abs(Fraction(p['event_frame'],10)-Fraction(str(r['time_s']))) for p,r in pairs)
            best=(0,Fraction(0))
            for count in range(1,4):
                for picks in itertools.combinations(range(3),count):
                    for targets in itertools.permutations(range(3),count):
                        pairs=[(ps[i],refs[j]) for i,j in zip(picks,targets)]
                        if all(abs(Fraction(p['event_frame'],10)-Fraction(str(r['time_s'])))<=Fraction('0.2') for p,r in pairs):
                            score=(count,-cost(pairs))
                            if score>best: best=score
            self.assertEqual((len(chosen),-cost(chosen)),best)


class OperationalProperties(unittest.TestCase):
    def test_hints_status_participates_in_identity(self):
        self.assertNotEqual(identity({},metadata(),c.config(),[],'provided'),identity({},metadata(),c.config(),[],'optional_file_missing'))

    def test_concurrent_cache_writers_then_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); path=write_inputs(tmp/'input'); cache=tmp/'cache'
            ctx=multiprocessing.get_context('spawn'); q=ctx.Queue()
            workers=[ctx.Process(target=cache_worker,args=(path,tmp/f'run{i}',cache,q)) for i in range(2)]
            workers.append(ctx.Process(target=cache_reader,args=(path,cache,q)))
            for p in workers:p.start()
            for p in workers:p.join(30); self.assertEqual(p.exitcode,0)
            results=[q.get(timeout=2) for _ in workers]
            self.assertTrue(all(r[0]=='ok' for r in results),results)
            self.assertEqual(results[0][1],results[1][1])
            cache_worker(path,tmp/'reader',cache,q)
            self.assertEqual(q.get(timeout=2)[0],'ok')
            self.assertTrue(_cache_valid(cache/results[0][1],results[0][1]))
            self.assertEqual(sha(tmp/'run0/events.jsonl'),sha(tmp/'reader/events.jsonl'))

    def test_duration_and_frame_rate_are_explicit(self):
        m=metadata(16); m['fps']=10
        with self.assertRaises(c.InputError): Detector(m)
        d=Detector(m,{'persistence_mode':'duration','persistence_seconds':.12})
        self.assertEqual(d.cfg['persistence_frames'],3)
        d=Detector(m,{'reference_fps':10,'persistence_frames':3})
        self.assertEqual(d.cfg['reference_fps'],10)
        with self.assertRaises(c.InputError): c.config({'persistence_mode':'duration','persistence_seconds':.1,'persistence_frames':3},25)

    def test_unrelated_old_player_does_not_delay_control(self):
        frames=sequence()
        # A remote old observation was not used in current candidate motion checks.
        frames[0]['players'].append(point(0,80.,track_id='remote',team='A',available_at_frame=100))
        _,ev,_=detect(metadata(16),frames)
        acquired=next(e for e in ev if e['type']=='possible_control_acquisition')
        self.assertLess(acquired['available_at_frame'],100)
        self.assertFalse(any('remote@0' in ref for ref in acquired['dependencies']['point_refs']))

    def test_extreme_numbers_and_clip_origin(self):
        for value in (True,10**1000,float('inf')):
            with self.assertRaises(c.InputError): c.number(value,'number')
        m=metadata(); m['is_clip']=True
        with self.assertRaises(c.InputError): c.manifest(m)
        m['origin']={'timeline':'tactical','first_frame':1000,'fps':25}
        c.manifest(m)


class HardenedBoundaries(unittest.TestCase):
    def test_cli_references_with_invalid_roots_are_audited_without_traceback(self):
        from src.ball_events.__main__ import main
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); fixture=json.loads((Path(__file__).parent/'fixtures/ball_events/causal-v2.json').read_text())
            from src.ball_events.artifacts import write_json
            write_json(tmp/'manifest.json',{'provenance':{'input_metadata':fixture['manifest']}})
            (tmp/'events.jsonl').write_text('\n'.join(canonical(e) for e in fixture['expected_events'])+'\n')
            for i,ref in enumerate(([],{'metadata':[],'events':[]})):
                write_json(tmp/'reference.json',ref); stream=io.StringIO()
                with contextlib.redirect_stderr(stream):
                    status=main(['evaluate-events','--events',str(tmp/'events.jsonl'),'--fps','25','--reference',str(tmp/'reference.json'),'--out',str(tmp/f'report{i}.json')])
                self.assertEqual(status,1);self.assertNotIn('Traceback',stream.getvalue())
                self.assertFalse((tmp/f'report{i}.json').exists())
            logs=[json.loads(p.read_text()) for p in (tmp/'_experiments').glob('*.json')]
            self.assertEqual([r['status'] for r in logs],['rejected','rejected'])

    def test_cache_read_fault_recalculates_and_permission_preflight_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=write_inputs(tmp/'input');m,f,h,files,status=standardized(path)
            original=run(m,f,h,files,tmp/'first',hypothesis='x',cache_dir=tmp/'cache',hints_status=status)
            with patch('src.ball_events.artifacts.shutil.copyfile',side_effect=OSError(errno.EACCES,'simulated')):
                result=run(m,f,h,files,tmp/'second',hypothesis='x',cache_dir=tmp/'cache',hints_status=status)
            self.assertFalse(result['cache_hit'])
            self.assertEqual(sha(tmp/'first/events.jsonl'),sha(tmp/'second/events.jsonl'))
            with patch('src.ball_events.artifacts.tempfile.TemporaryFile',side_effect=OSError(errno.EACCES,'simulated')):
                with self.assertRaises(c.InputError):run(m,f,h,files,tmp/'blocked',hypothesis='x',cache_dir=tmp/'blocked-cache')
            self.assertFalse((tmp/'blocked').exists())

    def test_cache_overlap_and_hint_status_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);path=write_inputs(tmp/'input');m,f,h,files,status=standardized(path)
            with self.assertRaises(c.InputError):run(m,f,h,files,tmp/'out',hypothesis='x',cache_dir=tmp/'input')
            a=run(m,f,h,files,tmp/'a',hypothesis='x',cache_dir=tmp/'cache',hints_status='provided')
            b=run(m,f,h,files,tmp/'b',hypothesis='x',cache_dir=tmp/'cache',hints_status='optional_file_missing')
            self.assertNotEqual(a['cache_key'],b['cache_key']);self.assertFalse(b['cache_hit'])
            self.assertEqual(json.loads((tmp/'b/quality.json').read_text())['hints_status'],'optional_file_missing')

    def test_namespace_xml_names_frequency_and_diagnostics(self):
        from starter.load_atd import inspect_atd_metadata,iter_aligned_points
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);meta=tmp/'meta.xml';raw=tmp/'raw.txt'
            xml='<Root xmlns="urn:fixture"><FrameRate>25</FrameRate><FieldSize><Width>105</Width><Height>68</Height></FieldSize><Team id="h"><Name>Barcelona</Name></Team><Team id="a"><Name>Athletic Club</Name></Team><Player id="p1" teamId="h"/><Player id="p2" teamId="a"/></Root>'
            meta.write_text(xml);raw.write_text('1:0.2,0.3,0;0.6,0.5,0:1.4,0.3,0\n')
            diag=inspect_atd_metadata(meta,[105.3,68],25,{'H':'Barcelona','A':'Athletic Club'})
            self.assertTrue(diag['dimensions_differ']);self.assertFalse(diag['orientation_verified'])
            meta.write_text(xml.replace('Athletic Club', 'Atletic Club'))
            inspect_atd_metadata(meta,[105.3,68],25,{'H':'Barcelona','A':'Athletic Club'})
            with self.assertRaises(c.InputError):
                inspect_atd_metadata(meta,[105.3,68],25,{'H':'Barcelona','A':'Other Club'})
            meta.write_text(xml)
            frame=list(iter_aligned_points(meta,raw,1,[105.3,68]))[0]
            self.assertIn('implausible_position',frame['ball']['quality_flags'])
            self.assertGreater(frame['ball']['x_m'],105.3)
            for bad in (xml.replace('<Name>Barcelona</Name>','<Name> </Name>'),xml.replace('id="p1"','id=""'),xml.replace('>25<','>10<')):
                meta.write_text(bad)
                with self.assertRaises(c.InputError):inspect_atd_metadata(meta,[105.3,68],25)

    def test_quality_contains_player_flags_and_coverage(self):
        frames=sequence();frames[1]['players'][0]['quality_flags']=['identity_uncertain']
        _,_,q=detect(metadata(16),frames)
        self.assertEqual(q['player_flag_observations']['synthetic:identity_uncertain'],1)
        self.assertEqual(q['coverage_by_team']['H_frames_with_player'],16)
        self.assertGreater(q['ineligible_reasons']['candidate_producer_flags'],0)

    def test_deterministic_fixture_and_schema_validation(self):
        root=Path(__file__).resolve().parents[1]
        fixture=json.loads((root/'tests/fixtures/ball_events/causal-v2.json').read_text())
        _,events,q=detect(fixture['manifest'],fixture['frames'],hints=fixture['hints'])
        self.assertEqual(events,fixture['expected_events']);self.assertEqual(q,fixture['expected_quality'])
        validate_events(events,fixture['manifest'])
        try:from jsonschema import Draft202012Validator
        except ImportError:self.skipTest('Install requirements-dev.txt for JSON Schema validation')
        schemas={p.stem.replace('.schema',''):json.loads(p.read_text()) for p in (root/'schemas').glob('*.json')}
        for s in schemas.values():Draft202012Validator.check_schema(s)
        Draft202012Validator(schemas['manifest-v1']).validate(fixture['manifest'])
        for f in fixture['frames']:
            Draft202012Validator(schemas['envelope-v1']).validate(f)
            c.envelope(f);c.point(f['ball'],f['frame'],fixture['manifest'])
            for p in f['players']:c.point(p,f['frame'],fixture['manifest'],True)
        for e in events:Draft202012Validator(schemas['event-v2']).validate(e)
        for key,value in (('contract_version','v1'),('type','wrong'),('components',[])):
            bad=copy.deepcopy(events[0]);bad[key]=value
            self.assertFalse(Draft202012Validator(schemas['event-v2']).is_valid(bad))
            with self.assertRaises(c.InputError):validate_events([bad],fixture['manifest'])

    def test_visual_crop_and_drain_are_accessible(self):
        from src.ball_events.visual import render_svg,event_table
        fixture=json.loads((Path(__file__).parent/'fixtures/ball_events/causal-v2.json').read_text())
        meta=fixture['manifest'];meta['is_clip']=True;meta['origin']={'timeline':'tactical','first_frame':1000,'fps':25}
        text=io.StringIO();render_svg(text,iter(fixture['expected_events']),[('observed',0,32)],meta,0,16)
        import xml.etree.ElementTree as ET
        ET.fromstring(text.getvalue())
        self.assertIn('1000.00',text.getvalue());self.assertIn('fora do intervalo',text.getvalue())
        table=io.StringIO();event_table(table,fixture['expected_events'])
        self.assertIn('possible_control_loss',table.getvalue());self.assertIn('parent_event_id',table.getvalue())
        check=replay(fixture['expected_events'],meta,drain=True)
        self.assertTrue(check['pending_at_end']);self.assertEqual(len(check['deliveries']),len(fixture['expected_events']))

    def test_new_segment_cannot_inherit_parent_and_chunk_results_match(self):
        frames=sequence();meta=metadata(16);meta['continuity_starts']=[9]
        expected=detect(meta,frames)
        self.assertFalse(any(e['type']=='possible_control_loss' for e in expected[1]))
        d=Detector(meta);features=[];events=[]
        for chunk in (frames[:5],frames[5:11],frames[11:]):
            for f in chunk:
                a,b=d.feed(f);features.append(a);events.extend(b)
        self.assertEqual((features,events,d.finish()),expected)


class CliSuccessTests(unittest.TestCase):
    def test_causal_demo_render_replay_and_evaluation(self):
        from src.ball_events.__main__ import main
        import contextlib
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['demo','--out',str(tmp/'demo')]),0)
                self.assertEqual(main(['render','--run',str(tmp/'demo/run'),'--start','0','--end','16','--out',str(tmp/'view')]),0)
                self.assertEqual(main(['replay','--run',str(tmp/'demo/run'),'--drain','--out',str(tmp/'replay')]),0)
                ref=reference([7/25]); (tmp/'ref.json').write_text(json.dumps(ref))
                self.assertEqual(main(['evaluate-events','--events',str(tmp/'demo/run/events.jsonl'),'--fps','25','--reference',str(tmp/'ref.json'),'--out',str(tmp/'report.json')]),0)
            events=[json.loads(line) for line in (tmp/'demo/run/events.jsonl').read_text().splitlines()]
            parent=next(e for e in events if e['type']=='possible_control_acquisition' and e['event_frame']==1)
            loss=next(e for e in events if e['type']=='possible_control_loss')
            self.assertEqual(parent['available_at_frame'],100);self.assertEqual(loss['available_at_frame'],100)
            self.assertIn('possible_control_loss',(tmp/'view/events-table.csv').read_text())
            report=json.loads((tmp/'report.json').read_text());self.assertIn('provenance',report)
            self.assertEqual(report['matching_version'],'max-cardinality-min-absolute-error-v2')


class DependencyPrecisionTests(unittest.TestCase):
    def test_dispute_does_not_wait_for_ball_two_frames_earlier(self):
        frames=[dict(frame=f,ball=point(f,x),players=[point(f,19.5,track_id='h',team='H'),point(f,20.5,track_id='a',team='A')])
                for f,x in enumerate([22.,21.8,21.4])]
        frames[0]['ball']['available_at_frame']=100
        _,ev,_=detect(metadata(3),frames)
        dispute=next(e for e in ev if e['type']=='possible_dispute')
        self.assertEqual(dispute['event_frame'],2);self.assertEqual(dispute['available_at_frame'],2)
        self.assertNotIn('synthetic:ball@0',dispute['dependencies']['point_refs'])

    def test_touch_carries_previous_endpoint_jump_check(self):
        frames=[dict(frame=f,ball=point(f,x),players=[point(f,20.,track_id='h',team='H'),point(f,26.,track_id='a',team='A')])
                for f,x in enumerate([19.2,19.6,20.,19.6])]
        frames[0]['ball']['available_at_frame']=100
        _,ev,_=detect(metadata(4),frames)
        touch_event=next(e for e in ev if e['type']=='possible_touch' and e['event_frame']==2)
        self.assertEqual(touch_event['available_at_frame'],100)
        self.assertIn('synthetic:ball@0',touch_event['dependencies']['point_refs'])


class AdditionalContractChecks(unittest.TestCase):
    def test_type_timing_dependency_namespace_and_metric_field_coverage(self):
        e=touch();e.update(confirmed_at_frame=27,support_end_frame=28,available_at_frame=27,lookahead_frames=2,availability_delay_frames=2)
        e['dependencies'].update(support_end_frame=28,available_at_frame=27)
        with self.assertRaises(c.InputError):validate_events([e],metadata(50))
        for point_ref in ('undeclared:ball@25','synthetic:ball@99999'):
            e=touch();e['dependencies']['point_refs']=[point_ref]
            with self.assertRaises(c.InputError):validate_events([e],metadata(50))
        with self.assertRaises(c.InputError):c.number(1e-300,'fps',True)
        r=evaluate_events({'fps':25,'manifest':metadata(50),'events':[touch()]},reference([1.,1.1]))['tolerances_s']['0.2']
        self.assertEqual(r['reference_recall'],.5)
        self.assertEqual(r['metrics']['availability_delay_s']['coverage'],1.)


if __name__=='__main__': unittest.main()
