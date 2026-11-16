import unittest
from src.team_identification.spaced_frames import SpacedFourFrameUniforms, POLICY
from src.team_identification.temporal_windows import decide
from tests.test_four_frame_uniforms import Scores


class SpacedFrameTests(unittest.TestCase):
    def test_waits_for_full_continuity_then_samples_four_offsets(self):
        stream=SpacedFourFrameUniforms(Scores())
        for f in range(16):
            result=stream.feed('p',f,'H' if f in (0,5,10,15) else 'A')
            if f<15:self.assertEqual(result['label'],'Incerto')
        self.assertEqual(result['frame_ids'],[0,5,10,15])
        self.assertEqual(result['label'],'Barcelona')
        self.assertEqual(result['policy'],POLICY)
        self.assertEqual(result['window_span_s'],.6)
        self.assertEqual(stream.feed('p',16,'H')['frame_ids'],[1,6,11,16])
        self.assertEqual(len(stream.tracks['p']),16)

    def test_gap_ambiguity_and_retirement_restart_warmup(self):
        for mode in ('gap','ambiguous','retire','reset'):
            stream=SpacedFourFrameUniforms(Scores())
            for f in range(16):stream.feed('p',f,'H')
            if mode=='retire':stream.retain(set())
            if mode=='reset':stream.reset()
            if mode=='ambiguous':stream.feed('p',16,'H',association_verified=False)
            f=17 if mode in ('gap','ambiguous') else 16
            result=stream.feed('p',f,'A')
            self.assertEqual(result['label'],'Incerto')
            self.assertEqual(result['frames_observed'],1)

    def test_future_pixels_are_not_read_and_do_not_become_votes(self):
        stream=SpacedFourFrameUniforms(Scores())
        for f in range(16):result=stream.feed('p',f,'future' if f==15 else 'H',available_at_frame=f+1 if f==15 else f)
        self.assertEqual(result['label'],'Incerto')
        self.assertEqual(result['reason'],'observation_unavailable')

    def test_missing_descriptor_requires_three_of_four(self):
        for missing,expected in [({0},'Barcelona'),({0,5},'Incerto')]:
            stream=SpacedFourFrameUniforms(Scores())
            for f in range(16):result=stream.feed('p',f,'missing' if f in missing else 'H')
            self.assertEqual(result['label'],expected)

    def test_matches_frozen_experiment_combination(self):
        stream=SpacedFourFrameUniforms(Scores());rows=[]
        for f in range(25):
            value='A' if f%7==0 else 'H'
            rows.append(dict(identity='p',frame=f,available_at_frame=f,association_verified=True,crop=value))
            actual=stream.feed('p',f,value)
            if f>=15:
                expected=decide(Scores(),'p',f,rows,[-15,-10,-5,0])
                for key in ['label','distance','margin','supporting_frames','valid_frames','frame_ids']:
                    self.assertEqual(actual[key],expected[key])
        with self.assertRaises(ValueError):stream.feed('p',24,'H')

class OverlayPolicyTests(unittest.TestCase):
    def test_spaced_metadata_and_legacy_loading(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from src.team_identification.manifests import write_json
        from src.team_identification.overlay import load_overlay, SCHEMA
        from src.team_identification.four_frames import POLICY as LEGACY
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            artifact=dict(examples=[dict(label=c,source_video_sha256='hash') for c in ['Barcelona','Athletic','Arbitro','Goleiro Barcelona','Goleiro Athletic']])
            write_json(root/'model.json',artifact)
            (root/'observations.jsonl').write_text('{"frame":0}\n')
            meta=dict(timeline='tactical',fps=25,n_frames=10)
            package=dict(schema=SCHEMA,**meta,status='complete',inference_mode='full_frame',
                source_video_sha256='hash',atd_raw_sha256='hash',homography_sha256='hash',
                supervised_model='model.json',supervised_model_sha256='hash',uniform_policy=POLICY,
                uniform_window_frames=4,uniform_offsets=[-15,-10,-5,0],uniform_window_span_s=.6,
                uniform_continuity_window_frames=16,observations='observations.jsonl',observations_sha256='hash',
                start_frame=0,end_frame_exclusive=1)
            with patch('src.team_identification.overlay.sha256',return_value='hash'),patch('src.team_identification.supervised_uniforms.SupervisedUniforms'):
                write_json(root/'overlay.json',package)
                rows,_=load_overlay(root/'overlay.json',meta,'video','atd','homography')
                self.assertEqual(set(rows),{0})
                for key,bad in [('uniform_offsets',[-3,-2,-1,0]),('uniform_window_span_s',.12),('uniform_continuity_window_frames',4)]:
                    write_json(root/'overlay.json',dict(package,**{key:bad}))
                    with self.assertRaises(ValueError):load_overlay(root/'overlay.json',meta,'video','atd','homography')
                package['uniform_policy']=LEGACY
                for key in ['uniform_offsets','uniform_window_span_s','uniform_continuity_window_frames']:package.pop(key)
                write_json(root/'overlay.json',package)
                self.assertEqual(set(load_overlay(root/'overlay.json',meta,'video','atd','homography')[0]),{0})
