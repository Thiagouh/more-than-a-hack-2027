import unittest
import numpy as np
from src.team_identification.temporal_windows import CONFIGURATIONS, decide, check_separation

class Model:
    max_distance=.35
    min_margin=.1
    def predict(self, crop):
        return {} if crop is None else {'class_distances': {'Barcelona':float(crop), 'Athletic':.8}}

class TemporalWindowsTest(unittest.TestCase):
    def rows(self):
        return [dict(frame=f,identity='p',association_verified=True,available_at_frame=f,crop=.1) for f in range(51)]
    def test_all_spacings_and_bounds(self):
        for offsets in CONFIGURATIONS.values():
            r=decide(Model(),'p',50,self.rows(),offsets)
            self.assertEqual(r['label'],'Barcelona')
            self.assertEqual(r['frame_ids'],[50+o for o in offsets])
            self.assertEqual(r['window_span_s'],-offsets[0]/25)
    def test_intermediate_loss_and_slot_reuse(self):
        for mutation in ['missing','identity','association_verified']:
            rows=self.rows()
            if mutation=='missing':del rows[12]
            else:rows[12][mutation]='q' if mutation=='identity' else False
            self.assertEqual(decide(Model(),'p',50,rows,CONFIGURATIONS['twelve_spaced'])['label'],'Incerto')
            self.assertEqual(decide(Model(),'p',50,rows,CONFIGURATIONS['four_consecutive'])['label'],'Barcelona')
    def test_causality_and_availability(self):
        with self.assertRaises(ValueError):decide(Model(),'p',50,self.rows(),[-1,0,1])
        rows=self.rows();rows[49]['available_at_frame']=51
        self.assertEqual(decide(Model(),'p',50,rows,[-3,-2,-1,0])['reason'],'observation_unavailable')
        rows=self.rows()+[dict(frame=51)]
        self.assertEqual(decide(Model(),'p',50,rows,[-3,-2,-1,0])['label'],'Barcelona')
    def test_missing_descriptors_and_rounding(self):
        rows=self.rows()
        for f in [0,5,9,14]:rows[f]['crop']=None
        r=decide(Model(),'p',50,rows,CONFIGURATIONS['twelve_spaced'])
        self.assertEqual(r['required_support'],9)
        self.assertEqual(r['reason'],'insufficient_valid_uniform_frames')
        self.assertEqual(decide(Model(),'p',2,self.rows(),[-3,-2,-1,0])['reason'],'window_before_source')
    def test_leakage_full_window(self):
        for f in [950,1049,1799]:
            with self.assertRaises(ValueError):check_separation(1000,1050,'s',[dict(frame=f,source_video_sha256='s')],[],{})
        with self.assertRaises(ValueError):check_separation(1000,1050,'s',[],[999],{})
        with self.assertRaises(ValueError):check_separation(1000,1050,'s',[],[],{'train':[(900,1001)]})
        check_separation(1000,1050,'s',[dict(frame=250,source_video_sha256='s')],[],{})

if __name__=='__main__':unittest.main()

class ConsensusTests(unittest.TestCase):
    def test_support_is_fraction_of_planned_not_available(self):
        class VotingModel(Model):
            def predict(self,crop):return {} if crop is None else {'class_distances': {'Barcelona':crop,'Athletic':.3}}
        rows=TemporalWindowsTest().rows()
        for f in [0,5,9]:rows[f]['crop']=.4
        result=decide(VotingModel(),'p',50,rows,CONFIGURATIONS['twelve_spaced'])
        self.assertEqual(result['supporting_frames'],9)
        self.assertEqual(result['label'],'Barcelona')
        rows[14]['crop']=.4
        result=decide(VotingModel(),'p',50,rows,CONFIGURATIONS['twelve_spaced'])
        self.assertEqual(result['supporting_frames'],8)
        self.assertEqual(result['label'],'Incerto')
    def test_duplicate_frames_and_nonfinite(self):
        rows=TemporalWindowsTest().rows()
        with self.assertRaises(ValueError):decide(Model(),'p',50,rows+[rows[0]],[-3,-2,-1,0])
        rows[50]['crop']=float('nan')
        with self.assertRaises(ValueError):decide(Model(),'p',50,rows,[-3,-2,-1,0])
    def test_legacy_equivalence_when_inputs_available(self):
        from src.team_identification.four_frames import FourFrameUniforms
        model=Model();stream=FourFrameUniforms(model)
        for f in range(47,51):legacy=stream.feed('p',f,.1)
        current=decide(model,'p',50,TemporalWindowsTest().rows(),[-3,-2,-1,0])
        for key in ['label','team','role','distance','margin','class_distances','supporting_frames']:
            self.assertEqual(current[key],legacy[key])
