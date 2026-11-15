import copy
import json
import tempfile
import unittest
from pathlib import Path
from src.team_identification.goalkeeper_context import infer_goalkeeper_team,CONTEXT_PROFILE
from src.team_identification.manifests import read_json,write_json,freeze,load_frozen
from src.team_identification.pilot import sha256
from src.team_identification.reference_review import completeness
from scripts.apply_goalkeeper_context import apply
from tests import test_reference_review as fixtures

POLICY=Path(__file__).resolve().parents[1]/'configs/team-pilot-goalkeeper-context.json'


class GoalkeeperContextTests(unittest.TestCase):
    def setUp(self):
        self.policy=read_json(POLICY)
        self.evidence=dict(goalkeeper_candidate=True,distinctive_kit=True,near_defended_goal=True,
            period_verified=True,pitch_end_verified=True,orientation_verified=True,half=1,
            pitch_end='left',view='wide_pitch',play_origin_verified=True,kit_family='green',
            notes='Synthetic reviewed wide view near defended goal')

    def test_halves_and_ends_follow_user_mapping(self):
        for half,side,team in ((1,'left','H'),(1,'right','A'),(2,'left','A'),(2,'right','H')):
            result=infer_goalkeeper_team(self.policy,dict(self.evidence,half=half,pitch_end=side))
            self.assertEqual(result['classification'],team)
            self.assertEqual(result['role'],'goalkeeper')
            self.assertFalse(result['visual_reference_verified'])
            self.assertFalse(result['calibrated_probability'])

    def test_color_does_not_determine_team(self):
        for kit in ('green','teal'):
            self.assertEqual(infer_goalkeeper_team(self.policy,dict(self.evidence,kit_family=kit))['team'],'H')
        self.assertEqual(infer_goalkeeper_team(self.policy,dict(self.evidence,kit_family='yellow'))['classification'],'uncertain')

    def test_unknown_or_unverified_context_abstains(self):
        for field in ('goalkeeper_candidate','distinctive_kit','near_defended_goal','period_verified',
                      'pitch_end_verified','orientation_verified','play_origin_verified'):
            self.assertEqual(infer_goalkeeper_team(self.policy,dict(self.evidence,**{field:False}))['classification'],'uncertain',field)
        for field,value in (('half',None),('half',True),('half',3),('pitch_end','center'),('view','closeup'),('notes','')):
            self.assertEqual(infer_goalkeeper_team(self.policy,dict(self.evidence,**{field:value}))['classification'],'uncertain')

    def test_unconfirmed_policy_rejected(self):
        policy=copy.deepcopy(self.policy);policy['confirmed_by']='inferred'
        with self.assertRaises(ValueError):infer_goalkeeper_team(policy,self.evidence)

    def test_context_profile_waives_only_goalkeeper_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,spec=fixtures.ReferenceReviewTests().candidate(root)
            spec.update(reference_profile=CONTEXT_PROFILE,goalkeeper_context_policy=self.policy)
            spec['references']=[r for r in spec['references'] if r['role']!='goalkeeper']
            self.assertTrue(completeness(spec)['ready'])
            write_json(path,spec);freeze(path,root/'frozen',require_complete=True)
            _,loaded=load_frozen(root/'frozen/frozen-references.json')
            self.assertEqual(loaded['reference_profile'],CONTEXT_PROFILE)
            self.assertEqual(len(loaded['references']),5)
            spec['references'][0]['approved']=False
            self.assertFalse(completeness(spec)['ready'])
            spec['references'][0]['approved']=True;spec.pop('goalkeeper_context_policy')
            self.assertFalse(completeness(spec)['ready'])

    def test_sidecar_preserves_visual_and_flags_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);detections=root/'detections.jsonl';hints=root/'context.json'
            detections.write_text(json.dumps(dict(frame=1,image_sha256='1'*64,detections=[
                dict(detection_index=0,classification='A',role='player')]))+'\n')
            before=sha256(detections)
            write_json(hints,dict(detections_sha256=before,reviewer='synthetic-test',candidates=[
                dict(self.evidence,frame=1,detection_index=0,image_sha256='1'*64)]))
            rows=apply(detections,hints,POLICY,root/'out')
            self.assertEqual(sha256(detections),before)
            self.assertEqual(rows[0]['contextual_hypothesis']['team'],'H')
            self.assertTrue(rows[0]['conflicts_with_visual'])
            self.assertEqual(rows[0]['visual_decision']['classification'],'A')
            spec=read_json(root/'out/contextual-goalkeepers.json')
            self.assertNotIn('evaluation_mode',spec)
            self.assertFalse(spec['visual_decisions_overwritten'])

    def test_context_must_match_image_and_original_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);detections=root/'detections.jsonl';hints=root/'context.json'
            detections.write_text(json.dumps(dict(frame=1,image_sha256='1'*64,detections=[dict(detection_index=0)])))
            write_json(hints,dict(detections_sha256=sha256(detections),reviewer='test',candidates=[
                dict(self.evidence,frame=1,detection_index=0,image_sha256='2'*64)]))
            with self.assertRaisesRegex(ValueError,'image binding'):apply(detections,hints,POLICY,root/'out')
            self.assertFalse((root/'out').exists())
            spec=read_json(hints);spec['detections_sha256']='0'*64;write_json(hints,spec)
            with self.assertRaisesRegex(ValueError,'unchanged'):apply(detections,hints,POLICY,root/'out')


if __name__=='__main__':unittest.main()
