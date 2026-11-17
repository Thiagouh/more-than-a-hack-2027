"""Automatic support cannot become fabricated observations or unbound inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.frente4.event_support import supported_event_teams,load_automatic_bundle
from src.ball_events.artifacts import ARTIFACTS,canonical
from src.ball_events.contract import config
from src.ball_events.demo import metadata,frames
from src.ball_events.detector import Detector


def event(team='H',start=3,confirmed=5,left=2,right=6,**extra):
    return dict(type='possible_control_acquisition',event_frame=start,confirmed_at_frame=confirmed,
        support_start_frame=left,support_end_frame=right,team=team,strength='high',quality_flags=[],**extra)


class EventSupportTests(unittest.TestCase):
    def test_actual_support_is_not_held_for_twelve_frames(self):
        source=[event()];before=copy.deepcopy(source)
        signal,audit=supported_event_teams(source,30)
        self.assertEqual(np.flatnonzero(signal>=0).tolist(),[3,4,5])
        self.assertEqual(audit['unsupported_hold_frames'],0)
        self.assertEqual(source,before)

    def test_removed_velocity_dependency_rejects_the_whole_event(self):
        valid=np.ones(20,bool);valid[2]=False
        signal,audit=supported_event_teams([event()],20,ball_valid=valid)
        self.assertTrue((signal<0).all())
        self.assertEqual(audit['rejected_ball_support'],1)

    def test_conflicts_remain_unknown_even_with_later_duplicate(self):
        source=[event(),event('A',4,6,3,7),event()]
        signal,audit=supported_event_teams(source,20)
        self.assertEqual(signal[3:7].tolist(),[0,-1,-1,1])
        self.assertEqual(audit['conflicting_frames'],2)

    def test_flags_strength_and_invalid_masks_cannot_certify_support(self):
        weak=event();weak['strength']='low'
        flagged=event();flagged['quality_flags']=['ball_jump']
        signal,audit=supported_event_teams([weak,flagged],20)
        self.assertTrue((signal<0).all())
        self.assertEqual(audit['rejected_strength'],1)
        self.assertEqual(audit['rejected_quality_flags'],1)
        with self.assertRaises(ValueError):supported_event_teams([event()],20,ball_valid=np.ones(20))
        with self.assertRaises(ValueError):supported_event_teams([event(confirmed=20,right=21)],20)


def write_bundle(root):
    meta=metadata(12);cfg=config(fps=25);detector=Detector(meta,cfg)
    batch=[]
    for row in frames(12):batch.extend(detector.feed(row)[1])
    detector.finish()
    for name in ARTIFACTS:(root/name).write_text('')
    (root/'events.jsonl').write_text(''.join(canonical(e)+'\n' for e in batch))
    repo=Path(__file__).resolve().parents[1]
    kernel=[*sorted((repo/'src/ball_events').glob('*.py')),repo/'starter/load_atd.py',repo/'games.py']
    inputs=dict(atd_raw='0'*64,atd_meta='1'*64)
    provenance=dict(identity_version='ball-events-run-v2',contract_version='v2',input_metadata=copy.deepcopy(meta),
        configuration=copy.deepcopy(cfg),input_sha256=copy.deepcopy(inputs),hints=[],commit='synthetic-test',
        code_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in kernel})
    manifest=dict(contract_version='v2',provenance=provenance,
        cache_key=hashlib.sha256(canonical(provenance).encode()).hexdigest(),
        artifact_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ARTIFACTS})
    (root/'manifest.json').write_text(canonical(manifest))
    return meta,cfg,inputs,manifest


def rewrite_manifest(root,manifest):
    manifest['cache_key']=hashlib.sha256(canonical(manifest['provenance']).encode()).hexdigest()
    (root/'manifest.json').write_text(canonical(manifest))


class BundleBindingTests(unittest.TestCase):
    def test_hash_bound_synthetic_bundle_loads_and_changed_artifact_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,cfg,inputs,_=write_bundle(root)
            events,binding=load_automatic_bundle(root,meta,inputs,cfg)
            self.assertGreater(len(events),0)
            self.assertFalse(binding['training'])
            with (root/'events.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'hash-verified'):load_automatic_bundle(root,meta,inputs,cfg)

    def test_extra_reference_input_or_wrong_kernel_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,cfg,inputs,manifest=write_bundle(root)
            manifest['provenance']['input_sha256']['ground_truth']='2'*64
            rewrite_manifest(root,manifest)
            with self.assertRaisesRegex(ValueError,'input hashes'):load_automatic_bundle(root,meta,inputs,cfg)
            meta,cfg,inputs,manifest=write_bundle(root)
            manifest['provenance']['code_sha256']['src/ball_events/detector.py']='0'*64
            rewrite_manifest(root,manifest)
            with self.assertRaisesRegex(ValueError,'kernel differs'):load_automatic_bundle(root,meta,inputs,cfg)

    def test_metadata_and_configuration_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);meta,cfg,inputs,manifest=write_bundle(root)
            changed=dict(meta,n_frames=13)
            with self.assertRaisesRegex(ValueError,'metadata differs'):load_automatic_bundle(root,changed,inputs,cfg)
            changed=dict(cfg,relative_speed_mps=4.)
            with self.assertRaisesRegex(ValueError,'configuration differ'):load_automatic_bundle(root,meta,inputs,changed)


if __name__=='__main__':unittest.main()
