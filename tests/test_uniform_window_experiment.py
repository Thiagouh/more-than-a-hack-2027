"""Integration guards around immutable held-out artifacts."""
import tempfile
import unittest
from pathlib import Path
from src.team_identification.manifests import write_json,read_json
from scripts.compare_uniform_windows import history,score
from src.team_identification.temporal_windows import CONFIGURATIONS

class ExperimentGuards(unittest.TestCase):
    def test_history_includes_promoted_and_sequence_interior(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'uniform-old').mkdir()
            write_json(root/'uniform-old'/'model.json',dict(examples=[dict(frame=123,source_video_sha256='s',original_split='test')],sequences=[dict(frames=[dict(frame=124,source_video_sha256='s')])]))
            frames,files=history(root,'s')
            self.assertEqual(frames,[123,124]);self.assertEqual(len(files),1)
            self.assertEqual(history(root,'other')[0],[])
    def test_score_once_per_sequence_and_labels_after_commit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pred=root/'pred.json';answers=root/'answers.json'
            rows=[dict(id=1,label='Barcelona',reason='temporal_consensus'),dict(id=2,label='Incerto',reason='identity_unverified'),dict(id=3,label='Athletic',reason='temporal_consensus')]
            write_json(pred,dict(batch_sha256='bound',committed_at_utc='2026-11-16T12:00:00+00:00',
                predictions={name:rows for name in CONFIGURATIONS},temporal_groups=1,warmup_s={name:-v[0]/25 for name,v in CONFIGURATIONS.items()}))
            data=dict(batch_sha256='bound',reviewer='user',reviewed_at_utc='2026-11-16T12:00:01+00:00',labels={'1':'Barcelona','2':'Athletic','3':'Barcelona'})
            write_json(answers,data);score(pred,answers,root/'report')
            report=read_json(root/'report'/'comparison.json')
            for result in report['results'].values():
                self.assertEqual((result['correct'],result['abstentions'],result['wrong']),(1,1,1))
                self.assertEqual(result['coverage'],2/3);self.assertEqual(result['accepted_accuracy'],.5)
            data['excluded_cases']={'3':'User: out of scope'};write_json(answers,data)
            score(pred,answers,root/'excluded-report')
            excluded=read_json(root/'excluded-report'/'comparison.json')
            for result in excluded['results'].values():
                self.assertEqual((result['correct'],result['abstentions'],result['wrong'],result['excluded']),(1,1,0,1))
                self.assertEqual(result['coverage'],.5);self.assertEqual(result['accepted_accuracy'],1.)
                self.assertEqual(result['by_category']['Barcelona']['excluded'],1)
            self.assertTrue(excluded['exclusions_selected_after_labels'])
            partial=dict(data,labels={'1':'Barcelona','3':'Barcelona'},pending_cases=[2])
            write_json(answers,partial);score(pred,answers,root/'partial-report')
            for result in read_json(root/'partial-report'/'comparison.json')['results'].values():
                self.assertEqual((result['evaluated'],result['pending'],result['excluded']),(1,1,1))
                self.assertEqual(result['coverage'],1.)
            unlabeled_exclusion=dict(data,labels={'1':'Barcelona','2':'Athletic'})
            write_json(answers,unlabeled_exclusion);score(pred,answers,root/'unlabeled-exclusion')
            complete=read_json(root/'unlabeled-exclusion'/'comparison.json')
            self.assertEqual(complete['status'],'complete')
            for result in complete['results'].values():
                self.assertEqual((result['evaluated'],result['excluded'],result['pending']),(2,1,0))
                self.assertIsNone(result['cases'][2]['truth'])
            data['excluded_cases']={'999':'unknown case'};write_json(answers,data)
            with self.assertRaises(ValueError):score(pred,answers,root/'invalid-exclusion')
            data.pop('excluded_cases')
            data['reviewed_at_utc']='2026-11-16T11:00:00+00:00';write_json(answers,data)
            with self.assertRaises(ValueError):score(pred,answers,root/'invalid')
            data['batch_sha256']='other';write_json(answers,data)
            with self.assertRaises(ValueError):score(pred,answers,root/'invalid')

    def test_review_omits_ambiguous_proposals(self):
        from scripts.compare_uniform_windows import review_batch
        from src.team_identification.pilot import sha256
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);crop=root/'crop.png';Image.new('RGB',(20,50),'red').save(crop)
            batch=root/'batch.json';review=root/'review.json'
            rows=[dict(frame=f,crop='crop.png',crop_sha256=sha256(crop),box=[0,0,20,50],association_verified=False) for f in range(51)]
            write_json(batch,dict(sequences=[dict(id=1,decision_frame=50,frames=rows)]))
            write_json(review,dict(batch_sha256=sha256(batch),reviewer='identity only',cases={'1':dict(verified_frames=list(range(25)),ambiguous_frames=list(range(25,51)))}))
            review_batch(batch,review,root/'ready')
            sanitized=read_json(root/'ready'/'batch.json')['sequences'][0]['frames']
            self.assertIsNotNone(sanitized[24]['crop'])
            self.assertTrue(sanitized[24]['association_verified'])
            self.assertIsNone(sanitized[25]['crop']);self.assertIsNone(sanitized[25]['box'])
            self.assertFalse(sanitized[25]['association_verified'])
            self.assertIsNotNone(sanitized[25]['proposed_crop'])
            self.assertEqual(read_json(root/'ready'/'identity-reviewed.json')['batch_sha256'],sha256(root/'ready'/'batch.json'))

if __name__=='__main__':unittest.main()
