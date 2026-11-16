from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.team_identification.supervised_uniforms import SupervisedUniforms, fit
from scripts.test_supervised_uniforms import score
from src.team_identification.manifests import write_json


def example(name,label,color,split='reference'):
    rgb=np.zeros((40,20,3),dtype=np.uint8);rgb[:]=color
    return dict(example_id=name,label=label,split=split,reviewed=True,reviewer='synthetic test',rgb=rgb)


class SupervisedUniformTests(unittest.TestCase):
    def test_learns_explicit_visual_classes_without_using_position(self):
        a=example('h','Barcelona',[0,0,255]);b=example('a','Athletic',[255,0,0])
        model=SupervisedUniforms(fit([a,b]))
        self.assertEqual(model.predict(a['rgb'])['label'],'Barcelona')
        self.assertEqual(model.predict(b['rgb'])['label'],'Athletic')

    def test_test_labels_and_unreviewed_examples_cannot_fit(self):
        for change in [dict(split='test'),dict(reviewed=False)]:
            a=example('h','Barcelona',[0,0,255]);a.update(change)
            with self.assertRaises(ValueError):fit([a,example('a','Athletic',[255,0,0])])

    def test_duplicate_training_ids_rejected(self):
        with self.assertRaises(ValueError):fit([example('same','Barcelona',[0,0,255]),example('same','Athletic',[255,0,0])])

    def test_ambiguous_color_abstains_instead_of_forcing_a_team(self):
        model=SupervisedUniforms(fit([example('h','Barcelona',[0,0,255]),example('a','Athletic',[0,0,255])]))
        self.assertEqual(model.predict(np.full((40,20,3),[0,0,255],dtype=np.uint8))['label'],'Incerto')

    def test_small_crop_abstains_and_other_is_not_a_team(self):
        model=SupervisedUniforms(fit([example('h','Barcelona',[0,0,255]),example('other','Outro',[255,0,0])]))
        self.assertEqual(model.predict(np.zeros((10,5,3),dtype=np.uint8))['label'],'Incerto')
        result=model.predict(np.full((40,20,3),[255,0,0],dtype=np.uint8))
        self.assertIsNone(result['team']);self.assertEqual(result['role'],'other')

    def test_scoring_preserves_abstention_and_requires_committed_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            predictions=dict(status='committed_before_user_labels',batch_sha256='a'*64,model_sha256='b'*64,
                committed_at_utc='2026-01-01T00:00:00+00:00',predictions=[dict(id=i,label=l,group_id='group-1',crop_sha256=str(i))
                    for i,l in [(1,'Barcelona'),(2,'Incerto'),(3,'Athletic')]])
            labeled=dict(status='user_labeled',original_batch_sha256='a'*64,reviewed_at_utc='2026-01-02T00:00:00+00:00',
                examples=[dict(id=i,label=l,split='test',reviewed=True,crop_sha256=str(i))
                    for i,l in [(1,'Barcelona'),(2,'Barcelona'),(3,'Athletic')]])
            write_json(root/'predictions.json',predictions);write_json(root/'labels.json',labeled)
            result=score(root/'predictions.json',root/'labels.json',root/'report')
            self.assertAlmostEqual(result['metrics']['accuracy'],2/3)
            self.assertAlmostEqual(result['metrics']['coverage'],2/3)
            self.assertEqual(result['metrics']['accuracy_accepted'],1.)
            labeled['reviewed_at_utc']='2025-01-01T00:00:00+00:00';write_json(root/'labels.json',labeled)
            with self.assertRaises(ValueError):score(root/'predictions.json',root/'labels.json',root/'invalid')


if __name__=='__main__':unittest.main()
