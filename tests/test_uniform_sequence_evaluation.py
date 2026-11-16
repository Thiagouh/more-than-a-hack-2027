from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scripts.prepare_uniform_sequences import prepare,record
from scripts.test_uniform_sequences import predict,score
from src.team_identification.manifests import read_json,write_json
from src.team_identification.pilot import sha256
from src.team_identification.supervised_uniforms import fit


class SequenceEvaluationTests(unittest.TestCase):
    def fixture(self,root):
        entries=[]
        for f in range(50000,50004):
            image=Image.new('RGB',(60,90),'green');image.paste('blue',(15,10,35,70))
            path=root/f'{f}.png';image.save(path)
            entries.append(dict(frame=f,image=path.name,image_sha256=sha256(path),source_video_sha256='a'*64))
        write_json(root/'extraction.json',dict(source_video_sha256='a'*64,frames=entries))
        write_json(root/'proposals.json',dict(split='test',sequences=[dict(id=136,start_frame=50000,box=[12,7,38,74])]))
        prepare(root/'proposals.json',root/'extraction.json',root/'batch')
        path=root/'batch/batch.json';batch=read_json(path)
        batch['sequences'][0].update(association_reviewed=True,association_reviewer='synthetic continuity review')
        write_json(path,batch)
        references=[dict(example_id=name,label=label,split='reference',reviewed=True,reviewer='synthetic',
            frame=13000,source_video_sha256='a'*64,rgb=np.full((40,20,3),color,dtype=np.uint8))
            for name,label,color in [('h','Barcelona',[0,0,255]),('a','Athletic',[255,0,0])]]
        write_json(root/'model.json',fit(references))
        return path

    def test_scores_one_sequence_not_four_samples_and_commits_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=self.fixture(root)
            predict(root/'model.json',batch,root/'predictions')
            p=read_json(root/'predictions/predictions.json')
            self.assertEqual(p['predictions'][0]['frame_ids'],list(range(50000,50004)))
            self.assertEqual([r['label'] for r in p['predictions'][0]['per_frame'][:3]],['Incerto']*3)
            answers=dict(batch_sha256=sha256(batch),reviewer='user',labels={'136':'Barcelona'})
            write_json(root/'answers.json',answers)
            labeled=record(batch,root/'answers.json',root/'labels')
            result=score(root/'predictions/predictions.json',labeled,root/'report')
            self.assertEqual(result['metrics']['examples'],1)
            self.assertEqual(result['metrics']['correct'],1)
            self.assertTrue(read_json(root/'report/comparison.json')['same_cases'])
            self.assertEqual(read_json(labeled)['examples'][0]['split'],'test')

    def test_test_label_cannot_be_supplied_before_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=self.fixture(root);spec=read_json(batch)
            spec['sequences'][0]['label']='Barcelona';write_json(batch,spec)
            with self.assertRaises(ValueError):predict(root/'model.json',batch,root/'invalid')
            self.assertFalse((root/'invalid').exists())

    def test_near_training_frame_is_rejected_even_when_test_pool_allows_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);batch=self.fixture(root);model=read_json(root/'model.json')
            model['examples'][0]['frame']=49999;write_json(root/'model.json',model)
            with self.assertRaises(ValueError):predict(root/'model.json',batch,root/'invalid')


if __name__=='__main__':unittest.main()
