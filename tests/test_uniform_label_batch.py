import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from scripts.prepare_uniform_label_batch import prepare, record
from src.team_identification.manifests import write_json, read_json
from src.team_identification.pilot import sha256


class LabelBatchTests(unittest.TestCase):
    def inputs(self, root, frame=9750):
        Image.new('RGB',(40,60),'blue').save(root/'source.png')
        write_json(root/'extraction.json',dict(source_video_sha256='a'*64,frames=[dict(frame=frame,
            image='source.png',image_sha256=sha256(root/'source.png'),source_video_sha256='a'*64)]))
        write_json(root/'proposals.json',dict(examples=[dict(id=12,frame=frame,box=[5,5,25,55])]))

    def test_explicit_user_class_never_approves_mask_or_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root)
            prepare(root/'proposals.json',root/'extraction.json',root/'batch')
            answers=read_json(root/'batch/answers-template.json')
            answers.update(reviewer='user',labels={'12':'Barcelona'})
            write_json(root/'answers.json',answers)
            result=record(root/'batch/batch.json',root/'answers.json',root/'labeled')
            row=read_json(result)['examples'][0]
            self.assertEqual((row['team'],row['role']),('H','player'))
            self.assertFalse(row['mask_approved'])
            self.assertFalse(row['geometry_approved'])

    def test_reserved_evaluation_frames_cannot_become_training_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root,16000)
            with self.assertRaises(ValueError):prepare(root/'proposals.json',root/'extraction.json',root/'batch')
            self.assertFalse((root/'batch').exists())

    def test_gameplay_requires_explicit_development_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root,12000)
            with self.assertRaises(ValueError):prepare(root/'proposals.json',root/'extraction.json',root/'invalid')
            spec=read_json(root/'proposals.json');spec['sampling_pool']='development_gameplay'
            write_json(root/'proposals.json',spec)
            prepare(root/'proposals.json',root/'extraction.json',root/'batch')
            self.assertEqual(read_json(root/'batch/batch.json')['sampling_pool'],'development_gameplay')

    def test_gameplay_pool_still_rejects_held_out_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root,16000)
            spec=read_json(root/'proposals.json');spec['sampling_pool']='development_gameplay'
            write_json(root/'proposals.json',spec)
            with self.assertRaises(ValueError):prepare(root/'proposals.json',root/'extraction.json',root/'batch')

    def test_additional_pool_accepts_second_half_and_excludes_evaluation(self):
        for frame,allowed in [(120000,True),(125000,True),(25000,True),(16000,False),(17000,False),(126000,False)]:
            with self.subTest(frame=frame), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);self.inputs(root,frame)
                spec=read_json(root/'proposals.json');spec['sampling_pool']='additional_gameplay'
                write_json(root/'proposals.json',spec)
                if allowed:
                    prepare(root/'proposals.json',root/'extraction.json',root/'batch')
                else:
                    with self.assertRaises(ValueError):prepare(root/'proposals.json',root/'extraction.json',root/'batch')

    def test_changed_crop_and_incomplete_answers_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.inputs(root)
            prepare(root/'proposals.json',root/'extraction.json',root/'batch')
            answers=read_json(root/'batch/answers-template.json');answers['reviewer']='user'
            write_json(root/'answers.json',answers)
            with self.assertRaises(ValueError):record(root/'batch/batch.json',root/'answers.json',root/'labeled')
            answers['labels']={'12':'Incerto'};write_json(root/'answers.json',answers)
            Image.new('RGB',(20,50),'red').save(root/'batch/crops/012.png')
            with self.assertRaises(ValueError):record(root/'batch/batch.json',root/'answers.json',root/'labeled')


if __name__=='__main__':unittest.main()
