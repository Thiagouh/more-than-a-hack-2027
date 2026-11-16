from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from src.ball_events.demo import metadata,point
from src.team_identification.four_frames import FourFrameUniforms
from src.team_identification.supervised_uniforms import SupervisedUniforms,fit
from src.team_identification.pipeline import IdentityPipeline
from src.team_possession.collective import CollectiveModel
from src.team_identification.manifests import write_json,read_json
from src.team_identification.pilot import sha256
from scripts.prepare_uniform_sequences import prepare,record


class Scores:
    max_distance=.35
    min_margin=.10
    def predict(self,value):
        if value=='future':raise AssertionError('Future pixels must not be read')
        if value=='missing':return dict(label='Incerto')
        distances={'Barcelona':.1,'Athletic':.6} if value=='H' else {'Barcelona':.6,'Athletic':.1}
        return dict(label='Barcelona' if value=='H' else 'Athletic',class_distances=distances)


class FourFrameTests(unittest.TestCase):
    def test_waits_for_four_consecutive_frames_then_uses_rolling_window(self):
        stream=FourFrameUniforms(Scores())
        results=[stream.feed('same',f,'H') for f in range(5)]
        self.assertEqual([r['label'] for r in results[:3]],['Incerto']*3)
        self.assertEqual(results[3]['label'],'Barcelona')
        self.assertEqual(results[4]['frame_ids'],[1,2,3,4])
        self.assertEqual(results[4]['available_at_frame'],4)

    def test_one_bad_frame_does_not_outvote_three_good_frames(self):
        stream=FourFrameUniforms(Scores())
        for f,value in enumerate(['H','A','H','H']):result=stream.feed('same',f,value)
        self.assertEqual(result['label'],'Barcelona')
        self.assertEqual(result['supporting_frames'],3)

    def test_conflicting_teams_and_insufficient_images_abstain(self):
        for values in [['H','A','H','A'],['H','missing','missing','H']]:
            stream=FourFrameUniforms(Scores())
            for f,value in enumerate(values):result=stream.feed('same',f,value)
            self.assertEqual(result['label'],'Incerto')

    def test_identities_gaps_and_uncertain_association_never_mix(self):
        stream=FourFrameUniforms(Scores())
        stream.feed('a',0,'H');stream.feed('a',1,'H')
        stream.feed('b',2,'A');r=stream.feed('b',3,'A')
        self.assertEqual(r['frames_observed'],2)
        self.assertEqual(stream.feed('a',4,'H')['frames_observed'],1)
        self.assertEqual(stream.feed('a',5,'H',association_verified=False)['label'],'Incerto')
        self.assertEqual(stream.feed('a',6,'H')['frames_observed'],1)

    def test_future_crop_cannot_affect_current_decision(self):
        stream=FourFrameUniforms(Scores())
        for f,value in enumerate(['H','H','H','future']):
            result=stream.feed('a',f,value,available_at_frame=f+1 if value=='future' else f)
        self.assertEqual(result['label'],'Barcelona')
        self.assertEqual(result['valid_frames'],3)

    def test_duplicate_and_backward_frames_rejected(self):
        stream=FourFrameUniforms(Scores());stream.feed('a',3,'H')
        for f in [3,2]:
            with self.assertRaises(ValueError):stream.feed('a',f,'H')

    def test_pipeline_and_possession_share_four_frame_result(self):
        def reference(ident,label,color):
            return dict(example_id=ident,label=label,split='reference',reviewed=True,reviewer='synthetic',
                        rgb=np.full((50,20,3),color,dtype=np.uint8))
        model=SupervisedUniforms(fit([reference('h','Barcelona',[0,0,255]),reference('a','Athletic',[255,0,0])]))
        meta=metadata(10);pipeline=IdentityPipeline(meta,model,uniform_policy='supervised-four-consecutive-frames-v1')
        image=Image.new('RGB',(150,150),'blue');mask=np.zeros((150,150),bool);mask[20:70,40:55]=True
        h=np.array([[105.3,0.,27.],[0.,68.,49.],[0.,0.,1.]])
        rows=[];raw=[]
        for f in range(4):
            envelope=dict(frame=f,ball=point(f,20.),players=[point(f,20.,track_id='slot',team='A')]);raw.append(envelope)
            rows.append(pipeline.step(envelope,image,[mask],dict(valid=True,matrix=h,reprojection_error=1.)))
        self.assertEqual([r['players'][0]['team'] for r in rows],[None,None,None,'H'])
        collective=CollectiveModel(meta,team_identity={r['frame']:r for r in rows})
        for envelope in raw:state,*_=collective.feed(envelope)
        self.assertEqual(state['team_identity_players'][0]['team'],'H')
        self.assertEqual(rows[-1]['players'][0]['four_frame_uniforms']['frame_ids'],[0,1,2,3])

    def test_default_pipeline_uses_spaced_policy_for_possession(self):
        def reference(ident,label,color):
            return dict(example_id=ident,label=label,split='reference',reviewed=True,reviewer='synthetic',
                        rgb=np.full((50,20,3),color,dtype=np.uint8))
        model=SupervisedUniforms(fit([reference('h','Barcelona',[0,0,255]),reference('a','Athletic',[255,0,0])]))
        meta=metadata(30);pipeline=IdentityPipeline(meta,model)
        image=Image.new('RGB',(150,150),'blue');mask=np.zeros((150,150),bool);mask[20:70,40:55]=True
        h=np.array([[105.3,0.,27.],[0.,68.,49.],[0.,0.,1.]])
        rows=[];raw=[]
        for f in range(16):
            envelope=dict(frame=f,ball=point(f,20.),players=[point(f,20.,track_id='slot',team='A')]);raw.append(envelope)
            rows.append(pipeline.step(envelope,image,[mask],dict(valid=True,matrix=h,reprojection_error=1.)))
        self.assertEqual([r['players'][0]['team'] for r in rows],[None]*15+['H'])
        collective=CollectiveModel(meta,team_identity={r['frame']:r for r in rows})
        for envelope in raw:state,*_=collective.feed(envelope)
        self.assertEqual(state['team_identity_players'][0]['team'],'H')
        self.assertEqual(rows[-1]['players'][0]['four_frame_uniforms']['frame_ids'],[0,5,10,15])


class SequenceLabelTests(unittest.TestCase):
    def test_four_crop_label_propagation_requires_association_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);entries=[]
            for f in range(13010,13014):
                im=Image.new('RGB',(60,90),'green');im.paste('blue',(15,10,35,70));path=root/f'{f}.png';im.save(path)
                entries.append(dict(frame=f,image=path.name,image_sha256=sha256(path),source_video_sha256='a'*64))
            write_json(root/'extraction.json',dict(source_video_sha256='a'*64,frames=entries))
            write_json(root/'proposals.json',dict(sequences=[dict(id=120,start_frame=13010,box=[12,7,38,74])]))
            prepare(root/'proposals.json',root/'extraction.json',root/'batch')
            batch=root/'batch/batch.json';spec=read_json(batch)
            answers=read_json(root/'batch/answers-template.json');answers.update(reviewer='user',labels={'120':'Barcelona'})
            write_json(root/'answers.json',answers)
            with self.assertRaises(ValueError):record(batch,root/'answers.json',root/'invalid')
            spec['sequences'][0].update(association_reviewed=True,association_reviewer='synthetic continuity review')
            write_json(batch,spec);answers['batch_sha256']=sha256(batch);write_json(root/'answers.json',answers)
            labeled=read_json(record(batch,root/'answers.json',root/'labeled'))
            self.assertEqual(labeled['independent_sequences'],1)
            self.assertEqual(labeled['crop_count'],4)
            self.assertTrue(all(r['label']=='Barcelona' and not r['geometry_approved'] for r in labeled['examples']))


if __name__=='__main__':unittest.main()
