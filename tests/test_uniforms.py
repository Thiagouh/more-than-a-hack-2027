"""Synthetic colours only; no match assets or ground truth."""
import unittest
import numpy as np
from src.ball_events.uniforms import UniformClassifier, histogram, resolve_player
from src.ball_events.contract import InputError
from src.ball_events import detect
from src.ball_events.demo import metadata, point
from src.ball_events.events import validate_events


def crop(color):
    return np.full((100, 40, 3), color, dtype=np.uint8)


def classifier():
    return UniformClassifier('synthetic', [dict(team=t, role=r, crops=[crop(c)]) for t,r,c in [
        ('H','player',(220,20,20)), ('A','player',(20,20,220)),
        ('H','goalkeeper',(20,220,20)), ('A','goalkeeper',(220,220,20)),
        (None,'referee',(15,15,15))]])


class UniformTests(unittest.TestCase):
    def test_both_goalkeepers_and_referee(self):
        for color, team, role in [((20,220,20),'H','goalkeeper'), ((220,220,20),'A','goalkeeper'), ((15,15,15),None,'referee')]:
            model=classifier()
            for f in range(3):
                result=model.classify(crop(color),'track',f,12 if f==0 else f)
            self.assertEqual((result['team'],result['role']),(team,role))
            self.assertEqual(result['available_at_frame'],12)
            p=point(2,20,track_id='track')
            resolved, diagnostic=resolve_player(p,result)
            if team:
                self.assertEqual(resolved['team'],team)
                self.assertEqual(resolved['support_start_frame'],0)
                self.assertEqual(resolved['available_at_frame'],12)
                self.assertFalse(diagnostic['excluded'])
            else:
                self.assertIsNone(resolved)

    def test_metadata_precedence(self):
        model=classifier()
        for f in range(3): result=model.classify(crop((220,220,20)),'g',f)
        p=point(2,20,track_id='g',team='H')
        resolved,diagnostic=resolve_player(p,result)
        self.assertEqual(resolved['team'],'H')
        self.assertTrue(diagnostic['visual_conflict'])
        self.assertEqual(resolved.get('support_start_frame',2),2)

    def test_ambiguity_templates_and_unknown(self):
        green=crop((20,220,20))
        model=UniformClassifier('s',[dict(team=t,role='goalkeeper',crops=[green]) for t in ('H','A')])
        for f in range(6): result=model.classify(green,'x',f)
        self.assertEqual(result['role'],'unknown')
        model=classifier()
        model.patterns.append(model.patterns[0])
        for f in range(3): result=model.classify(crop((220,20,20)),'x',f)
        self.assertEqual(result['team'],'H')
        self.assertEqual(model.classify(np.zeros((5,5,3)),'x',3)['role'],'unknown')
        self.assertEqual(model.classify(crop((220,20,20)),'x',20)['role'],'unknown')
        with self.assertRaises(InputError): model.classify(green,'x',20)
        with self.assertRaises(InputError): histogram(np.full((30,20,3),np.nan))
        model.reset()
        self.assertEqual(model.history,{})

    def test_stripes_luminosity_and_brightness(self):
        striped=crop((220,20,20)); striped[:,::2]=[20,20,220]
        self.assertAlmostEqual(float(histogram(striped).sum()),1.)
        model=UniformClassifier('s',[dict(team='H',role='player',crops=[striped]),
                                    dict(team='A',role='player',crops=[crop((240,240,240))])])
        for f in range(3): result=model.classify((striped*.95).astype('uint8'),'x',f)
        self.assertEqual(result['team'],'H')


class TeamControlTests(unittest.TestCase):
    def frames(self):
        frames=[]
        for f in range(10):
            players=[point(f,20 if f<3 else 22,track_id='h1',team='H'),
                     point(f,22 if f<3 else 20,track_id='h2',team='H'),
                     point(f,26,track_id='a',team='A')]
            frames.append(dict(frame=f,ball=point(f,20),players=players))
        return frames

    def test_team_handoff_preserves_evidence_and_availability(self):
        frames=self.frames()
        # Handoff movement must be physically compatible, including its first endpoint.
        for f in frames:
            i=f['frame']; f['players'][0]['x_m']=20 if i<3 else 20.1
            f['players'][1]['x_m']=20.1 if i<3 else 20
        frames[2]['players'][0]['available_at_frame']=18
        meta=metadata(10)
        _,events,_=detect(meta,frames, {'duplicate_m': .05})
        acquisitions=[e for e in events if e['type']=='possible_control_acquisition']
        self.assertEqual(len(acquisitions),1)
        self.assertEqual(acquisitions[0]['track_ids'],['synthetic:h1','synthetic:h2'])
        self.assertEqual(acquisitions[0]['available_at_frame'],18)
        validate_events(events,meta)

    def test_replacement_no_loss_then_team_separation(self):
        frames=self.frames()
        for f in frames:
            i=f['frame']; f['players']= [point(i,20 if i<6 else 24,track_id='old' if i<5 else 'new',team='H'), point(i,30,track_id='a',team='A')]
        _,events,_=detect(metadata(10),frames)
        losses=[e for e in events if e['type']=='possible_control_loss']
        self.assertEqual(len(losses),1)
        self.assertGreater(losses[0]['event_frame'],5)
        self.assertIsNotNone(losses[0]['parent_event_id'])
        validate_events(events, metadata(10))


    def test_missing_coverage_never_manufactures_team_loss(self):
        frames=self.frames()
        for f in frames:
            i=f['frame']
            f['players']=[point(i,20,track_id='h',team='H'), point(i,26,track_id='a',team='A')]
            if i>=5:
                f['players']=f['players'][1:]
        _,events,_=detect(metadata(10),frames)
        self.assertEqual(sum(e['type']=='possible_control_acquisition' for e in events),1)
        self.assertFalse(any(e['type']=='possible_control_loss' for e in events))

    def test_goalkeepers_resolve_before_control(self):
        for color,team in [((20,220,20),'H'),((220,220,20),'A')]:
            model=classifier(); frames=[]
            for f in range(9):
                result=model.classify(crop(color),'keeper',f)
                p,_=resolve_player(point(f,20,track_id='keeper'),result)
                players=[point(f,26,track_id='opponent',team='A' if team=='H' else 'H')]
                if p: players.append(p)
                frames.append(dict(frame=f,ball=point(f,20),players=players))
            _,events,_=detect(metadata(9),frames)
            self.assertEqual([e['team'] for e in events if e['type']=='possible_control_acquisition'],[team])
