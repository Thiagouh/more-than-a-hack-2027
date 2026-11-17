"""Synthetic source contracts for the external audit, without network access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.evaluate_ball_quality_external import load_external,inject_wrong_object_faults,SOURCE_COMMIT


def write_source(root,groups=('home team','away team'),duplicate=False):
    match=dict(id=1874553,home_team=dict(id=1),away_team=dict(id=2),
               pitch_length=104,pitch_width=68,
               players=[dict(id=10,team_id=1),dict(id=20,team_id=2)])
    rows=[]
    for i,group in enumerate(groups):
        rows.append(dict(frame=0 if duplicate else i*2,period=1,
            ball_data=dict(x=0,y=0,is_detected=i==0),possession=dict(group=group),
            player_data=[dict(player_id=10,x=-1,y=0),dict(player_id=20,x=1,y=0)]))
    metadata=json.dumps(match).encode();tracking=('\n'.join(map(json.dumps,rows))+'\n').encode()
    (root/'match.json').write_bytes(metadata);(root/'tracking.jsonl').write_bytes(tracking)
    (root/'source.json').write_text(json.dumps(dict(match_id='1874553',commit=SOURCE_COMMIT,
        metadata_sha256=hashlib.sha256(metadata).hexdigest(),
        tracking_sha256=hashlib.sha256(tracking).hexdigest())))


class ExternalSourceTests(unittest.TestCase):
    def test_faults_are_reproducible_and_do_not_mutate_original_inputs(self):
        n=3000;ball=np.tile([40.,30.],(n,1)).astype(np.float32)
        players=np.stack([ball,np.tile([80.,30.],(n,1))],axis=1)
        tr=dict(ball_xy=ball,players_xy=players,n_frames=n,fps=10.,pitch=(105,68),team_of_slot=np.array([0,1]))
        truth=np.zeros(n,np.int8);period=np.ones(n,bool)
        a,mask,summary=inject_wrong_object_faults(tr,truth,period,1874553)
        b,other,second=inject_wrong_object_faults(tr,truth,period,1874553)
        self.assertEqual(summary,second)
        self.assertEqual(summary['created_windows'],20)
        np.testing.assert_array_equal(a['ball_xy'],b['ball_xy'])
        np.testing.assert_array_equal(mask,other)
        np.testing.assert_array_equal(tr['ball_xy'],ball)
        self.assertTrue((a['ball_xy'][mask,0]==80).all())
        self.assertTrue((a['ball_xy'][~mask,0]==40).all())
        self.assertEqual(int(mask.sum()),375)

    def test_coordinates_missing_frames_and_observation_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_source(root)
            tr,truth,period,_,_=load_external(root)
            self.assertEqual(tr['n_frames'],3)
            np.testing.assert_array_equal(tr['ball_xy'][0],[52,34])
            self.assertTrue(np.isnan(tr['ball_xy'][1:]).all())
            np.testing.assert_array_equal(truth,[0,-1,1])
            np.testing.assert_array_equal(period,[True,False,True])
            self.assertEqual(tr['fps'],10)

    def test_reference_labels_do_not_enter_tracking_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_source(root)
            first,truth,_,_,_=load_external(root)
            write_source(root,groups=('away team','home team'))
            second,other,_,_,_=load_external(root)
            self.assertFalse(np.array_equal(truth,other))
            self.assertEqual(set(first),set(second))
            self.assertNotIn('possession',first)
            for key in first:np.testing.assert_array_equal(first[key],second[key])

    def test_changed_source_and_duplicate_frames_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_source(root)
            with (root/'tracking.jsonl').open('a') as stream:stream.write('\n')
            with self.assertRaisesRegex(ValueError,'source changed'):load_external(root)
            write_source(root,duplicate=True)
            with self.assertRaisesRegex(ValueError,'unique and increasing'):load_external(root)


if __name__=='__main__':unittest.main()
