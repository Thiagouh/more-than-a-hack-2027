"""Physics and provenance contracts, using only synthetic trajectories."""
import unittest
import numpy as np
from src.frente4.ball_quality import clean_ball_observations
from src.frente4.deadball import dead_ball_evidence,REASON_LOST_OUTSIDE
from src.frente4.features import speed_from_positions


def tracking(xy,fps=25):
    return dict(ball_xy=np.asarray(xy,dtype=np.float32),n_frames=len(xy),fps=fps,pitch=(105.,68.))


class BallQualityTests(unittest.TestCase):
    def test_plausible_fast_shot_is_retained_exactly(self):
        raw=tracking([[20+1.4*i,30] for i in range(8)])
        result,starts,summary=clean_ball_observations(raw)
        np.testing.assert_array_equal(result['ball_xy'],raw['ball_xy'])
        self.assertEqual(summary['removed_observations'],0)
        self.assertFalse(starts.any())

    def test_impossible_jump_is_removed_without_mutating_or_filling(self):
        raw=tracking([[50,30]]*12);raw['ball_xy'][5,0]=80
        before=raw['ball_xy'].copy()
        result,starts,summary=clean_ball_observations(raw)
        np.testing.assert_array_equal(raw['ball_xy'],before)
        self.assertTrue(np.isnan(result['ball_xy'][2:9]).all())
        kept=np.isfinite(result['ball_xy']).all(axis=1)
        np.testing.assert_array_equal(result['ball_xy'][kept],before[kept])
        self.assertEqual(summary['removed_observations'],7)
        self.assertEqual(summary['interpolated_observations'],0)
        self.assertFalse(starts.any())

    def test_inside_gap_does_not_imply_dead_ball(self):
        raw=tracking([[50,30]]*12);raw['ball_xy'][4:8]=np.nan
        result,starts,summary=clean_ball_observations(raw)
        self.assertTrue(np.isnan(result['ball_xy'][4:8]).all())
        self.assertFalse(starts.any())
        self.assertEqual(summary['kept_observations'],8)

    def test_outside_gap_uses_existing_boundary_margin(self):
        raw=tracking([[104,30],[104.5,30],[105.6,30],[np.nan,np.nan],[np.nan,np.nan]])
        _,starts,_=clean_ball_observations(raw)
        self.assertEqual(np.flatnonzero(starts).tolist(),[3])
        raw['ball_xy'][2,0]=105.2
        _,starts,_=clean_ball_observations(raw)
        self.assertFalse(starts.any())

    def test_extreme_outside_and_partial_coordinates_cannot_trigger(self):
        raw=tracking([[999,30],[np.nan,np.nan],[50,np.nan],[50,30]])
        result,starts,summary=clean_ball_observations(raw)
        self.assertTrue(np.isnan(result['ball_xy'][:3]).all())
        self.assertFalse(starts.any())
        self.assertEqual(summary['removed_extreme_outside_observations'],1)
        self.assertEqual(summary['partial_coordinate_frames'],1)
        second,_,_=clean_ball_observations(result)
        np.testing.assert_array_equal(second['ball_xy'],result['ball_xy'])

    def test_short_and_empty_inputs_keep_their_original_length(self):
        for n in range(5):
            raw=tracking(np.full((n,2),30,dtype=np.float32))
            result,starts,_=clean_ball_observations(raw)
            self.assertEqual(len(result['ball_xy']),n)
            self.assertEqual(len(starts),n)

    def test_fps_controls_physical_speed_and_bad_alignment_is_rejected(self):
        raw=tracking([[20,30],[21,30],[22,30]],fps=25)
        result,_,_=clean_ball_observations(raw)
        self.assertTrue(np.isfinite(result['ball_xy']).all())
        raw['fps']=50
        result,_,_=clean_ball_observations(raw)
        self.assertTrue(np.isnan(result['ball_xy']).all())
        raw['n_frames']=4
        with self.assertRaises(ValueError):clean_ball_observations(raw)

    def test_lost_outside_opens_d_until_observed_moving_restart(self):
        xy=np.full((80,2),np.nan,dtype=np.float32)
        xy[:3]=[[104,30],[104.5,30],[105.6,30]]
        xy[40:]=[[50+.2*i,30] for i in range(40)]
        raw=tracking(xy);cleaned,starts,_=clean_ball_observations(raw)
        ball=cleaned['ball_xy']
        feat=dict(ball_observed=np.isfinite(ball).all(axis=1),ball_x=ball[:,0],
                  ball_y=ball[:,1],ball_speed=speed_from_positions(ball,25))
        cfg=dict(use_stillness=False,use_set_piece_tags=False)
        baseline=dead_ball_evidence(feat,[(0,79)],{1:0},raw['pitch'],cfg=cfg)
        self.assertFalse(baseline['dead'].any())
        result=dead_ball_evidence(feat,[(0,79)],{1:0},raw['pitch'],cfg=cfg,lost_outside_starts=starts)
        self.assertTrue(result['dead'][3:41].all())
        self.assertFalse(result['dead'][41:].any())
        self.assertEqual(result['reason'][3],REASON_LOST_OUTSIDE)


if __name__=='__main__':unittest.main()
