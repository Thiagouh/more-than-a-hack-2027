"""Quality gates must remain identical when computations use tight person boxes."""
import unittest
import numpy as np
from src.team_identification.pilot import torso_evidence,mask_flags,binary_mask


class SpatialOptimizationTests(unittest.TestCase):
    def test_torso_is_invariant_to_empty_padding_and_returns_full_size_mask(self):
        rng=np.random.default_rng(17)
        for index in range(20):
            h,w=int(rng.integers(40,100)),int(rng.integers(10,30))
            image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
            mask=rng.random((h,w))>.08;mask[0,0]=mask[-1,-1]=True
            excluded=rng.random((h,w))>.95
            expected=torso_evidence(image,mask,excluded,return_mask=True)
            padded=np.zeros((h+31,w+49,3),dtype=np.uint8)
            padded[13:13+h,19:19+w]=image
            bigmask=np.zeros(padded.shape[:2],bool);bigmask[13:13+h,19:19+w]=mask
            bigexcluded=np.zeros_like(bigmask);bigexcluded[13:13+h,19:19+w]=excluded
            actual=torso_evidence(padded,bigmask,bigexcluded,return_mask=True)
            self.assertEqual(expected[1:],actual[1:])
            if expected[0] is not None:
                np.testing.assert_array_equal(expected[0],actual[0][13:13+h,19:19+w])
                self.assertEqual(actual[0].shape,bigmask.shape)
                self.assertEqual(actual[0].sum(),expected[0].sum())
            a=torso_evidence(image,mask,excluded);b=torso_evidence(padded,bigmask,bigexcluded)
            if a[0] is not None:np.testing.assert_array_equal(a[0],b[0])

    def test_overlap_flags_match_full_frame_intersections(self):
        masks=np.zeros((6,100,150),bool)
        masks[0,5:45,10:30]=True;masks[1]=masks[0]
        masks[2,10:40,12:25]=True;masks[3,60:95,110:135]=True
        masks[4,30:55,15:45]=True
        expected=[None]*len(masks);areas=[int(m.sum()) for m in masks]
        for i in range(len(masks)):
            for j in range(i):
                intersection=int((masks[i]&masks[j]).sum())
                if not min(areas[i],areas[j]):continue
                if intersection/(areas[i]+areas[j]-intersection)>=.85:expected[i]=expected[j]='duplicate_mask'
                elif intersection/min(areas[i],areas[j])>=.8:expected[i]=expected[j]='merged_or_contained_mask'
        self.assertEqual(mask_flags(masks),expected)

    def test_external_invalid_rgb_and_nonbinary_masks_still_fail(self):
        image=np.zeros((100,100,3));mask=np.zeros((100,100),bool);mask[20:80,30:50]=True
        image[0,0,0]=float('nan')
        with self.assertRaises(ValueError):torso_evidence(image,mask)
        with self.assertRaises(ValueError):binary_mask(np.full((10,10),2),(10,10))
