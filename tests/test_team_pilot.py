import unittest
import numpy as np
from src.team_identification.pilot import TeamClassifier

class TeamPilotTests(unittest.TestCase):
    def test_team_without_identity_and_overlap_abstention(self):
        red=np.zeros((60,30,3),dtype=np.uint8);red[:,:,0]=255
        blue=np.zeros_like(red);blue[:,:,2]=255
        mask=np.ones((60,30),dtype=bool)
        c=TeamClassifier([(red,mask,'H','player'),(blue,mask,'A','player')])
        self.assertEqual(c.classify(red,[mask])[0]['team'],'H')
        self.assertEqual(c.classify(blue,[mask])[0]['team'],'A')
        self.assertTrue(all(r['team'] is None for r in c.classify(red,[mask,mask])))
        self.assertEqual(c.classify(red,[mask])[0]['team'],'H')

    def test_missing_team_reference_rejected(self):
        with self.assertRaises(ValueError):TeamClassifier([])

    def test_small_crop_abstains(self):
        red=np.full((60,30,3),(255,0,0),dtype=np.uint8)
        blue=np.full((60,30,3),(0,0,255),dtype=np.uint8)
        mask=np.ones((60,30),dtype=bool)
        c=TeamClassifier([(red,mask,'H','player'),(blue,mask,'A','player')])
        small=np.zeros_like(mask);small[:4,:4]=True
        self.assertIsNone(c.classify(red,[small])[0]['team'])
