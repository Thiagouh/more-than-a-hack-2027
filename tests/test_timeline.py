import sys
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from desenhar_posses import intervals, render_svg
from possession_metrics import label_codes


class TimelineTests(unittest.TestCase):
    def test_crop_preserves_absolute_frames_and_x(self):
        codes = label_codes(list('HHXXAAD'))
        self.assertEqual(intervals(codes, 1, 5), [(1, 2, 'H'), (2, 4, 'X'), (4, 6, 'A')])
        svg = render_svg([('a<&', codes)], 1, 5, 25)
        ET.fromstring(svg)
        self.assertIn('X: frames 2–3', svg)
        self.assertIn('0.04', svg)
        self.assertIn('a&lt;&amp;', svg)

    def test_bad_ranges_and_coverage(self):
        codes = label_codes(list('HA'))
        for start, end in [(-1, 1), (1, 0), (0, 2)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                render_svg([('a', codes)], start, end)
        with self.assertRaises(ValueError):
            render_svg([('a', codes), ('b', codes[:1])])
        for fps in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                render_svg([('a', codes)], fps=fps)


if __name__ == '__main__':
    unittest.main()
