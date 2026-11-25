"""Unit tests for Metro Exit Counter core logic (no GPU / video / model needed).

Run:  python -m pytest -q        (or)   python tests/test_core.py
"""
import os
import sys
import types
import unittest
from pathlib import Path

# Skip auto-install / restart logic and avoid loading heavy ultralytics.
os.environ["NUMPY_FIXED"] = "1"
if "ultralytics" not in sys.modules:
    try:
        import ultralytics  # noqa: F401
    except Exception:
        stub = types.ModuleType("ultralytics")
        stub.YOLO = object
        sys.modules["ultralytics"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import main as mec


class GeometryTests(unittest.TestCase):
    def test_point_in_poly(self):
        sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertTrue(mec.point_in_poly(5, 5, sq))
        self.assertFalse(mec.point_in_poly(15, 5, sq))
        self.assertFalse(mec.point_in_poly(-1, -1, sq))

    def test_scale_pts(self):
        out = mec.scale_pts(np.array([[0.5, 0.5]]), 200, 100)
        self.assertEqual(out.tolist(), [[100, 50]])

    def test_side_of_line_opposite_signs(self):
        a, b = (0, 0), (10, 0)
        self.assertGreater(mec.side_of_line((5, -3), a, b) * mec.side_of_line((5, 3), a, b) * -1, 0)

    def test_within_segment(self):
        a, b = (0, 0), (10, 0)
        self.assertTrue(mec.within_segment((5, 2), a, b))
        self.assertFalse(mec.within_segment((30, 2), a, b))


class TrackerTests(unittest.TestCase):
    def test_centroid_tracker_keeps_ids(self):
        t = mec.CentroidTracker()
        ids1 = t.update([(0, 0, 10, 10), (100, 100, 110, 110)])
        ids2 = t.update([(2, 2, 12, 12), (102, 101, 112, 111)])
        self.assertEqual(ids1.tolist(), ids2.tolist())

    def test_centroid_tracker_new_id(self):
        t = mec.CentroidTracker()
        t.update([(0, 0, 10, 10)])
        ids = t.update([(500, 500, 510, 510)])
        self.assertEqual(ids[0], 2)


class StatsTests(unittest.TestCase):
    def test_peak_and_timeline(self):
        s = mec.Stats(fps=10)
        for f, n in [(1, 2), (5, 7), (12, 3)]:
            s.tick(f, n)
        self.assertEqual(s.peak, 7)
        self.assertGreaterEqual(len(s.timeline), 2)

    def test_avg_dwell(self):
        s = mec.Stats(fps=10)
        tr = mec.Track(1, first=0, last=100, trail=[], hits=50)
        s.finish_track(tr)
        self.assertAlmostEqual(s.avg_dwell([]), 10.0)


class HeatmapTests(unittest.TestCase):
    def test_heatmap_blend_shape(self):
        h = mec.Heatmap(64, 48)
        for _ in range(20):
            h.add(30, 20)
        frame = np.zeros((48, 64, 3), np.uint8)
        self.assertEqual(h.blend(frame).shape, frame.shape)

    def test_density_levels(self):
        self.assertEqual(mec.density_level(1, 4, 8)[0], "LOW")
        self.assertEqual(mec.density_level(5, 4, 8)[0], "MEDIUM")
        self.assertEqual(mec.density_level(9, 4, 8)[0], "HIGH")


if __name__ == "__main__":
    unittest.main()
