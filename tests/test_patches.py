import unittest
import numpy as np
from inspectai.patches import score_maps, heatmap_png


class PatchTests(unittest.TestCase):
    def test_local_change_remains_local(self):
        mean = np.zeros((2,3,3),dtype=np.float32)
        x = np.zeros((1,2,3,3),dtype=np.float32)
        x[:,:,1,2] = 2
        scores = score_maps(x, mean, np.ones_like(mean))
        self.assertEqual(scores[0,1,2],2)
        self.assertEqual(np.count_nonzero(scores),1)

    def test_invalid_variance_rejected(self):
        with self.assertRaises(ValueError):
            score_maps(np.zeros((1,2,3,3)),np.zeros((2,3,3)),np.zeros((2,3,3)))

    def test_constant_map_renders(self):
        import base64
        self.assertTrue(base64.b64decode(heatmap_png(np.zeros((3,3)))).startswith(b'\x89PNG'))
