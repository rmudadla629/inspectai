import unittest
import numpy as np
from PIL import Image
from inspectai.simple_features import encode_pixels
from inspectai.compare import metrics


class ComparisonTests(unittest.TestCase):
    def test_feature_shape_and_norm(self):
        features = encode_pixels([Image.new('RGB', (60, 80), 'white'), Image.new('RGB', (60, 80), 'black')])
        self.assertEqual(features.shape, (2, 1024))
        self.assertTrue(np.isfinite(features).all())
        self.assertAlmostEqual(float(np.linalg.norm(features[0])), 1)
        self.assertEqual(float(np.linalg.norm(features[1])), 0)

    def test_identical_images_have_identical_features(self):
        image = Image.new('RGB', (40, 40), 'red')
        features = encode_pixels([image, image.copy()])
        np.testing.assert_array_equal(features[0], features[1])

    def test_metrics_count_misses_and_false_alarms(self):
        result = metrics([0, 0, 1, 1], np.array([.1, .7, .4, .9]), .5)
        self.assertEqual(result['confusion'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
        self.assertEqual(result['defect_recall'], .5)
        self.assertEqual(result['false_positive_rate'], .5)
