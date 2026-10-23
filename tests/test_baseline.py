from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from inspectai.baseline import anomaly_scores
from inspectai.web import app


class BaselineTests(unittest.TestCase):
    def test_scores_order_by_distance(self):
        scores = anomaly_scores([[1., 0.], [0., 1.]], [[1., 0.]])
        self.assertEqual(scores[0], 0)
        self.assertGreater(scores[1], scores[0])

    def test_invalid_features_rejected(self):
        with self.assertRaises(ValueError):
            anomaly_scores([[float('nan')]], [[1.]])
        with self.assertRaises(ValueError):
            anomaly_scores([[1., 2.]], [[1.]])

    def test_no_model_does_not_fake_prediction(self):
        from inspectai.baseline import predict
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                predict(Image.new('RGB', (32, 32)), Path(directory) / 'missing.npz')

    def test_upload_rejects_invalid_file(self):
        response = app.test_client().post('/inspect', data={'image': (BytesIO(b'not an image'), 'bad.png')})
        self.assertEqual(response.status_code, 400)

    @patch('inspectai.web.predict', side_effect=FileNotFoundError('Baseline missing'))
    def test_valid_image_without_model(self, model):
        buffer = BytesIO()
        Image.new('RGB', (32, 32)).save(buffer, format='PNG')
        buffer.seek(0)
        response = app.test_client().post('/inspect', data={'image': (buffer, 'sample.png')})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b'Baseline missing', response.data)
