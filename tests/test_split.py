from pathlib import Path
import tempfile
import unittest
from inspectai.train import split_normal


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(20):
                path = Path(directory) / f'{i}.png'
                path.write_bytes(str(i).encode())
                paths.append(path)
            references, calibration = split_normal(paths)
            self.assertEqual(len(references), 16)
            self.assertEqual(len(calibration), 4)
            self.assertFalse(set(references) & set(calibration))
            self.assertEqual((references, calibration), split_normal(paths))

    def test_exact_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i in range(20):
                path = Path(directory) / f'{i}.png'
                path.write_bytes(b'duplicate')
                paths.append(path)
            with self.assertRaises(ValueError):
                split_normal(paths)
