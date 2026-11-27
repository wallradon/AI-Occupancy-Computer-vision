import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from extractor import extract, output_size


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / 'sample.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 90))
        self.assertTrue(writer.isOpened())
        for index in range(50):
            writer.write(np.full((90, 160, 3), index * 4, dtype=np.uint8))
        writer.release()

    def test_interval_resize_and_unicode_destination(self):
        folder, count, cancelled = extract(self.video, self.root / 'ภาพ', interval=2,
                                           bounds=(80, 80), image_format='png')
        self.assertEqual(count, 3)
        self.assertFalse(cancelled)
        files = sorted(folder.glob('*.png'))
        for index, file in enumerate(files):
            frame = cv2.imdecode(np.frombuffer(file.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(frame.shape[:2], (45, 80))
            self.assertAlmostEqual(float(frame.mean()), index * 80, delta=6)

    def test_short_video_and_no_overwrite(self):
        first, count, _ = extract(self.video, self.root, interval=30)
        second, _, _ = extract(self.video, self.root, interval=30)
        self.assertEqual(count, 1)
        self.assertNotEqual(first, second)
        self.assertEqual(len(list(first.glob('*.jpg'))), 1)

    def test_cancellation_preserves_saved_images(self):
        cancel = threading.Event()
        folder, count, cancelled = extract(self.video, self.root, interval=1, cancel=cancel,
                                           progress=lambda *args: cancel.set())
        self.assertTrue(cancelled)
        self.assertEqual(count, 1)
        self.assertEqual(len(list(folder.glob('*.jpg'))), 1)

    def test_dimensions_and_invalid_settings(self):
        self.assertEqual(output_size(1080, 1920, (1920, 1080)), (608, 1080))
        self.assertEqual(output_size(160, 90, (80, 80), False), (80, 80))
        for interval in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                extract(self.video, self.root, interval=interval)
        with self.assertRaises(ValueError):
            extract(self.video, self.root, bounds=(0, 100))


if __name__ == '__main__':
    unittest.main()
