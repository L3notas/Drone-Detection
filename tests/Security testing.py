import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection'))))

import main

class SecurityTestDroneDetector(unittest.TestCase):
    def setUp(self):
        self.model_path = 'C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt'
        self.detection_threshold = 0.6
        self.audio_threshold = 0.01

        patcher = patch('main.YOLO')
        self.addCleanup(patcher.stop)
        self.mock_YOLO = patcher.start()

        self.mock_model = MagicMock()
        self.mock_YOLO.return_value = self.mock_model

        self.detector = DroneDetector(
            self.model_path,
            self.detection_threshold,
            self.audio_threshold
        )

    def test_detect_with_corrupted_frame(self):
        frame = None

        with patch('main.messagebox') as mock_messagebox:
            detections = self.detector.detect(frame)
            mock_messagebox.showerror.assert_called_once()
            self.assertEqual(detections, [])

    def test_detect_with_invalid_frame_type(self):
        """Test detect method with invalid frame data type."""
        frame = "This is not an image"

        with patch('main.messagebox') as mock_messagebox:
            detections = self.detector.detect(frame)
            mock_messagebox.showerror.assert_called_once()
            self.assertEqual(detections, [])

    def test_detect_audio_with_invalid_data(self):
        """Test detect_audio method with invalid audio data."""
        with patch('main.sd') as mock_sd:
            mock_sd.rec.return_value = "Invalid audio data"
            mock_sd.wait.return_value = None

            with patch('main.messagebox') as mock_messagebox:
                detection = self.detector.detect_audio()
                mock_messagebox.showerror.assert_called_once()
                self.assertIsNone(detection)

    def test_handle_large_frame(self):
        """Test detect method with an excessively large image frame."""
        frame = np.random.randint(0, 256, (10000, 10000, 3), dtype=np.uint8)

        with patch('main.messagebox') as mock_messagebox:
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                mock_messagebox.showerror.assert_called_once()
                self.assertIsInstance(e, Exception)

    def test_model_loading_with_invalid_path(self):
        """Test model loading with an invalid model path."""
        with patch('main.messagebox') as mock_messagebox:
            with self.assertRaises(SystemExit):
                invalid_detector = DroneDetector(
                    'invalid/path/to/model.pt',
                    self.detection_threshold,
                    self.audio_threshold
                )
            mock_messagebox.showerror.assert_called_once()

if __name__ == '__main__':
    unittest.main()
