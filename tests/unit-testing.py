import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import the DroneDetector class from your module
from main.py import DroneDetector

class TestDroneDetector(unittest.TestCase):
    def setUp(self):
        self.model_path = 'path/to/model.pt'
        self.detection_threshold = 0.6
        self.audio_threshold = 0.01

        patcher = patch('drone_detection.YOLO')
        self.addCleanup(patcher.stop)
        self.mock_YOLO = patcher.start()

        self.mock_model = MagicMock()
        self.mock_YOLO.return_value = self.mock_model

        self.detector = DroneDetector(
            self.model_path,
            self.detection_threshold,
            self.audio_threshold
        )

    def test_initialization(self):
        """Test that the DroneDetector initializes with correct parameters."""
        self.assertEqual(self.detector.detection_threshold, self.detection_threshold)
        self.assertEqual(self.detector.audio_threshold, self.audio_threshold)
        self.assertFalse(self.detector.audio_detection_active)
        self.mock_YOLO.assert_called_once_with(self.model_path)

    def test_detect_no_detections(self):
        """Test detect method when no objects are detected."""
        mock_results = [MagicMock()]
        mock_results[0].boxes = []
        self.mock_model.return_value = mock_results

        # dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # call detct
        detections = self.detector.detect(frame)

        # assertions
        self.assertEqual(detections, [])

    def test_detect_with_detections(self):
        """Test detect method when objects are detected."""
        # prompt a return of detection
        mock_box = MagicMock()
        mock_box.conf = [0.75]
        mock_box.xyxy = [np.array([100, 150, 200, 250])]
        mock_box.cls = [0]

        mock_results = [MagicMock()]
        mock_results[0].boxes = [mock_box]
        self.mock_model.return_value = mock_results

        # dummy frame 
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # call detect method
        detections = self.detector.detect(frame)

        # assertions
        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection['confidence'], 0.75)
        self.assertEqual(detection['position'], (100, 150, 200, 250))
        self.assertEqual(detection['label'], 'drone 0.75')
        self.assertEqual(detection['type'], 'image')

    def test_detect_audio_below_threshold(self):
        """Test detect_audio method when audio is below threshold."""
        with patch('drone_detection.sd') as mock_sd:
            mock_sd.rec.return_value = np.zeros((22050, 1))
            mock_sd.wait.return_value = None

            detection = self.detector.detect_audio()

            self.assertIsNone(detection)

    def test_detect_audio_above_threshold(self):
        """Test detect_audio method when audio is above threshold."""
        with patch('drone_detection.sd') as mock_sd:
            audio_data = np.ones((22050, 1)) * 0.02  
            mock_sd.rec.return_value = audio_data
            mock_sd.wait.return_value = None

            detection = self.detector.detect_audio()

            self.assertIsNotNone(detection)
            self.assertEqual(detection['type'], 'audio')
            self.assertGreaterEqual(detection['confidence'], self.detector.audio_threshold)

    def test_detect_with_invalid_confidence(self):
        """Test detect method with invalid confidence value."""
        mock_box = MagicMock()
        mock_box.conf = ['invalid']  
        mock_box.xyxy = [np.array([100, 150, 200, 250])]
        mock_box.cls = [0]

        mock_results = [MagicMock()]
        mock_results[0].boxes = [mock_box]
        self.mock_model.return_value = mock_results

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch('drone_detection.messagebox') as mock_messagebox:
            detections = self.detector.detect(frame)

            mock_messagebox.showerror.assert_called_once()
            self.assertEqual(detections, [])

    def test_detect_audio_exception_handling(self):
        """Test detect_audio method handling exceptions."""
        with patch('drone_detection.sd') as mock_sd:
            mock_sd.rec.side_effect = Exception("Microphone not accessible")
            with patch('drone_detection.messagebox') as mock_messagebox:
                detection = self.detector.detect_audio()

                mock_messagebox.showerror.assert_called_once_with(
                    "Error", "An error occurred during audio detection: Microphone not accessible"
                )
                self.assertIsNone(detection)

if __name__ == '__main__':
    unittest.main()
