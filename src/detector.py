# src/detector.py

import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
import sounddevice as sd
from tkinter import messagebox

class DroneDetector:
    """
    Class to handle drone detection logic using the YOLO model and audio detection.
    """

    def __init__(self, model_path, detection_threshold, audio_threshold):
        """
        Initialize the DroneDetector with a YOLO model and detection parameters.
        """
        try:
            self.model = YOLO(model_path)
            self.class_names = ["drone"]
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load the model: {e}")
            exit()
        self.detection_threshold = detection_threshold
        self.audio_threshold = audio_threshold
        self.audio_detection_active = False

    def detect(self, frame):
        """
        Perform drone detection on the given image frame using the YOLO model.
        """
        detections = []
        results = self.model(frame)
        boxes = results[0].boxes
        for box in boxes:
            try:
                conf = box.conf[0]
                if not isinstance(conf, float):
                    raise TypeError("Confidence value is not a float")
                if conf >= self.detection_threshold:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    # Round up the confidence to two decimal places
                    conf = math.ceil((conf * 100)) / 100
                    cls = int(box.cls[0])
                    if cls < 0 or cls >= len(self.class_names):
                        raise ValueError("Class index out of range")
                    class_name = self.class_names[cls]
                    label = f"{class_name} {conf}"
                    detection_time = time.strftime("%c", time.localtime())
                    detections.append({
                        "time": detection_time,
                        "confidence": conf,
                        "position": (x1, y1, x2, y2),
                        "label": label,
                        "type": "image",
                    })
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred processing detection: {e}")
        return detections

    def detect_audio(self, duration=0.5):
        """
        Perform audio detection using the microphone.
        """
        try:
            # Record audio for the given duration
            audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
            sd.wait()
            # Compute the RMS (root mean square) of the audio signal
            rms = np.sqrt(np.mean(np.square(audio_data)))
            if rms >= self.audio_threshold:
                detection_time = time.strftime("%c", time.localtime())
                return {
                    "time": detection_time,
                    "confidence": rms,
                    "label": f"Audio Detection {rms:.2f}",
                    "type": "audio",
                }
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during audio detection: {e}")
        return None
