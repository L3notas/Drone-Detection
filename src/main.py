# src/main.py

"""
Drone Detection System with OOP and Audio Detection
This program uses a YOLO model to detect drones from a camera feed and basic audio detection.
It provides a GUI for starting, pausing, and stopping the detection,
and displays detections, current alerts, and alert history.
"""

import os
import sys
import threading
import time
import math
import configparser
import locale
import xml.etree.ElementTree as ET
from tkinter import (
    Tk,
    Canvas,
    Toplevel,
    Label,
    Scale,
    HORIZONTAL,
    StringVar,
    OptionMenu,
    Scrollbar,
    Listbox,
    Menu,
    messagebox,
)
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import winsound
import numpy as np
from detector import DroneDetector
import sounddevice as sd

# Ensure the src directory is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class DroneDetectionApp:
    """
    Main application class for the Drone Detection System.
    """

    def __init__(self, root):
        """
        Initialize the DroneDetectionApp, load settings, and set up the GUI.
        """
        self.root = root
        self.root.title("Drone Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#FFFFFF")
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        # Get the base directory of the project
        self.base_dir = os.path.dirname(current_dir)

        # Paths to configuration and model files
        self.config_path = os.path.join(self.base_dir, 'config', 'settings.ini')
        self.model_path = os.path.join(self.base_dir, 'models', 'best.pt')

        # Initialize variables
        self.detection_times = []
        self.detection_details = []
        self.cap = None
        self.camera_feed_active = False
        self.start_time = None
        self.elapsed_time = 0
        self.running = False
        self.video_label = None

        # Load settings
        self.config = configparser.ConfigParser()
        self.load_settings()

        # Initialize the detector
        self.detector = DroneDetector(
            self.model_path,
            self.detection_threshold,
            self.audio_threshold,
        )

        # Apply theme
        self.apply_theme()

        # Set up GUI elements
        self.setup_gui()

        # Update language after GUI elements are created
        self.update_language()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_main_window_closing)

        # Ensure alert history directory exists
        self.alert_history_dir = os.path.join(self.base_dir, 'alert_history')
        if not os.path.exists(self.alert_history_dir):
            os.makedirs(self.alert_history_dir)

    def load_settings(self):
        """
        Load settings from 'settings.ini' configuration file.
        """
        if not os.path.exists(self.config_path):
            # If settings.ini doesn't exist, copy from settings.example.ini
            example_config_path = os.path.join(
                os.path.dirname(self.config_path), 'settings.example.ini'
            )
            if os.path.exists(example_config_path):
                import shutil
                shutil.copy(example_config_path, self.config_path)
            else:
                messagebox.showerror("Error", "Configuration file not found.")
                exit()
        self.config.read(self.config_path)
        self.detection_threshold = self.config.getfloat(
            "Settings", "detection_threshold", fallback=0.60
        )
        self.audio_threshold = self.config.getfloat(
            "Settings", "audio_threshold", fallback=0.01
        )
        self.language = self.config.get("Settings", "language", fallback="English")
        self.dark_mode = self.config.getboolean("Settings", "dark_mode", fallback=False)
        # Set locale based on language
        if self.language == "French":
            try:
                locale.setlocale(locale.LC_TIME, "fr_FR")
            except locale.Error:
                locale.setlocale(locale.LC_TIME, "")
        else:
            locale.setlocale(locale.LC_TIME, "")

    def save_settings(self):
        """
        Save current settings to 'settings.ini' configuration file.
        """
        self.config["Settings"] = {
            "detection_threshold": str(self.detection_threshold),
            "audio_threshold": str(self.audio_threshold),
            "language": self.language,
            "dark_mode": str(self.dark_mode),
        }
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def apply_theme(self):
        """
        Apply GUI theme based on the 'dark_mode' setting.
        """
        if self.dark_mode:
            self.style.theme_use("alt")
            self.root.configure(bg="#2E2E2E")
            self.style.configure("TFrame", background="#2E2E2E")
            self.style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF")
            self.style.configure("TButton", background="#3E3E3E", foreground="#FFFFFF")
            if self.video_label:
                self.video_label.config(bg="#66677E")
        else:
            self.style.theme_use("clam")
            self.root.configure(bg="#FFFFFF")
            self.style.configure("TFrame", background="#FFFFFF")
            self.style.configure("TLabel", background="#FFFFFF", foreground="#000000")
            self.style.configure("TButton", background="#E0E0E0", foreground="#000000")
            if self.video_label:
                self.video_label.config(bg="#66677E")

    def format_time(self, seconds):
        """
        Format elapsed time in seconds into a HH:MM:SS string.
        """
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02}:{int(mins):02}:{int(secs):02}"

    def update_time(self):
        """
        Update the time label with the elapsed time.
        """
        if self.running:
            current_time = time.time()
            self.elapsed_time = int(current_time - self.start_time)
            self.time_label.config(text=self.format_time(self.elapsed_time))
            self.root.after(1000, self.update_time)  # Schedule the next update after 1 second

    def start_timer_and_detection(self):
        """
        Start the timer and begin the drone detection process.
        """
        if not self.running:
            self.running = True
            if self.start_time is None:
                self.start_time = time.time() - self.elapsed_time
            self.update_time()
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Cannot access the camera.")
                    self.running = False
                    return
            threading.Thread(target=self.run_detection, daemon=True).start()
            threading.Thread(target=self.run_audio_detection, daemon=True).start()

    def pause_timer_and_detection(self):
        """
        Pause the timer and the drone detection process without resetting.
        """
        self.running = False

    def stop_timer_and_detection(self):
        """
        Stop the timer and the drone detection process.
        """
        self.running = False
        self.start_time = None
        self.elapsed_time = 0
        self.time_label.config(text="00:00:00")
        self.export_detections_to_xml()
        self.detection_times.clear()
        self.detection_details.clear()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def run_detection(self):
        """
        Run the drone image detection process in a separate thread.
        """
        while self.running:
            try:
                success, img = self.cap.read()
                if not success:
                    break  # Break the loop if frame capture fails
                detections = self.detector.detect(img)
                for detection in detections:
                    self.detection_times.append(detection["time"])
                    self.detection_details.append(detection)
                    self.save_detection_snapshot(img)  # Save a snapshot when detection occurs
                    winsound.Beep(1000, 500)  # Play a beep sound
                time.sleep(0.1)  # Sleep briefly to reduce CPU usage
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                break

    def run_audio_detection(self):
        """
        Run the audio detection process in a separate thread.
        """
        while self.running:
            detection = self.detector.detect_audio()
            if detection:
                self.detection_times.append(detection["time"])
                self.detection_details.append(detection)
                winsound.Beep(1500, 500)  # Play a different beep sound for audio detection
            time.sleep(0.5)  # Sleep to control detection frequency

    def toggle_camera_feed(self):
        """
        Toggle the display of the camera feed in the GUI.
        """
        if not self.camera_feed_active:
            self.camera_feed_active = True
            self.btn_camera_feed.config(
                text="Hide Camera Feed" if self.language == "English" else "Cacher Flux Vidéo"
            )
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Cannot access the camera.")
                    self.camera_feed_active = False
                    self.btn_camera_feed.config(
                        text="Camera Feed" if self.language == "English" else "Flux Vidéo"
                    )
                    return
            self.create_video_feed_label()
            threading.Thread(target=self.update_video_feed, daemon=True).start()
        else:
            self.camera_feed_active = False
            self.btn_camera_feed.config(
                text="Camera Feed" if self.language == "English" else "Flux Vidéo"
            )
            if self.video_label:
                self.video_label.imgtk = None
                self.video_label.config(image='', bg="#66677E")

    def create_video_feed_label(self):
        """
        Create or update the label widget used to display the video feed.
        """
        if not self.video_label:
            self.video_label = Label(self.root, bg="#66677E")
            self.video_label.place(x=0, y=400, width=600, height=200)
            self.root.bind('<Configure>', self.on_window_resize)
        else:
            self.video_label.config(bg="#66677E")

    def update_video_feed(self):
        """
        Continuously update the video feed in the GUI.
        """
        while self.camera_feed_active:
            try:
                success, frame = self.cap.read()
                if not success:
                    break
                detections = self.detector.detect(frame)
                for detection in detections:
                    x1, y1, x2, y2 = detection["position"]
                    label = detection["label"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                    )
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                # Resize the image to fit the video label
                img = img.resize((self.video_label.winfo_width(), self.video_label.winfo_height()))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                time.sleep(0.03)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                break

    def on_window_resize(self, event):
        """
        Adjust the size of the video feed label when the window is resized.
        """
        if self.camera_feed_active:
            self.video_label.config(
                width=self.root.winfo_width(),
                height=int(self.root.winfo_height() * 0.33)
            )

    def save_detection_snapshot(self, frame):
        """
        Save a snapshot image of the current frame when a detection occurs.
        """
        snapshot_dir = os.path.join(self.base_dir, 'snapshots')
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        filepath = os.path.join(snapshot_dir, filename)
        try:
            cv2.imwrite(filepath, frame)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save snapshot: {e}")

    def export_detections_to_xml(self):
        """
        Export the current detection details to an XML file.
        """
        if not self.detection_times:
            print("No detections to export.")
            return
        root = ET.Element("Detections")
        for idx, detail in enumerate(self.detection_details, 1):
            detection = ET.SubElement(root, "Detection", id=str(idx))
            time_elem = ET.SubElement(detection, "Time")
            time_elem.text = detail["time"]
            conf_elem = ET.SubElement(detection, "Confidence")
            conf_elem.text = str(detail["confidence"])
            if detail["type"] == "image":
                pos_elem = ET.SubElement(detection, "Position")
                pos_elem.text = str(detail["position"])
            type_elem = ET.SubElement(detection, "Type")
            type_elem.text = detail["type"]
        tree = ET.ElementTree(root)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        xml_filename = os.path.join(self.alert_history_dir, f"detections_{timestamp}.xml")
        try:
            tree.write(xml_filename)
            print(f"Detections exported to {xml_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export detections: {e}")

    # ... [Include all other methods from your original code, adjusting paths as necessary] ...

    def open_settings_window(self):
        """
        Open the settings window where users can adjust detection thresholds and language.
        """
        settings_window = Toplevel(self.root)
        settings_window.title("Settings" if self.language == "English" else "Paramètres")
        settings_window.geometry("400x400")
        settings_window.configure(bg="#FFFFFF")

        def on_threshold_change(val):
            """
            Handle changes to the detection threshold slider.
            """
            try:
                self.detection_threshold = float(val) / 100
                if not (0.0 <= self.detection_threshold <= 1.0):
                    raise ValueError("Detection threshold must be between 0 and 100")
                self.save_settings()
                self.detector.detection_threshold = self.detection_threshold
            except ValueError:
                messagebox.showerror("Error", "Invalid threshold value.")

        def on_audio_threshold_change(val):
            """
            Handle changes to the audio threshold slider.
            """
            try:
                self.audio_threshold = float(val) / 1000
                if self.audio_threshold < 0:
                    raise ValueError("Audio threshold must be positive")
                self.save_settings()
                self.detector.audio_threshold = self.audio_threshold
            except ValueError:
                messagebox.showerror("Error", "Invalid audio threshold value.")

        Label(
            settings_window,
            text="Detection Threshold:",
            font=("Roboto", 12),
            bg="#FFFFFF",
        ).pack(pady=10)
        threshold_scale = Scale(
            settings_window,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            command=on_threshold_change,
        )
        threshold_scale.set(self.detection_threshold * 100)
        threshold_scale.pack()

        Label(
            settings_window,
            text="Audio Threshold:",
            font=("Roboto", 12),
            bg="#FFFFFF",
        ).pack(pady=10)
        audio_threshold_scale = Scale(
            settings_window,
            from_=0,
            to=100,
            orient=HORIZONTAL,
            command=on_audio_threshold_change,
        )
        audio_threshold_scale.set(self.audio_threshold * 1000)
        audio_threshold_scale.pack()

        def on_language_change(selection):
            """
            Handle changes to the language selection.
            """
            if selection not in ["English", "French"]:
                messagebox.showerror("Error", "Unsupported language selected.")
                return
            self.language = selection
            self.save_settings()
            self.update_language()

        Label(
            settings_window,
            text="Select Language:",
            font=("Roboto", 12),
            bg="#FFFFFF",
        ).pack(pady=10)
        language_var = StringVar(settings_window)
        language_var.set(self.language)
        language_options = ["English", "French"]
        language_menu = OptionMenu(
            settings_window, language_var, *language_options, command=on_language_change
        )
        language_menu.pack()

        close_button_text = "Close" if self.language == "English" else "Fermer"
        ttk.Button(
            settings_window, text=close_button_text, command=settings_window.destroy
        ).pack(pady=20)

    # ... [Include the rest of your methods, adjusting paths and keeping functionality intact] ...

    def run(self):
        """
        Start the main GUI event loop.
        """
        self.root.mainloop()

if __name__ == "__main__":
    app = DroneDetectionApp(Tk())
    app.run()
