
"""
Drone Detection System with OOP and Audio Detection
This program uses a YOLO model to detect drones from a camera feed and basic audio detection.
It provides a GUI for starting, pausing, and stopping the detection,
and displays detections, current alerts, and alert history.
"""

import threading
import time
import os
import math
import cv2
import winsound
import configparser
import locale
import xml.etree.ElementTree as ET
import numpy as np
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
from ultralytics import YOLO
from pathlib import Path
import sounddevice as sd


class DroneDetector:
    """
    Class to handle drone detection logic using the YOLO model and audio detection.
    """

    def __init__(self, model_path, detection_threshold, audio_threshold):
        """
        Initialize the YOLO model and set detection parameters.
        Args:
            model_path (str): Path to the YOLO model weights.
            detection_threshold (float): Confidence threshold for image detection.
            audio_threshold (float): Threshold for audio detection.
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
        Perform detection on the given frame.
        Args:
            frame (numpy.ndarray): The image frame to perform detection on.
        Returns:
            list: A list of detection details.
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
        Args:
            duration (float): Duration to record audio in seconds.
        Returns:
            dict or None: Detection details if audio exceeds threshold, else None.
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


class DroneDetectionApp:
    """
    Main application class for the Drone Detection System.
    """

    def __init__(self, root):
        """
        Initialize the application, load settings, and set up the GUI.
        Args:
            root (Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Drone Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#FFFFFF")
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

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
            "C:/Users/liam_/OneDrive/Desktop/Applied Computing/runs/detect/train19/weights/best.pt",
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
        self.alert_history_dir = "alert_history"
        if not os.path.exists(self.alert_history_dir):
            os.makedirs(self.alert_history_dir)

    def load_settings(self):
        """
        Load settings from 'settings.ini' configuration file.
        Updates instance variables: detection_threshold, language, dark_mode, audio_threshold.
        """
        self.config.read("settings.ini")
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
        with open("settings.ini", "w") as configfile:
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
        Format elapsed time in seconds into HH:MM:SS string.
        Args:
            seconds (int): The number of seconds elapsed.
        Returns:
            str: Formatted time string.
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
            self.root.after(1000, self.update_time)

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
        Run the drone detection process.
        """
        while self.running:
            try:
                success, img = self.cap.read()
                if not success:
                    break
                detections = self.detector.detect(img)
                for detection in detections:
                    self.detection_times.append(detection["time"])
                    self.detection_details.append(detection)
                    self.save_detection_snapshot(img)
                    winsound.Beep(1000, 500)
                time.sleep(0.1)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
                break

    def run_audio_detection(self):
        """
        Run the audio detection process.
        """
        while self.running:
            detection = self.detector.detect_audio()
            if detection:
                self.detection_times.append(detection["time"])
                self.detection_details.append(detection)
                winsound.Beep(1500, 500)
            time.sleep(0.5)

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
        Args:
            event: The resize event.
        """
        if self.camera_feed_active:
            self.video_label.config(
                width=self.root.winfo_width(),
                height=int(self.root.winfo_height() * 0.33)
            )

    def save_detection_snapshot(self, frame):
        """
        Save a snapshot image of the current frame when a detection occurs.
        Args:
            frame (numpy.ndarray): The image frame to save.
        """
        snapshot_dir = "snapshots"
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        try:
            cv2.imwrite(os.path.join(snapshot_dir, filename), frame)
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

    def open_settings_window(self):
        """
        Open the settings window where users can adjust detection threshold and language.
        """
        settings_window = Toplevel(self.root)
        settings_window.title("Settings" if self.language == "English" else "Paramètres")
        settings_window.geometry("400x400")
        settings_window.configure(bg="#FFFFFF")

        def on_threshold_change(val):
            try:
                self.detection_threshold = float(val) / 100
                if not (0.0 <= self.detection_threshold <= 1.0):
                    raise ValueError("Detection threshold must be between 0 and 100")
                self.save_settings()
                self.detector.detection_threshold = self.detection_threshold
            except ValueError:
                messagebox.showerror("Error", "Invalid threshold value.")

        def on_audio_threshold_change(val):
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

    def update_language(self):
        """
        Update the text of GUI elements based on the selected language.
        """
        if self.language == "English":
            self.btn_start.config(text="Start")
            self.btn_pause.config(text="Pause")
            self.btn_stop.config(text="Stop")
            self.btn_camera_feed.config(
                text="Camera Feed" if not self.camera_feed_active else "Hide Camera Feed"
            )
            self.btn_current_alerts.config(text="Current Alerts")
            self.btn_settings.config(text="Settings")
            self.btn_alert_history.config(text="Alert History")
            self.time_label.config(text=self.format_time(self.elapsed_time))
        elif self.language == "French":
            self.btn_start.config(text="Démarrer")
            self.btn_pause.config(text="Pause")
            self.btn_stop.config(text="Arrêter")
            self.btn_camera_feed.config(
                text="Flux Vidéo" if not self.camera_feed_active else "Cacher Flux Vidéo"
            )
            self.btn_current_alerts.config(text="Alertes Actuelles")
            self.btn_settings.config(text="Paramètres")
            self.btn_alert_history.config(text="Historique des Alertes")
            self.time_label.config(text=self.format_time(self.elapsed_time))
        # Update other labels if necessary

    def open_current_alerts_window(self):
        """
        Open a window displaying current detections in the ongoing detection cycle.
        """
        current_alerts_window = Toplevel(self.root)
        if self.language == "French":
            current_alerts_window.title("Alertes Actuelles")
            no_detections_text = "Aucune détection dans le cycle actuel."
            detection_label_text = "Détection"
            close_button_text = "Fermer"
        else:
            current_alerts_window.title("Current Alerts")
            no_detections_text = "No detections in the current cycle."
            detection_label_text = "Detection"
            close_button_text = "Close"

        current_alerts_window.geometry("400x300")
        current_alerts_window.configure(bg="#FFFFFF")

        if not self.detection_times:
            Label(
                current_alerts_window,
                text=no_detections_text,
                font=("Roboto", 12),
                bg="#FFFFFF",
            ).pack(pady=20)
        else:
            scrollbar = Scrollbar(current_alerts_window)
            scrollbar.pack(side="right", fill="y")

            listbox = Listbox(
                current_alerts_window,
                yscrollcommand=scrollbar.set,
                font=("Roboto", 12),
            )
            for idx, detail in enumerate(self.detection_details, 1):
                if detail["type"] == "image":
                    info = f"{detection_label_text} {idx}: {detail['time']}, Confidence: {detail['confidence']}, Position: {detail['position']}"
                else:
                    info = f"{detection_label_text} {idx}: {detail['time']}, Audio Level: {detail['confidence']}"
                listbox.insert("end", info)
            listbox.pack(side="left", fill="both", expand=True)

            scrollbar.config(command=listbox.yview)

        ttk.Button(
            current_alerts_window,
            text=close_button_text,
            command=current_alerts_window.destroy,
        ).pack(pady=10)

    def open_alert_history_window(self):
        """
        Open a window displaying historical detections from previous detection cycles.
        """
        alert_history_window = Toplevel(self.root)
        if self.language == "French":
            alert_history_window.title("Historique des Alertes")
            no_history_text = "Aucun historique d'alertes."
            detection_label_text = "Détection"
            cycle_label_text = "Cycle"
            close_button_text = "Fermer"
        else:
            alert_history_window.title("Alert History")
            no_history_text = "No alert history available."
            detection_label_text = "Detection"
            cycle_label_text = "Cycle"
            close_button_text = "Close"

        alert_history_window.geometry("400x400")
        alert_history_window.configure(bg="#FFFFFF")

        xml_files = [
            f for f in os.listdir(self.alert_history_dir) if f.endswith(".xml")
        ]
        if not xml_files:
            Label(
                alert_history_window,
                text=no_history_text,
                font=("Roboto", 12),
                bg="#FFFFFF",
            ).pack(pady=20)
            ttk.Button(
                alert_history_window,
                text=close_button_text,
                command=alert_history_window.destroy,
            ).pack(pady=10)
            return

        scrollbar = Scrollbar(alert_history_window)
        scrollbar.pack(side="right", fill="y")

        listbox = Listbox(
            alert_history_window,
            yscrollcommand=scrollbar.set,
            font=("Roboto", 12),
        )

        for xml_file in sorted(xml_files):
            try:
                tree = ET.parse(os.path.join(self.alert_history_dir, xml_file))
                root = tree.getroot()
                timestamp = xml_file.replace("detections_", "").replace(".xml", "")
                try:
                    cycle_time = time.strftime(
                        "%c", time.strptime(timestamp, "%Y%m%d-%H%M%S")
                    )
                except ValueError:
                    cycle_time = timestamp
                listbox.insert("end", f"{cycle_label_text} {cycle_time}:")
                for detection in root.findall("Detection"):
                    detection_id = detection.get("id")
                    detection_time = detection.find("Time").text
                    confidence = detection.find("Confidence").text
                    detection_type = detection.find("Type").text
                    if detection_type == "image":
                        position = detection.find("Position").text
                        info = f"  {detection_label_text} {detection_id}: {detection_time}, Confidence: {confidence}, Position: {position}"
                    else:
                        info = f"  {detection_label_text} {detection_id}: {detection_time}, Audio Level: {confidence}"
                    listbox.insert("end", info)
                listbox.insert("end", "")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load alert history: {e}")
        listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        ttk.Button(
            alert_history_window,
            text=close_button_text,
            command=alert_history_window.destroy,
        ).pack(pady=10)

    def on_main_window_closing(self):
        """
        Handler for the main window closing event.
        """
        self.running = False
        self.camera_feed_active = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def setup_gui(self):
        """
        Set up the GUI elements.
        """
        canvas = Canvas(
            self.root,
            bg="#FFFFFF",
            height=600,
            width=800,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )
        canvas.place(x=0, y=0)

        # Background rectangles for layout
        canvas.create_rectangle(0, 0, 600, 600, fill="#66677E", outline="")
        canvas.create_rectangle(600, 0, 800, 600, fill="#3E3E4D", outline="")

        # Label for showing time elapsed
        self.time_label = Label(
            self.root, text="00:00:00", bg="#D7DFF6", font=("Roboto", 18, "bold"), anchor="w"
        )
        self.time_label.place(x=620, y=386, width=150, height=40)

        # Buttons
        self.btn_camera_feed = ttk.Button(
            self.root,
            text="Camera Feed",
            command=self.toggle_camera_feed,
        )
        self.btn_camera_feed.place(x=36, y=38, width=155, height=155)

        self.btn_current_alerts = ttk.Button(
            self.root,
            text="Current Alerts",
            command=self.open_current_alerts_window,
        )
        self.btn_current_alerts.place(x=223, y=38, width=155, height=155)

        self.btn_settings = ttk.Button(
            self.root, text="Settings", command=self.open_settings_window
        )
        self.btn_settings.place(x=36, y=220, width=155, height=155)

        self.btn_alert_history = ttk.Button(
            self.root,
            text="Alert History",
            command=self.open_alert_history_window,
        )
        self.btn_alert_history.place(x=223, y=220, width=155, height=155)

        # Timer Buttons
        self.btn_start = ttk.Button(
            self.root,
            text="Start",
            command=lambda: threading.Thread(
                target=self.start_timer_and_detection, daemon=True
            ).start(),
        )
        self.btn_start.place(x=620, y=16, width=150, height=40)

        self.btn_pause = ttk.Button(
            self.root,
            text="Pause",
            command=self.pause_timer_and_detection,
        )
        self.btn_pause.place(x=620, y=75, width=150, height=40)

        self.btn_stop = ttk.Button(
            self.root,
            text="Stop",
            command=self.stop_timer_and_detection,
        )
        self.btn_stop.place(x=620, y=134, width=150, height=40)

        # Menu Bar
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)

        # Help Menu
        help_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Help" if self.language == "English" else "Aide", menu=help_menu)

        def show_help():
            help_text = (
                "Drone Detection System Help\n\n"
                "Start: Begin the detection cycle.\n"
                "Pause: Pause the detection without resetting.\n"
                "Stop: Stop the detection and reset the timer.\n"
                "Settings: Configure detection threshold and language.\n"
                "Current Alerts: View detections from the current cycle.\n"
                "Alert History: View detections from previous cycles.\n"
                "Camera Feed: Toggle the video feed display.\n"
                "Dark Mode: Toggle between light and dark themes."
            )
            messagebox.showinfo("Help", help_text)

        help_menu.add_command(
            label="View Help" if self.language == "English" else "Voir l'aide",
            command=show_help
        )

        # Theme Menu
        theme_menu = Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Theme", menu=theme_menu)

        def toggle_dark_mode():
            self.dark_mode = not self.dark_mode
            self.apply_theme()
            self.save_settings()

        theme_menu.add_command(
            label="Toggle Dark Mode" if self.language == "English" else "Basculer en mode sombre",
            command=toggle_dark_mode
        )

    def run(self):
        """
        Start the main GUI event loop.
        """
        self.root.mainloop()


if __name__ == "__main__":
    app = DroneDetectionApp(Tk())
    app.run()