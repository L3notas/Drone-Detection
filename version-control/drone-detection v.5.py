
import threading
import time
import cv2
import winsound
from tkinter import Tk, Label, Canvas, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path, detection_threshold):
        self.model = YOLO(model_path)
        self.detection_threshold = detection_threshold

    def detect(self, frame):
        detections = []
        results = self.model(frame)
        boxes = results[0].boxes
        for box in boxes:
            conf = box.conf[0]
            if conf >= self.detection_threshold:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = round(conf * 100, 2)
                label = f"Drone {conf}%"
                detections.append({
                    "confidence": conf,
                    "position": (x1, y1, x2, y2),
                    "label": label,
                })
        return detections

class DroneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#FFFFFF")
        self.detector = DroneDetector('C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt', 0.6)
        self.cap = None
        self.camera_feed_active = False
        self.video_label = None
        self.running = False
        self.start_time = None
        self.elapsed_time = 0

        self.setup_gui()

    def setup_gui(self):
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

        self.time_label = Label(
            self.root, text="00:00:00", bg="#D7DFF6", font=("Roboto", 18, "bold"), anchor="w"
        )
        self.time_label.place(x=620, y=386, width=150, height=40)

        self.btn_camera_feed = ttk.Button(
            self.root,
            text="Camera Feed",
            command=self.toggle_camera_feed
        )
        self.btn_camera_feed.place(x=36, y=38, width=155, height=155)

        self.btn_start = ttk.Button(
            self.root,
            text="Start",
            command=lambda: threading.Thread(
                target=self.start_detection, daemon=True
            ).start(),
        )
        self.btn_start.place(x=620, y=16, width=150, height=40)

        self.btn_pause = ttk.Button(
            self.root,
            text="Pause",
            command=self.pause_detection,
        )
        self.btn_pause.place(x=620, y=75, width=150, height=40)

        self.btn_stop = ttk.Button(
            self.root,
            text="Stop",
            command=self.stop_detection,
        )
        self.btn_stop.place(x=620, y=134, width=150, height=40)

    def toggle_camera_feed(self):
        if not self.camera_feed_active:
            self.camera_feed_active = True
            self.cap = cv2.VideoCapture(0)
            self.create_video_feed_label()
            threading.Thread(target=self.update_video_feed, daemon=True).start()
        else:
            self.camera_feed_active = False
            if self.cap:
                self.cap.release()
            if self.video_label:
                self.video_label.imgtk = None
                self.video_label.config(image='')

    def create_video_feed_label(self):
        if not self.video_label:
            self.video_label = Label(self.root, bg="#66677E")
            self.video_label.place(x=0, y=400, width=600, height=200)
            self.root.bind('<Configure>', self.on_window_resize)
        else:
            self.video_label.config(bg="#66677E")

    def update_video_feed(self):
        while self.camera_feed_active:
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

    def on_window_resize(self, event):
        if self.camera_feed_active:
            self.video_label.config(
                width=self.root.winfo_width(),
                height=int(self.root.winfo_height() * 0.33)
            )

    def start_detection(self):
        if not self.running:
            self.running = True
            if self.start_time is None:
                self.start_time = time.time() - self.elapsed_time
            self.update_time()
            threading.Thread(target=self.run_detection, daemon=True).start()

    def update_time(self):
        if self.running:
            current_time = time.time()
            self.elapsed_time = int(current_time - self.start_time)
            self.time_label.config(text=self.format_time(self.elapsed_time))
            self.root.after(1000, self.update_time)

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02}:{int(mins):02}:{int(secs):02}"

    def run_detection(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                break
            detections = self.detector.detect(frame)
            if detections:
                winsound.Beep(1000, 500)
            time.sleep(0.1)

    def pause_detection(self):
        self.running = False

    def stop_detection(self):
        self.running = False
        self.start_time = None
        self.elapsed_time = 0
        self.time_label.config(text="00:00:00")
        if self.cap and self.cap.isOpened():
            self.cap.release()

def main():
    root = Tk()
    app = DroneDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
