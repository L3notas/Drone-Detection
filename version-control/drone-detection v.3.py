#Version 3, has detection within bare minimum GUI components
import tkinter as tk
from tkinter import ttk
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk

class DroneDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Detection System")
        self.root.geometry("800x600")
        self.root.configure(bg="#FFFFFF")
        self.model = YOLO('C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt')
        self.cap = None
        self.camera_feed_active = False
        self.video_label = tk.Label(self.root)
        self.video_label.place(x=0, y=400, width=600, height=200)

        btn_camera_feed = ttk.Button(
            self.root,
            text="Camera Feed",
            command=self.toggle_camera_feed
        )
        btn_camera_feed.place(x=36, y=38, width=155, height=155)

    def toggle_camera_feed(self):
        if not self.camera_feed_active:
            self.camera_feed_active = True
            self.cap = cv2.VideoCapture(0)
            self.update_video_feed()
        else:
            self.camera_feed_active = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')

    def update_video_feed(self):
        if self.camera_feed_active:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(frame)
                boxes = results[0].boxes

                for box in boxes:
                    conf = box.conf[0]
                    if conf >= 0.6:
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.root.after(30, self.update_video_feed)

def main():
    root = tk.Tk()
    app = DroneDetectionApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
