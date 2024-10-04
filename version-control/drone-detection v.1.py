import cv2
import winsound
from ultralytics import YOLO

def main():
    model = YOLO('C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt')
    cap = cv2.VideoCapture(0)
    detection_threshold = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes
        detected = False

        for box in boxes:
            conf = box.conf[0]
            if conf >= detection_threshold:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                detected = True

        cv2.imshow('Drone Detection', frame)

        if detected:
            winsound.Beep(1000, 500)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
