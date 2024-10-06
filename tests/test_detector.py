import cv2
from ultralytics import YOLO

# Load the YOLOv8 model (use the appropriate model file or path here)
model = YOLO('C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt')  # 'yolov8n.pt' is the YOLOv8 nano model

# Open the video file
video_path = 'C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/tests/Test Video/drone video 5 cloudy good lighting.mp4'  # repace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video details
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Set up the video writer to save the output (optional)
output_path = 'output_video.mp4'  # Path to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 detection on the frame
    results = model(frame)
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Display the frame with detections (optional)
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # Write the frame to the output video file (optional)
    out.write(annotated_frame)
    
    # Press 'q' to quit early (optional)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
