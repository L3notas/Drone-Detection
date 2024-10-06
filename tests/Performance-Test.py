import multiprocessing
import time
import numpy as np
import cProfile
import pstats
import io
import sys
import os

# Add the 'src' directory to sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

from main import DroneDetector

def generate_synthetic_frame(width=640, height=480):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def detection_worker(model_path, detection_threshold, audio_threshold, num_frames, worker_id):
    pr = cProfile.Profile()
    pr.enable()

    detector = DroneDetector(model_path, detection_threshold, audio_threshold)

    for _ in range(num_frames):
        frame = generate_synthetic_frame()
        detections = detector.detect(frame)
        time.sleep(0.01)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    with open(f'performance_profile_worker_{worker_id}.txt', 'w') as f:
        f.write(s.getvalue())

def performance_test():
    model_path = 'C:/Users/liam_/OneDrive/Documents/Drone-Detection/Drone-Detection/models/best.pt'  # Update with your actual model path
    detection_threshold = 0.6
    audio_threshold = 0.01

    num_workers = multiprocessing.cpu_count()
    num_frames_per_worker = 100

    start_time = time.time()

    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(
            target=detection_worker,
            args=(model_path, detection_threshold, audio_threshold, num_frames_per_worker, i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Frames Processed: {num_workers * num_frames_per_worker}")
    print(f"Frames per Second: {(num_workers * num_frames_per_worker) / total_time:.2f}")

if __name__ == '__main__':
    performance_test()
