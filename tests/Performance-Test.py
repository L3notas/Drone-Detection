import multiprocessing
import time
import numpy as np
import cProfile
import pstats
import io
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import DroneDetector

def generate_synthetic_frame(width=640, height=480):
    """
    Generate a synthetic image frame with random pixel values.
    """
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

def detection_worker(detector, num_frames, frame_queue):
    """
    Worker function to simulate detection on synthetic frames.
    """
    for _ in range(num_frames):
        frame = generate_synthetic_frame()
        detections = detector.detect(frame)
        # Optionally, process detections
        time.sleep(0.01)  # Simulate processing time
    frame_queue.put('done')

def performance_test():
    """
    Run the performance test by simulating multiple detection workers.
    """
    model_path = 'path/to/your/model.pt'
    detection_threshold = 0.6
    audio_threshold = 0.01

    detector = DroneDetector(model_path, detection_threshold, audio_threshold)

    num_workers = multiprocessing.cpu_count()  # Use number of CPU cores
    num_frames_per_worker = 100  # Total frames each worker will process

    frame_queue = multiprocessing.Queue()
    workers = []

    start_time = time.time()

    # Start profiling
    pr = cProfile.Profile()
    pr.enable()

    # Start worker processes
    for _ in range(num_workers):
        p = multiprocessing.Process(target=detection_worker, args=(detector, num_frames_per_worker, frame_queue))
        workers.append(p)
        p.start()

    # Wait for all workers to finish
    completed_workers = 0
    while completed_workers < num_workers:
        msg = frame_queue.get()
        if msg == 'done':
            completed_workers += 1

    # Stop profiling
    pr.disable()

    end_time = time.time()
    total_time = end_time - start_time

    # Output profiling stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats()

    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Frames Processed: {num_workers * num_frames_per_worker}")
    print(f"Frames per Second: {(num_workers * num_frames_per_worker) / total_time:.2f}")

    # Save profiling results to a file
    with open('performance_profile.txt', 'w') as f:
        f.write(s.getvalue())

if __name__ == '__main__':
    performance_test()
