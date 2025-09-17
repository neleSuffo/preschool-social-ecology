import cv2
import logging
import argparse
import sqlite3
from typing import List, Callable
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from constants import DataPaths, FaceDetection
from config import FaceConfig
from models.proximity.estimate_proximity import calculate_proximity
from utils import get_video_id, extract_frame_number, get_frame_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_video(
    video_name: str,
    frame_step: int,
    model,
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    task: str,
    process_frame_func: Callable
):
    """
    Generic video processing function.
    
    Args:
        video_name: Name of the video.
        frame_step: Process every nth frame.
        model: Model to use (CNNEncoder+RNN or YOLO).
        cursor: SQLite cursor.
        conn: SQLite connection.
        task: String to show in progress bar (e.g., "Classifying" or "Detecting Faces").
        process_frame_func: Function that processes a single frame and returns a metric.
                            For classification: returns bool (success).
                            For detection: returns int (number of faces).
    """
    logging.info(f"Processing video: {video_name}")

    # Get video_id
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        return

    # Frames directory
    frames_dir = DataPaths.IMAGES_INPUT_DIR / video_name
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return
    
    frame_files = get_frame_paths(frames_dir)
    
    if not frame_files:
        logging.warning(f"No frame files found for video: {video_name}")
        return

    # Filter frames (respecting frame_step)
    frames_to_process = []
    for frame_file in frame_files:
        try:
            frame_number = extract_frame_number(frame_file.name)
            if frame_step > 1 and frame_number % frame_step != 0:
                continue
            frames_to_process.append((frame_file, frame_number))
        except Exception:
            logging.warning(f"Could not extract frame number from: {frame_file.name}")
            continue

    processed_frames = 0
    metric_sum = 0  # success count for classification, face count for detection

    # Process frames with progress bar
    with tqdm(frames_to_process, desc=f"{task} {video_name}", unit="frames") as pbar:
        for frame_file, frame_number in pbar:
            # Task-specific frame processing
            result = process_frame_func(frame_file, video_id, frame_number, model, cursor)
            if isinstance(result, bool):  # classification success
                metric_sum += int(result)
            elif isinstance(result, int):  # number of faces
                metric_sum += result

            processed_frames += 1

            # Update progress bar
            pbar.set_postfix({
                'metric': metric_sum,
                'current_frame': frame_number
            })

            # Commit every 100 frames
            if processed_frames % 100 == 0:
                conn.commit()

    conn.commit()
    logging.info(f"Processed video {video_name}: {processed_frames} frames classified")

def process_frame(frame_path: Path, video_id: int, frame_number: int, 
                 model: YOLO, cursor: sqlite3.Cursor) -> int:
    """
    Process a single frame for face detection
    
    Returns:
    -------
    int: Number of faces detected
    """
    # Read frame
    frame = cv2.imread(str(frame_path))
    if frame is None:
        logging.warning(f"Failed to read frame: {frame_path}")
        return 0
    
    # Run YOLO face detection
    results = model.predict(frame, verbose=False)
    
    face_count = 0
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Extract detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Calculate proximity
                proximity = calculate_proximity([x1, y1, x2, y2], class_id)
                if proximity is None:
                    continue
                # Insert into FaceDetections table
                cursor.execute('''
                    INSERT INTO FaceDetections 
                    (video_id, frame_number, model_id, confidence_score, 
                     x_min, y_min, x_max, y_max, age_class, proximity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_id, frame_number, FaceConfig.MODEL_ID, float(confidence), 
                      float(x1), float(y1), float(x2), float(y2), class_id, proximity))
                
                face_count += 1
    
    return face_count

def main(video_list: List[str], frame_step: int = 10):
    """
    Main function to process videos for face detection
    
    Parameters:
    ----------
    video_list : List[str]
        List of video names to process
    frame_step : int
        Step size for frame processing (default: 10)
    """
    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    # Load YOLO face detection model
    model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    logging.info(f"Loaded face detection model from {FaceDetection.TRAINED_WEIGHTS_PATH}")
    
    # Process each video
    for video_name in video_list:
        try:
            process_video(video_name, frame_step, model, cursor, conn, task="Detecting Faces", process_frame_func=process_frame)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO face detection on extracted frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                       help="List of video names to process")
    parser.add_argument("--frame_step", type=int, default=10, 
                       help="Frame step for processing (default: 10)")
    
    args = parser.parse_args()
    main(args.video_list, args.frame_step)