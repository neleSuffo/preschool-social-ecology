import cv2
import logging
import argparse
import sqlite3
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from ultralytics import YOLO
from constants import DataPaths, FaceDetection
from models.proximity.estimate_proximity import calculate_proximity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_video_id(video_name: str, cursor: sqlite3.Cursor) -> int:
    """Get video_id from Videos table using video name"""
    cursor.execute('SELECT video_id FROM Videos WHERE video_name = ?', (video_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        logging.error(f"Video {video_name} not found in Videos table")
        return None

def process_frame(frame_path: Path, video_id: int, frame_number: int, 
                 face_model: YOLO, cursor: sqlite3.Cursor) -> int:
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
    results = face_model.predict(frame, verbose=False)
    
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
                ''', (video_id, frame_number, 1, float(confidence), 
                      float(x1), float(y1), float(x2), float(y2), class_id, proximity))
                
                face_count += 1
    
    return face_count

def process_video(video_name: str, frame_step: int, 
                 face_model: YOLO, cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """Process all frames for a video"""
    logging.info(f"Processing video: {video_name}")
    
    # Get video_id from database
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        return
    
    # Find frame files for this video
    frames_dir = DataPaths.IMAGES_INPUT_DIR / video_name
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return
    
    frame_files = sorted(frames_dir.glob(f"{video_name}_*.jpg"))
    if not frame_files:
        logging.warning(f"No frame files found for video: {video_name}")
        return
        
    processed_frames = 0
    total_faces = 0
    
    for frame_file in frame_files:
        # Extract frame number from filename: video_name_000123.jpg -> 123
        try:
            frame_number = int(frame_file.stem.split('_')[-1])
        except ValueError:
            logging.warning(f"Could not extract frame number from: {frame_file}")
            continue
        
        # Skip frames according to frame_step
        if frame_step > 1 and frame_number % frame_step != 0:
            continue
        
        # Process frame
        face_count = process_frame(frame_file, video_id, frame_number, face_model, cursor)
        total_faces += face_count
        processed_frames += 1
        
        # Commit every 100 frames
        if processed_frames % 100 == 0:
            conn.commit()
    
    conn.commit()
    logging.info(f"Processed {processed_frames} frames, detected {total_faces} faces")

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
    face_model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    logging.info(f"Loaded face detection model from {FaceDetection.TRAINED_WEIGHTS_PATH}")
    
    # Process each video
    for video_name in video_list:
        try:
            process_video(video_name, frame_step, face_model, cursor, conn)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()
    logging.info("Face detection processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO face detection on extracted frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                       help="List of video names to process")
    parser.add_argument("--frame_step", type=int, default=10, 
                       help="Frame step for processing (default: 10)")
    
    args = parser.parse_args()
    main(args.video_list, args.frame_step)