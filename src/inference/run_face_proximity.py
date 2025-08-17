import cv2
import logging
import argparse
import sqlite3
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from constants import DataPaths, FaceDetection
from config import FaceConfig
from models.proximity.estimate_proximity import calculate_proximity
from inference import process_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info("Face detection processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO face detection on extracted frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                       help="List of video names to process")
    parser.add_argument("--frame_step", type=int, default=10, 
                       help="Frame step for processing (default: 10)")
    
    args = parser.parse_args()
    main(args.video_list, args.frame_step)