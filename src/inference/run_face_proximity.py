import cv2
import logging
import argparse
import sqlite3
import pandas as pd
import numpy as np
import subprocess
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from ultralytics import YOLO
from constants import YoloPaths, DetectionPaths, VTCPaths
from config import YoloConfig, DetectionPipelineConfig
from estimate_proximity import get_proximity
from models.yolo_detections.inference import (
    calculate_detection_scores,
    is_face_inside_person,
    custom_nms,
)

def extract_id_from_filename(filename: str) -> str:
    """
    This function extracts the ID from a filename.
    
    Parameters:
    ----------
    filename : str
        Filename to extract the ID from
    
    Returns:
    -------
    id_part : str
        Extracted ID part of the filename
    """
    parts = filename.split('id')
    if len(parts) > 1:
        id_part = parts[1].split('_')[0]
        return id_part
    return None



def insert_video_record(video_path, cursor) -> int:
    """
    This function inserts a video record into the database if it doesn't already exist.
    It retrieves child information from the Subjects table using the video name.
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    video_id : int
        Video ID
    """
    # Extract just the video name from the path
    video_name = Path(video_path).name
    
    cursor.execute('SELECT video_id FROM Videos WHERE video_path = ?', (video_name,))
    existing_video = cursor.fetchone()
    
    if existing_video:
        logging.info(f"Video {video_name} already processed. Skipping.")
        return None
    else:
        # Get child information from Subjects table using video name
        cursor.execute('''
            SELECT child_id, birthday, age_at_recording, recording_date
            FROM Subjects 
            WHERE video_name = ?
        ''', (video_name,))
        result = cursor.fetchone()
        
        if result:
            child_id, birthday, age_at_recording, recording_date = result
            # Insert video record
            cursor.execute('''
                INSERT INTO Videos (video_path, child_id, recording_date, age_at_recording) 
                VALUES (?, ?, ?, ?)
            ''', (video_name, child_id, recording_date, age_at_recording))
            
            cursor.execute('SELECT video_id FROM Videos WHERE video_path = ?', (video_name,))
            return cursor.fetchone()[0]
        else:
            logging.error(f"No subject data found for video {video_name}")
            return None

def process_frame(frame: np.ndarray, frame_idx: int, video_id: int, detection_model: YOLO, gaze_model: YOLO, cursor: sqlite3.Cursor) -> dict:
    """
    This function processes a frame. It inserts the frame record and processes each object detected in the frame.
    The steps are as follows:
    1. Insert frame record
    2. Process each object detected in the frame
    
    Parameters:
    ----------
    frame : np.ndarray
        Frame to be processed
    frame_idx : int
        Frame index
    video_id : int
        Video ID
    detection_model : YOLO
        YOLO model for person, face, and object detection
    gaze_model : YOLO
        YOLO model for gaze classification
    cursor : sqlite3.Cursor
        SQLite cursor object
    
    Returns:
    -------
    detection_counts : dict
        Dictionary containing counts of each object class detected
    """
    cursor.execute('INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)', (video_id, frame_idx))

    # Initialize detection counts
    detection_counts = {name: 0 for name in YoloConfig.detection_mapping.values()}
    
    # run detections and NMS
    results = detection_model.predict(frame, iou=YoloConfig.best_iou)
    filtered_boxes = custom_nms(results, iou_threshold=YoloConfig.between_classes_iou)
    
    # Categorize detections
    people_boxes, face_boxes, other_boxes = [], [], []
    # First pass: categorize detections
    for box in filtered_boxes:
        x1, y1, x2, y2, class_id, confidence_score = box
        detection = {
            'box': (x1, y1, x2, y2),
            'class_id': class_id,
            'confidence': confidence_score,
            'class_name': detection_model.names[class_id]
        }
        
        if class_id in [0, 1]:  # Child or Adult
            people_boxes.append(detection)
        elif class_id in [2, 3]:  # Child or Adult face
            face_boxes.append(detection)
        else:
            other_boxes.append(detection)       

# Process face-person pairs
    for face in face_boxes:
        for person in people_boxes:
            if is_face_inside_person(face['box'], person['box']):
                scores = calculate_detection_scores(face, person, YoloConfig.ap_values)
                
                # Log scores for debugging
                for key, value in scores.items():
                    logging.debug(f"{key}: {value:.3f}")
                
                # Update classes based on combined scores
                if scores['adult_combined'] > scores['child_combined']:
                    face['class_id'] = 3  # Adult face
                    face['class_name'] = detection_model.names[3]
                    person['class_id'] = 1  # Adult
                    person['class_name'] = detection_model.names[1]
                else:
                    face['class_id'] = 2  # Child face
                    face['class_name'] = detection_model.names[2]
                    person['class_id'] = 0  # Child
                    person['class_name'] = detection_model.names[0]
    
    # Process all detections
    all_detections = people_boxes + face_boxes + other_boxes
    for detection in all_detections:
        x1, y1, x2, y2 = detection['box']
        class_id = detection['class_id']
        class_name = detection['class_name']
        
        # Update detection count
        if class_name in detection_counts:
            detection_counts[class_name] += 1
        
        # Initialize face-specific variables
        gaze_direction = None
        gaze_confidence = None
        proximity = None
            
        # Process faces for gaze and proximity
        if class_id in [2, 3]:  # child or adult face
            face_image = frame[y1:y2, x1:x2]
            gaze_direction, gaze_confidence = classify_gaze(gaze_model, face_image)
            proximity = get_proximity([x1, y1, x2, y2], class_name)

        # Insert detection record
        cursor.execute('''
            INSERT INTO Detections
            (video_id, frame_number, object_class, confidence_score,
            x_min, y_min, x_max, y_max, gaze_direction,
            gaze_confidence, proximity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, class_id, detection['confidence'], 
              x1, y1, x2, y2, gaze_direction, gaze_confidence, proximity))

    return detection_counts

def process_video(video_folder: Path, 
                  detection_model: YOLO, 
                  gaze_model: YOLO, 
                  cursor: sqlite3.Cursor, 
                  conn: sqlite3.Connection,
                  frame_skip: int):
    """
    Process frames from saved image files instead of video.
    """
    logging.info(f"Processing video: {video_folder}")
    
    # Convert Path to string for SQLite
    video_folder_str = str(video_folder)
    
    # Check if video exists in VideoStatistics table
    cursor.execute('''
        SELECT v.video_id, vs.processed_frames 
        FROM Videos v
        LEFT JOIN VideoStatistics vs ON v.video_id = vs.video_id
        WHERE v.video_path = ?
    ''', (video_folder_str,))
    result = cursor.fetchone()
    
    if result and result[1] is not None and result[1] > 0:  # Check if frames were actually processed
        logging.info(f"Video {video_folder} already processed with {result[1]} frames. Skipping.")
        return
    
    # Continue with video processing if not already processed
    video_id = insert_video_record(video_folder_str, cursor)
    
    # Skip if video already processed
    if video_id is None:
        return

    # Get corresponding frames directory
    frames_dir = DetectionPaths.images_input_dir / video_folder
    
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return

    # Get all frame images sorted by frame number
    frame_files = sorted(frames_dir.glob('*.jpg'))
    processed_frames = 0
    total_counts = {name: 0 for name in YoloConfig.detection_mapping.values()}

    for frame_file in frame_files:
        # Extract frame number from filename (last 6 digits before .jpg)
        padded_frame_idx = frame_file.stem[-6:]  # e.g. "000533"
        frame_idx = int(padded_frame_idx)  # automatically converts to 533

        # Skip frames according to frame_skip
        if frame_skip > 0 and frame_idx % frame_skip != 0:
            continue
            
        # Read frame
        frame = cv2.imread(str(frame_file))
        if frame is None:
            logging.warning(f"Failed to read frame: {frame_file}")
            continue

        # Process frame
        detection_counts = process_frame(
            frame, frame_idx, video_id, detection_model, gaze_model, cursor
        )
        
        # Update total counts
        for class_name, count in detection_counts.items():
            total_counts[class_name] += count
            
        conn.commit()
        processed_frames += 1

    # Store statistics in database
    cursor.execute('''
        INSERT INTO VideoStatistics (
            video_id, total_frames, processed_frames,
            child_count, adult_count, child_face_count, adult_face_count,
            book_count, toy_count, kitchenware_count, screen_count,
            food_count, other_object_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        video_id, len(frame_files), processed_frames,
        total_counts['infant/child'], total_counts['adult'],
        total_counts['infant/child face'], total_counts['adult face'],
        total_counts['book'], total_counts['toy'],
        total_counts['kitchenware'], total_counts['screen'],
        total_counts['food'], total_counts['other_object']
    ))
    conn.commit()
    
    logging.info(f"Processed {processed_frames} frames out of {len(frame_files)} total frames")

          
def main(num_videos_to_process: int = None,
        frame_skip: int = 10):
    """
    This function processes videos using YOLO models for person and face detection and gaze classification.
    It loads the age group data and processes either all videos or a balanced subset from each age group.
    
    Parameters:
    ----------
    num_videos_to_process : int, optional
        Number of videos to process per age group. If None, processes all available videos.
    frame_skip : int, optional
        Number of frames to skip between processing (default: 10)
    """
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load the age group data
    age_df = pd.read_csv('/home/nele_pauline_suffo/ProcessedData/age_group.csv')
    
    videos_input_dir = DetectionPaths.images_input_dir
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    # Initialize models once
    object_model = YOLO(YoloPaths.all_trained_weights_path)
    gaze_model = YOLO(YoloPaths.gaze_trained_weights_path)   
    
    detection_model_id = register_model(cursor, "detection", detection_model)
    gaze_model_id = register_model(cursor, "gaze", gaze_model)

    videos_to_not_process = DetectionPipelineConfig.videos_to_not_process
    if num_videos_to_process is None:
        # Process all video folders
        all_videos = [p for p in videos_input_dir.iterdir() 
                if p.is_dir() and extract_id_from_filename(p.name) is not None]
        videos_to_process = [v for v in all_videos if v.name not in videos_to_not_process]

        skipped = len(all_videos) - len(videos_to_process)
        logging.info(f"Found {len(all_videos)} video folders")
        logging.info(f"Skipping {skipped} videos from exclusion list")
        logging.info(f"Processing {len(videos_to_process)} videos")
    else:
        # Get balanced videos across age groups
        available_videos = get_balanced_videos(videos_input_dir, age_df, videos_per_group=num_videos_to_process // 3)
        videos_to_process = [v for v in available_videos if v.name not in videos_to_not_process]
        
        skipped = len(available_videos) - len(videos_to_process)
        logging.info(f"Selected {len(available_videos)} balanced videos")
        logging.info(f"Skipping {skipped} videos from exclusion list")
        logging.info(f"Processing {len(videos_to_process)} videos")
    
    # Process videos
    for video in videos_to_process:
        process_video(video, detection_model, gaze_model, cursor, conn, frame_skip)
    
    conn.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process a set of videos using YOLO models for person and face detection and gaze classification.")
    argparser.add_argument("--num_videos", type=int, help="Number of videos to process")
    argparse.add_argument("--frame_skip", type=int, default=10, help="Number of frames to skip between processing (default: 10)")
    args = argparser.parse_args()
    num_videos_to_process = args.num_videos
    frame_skip = args.frame_skip
    main(num_videos_to_process, frame_skip)