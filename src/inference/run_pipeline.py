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
from config import YoloConfig, DetectionPipelineConfig, VideoConfig
from estimate_proximity import get_proximity
from detection_pipeline.inference import is_face_inside_person

class DetectionClassMapping:
    # Final class IDs for database storage
    CHILD = 0
    ADULT = 1
    CHILD_FACE = 2
    ADULT_FACE = 3
    CHILD_BODY_PART = 4
    # Objects 5-10 remain unchanged

def map_detection_to_final_class(cls_id: int, model_type: str, age_cls: int = None) -> int:
    """Maps model-specific class IDs to final database class IDs."""
    if model_type == "object":
        return cls_id  # Object classes (5-10) remain unchanged
    elif model_type == "person_face":
        if cls_id == 0:  # Person
            return DetectionClassMapping.ADULT if age_cls == 0 else DetectionClassMapping.CHILD
        elif cls_id == 1:  # Face
            return DetectionClassMapping.ADULT_FACE if age_cls == 0 else DetectionClassMapping.CHILD_FACE
        elif cls_id == 2:  # Child body part
            return DetectionClassMapping.CHILD_BODY_PART
    return cls_id

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

def get_balanced_videos(videos_dir: Path, age_df: pd.DataFrame, videos_per_group: int) -> list:
    """
    Select a balanced number of videos from each age group, skipping videos without age data.
    
    Parameters:
    ----------
    videos_dir : Path
        Directory containing the video folders
    age_df : pd.DataFrame
        DataFrame with columns 'child_id' and 'age_group'
    videos_per_group : int
        Number of videos to select from each age group
        
    Returns:
    -------
    list
        List of selected video folder paths
    """
    selected_videos = []
    
    # Convert video folders to a list with their IDs
    available_videos = []
    skipped_videos = []
    
    # Look for folders (video directories) instead of MP4 files
    for video_folder in videos_dir.iterdir():
        if not video_folder.is_dir():
            continue
            
        video_id = extract_id_from_filename(video_folder.name)
        if video_id:
            try:
                # Check if ID exists in age_df and has a valid age group
                age_row = age_df[age_df['child_id'] == int(video_id)]
                if not age_row.empty and pd.notna(age_row['age_group'].iloc[0]):
                    available_videos.append((video_folder, video_id))
                else:
                    skipped_videos.append(video_folder.name)
            except (ValueError, KeyError):
                skipped_videos.append(video_folder.name)
    
    if skipped_videos:
        logging.info(f"Skipped {len(skipped_videos)} videos without age data")
        logging.debug(f"Skipped videos: {', '.join(skipped_videos)}")
    
    # Group videos by age
    videos_by_age = {3: [], 4: [], 5: []}
    for video_folder, video_id in available_videos:
        age_group = age_df[age_df['child_id'] == int(video_id)]['age_group'].iloc[0]
        if age_group in videos_by_age:
            videos_by_age[age_group].append(video_folder)
    
    # Log available videos per age group
    for age_group, videos in videos_by_age.items():
        logging.info(f"Age group {age_group}: {len(videos)} videos available")
    
    # Select balanced number of videos from each group
    for age_group in videos_by_age:
        videos = videos_by_age[age_group]
        if len(videos) >= videos_per_group:
            selected = np.random.choice(videos, size=videos_per_group, replace=False)
            selected_videos.extend(selected)
            logging.info(f"Selected {len(selected)} videos from age group {age_group}")
        else:
            logging.warning(f"Age group {age_group} has only {len(videos)} videos, using all available")
            selected_videos.extend(videos)
    
    return selected_videos

def classify_gaze(gaze_model: YOLO, face_image: np.ndarray) -> Tuple[int, int]:
    """
    This function classifies the gaze direction of a face image.
    
    Parameters:
    ----------
    gaze_model : YOLO
        YOLO model for gaze classification
    face_image : np.ndarray
        Face image to be classified
        
    Returns:
    -------
    gaze_direction : int
        Gaze direction class label
    gaze_confidence : int
        Confidence score of the gaze classification
    """
    results = gaze_model(face_image)
    
    result = results[0].probs
    # Extract detection results
    object_cls = result.top1  # Class labels (0 = no_gaze, 1 = gaze)
    conf = result.top1conf.item()  # Confidence scores
    return object_cls, conf

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

def process_frame(frame: np.ndarray, 
                 frame_idx: int, 
                 video_id: int, 
                 object_model: YOLO, 
                 person_face_model: YOLO,
                 gaze_cls_model: YOLO,
                 face_cls_model: YOLO,
                 person_cls_model: YOLO,
                 cursor: sqlite3.Cursor) -> Dict[str, int]:
    """Process a frame with object and person/face detection models.
    
    Parameters:
    ----------
    frame : np.ndarray
        The image frame to process.
    frame_idx : int
        The index of the frame in the video.
    video_id : int
        The ID of the video in the database.
    object_model : YOLO
        YOLO model for object detection.
    person_face_model : YOLO
        YOLO model for person and face detection.
    gaze_cls_model : YOLO
        YOLO model for gaze classification.
    face_cls_model : YOLO
        YOLO model for face classification.
    person_cls_model : YOLO
        YOLO model for person classification.
    cursor : sqlite3.Cursor
        SQLite cursor object.
    
    Returns:
    -------
    detection_counts : Dict[str, int]
        A dictionary with counts of detected classes.
    """
    cursor.execute('INSERT INTO Frames (video_id, frame_number) VALUES (?, ?)', (video_id, frame_idx))

    # Initialize detection counts
    detection_counts = {name: 0 for name in YoloConfig.detection_mapping.values()}
    
    # Run object detection model
    object_results = object_model(frame)
    # Process object detections (only classes 5-10)
    for r in object_results[0].boxes:
        if r.cls.item() in range(5, 11):  # Only process classes 5-10
            box = r.xyxy[0].cpu().numpy()
            cls_id = int(r.cls.item())
            conf = r.conf.item()
            
            final_cls_id = map_detection_to_final_class(cls_id, "object")
            
            cursor.execute('''
                INSERT INTO Detections
                (video_id, frame_number, object_class, confidence_score,
                x_min, y_min, x_max, y_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, frame_idx, final_cls_id, float(conf), 
                float(box[0]), float(box[1]), float(box[2]), float(box[3])))
                  
            class_name = object_model.names[cls_id]
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
    
    # Collect person and face detections first
    person_detections = []
    face_detections = []
    body_part_detections = []
    
    # First pass: collect all person/face/body part detections
    person_face_results = person_face_model(frame)
    for r in person_face_results[0].boxes:
        box = r.xyxy[0].cpu().numpy()
        cls_id = int(r.cls.item())
        conf = r.conf.item()
        
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]
        
        if cls_id == 0:  # Person
            person_cls_results = person_cls_model(roi)
            person_age_cls = int(person_cls_results[0].probs.top1)
            person_age_conf = float(person_cls_results[0].probs.top1conf)
            
            person_detections.append({
                'box': box,
                'age_cls': person_age_cls,
                'age_conf': person_age_conf,
                'conf': conf
            })
            
        elif cls_id == 1:  # Face
            face_cls_results = face_cls_model(roi)
            face_age_cls = int(face_cls_results[0].probs.top1)
            face_age_conf = float(face_cls_results[0].probs.top1conf)
            
            gaze_results = gaze_cls_model(roi)
            gaze_cls = int(gaze_results[0].probs.top1)
            gaze_conf = float(gaze_results[0].probs.top1conf)
            
            face_detections.append({
                'box': box,
                'age_cls': face_age_cls,
                'age_conf': face_age_conf,
                'gaze_cls': gaze_cls,
                'gaze_conf': gaze_conf,
                'conf': conf
            })
            
        elif cls_id == 2:  # Child body part
            body_part_detections.append({
                'box': box,
                'conf': conf
            })
    
    # Process and store person detections
    for person in person_detections:
        final_cls_id = map_detection_to_final_class(0, "person_face", person['age_cls'])
        
        cursor.execute('''
            INSERT INTO Detections
            (video_id, frame_number, object_class, confidence_score,
            x_min, y_min, x_max, y_max, age_class, age_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, final_cls_id, float(person['conf']),
            float(person['box'][0]), float(person['box'][1]), 
            float(person['box'][2]), float(person['box'][3]),
            person['age_cls'], float(person['age_conf'])))
        
        class_name = 'adult' if person['age_cls'] == 0 else 'infant/child'
        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
    
    # Process and store face detections with potential adjustments
    for face in face_detections:
        face_adjusted = False
        
        # Check if face is inside any person
        for person in person_detections:
            if is_face_inside_person(face['box'], person['box']):
                # If age classes don't match, adjust face age class
                if face['age_cls'] != person['age_cls']:
                    face['age_cls'] = person['age_cls']
                    face['age_conf'] = person['age_conf']  # Use person's confidence
                    face_adjusted = True
                break
            
        final_cls_id = map_detection_to_final_class(1, "person_face", face['age_cls'])
        class_name = 'adult face' if face['age_cls'] == 0 else 'infant/child face'
            
        # Calculate proximity after final class determination
        x1, y1, x2, y2 = face['box']
        proximity = get_proximity([x1, y1, x2, y2], class_name)
     
        cursor.execute('''
            INSERT INTO Detections
            (video_id, frame_number, object_class, confidence_score,
            x_min, y_min, x_max, y_max, age_class, age_confidence,
            gaze_direction, gaze_confidence, face_adjusted, proximity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, final_cls_id, float(face['conf']),
            float(x1), float(y1), float(x2), float(y2),
            face['age_cls'], float(face['age_conf']),
            face['gaze_cls'], float(face['gaze_conf']), 
            face_adjusted, float(proximity)))
        
        detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
    
    # Process and store child body part detections
    for body_part in body_part_detections:
        final_cls_id = map_detection_to_final_class(2, "person_face")  # Child body part is always class 4
        
        cursor.execute('''
            INSERT INTO Detections
            (video_id, frame_number, object_class, confidence_score,
            x_min, y_min, x_max, y_max)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (video_id, frame_idx, final_cls_id, float(body_part['conf']),
            float(body_part['box'][0]), float(body_part['box'][1]), 
            float(body_part['box'][2]), float(body_part['box'][3])))
        
        detection_counts['child body parts'] = detection_counts.get('child body parts', 0) + 1

    return detection_counts

def process_video(video_folder: Path, 
                  object_model: YOLO, 
                  person_face_model: YOLO,
                  gaze_cls_model: YOLO, 
                  face_cls_model: YOLO,
                  person_cls_model: YOLO,
                  cursor: sqlite3.Cursor, 
                  conn: sqlite3.Connection,
                  frame_skip: int):
    """
    Process frames from saved image files in a video folder.
    
    Parameters:
    ----------
    video_folder : Path
        Path to the video folder
    object_model : YOLO
        YOLO model for object detection
    person_face_model : YOLO
        YOLO model for person and face detection
    gaze_cls_model : YOLO
        YOLO model for gaze classification
    face_cls_model : YOLO
        YOLO model for face classification
    person_cls_model : YOLO
        YOLO model for person classification
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
    # Initialize total counts for each class
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
            frame, frame_idx, video_id, object_model, person_face_model, gaze_cls_model, face_cls_model, person_cls_model, cursor
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

def load_rttm_results(rttm_file: Path) -> pd.DataFrame:
    """
    Loads and parses RTTM file into a DataFrame.
    
    Parameters:
    ----------
    rttm_file : Path
        Path to the RTTM file containing all voice detection results
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with columns [audio_id, start_time, duration, label]
    """
    columns = ['type', 'audio_id', 'channel', 'start_time', 
               'duration', 'na1', 'na2', 'label', 'na3', 'na4']
    
    rttm_data = []
    with open(rttm_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                rttm_data.append(parts)
                
    df = pd.DataFrame(rttm_data, columns=columns)
    df['start_time'] = df['start_time'].astype(float)
    df['duration'] = df['duration'].astype(float)
    
    output_df = df[['audio_id', 'start_time', 'duration', 'label']]
    
    # save to CSV for future use
    output_df.to_pickle(VTCPaths.quantex_output_dir / 'vtc_quantex_audio_results.pkl')
    logging.info(f"Saved RTTM results to {VTCPaths.quantex_output_dir / 'vtc_quantex_audio_results.pkl'}")
    return output_df
 
def store_voice_detections(video_file_name: str, 
                           rttm_df: pd.DataFrame, 
                           fps: int = VideoConfig.fps):
    """
    Stores voice detections for a specific video from pre-loaded RTTM results.
    
    Parameters:
    ----------
    video_file_name : str
        The video file name used as the video_path in the Videos table.
    rttm_df : pd.DataFrame
        DataFrame containing all RTTM results with columns:
        [audio_id, start_time, duration, label]
    fps : int
        Frames per second of the video (default: 30)
    """
    # Get or create voice type classifier model ID
    vtc_model_id = register_model(cursor, "voice_type_classifier")
    
    # Insert video record if it does not exist
    cursor.execute("SELECT video_id FROM Videos WHERE video_path = ?", (video_file_name,))
    result = cursor.fetchone()
    if result:
        video_id = result[0]
    else:
        logging.error(f"Video {video_file_name} not found in Videos table")
        return
    
    # Filter RTTM results for this video
    video_rttm = rttm_df[rttm_df['audio_id'] == video_file_name]
    
    for _, row in video_rttm.iterrows():
        start_time = float(row['start_time'])
        duration = float(row['duration'])
        class_name = row['label']
        
        # Register voice class if not already present
        cursor.execute("""
            INSERT OR IGNORE INTO Classes (model_id, class_name) 
            VALUES (?, ?)
        """, (vtc_model_id, class_name))
        
        # Get class_id
        cursor.execute("SELECT class_id FROM Classes WHERE model_id = ? AND class_name = ?", 
                      (vtc_model_id, class_name))
        class_id = cursor.fetchone()[0]
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        num_frames = int(duration * fps)
        
        for frame_offset in range(num_frames):
            actual_frame = start_frame + frame_offset
            # Insert frame record if it doesn't exist
            cursor.execute('INSERT OR IGNORE INTO Frames (video_id, frame_number) VALUES (?, ?)', 
                         (video_id, actual_frame))
            
            # Insert detection record
            cursor.execute('''
                INSERT INTO Detections 
                (video_id, frame_number, object_class, confidence_score, 
                 x_min, y_min, x_max, y_max)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, actual_frame, class_id, None, 
                  None, None, None, None))
    
    logging.info(f"Stored voice detections for video {video_file_name}")
    
def run_voice_type_classifier(video_file_name):
    """
    This function runs the voice type classifier on the given video.
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file
    """
    # get corresponding audio file
    audio_file_name = video_file_name.replace(".MP4", "_16kHz.wav")
    audio_file_path = VTCPaths.quantex_output_folder / audio_file_name
     # Run voice type classifier using apply.sh.
    # This command changes to the 'projects/voice-type-classifier' directory, activates the pyannote conda environment,
    # and executes the apply.sh script.
    voice_command = (
        "cd /home/nele_pauline_suffo/projects/voice-type-classifier && " 
        f"conda run -n pyannote ./apply.sh {audio_file_path} --device=gpu"
    )
    subprocess.run(voice_command, shell=True, check=True)
    
    # Assume the classifier writes its output to an 'all.rttm' file in the designated results directory.
    vtc_results_dir = VTCPaths.vtc_results_dir
    rttm_file = vtc_results_dir / video_file_name / "all.rttm"
    
    if rttm_file.exists():
        store_voice_detections(video_file_name, rttm_file)
    else:
        logging.error(f"RTTM results file not found: {rttm_file}")
 
def register_model(cursor: sqlite3.Cursor, model_name: str, model: Optional[YOLO] = None) -> int:
    """
    Registers a model and its classes in the database.
    
    Parameters:
    ----------
    cursor : sqlite3.Cursor
        The SQLite database cursor.
    model_name : str
        The name of the model.
    model : Optional[YOLO]
        The YOLO model instance which contains a `model.names` mapping.
        If None, just registers the model without classes.
        
    Returns:
    -------
    model_id : int
        The model_id assigned in the database.
    """
    # Insert model into Models table if not exists
    cursor.execute("INSERT OR IGNORE INTO Models (model_name) VALUES (?)", (model_name,))
    cursor.execute("SELECT model_id FROM Models WHERE model_name = ?", (model_name,))
    model_id = cursor.fetchone()[0]
    
    # If model object provided, register its classes
    if model is not None:
        for class_id, class_name in model.model.names.items():
            cursor.execute('''
                INSERT OR IGNORE INTO Classes (class_id, model_id, class_name)
                VALUES (?, ?, ?)
            ''', (class_id, model_id, class_name))
    
    return model_id
          
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
    
    # Load the age group data and rttm file once
    age_df = pd.read_csv('/home/nele_pauline_suffo/ProcessedData/age_group.csv')
    rttm_file = VTCPaths.quantex_output_dir / "all.rttm"
    if rttm_file.exists():
        rttm_df = load_rttm_results(rttm_file)
        logging.info(f"Loaded voice detection results for {rttm_df['audio_id'].nunique()} videos")
    else:
        rttm_df = None
        logging.warning("No voice detection results found")
    
    videos_input_dir = DetectionPaths.images_input_dir
    conn = sqlite3.connect(DetectionPaths.detection_db_path)
    cursor = conn.cursor()
    
    # Initialize models once
    object_model = YOLO(YoloPaths.all_trained_weights_path)
    person_face_model = YOLO(YoloPaths.person_face_trained_weights_path)
    gaze_cls_model = YOLO(YoloPaths.gaze_trained_weights_path)   
    face_cls_model = YOLO(YoloPaths.face_trained_weights_path)
    person_cls_model = YOLO(YoloPaths.person_trained_weights_path)
    
    register_model(cursor, "object", object_model)
    register_model(cursor, "person_face", person_face_model)
    register_model(cursor, "gaze_cls", gaze_cls_model)
    register_model(cursor, "face_cls", face_cls_model)
    register_model(cursor, "person_cls", person_cls_model)

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
        process_video(video, 
                      object_model, 
                      person_face_model,
                      gaze_cls_model, 
                      face_cls_model,
                      person_cls_model,
                      cursor, 
                      conn, 
                      frame_skip)
        if rttm_df is not None:
            store_voice_detections(video.name, rttm_df, cursor)
    
    conn.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process a set of videos using YOLO models for person and face detection and gaze classification.")
    argparser.add_argument("--num_videos", type=int, help="Number of videos to process")
    argparse.add_argument("--frame_skip", type=int, default=10, help="Number of frames to skip between processing (default: 10)")
    args = argparser.parse_args()
    num_videos_to_process = args.num_videos
    frame_skip = args.frame_skip
    main(num_videos_to_process, frame_skip)