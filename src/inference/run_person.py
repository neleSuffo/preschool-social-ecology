import cv2
import logging
import argparse
import sqlite3
from typing import List, Callable
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from constants import DataPaths, Inference, PersonClassification
from config import PersonConfig
from utils import get_video_id, get_frame_paths, extract_frame_number, load_processed_videos, save_processed_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_PATH = Inference.PERSON_LOG_FILE_PATH

def process_video(
    video_name: str,
    frame_step: int,
    model: YOLO,
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    process_frame_func: Callable
):
    """
    Process a video for person detection
    
    Parameters
    ----------
    video_name: str
        Name of the video.
    frame_step: int
        Step size for frame processing.
    model: YOLO
        Model to use
    cursor: sqlite3.Cursor
        SQLite cursor.
    conn: sqlite3.Connection
        SQLite connection.
    process_frame_func: Callable[[Path, int, int, YOLO, sqlite3.Cursor], int]
        Function that processes a single frame and returns an int (number of faces detected).
    """
    logging.info(f"Processing video: {video_name}")

    # Get video_id
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        return

    # Frames directory
    frames_dir = DataPaths.QUANTEX_IMAGES_INPUT_DIR / video_name
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
    with tqdm(frames_to_process, desc=f"Processing {video_name}", unit="frames") as pbar:
        for frame_file, frame_number in pbar:
            # Process frame
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
    
    person_count = 0
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                # Extract detection info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Insert into PersonClassifications table
                cursor.execute('''
                    INSERT INTO PersonClassifications 
                    (video_id, frame_number, model_id, confidence_score, age_class)
                    VALUES (?, ?, ?, ?, ?)
                ''', (video_id, frame_number, PersonConfig.MODEL_ID, float(confidence), "adult"))
                
                person_count += 1

    return person_count

def main(video_list: List[str], frame_step: int = 10):
    """
    Main function to process videos for person detection
    
    Parameters:
    ----------
    video_list : List[str]
        List of video names to process
    frame_step : int
        Step size for frame processing (default: 10)
    """
    # Setup processing log file
    processed_videos = load_processed_videos(LOG_FILE_PATH)
    
    # Filter out already processed videos
    videos_to_process = [v for v in video_list if v not in processed_videos]
    skipped_videos = [v for v in video_list if v in processed_videos]
    
    if not videos_to_process:
        logging.info("All requested videos have already been processed!")
        return
    
    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    # Load YOLO face detection model
    model = YOLO(PersonClassification.TRAINED_WEIGHTS_PATH)
    
    # Process each video
    for video_name in video_list:
        try:
            process_video(video_name, frame_step, model, cursor, conn, process_frame_func=process_frame)
            save_processed_video(LOG_FILE_PATH, video_name)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()
    logging.info("Person detection processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO person detection on extracted frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                    help="List of video names to process")
    parser.add_argument("--frame_step", type=int, default=10, 
                    help="Frame step for processing (default: 10)")
    parser.add_argument("--force", action='store_true',
                    help="Force reprocessing of already processed videos")
    args = parser.parse_args()
    
    # Handle force reprocessing
    if args.force:
        logging.info("Force flag enabled - will reprocess all videos")
        if LOG_FILE_PATH.exists():
            # Create backup of current log
            backup_path = LOG_FILE_PATH.with_suffix('.txt.backup')
            import shutil
            shutil.copy2(LOG_FILE_PATH, backup_path)
            LOG_FILE_PATH.unlink()  # Remove current log
            
    main(args.video_list, args.frame_step)