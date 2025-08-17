import logging
import sqlite3
import re
from pathlib import Path
from typing import Callable
from tqdm import tqdm
from constants import DataPaths
from config import DataConfig

def get_video_id(video_name: str, cursor: sqlite3.Cursor) -> int:
    """Get video_id from Videos table using video name"""
    cursor.execute('SELECT video_id FROM Videos WHERE video_name = ?', (video_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        logging.error(f"Video {video_name} not found in Videos table")
        return None
    
def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename using regex"""
    # Try to extract numbers from filename (e.g., frame_001.jpg -> 1, video_name_0042.jpg -> 42)
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Use last number found
    else:
        return 0    

def get_frame_paths(video_frame_dir: Path):
    """
    Returns a sorted list of frame paths in video_frame_dir
    for all valid extensions defined in DataConfig.VALID_EXTENSIONS.
    """
    frame_paths = []
    for ext in DataConfig.VALID_EXTENSIONS:
        frame_paths.extend(video_frame_dir.glob(f"*{ext}"))

    # Sort by the numeric part at the end of the filename
    frame_paths = sorted(
        frame_paths,
        key=lambda x: int(x.stem.split('_')[-1])
    )
    return frame_paths
  
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
    logging.info(f"Processed {processed_frames} frames, metric total: {metric_sum}")  