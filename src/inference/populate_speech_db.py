import argparse
import numpy as np
import sqlite3
import logging
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from constants import AudioClassification, DataPaths, Inference
from config import AudioConfig, DataConfig
from utils import get_video_id, load_processed_videos, save_processed_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_PATH = Inference.SPEECH_LOG_FILE_PATH

def parse_rttm_snippets_with_label(rttm_path: Path, video_name: str) -> List[Dict]:
    """
    Parses a single large RTTM file, extracts speech segments (start, duration, label),
    and filters results for the specified video_name (fileID).
    
    Parameters
    ----------
    rttm_path : Path
        Path to the RTTM file containing VTC output (KCHI, KCDS, OHS labels).
    video_name : str
        Name of the video to filter snippets for (without extension).
    
    Returns
    -------
    snippets : List[Dict]
        List of dictionaries containing 'start_time', 'duration', and 'label'.
    """
    snippets = []
    
    # Patterns to match the video name exactly in the RTTM's fileID field (parts[1])
    match_patterns = {video_name, f"{video_name}.wav", f"{video_name}.mp4", f"{video_name}.MP4"}

    if not rttm_path.exists():
        logging.warning(f"Single large RTTM file not found: {rttm_path}.")
        return snippets
        
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Check for SPEAKER tag and minimum parts (9 for the label)
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                rttm_file_id = parts[1] # This is the video identifier
                label = parts[7]        # The classification label (e.g., KCDS)
                
                # Filter by video name and check if the label is one of the targets
                if rttm_file_id in match_patterns and label in AudioConfig.VALID_RTTM_CLASSES:
                    try:
                        start = float(parts[3])
                        duration = float(parts[4])
                        if duration > 0:
                            snippets.append({
                                'start_time': start, 
                                'duration': duration, 
                                'label': label
                            })
                    except ValueError:
                        logging.warning(f"Skipping malformed RTTM line: {line.strip()}")
    return snippets


def aggregate_and_save_results(snippets: List[Dict], db_cursor: sqlite3.Cursor, video_id: int):
    """
    Saves classification results to the database by applying the RTTM label 
    across its entire duration with a confidence score of 1.0 for that class.
    
    Parameters
    ----------
    snippets : List[Dict]
        List of dictionaries containing start_time, duration, and label.
    db_cursor : sqlite3.Cursor
        Database cursor for saving results.
    video_id : int
        ID of the video in the database.
    """
    
    # Iterate over each classified snippet
    for snippet in snippets:
        start_sec = snippet['start_time']
        duration = snippet['duration']
        label = snippet['label']
        end_sec = start_sec + duration
        
        # Initialize scores for all classes to 0
        scores = {cls: 0 for cls in AudioConfig.VALID_RTTM_CLASSES}
        
        # Set the score for the detected class to 1.0 (perfect confidence from RTTM)
        if label in scores:
            scores[label] = 1.0
        else:
            logging.warning(f"Skipping snippet with unexpected label: {label}")
            continue
        
        # Calculate the frame range covered by this snippet
        start_frame = int(np.floor(start_sec * DataConfig.FPS))
        end_frame = int(np.ceil(end_sec * DataConfig.FPS))

        # Find the first frame that is a multiple of FRAME_STEP_INTERVAL and >= start_frame
        first_frame = ((start_frame + DataConfig.FRAME_STEP_INTERVAL - 1) // DataConfig.FRAME_STEP_INTERVAL) * DataConfig.FRAME_STEP_INTERVAL

        # Insert records for all frames covered by this snippet
        for frame_number in range(first_frame, end_frame, DataConfig.FRAME_STEP_INTERVAL):
            db_cursor.execute("""
                INSERT INTO AudioClassifications (
                    video_id, frame_number, model_id,
                    has_kchi, kchi_confidence_score,
                    has_cds, cds_confidence_score,
                    has_ohs, ohs_confidence_score) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id, 
                frame_number, 
                MODEL_ID,  # Using the fixed Model ID
                int(scores[AudioConfig.VALID_RTTM_CLASSES[2]]),
                scores[AudioConfig.VALID_RTTM_CLASSES[2]],
                int(scores[AudioConfig.VALID_RTTM_CLASSES[1]]), 
                scores[AudioConfig.VALID_RTTM_CLASSES[1]],
                int(scores[AudioConfig.VALID_RTTM_CLASSES[0]]), 
                scores[AudioConfig.VALID_RTTM_CLASSES[0]]
            ))


def process_audio_file(video_name: str, cursor: sqlite3.Cursor):
    """
    Process a single video by parsing the VTC RTTM file and saving results to database.
    
    Parameters:
    ----------
    video_name : str
        Name of the video (without extension)
    cursor : sqlite3.Cursor
        Database cursor for saving results
    """
    logging.info(f"Processing VTC RTTM data for video: {video_name}")
    
    video_id = get_video_id(video_name, cursor)
    
    if video_id is None:
        logging.error(f"Video ID not found for {video_name}")
        return
    
    try:
        # 1. Parse RTTM file to get snippet boundaries and labels
        snippets_with_labels = parse_rttm_snippets_with_label(AudioClassification.VTC_RTTM_FILE, video_name)
        
        if not snippets_with_labels:
            logging.warning(f"No classified speech snippets found in VTC RTTM for {video_name}. Skipping.")
            return

        # 2. Save the snippet's RTTM classification across its entire frame range
        aggregate_and_save_results(snippets_with_labels, cursor, video_id)
        
    except Exception as e:
        logging.error(f"❌ Error processing RTTM for {video_name}: {e}")

def main():
    """
    Main function to process videos for audio classification
    """
    # Load the log file as the list of videos to add to the database
    videos_to_process = load_processed_videos(LOG_FILE_PATH)
    if not videos_to_process:
        logging.info("No videos found in log file to process!")
        return

    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()

    # Process each video
    for video_name in videos_to_process:
        try:
            process_audio_file(video_name, cursor)
            conn.commit()  # Commit after each video
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            conn.rollback() # Rollback if error occurred before commit
            continue

    conn.close()
    logging.info("RTTM-based audio classification processing completed")

if __name__ == "__main__":
    main()