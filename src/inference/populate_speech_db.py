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

def parse_rttm_file_to_snippets(rttm_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse the RTTM file and return a mapping from file_id (RTTM fileID) to
    a list of snippet dicts {start_time, duration, label}.
    
    Parameters
    ----------
    rttm_path : Path
        Path to the RTTM file containing VTC output (KCHI, KCDS, OHS labels).
    
    Returns
    -------
    snippets_by_file : Dict[str, List[Dict]]
        Mapping from file_id to list of snippet dictionaries.
    """
    snippets_by_file = {}

    if not Path(rttm_path).exists():
        logging.error(f"RTTM file not found: {rttm_path}")
        return snippets_by_file

    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8 and parts[0] == 'SPEAKER':
                file_id = parts[1]
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                except ValueError:
                    logging.warning(f"Skipping malformed RTTM line: {line.strip()}")
                    continue
                label = parts[7]
                if label not in AudioConfig.VALID_RTTM_CLASSES:
                    # skip labels we don't care about
                    continue
                if duration <= 0:
                    continue
                snippets_by_file.setdefault(file_id, []).append({
                    'start_time': start,
                    'duration': duration,
                    'label': label
                })

    return snippets_by_file


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
                AudioConfig.MODEL_ID,  # Using the fixed Model ID
                int(scores[AudioConfig.VALID_RTTM_CLASSES[2]]),
                scores[AudioConfig.VALID_RTTM_CLASSES[2]],
                int(scores[AudioConfig.VALID_RTTM_CLASSES[1]]), 
                scores[AudioConfig.VALID_RTTM_CLASSES[1]],
                int(scores[AudioConfig.VALID_RTTM_CLASSES[0]]), 
                scores[AudioConfig.VALID_RTTM_CLASSES[0]]
            ))

def main(selected_videos: List[str]):
    """
    Main function to process RTTM data for a list of selected videos.
    
    Parameters
    ----------
    selected_videos : List[str]
        List of video stems (filenames without extension) to process.
    """
    rttm_path = Path(AudioClassification.VTC_RTTM_FILE)
    all_snippets_by_file = parse_rttm_file_to_snippets(rttm_path)

    if not all_snippets_by_file:
        logging.info("No snippets found in RTTM file. Nothing to process.")
        return

    # Create a set of base file stems (without RTTM's file extensions) for quick lookup
    selected_stems = set(selected_videos)
    
    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    processed_count = 0

    # Iterate over all RTTM file_ids, but only process if they are in the selected_stems set
    for rttm_file_id, snippets in all_snippets_by_file.items():
        
        # Determine the base stem of the RTTM file_id (e.g., '387058' from '387058.MP4')
        base_stem = rttm_file_id.split('.')[0]
        
        if base_stem not in selected_stems:
            continue # Skip files not selected by the pipeline
            
        # Try mapping the base stem to a video_id in the DB
        video_id = get_video_id(base_stem, cursor)

        if video_id is None:
            logging.warning(f"Video ID not found for base stem '{base_stem}' (RTTM ID: {rttm_file_id}). Skipping.")
            continue

        try:
            aggregate_and_save_results(snippets, cursor, video_id)
            conn.commit()
            logging.info(f"Inserted {len(snippets)} snippets for video_id {video_id} (Stem: {base_stem})")
            processed_count += 1
        except Exception as e:
            logging.error(f"Error inserting snippets for {base_stem}: {e}")
            conn.rollback()

    conn.close()
    logging.info(f"RTTM-based speech classification processing completed for {processed_count} videos.")

if __name__ == "__main__":
    main()