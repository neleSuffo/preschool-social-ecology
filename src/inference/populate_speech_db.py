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

def main():
    """Main: parse RTTM and insert snippets for each file_id into the DB.

    The RTTM file is expected at AudioClassification.VTC_RTTM_FILE. For each
    distinct fileID (first field after SPEAKER), the script will attempt to
    map that fileID to a video_id in the DB via get_video_id(), then insert
    per-frame AudioClassifications for each snippet.
    """
    rttm_path = Path(AudioClassification.VTC_RTTM_FILE)
    snippets_by_file = parse_rttm_file_to_snippets(rttm_path)

    if not snippets_by_file:
        logging.info("No snippets found in RTTM file. Nothing to process.")
        return

    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()

    for file_id, snippets in snippets_by_file.items():
        # Try mapping RTTM fileID variants to DB video name (strip extensions if present)
        candidates = [file_id, file_id.replace('.wav', ''), file_id.replace('.mp4', ''), file_id.replace('.MP4', '')]
        video_id = None
        for cand in candidates:
            video_id = get_video_id(cand, cursor)
            if video_id is not None:
                break

        if video_id is None:
            logging.warning(f"Video ID not found for RTTM fileID '{file_id}'. Skipping {len(snippets)} snippets.")
            continue

        try:
            aggregate_and_save_results(snippets, cursor, video_id)
            conn.commit()
            logging.info(f"Inserted {len(snippets)} snippets for video_id {video_id} (RTTM fileID: {file_id})")
        except Exception as e:
            logging.error(f"Error inserting snippets for {file_id}: {e}")
            conn.rollback()

    conn.close()
    logging.info("RTTM-based audio classification processing completed")

if __name__ == "__main__":
    main()