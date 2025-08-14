import argparse
import logging
import sqlite3
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
#import run_pipeline, run_person, run_face_proximity, run_audio_classification
from pathlib import Path
from ultralytics import YOLO
from setup_detection_database import setup_detection_database
from constants import DataPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_balanced_videos(videos_per_group: int) -> list:
    """
    Select a balanced number of videos from each age group using the CSV data.
    
    Parameters:
    ----------
    videos_per_group : int
        Number of videos to select from each age group
        
    Returns:
    -------
    list
        List of selected video names
    """
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
    selected_videos = []
    
    # Group videos by age group
    videos_by_age = {3: [], 4: [], 5: []}
    
    for _, row in age_df.iterrows():
        video_name = row['video_name']
        age_group = row['age_group']
        
        if age_group in videos_by_age:
            videos_by_age[age_group].append(video_name)
    
    # Log available videos per age group
    for age_group, videos in videos_by_age.items():
        logging.info(f"Age group {age_group}: {len(videos)} videos available")
    
    # Select balanced number of videos from each group
    for age_group in videos_by_age:
        videos = videos_by_age[age_group]
        if len(videos) >= videos_per_group:
            selected = np.random.choice(videos, size=videos_per_group, replace=False)
            selected_videos.extend(selected)
        else:
            logging.warning(f"Age group {age_group} has only {len(videos)} videos, using all available")
            selected_videos.extend(videos)
    logging.info(f"Selected {len(selected_videos)} videos across all age groups")
    
    return selected_videos

def main(videos_per_group: int = None):
    """
    This function runs the detection pipeline. It first sets up the detection database and then runs the detection pipeline.
    
    Parameters:
    ----------
    num_videos_to_process: int
        The number of videos
    """
    # Setup the detection database which will hold the detection results (if it doesnt already exist)
    setup_detection_database()
    
    # Select videos to process
    selected_videos = get_balanced_videos(videos_per_group)
    if not selected_videos:
        logging.error("No videos selected for processing. Exiting.")
        return
    
    # Run the detection pipeline
    #run_person.main(selected_videos)
    #run_face_proximity.main(selected_videos)
    #run_audio_classification.main(selected_videos)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the detection pipeline')
    parser.add_argument('--num_videos_per_age', type=int, help='The number of videos to process per age group. If not specified, processes all videos.', default=None)
    args = parser.parse_args()
    main(args.num_videos_per_age)