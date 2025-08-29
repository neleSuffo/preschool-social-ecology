import argparse
import logging
import pandas as pd
import numpy as np
import run_face_proximity
from pathlib import Path
from setup_interaction_db import setup_interaction_db
from constants import DataPaths
from config import InferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_balanced_videos(videos_per_group: int) -> list:
    """
    Select a balanced number of videos from each age group using the CSV data.
    
    Parameters:
    ----------
    videos_per_group : int
        Number of videos to select from each age group (videos only contain video_names without extension)
        
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
    logging.info("Available videos per age group: " + ", ".join(f"{age_group}: {len(videos)} videos" for age_group, videos in videos_by_age.items()))
    
    # Select balanced number of videos from each group
    for age_group in videos_by_age:
        videos = videos_by_age[age_group]
        if len(videos) >= videos_per_group:
            selected = np.random.choice(videos, size=videos_per_group, replace=False)
            selected_videos.extend(selected)
        else:
            logging.warning(f"Age group {age_group} has only {len(videos)} videos, using all available")
            selected_videos.extend(videos)
    
    return selected_videos

def main(video_path: Path, db_path: Path, frame_step: int = InferenceConfig.SAMPLE_RATE, num_videos_per_age: int = None):
    """
    This function runs the detection pipeline. It first sets up the detection database and then runs the detection pipeline.
    
    Parameters:
    ----------
    video_path : Path
        Path to the video file to process
    db_path : Path
        Path to the database where results will be stored
    frame_step : int
        The step size for processing frames in the videos, default is 10
    num_videos_per_age : int
        The number of videos to process per age group. If not specified, processes all videos.
        If specified, it will select a balanced number of videos from each age group.
        
    Returns:
    -------
    bool
        True if the inference completed successfully, False otherwise
    """
    try:
        # Setup the detection database which will hold the detection results (if it doesnt already exist)
        setup_interaction_db(db_path = db_path)
        
        # Select videos to process
        if num_videos_per_age is None:
            # Process all videos from CSV
            age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
            selected_videos = age_df['video_name'].tolist()
            logging.info(f"Processing all {len(selected_videos)} videos from CSV")
        else:
            # Process balanced selection from each age group
            selected_videos = get_balanced_videos(num_videos_per_age)
            logging.info(f"Processing {len(selected_videos)} selected videos ({num_videos_per_age} per age group)")

        # Run the detection pipeline
        #run_person.main(selected_videos)
        run_face_proximity.main(selected_videos, frame_step)
        #run_audio_classification.main(selected_videos)
        
        logging.info("Inference pipeline completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error during inference pipeline: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the detection pipeline')
    parser.add_argument('--frame_step', type=int, default=10, help='Frame step for processing videos (default: 10)')
    parser.add_argument('--num_videos_per_age', type=int, help='The number of videos to process per age group. If not specified, processes all videos.', default=None)
    args = parser.parse_args()
    main(args.frame_step, args.num_videos_per_age)