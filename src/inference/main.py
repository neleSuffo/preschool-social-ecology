import argparse
import logging
import run_face_proximity
from pathlib import Path
from setup_interaction_db import setup_interaction_db
from constants import DataPaths
from config import InferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(video_path: Path , db_path: Path, frame_step: int):
    """
    This function runs the detection pipeline. It first sets up the detection database and then runs the detection pipeline.
    
    Parameters:
    ----------
    video_path : Path
        Path to either a single video file or a directory containing video files
    db_path : Path
        Path to the database where results will be stored
    frame_step : int
        The step size for processing frames in the videos, default is 10
        
    Returns:
    -------
    bool
        True if the inference completed successfully, False otherwise
    """
    try:
        # Setup the detection database which will hold the detection results (if it doesnt already exist)
        setup_interaction_db(db_path = db_path, video_path = video_path)

        # Check if video_path is a file or directory
        video_path = Path(video_path)
        
        if video_path.is_file():
            # Single video file processing
            if not video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.MP4']:
                logging.error(f"Unsupported video file extension: {video_path.suffix}")
                return False
            
            logging.info(f"Processing single video file: {video_path.name}")
            selected_videos = [video_path.stem]  # Use stem (filename without extension)
            
        elif video_path.is_dir():
            # Directory processing - find all video files
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}
            video_files = [f for f in video_path.iterdir() 
                        if f.is_file() and f.suffix in video_extensions]
            
            if not video_files:
                logging.error(f"No video files found in directory: {video_path}")
                logging.info(f"Looking for extensions: {', '.join(sorted(video_extensions))}")
                return False
            
            logging.info(f"Processing {len(video_files)} video files from directory: {video_path}")
            selected_videos = [f.stem for f in video_files]  # Use stems (filenames without extensions)
            
        else:
            logging.error(f"Path is neither a file nor a directory: {video_path}")
            return False

        # Run the detection pipeline
        logging.info(f"Starting detection pipeline for {len(selected_videos)} videos")
        #run_person.main(selected_videos)
        run_face_proximity.main(selected_videos, frame_step)
        #run_audio_classification.main(selected_videos)
        
        logging.info("Inference pipeline completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error during inference pipeline: {e}")
        return False

if __name__ == "__main__":
    main(video_path=DataPaths.VIDEOS_INPUT_DIR, db_path=DataPaths.INFERENCE_DB_PATH, frame_step=InferenceConfig.SAMPLE_RATE)