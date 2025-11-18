import logging
import argparse
import run_face_proximity
import populate_speech_db as run_speech_type
import run_person
from pathlib import Path
from setup_interaction_db import main as setup_interaction_db
from constants import DataPaths, Inference
from config import InferenceConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(video_path: Path, db_path: Path, frame_step: int, models: list = None):
    """
    This function runs the detection pipeline. It first sets up the detection database and then runs the detection pipeline.
    
    Parameters:
    ----------
    video_path : Path
        Path to a single video file, a directory containing video files, OR a .txt file 
        containing a list of video filenames.
    db_path : Path
        Path to the database where results will be stored
    frame_step : int
        The step size for processing frames in the videos, default is 10
    models : list, optional
        List of models to run. Options: 'person', 'face_proximity', 'speech_type', 'all'
        If None or 'all', all models will be run.
        
    Returns:
    -------
    bool
        True if the inference completed successfully, False otherwise
    """
    try:
        # Setup the detection database which will hold the detection results (if it doesnt already exist)
        setup_interaction_db(db_path = db_path)

        # Ensure video_path is a Path object
        video_path = Path(video_path)
        selected_videos = []
        
        # --- PATH CHECK LOGIC ADJUSTMENT ---
        if video_path.is_file():
            if video_path.suffix.lower() == '.txt':
                # Case 1: .txt file provided
                logging.info(f"Processing list of videos from text file: {video_path.name}")
                with open(video_path, 'r') as f:
                    # Read lines, strip whitespace, and filter out empty lines
                    selected_videos = [line.strip() for line in f if line.strip()]
                
                if not selected_videos:
                    logging.error(f"Text file is empty or contains no video names: {video_path}")
                    return False
                logging.info(f"Loaded {len(selected_videos)} video names from the list.")

            elif video_path.suffix.lower() in video_extensions:
                # Case 2: Single video file processing
                logging.info(f"Processing single video file: {video_path.name}")
                selected_videos = [video_path.stem]  # Use stem (filename without extension)
                
            else:
                logging.error(f"Unsupported file extension: {video_path.suffix}")
                return False
            
        elif video_path.is_dir():
            # Case 3: Directory processing - find all video files
            video_files = [f for f in video_path.iterdir() 
                        if f.is_file() and f.suffix in video_extensions]
            
            if not video_files:
                logging.error(f"No video files found in directory: {video_path}")
                logging.info(f"Looking for extensions: {', '.join(sorted(video_extensions))}")
                return False
            
            logging.info(f"Processing {len(video_files)} video files from directory: {video_path}")
            selected_videos = [f.stem for f in video_files]  # Use stems (filenames without extensions)
            
        else:
            logging.error(f"Path is neither a file, a directory, nor a valid text list: {video_path}")
            return False
        
        # Determine which models to run
        if models is None or 'all' in models:
            models_to_run = ['person', 'face_proximity', 'speech_type']
        else:
            models_to_run = models
        
        # Run selected models
        if 'person' in models_to_run:
            logging.info("Running person detection model")
            run_person.main(selected_videos)
        
        if 'face_proximity' in models_to_run:
            logging.info("Running face model with proximity heuristic")
            run_face_proximity.main(selected_videos, frame_step)
        
        if 'speech_type' in models_to_run:
            logging.info("Running speech type classification model")
            run_speech_type.main(selected_videos)
        
        logging.info("Inference pipeline completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error during inference pipeline: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline with selectable models")

    parser.add_argument("--video_path", type=Path, default=Inference.QUANTEX_VIDEOS_LIST_FILE, help="Path to video file or directory containing videos")
    parser.add_argument("--db_path", type=Path, default=DataPaths.INFERENCE_DB_PATH, help="Path to the database where results will be stored")
    parser.add_argument("--frame_step", type=int, default=InferenceConfig.SAMPLE_RATE, help="Frame step size for processing videos")
    parser.add_argument("--models", nargs='+', choices=['person', 'face_proximity', 'speech_type', 'all'], default=['all'],  help="Select which models to run. Options: person, face_proximity, speech_type, all")
    args = parser.parse_args()
    
    main(
        video_path=args.video_path, 
        models=args.models,
        db_path=args.db_path, 
        frame_step=args.frame_step,
    )