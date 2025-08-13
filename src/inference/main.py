import subprocess
import argparse
import run_pipeline
from setup_detection_database import setup_detection_database

def main(num_videos_to_process: int = None):
    """
    This function runs the detection pipeline. It first sets up the detection database and then runs the detection pipeline.
    
    Parameters:
    ----------
    num_videos_to_process: int
        The number of videos
    """
    # Setup the detection database which will hold the detection results (if it doesnt already exist)
    setup_detection_database()
    # Run the detection pipeline
    run_pipeline.main(num_videos_to_process)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the detection pipeline')
    parser.add_argument('--num_videos', type=int, help='The number of videos to process. If not specified, processes all videos.', default=None)
    args = parser.parse_args()
    main(args.num_videos)