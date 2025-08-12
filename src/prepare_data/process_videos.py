import logging
from utils import extract_every_nth_frame_from_videos_in_folder 
from constants import DataPaths
from config import DataConfig
from utils import extract_audio_from_videos_in_folder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None: 
    logging.info("Starting to extract frames from videos.")
    # Extract frames from video
    extract_every_nth_frame_from_videos_in_folder(DataPaths.VIDEOS_INPUT_DIR, 
                                                  DetectionPaths.images_input_dir, 
                                                  DataConfig.FRAME_STEP_INTERVAL, 
                                                  DataPaths.RAWFRAMES_EXTRACTION_ERROR_LOG,
                                                  DataPaths.PROCESSED_VIDEOS_LOG)
    # extract_every_nth_frame_from_videos_in_folder(DetectionPaths.childlens_videos_input_dir, 
    #                                               DetectionPaths.childlens_images_input_dir, DetectionParameters.frame_step_interval, 
    #                                               VideoParameters.childlens_rawframes_extraction_error_log,
    #                                               VideoParameters.childlens_processed_videos_log)
    logging.info("Finished extracting frames from videos.")
    
    logging.info("Starting to extract audio from videos.")
    #Extract audio from video
    #extract_audio_from_videos_in_folder(DetectionPaths.quantex_videos_input_dir, VTCPaths.quantex_audio_dir)
    logging.info("Finished extracting audio from videos.")
    
if __name__ == "__main__":
    main()