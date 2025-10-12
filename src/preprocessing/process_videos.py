import logging
import argparse
from pathlib import Path

from utils import extract_every_nth_frame_from_videos_in_folder, extract_audio_from_videos_in_folder
from constants import DataPaths, AudioClassification
from config import DataConfig


def main(dataset: str) -> None:
    """Main function to extract frames and audio for the selected dataset(s)."""
    logging.info(f"Starting extraction for dataset: {dataset.upper()}")

    if dataset in ["quantex", "both"]:
        logging.info("Extracting frames from QUANTEX videos...")
        extract_every_nth_frame_from_videos_in_folder(
            Path("/home/nele_pauline_suffo/ProcessedData/quantex_test_videos"),#DataPaths.QUANTEX_VIDEOS_INPUT_DIR,
            Path("/home/nele_pauline_suffo/ProcessedData/quantex_test_rawframes"),#DataPaths.QUANTEX_IMAGES_INPUT_DIR,
            1,#DataConfig.FRAME_STEP_INTERVAL,
            Path("/home/nele_pauline_suffo/ProcessedData/test_rawframes_extraction_error.log"),#DataPaths.QUANTEX_RAWFRAMES_EXTRACTION_ERROR_LOG,
            Path("/home/nele_pauline_suffo/ProcessedData/test_processed_videos.txt"),#DataPaths.QUANTEX_PROCESSED_VIDEOS_LOG,
        )
        #logging.info("Extracting audio from QUANTEX videos...")
        #extract_audio_from_videos_in_folder(
        #    DataPaths.QUANTEX_VIDEOS_INPUT_DIR,
        #    AudioClassification.QUANTEX_AUDIO_DIR,
        #)

    if dataset in ["childlens", "both"]:
        logging.info("Extracting frames from CHILDLENS videos...")
        extract_every_nth_frame_from_videos_in_folder(
            DataPaths.CHILDLENS_VIDEOS_INPUT_DIR,
            DataPaths.CHILDLENS_IMAGES_INPUT_DIR,
            DataConfig.FRAME_STEP_INTERVAL,
            DataPaths.CHILDLENS_RAWFRAMES_EXTRACTION_ERROR_LOG,
            DataPaths.CHILDLENS_PROCESSED_VIDEOS_LOG,
        )
        logging.info("Extracting audio from CHILDLENS videos...")
        extract_audio_from_videos_in_folder(
            DataPaths.CHILDLENS_VIDEOS_INPUT_DIR,
            AudioClassification.CHILDLENS_AUDIO_DIR,
        )

    logging.info(f"Finished processing dataset: {dataset.upper()}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Argument parser ---
    parser = argparse.ArgumentParser(
        description="Extract frames and audio from videos for the specified dataset(s)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["childlens", "quantex", "both"],
        help="Which dataset to process (childlens, quantex, or both).",
    )
    args = parser.parse_args()

    main(args.dataset)