import logging
from pathlib import Path
from typing import Set, List, Tuple
from tqdm import tqdm
import cv2
from constants import DataPaths
from config import DataConfig

# Configure logging globally
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_processed_videos(processed_videos_file: Path) -> Set[str]:
    """Return a set of already processed video filenames."""
    if processed_videos_file.exists():
        return {line.strip() for line in processed_videos_file.read_text().splitlines()}
    return set()

def mark_video_as_processed(video_path: Path, processed_videos_file: Path) -> None:
    """Append a processed video name to the tracking file."""
    processed_videos_file.parent.mkdir(parents=True, exist_ok=True)
    with processed_videos_file.open("a") as f:
        f.write(f"{video_path.name}\n")

def generate_image_name(video_path: Path, frame_idx: int) -> str:
    """Generate standardized frame filename."""
    return f"{video_path.stem}_{frame_idx:06d}.jpg"

def extract_every_nth_frame(video_path: Path, output_folder: Path, frame_interval: int, error_log_file: Path) -> None:
    """
    Extract every nth frame from a single video, ensuring exact frame positioning.
    """
    output_folder.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logging.error(f"Unable to open video file: {video_path}")
        error_log_file.write_text(f"Failed to open video: {video_path}\n")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    failed_frames: List[Tuple[str, int]] = []
    saved_frames = 0

    with tqdm(total=total_frames // frame_interval, desc=f"Extracting {video_path.stem}") as pbar:
        for frame_idx in range(0, total_frames, frame_interval):
            frame_name = generate_image_name(video_path, frame_idx)
            output_path = output_folder / frame_name

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                logging.error(f"Failed to read frame {frame_idx} in {video_path}")
                failed_frames.append((video_path.name, frame_idx))
            else:
                try:
                    cv2.imwrite(str(output_path), frame)
                    saved_frames += 1
                except Exception as e:
                    logging.error(f"Error saving frame {frame_idx} from {video_path}: {e}")
                    failed_frames.append((video_path.name, frame_idx))

            pbar.update(1)

    cap.release()

    if failed_frames:
        with error_log_file.open("a") as error_log:
            error_log.writelines(f"{video},{idx}\n" for video, idx in failed_frames)

    logging.info(f"Extraction complete for {video_path} | Saved: {saved_frames} | Failed: {len(failed_frames)}")

def extract_every_nth_frame_from_videos_in_folder(
    input_folder: Path,
    output_root_folder: Path,
    frame_step: int,
    error_log_file: Path,
    processed_videos_file: Path,
) -> None:
    """
    Extract frames from all videos in a folder that haven't been processed yet.
    """
    output_root_folder.mkdir(parents=True, exist_ok=True)
    video_files = sorted(input_folder.glob("*.MP4"))

    if not video_files:
        logging.info(f"No .MP4 files found in: {input_folder}")
        return

    processed_videos = get_processed_videos(processed_videos_file)
    videos_to_process = [v for v in video_files if v.name not in processed_videos]

    logging.info(f"Total videos: {len(video_files)} | Processed: {len(processed_videos)} | Remaining: {len(videos_to_process)}")

    for video_path in videos_to_process:
        logging.info(f"Processing: {video_path}")
        video_output_folder = output_root_folder / video_path.stem
        extract_every_nth_frame(video_path, video_output_folder, frame_step, error_log_file)
        mark_video_as_processed(video_path, processed_videos_file)

    logging.info(f"All videos processed. Frames saved to {output_root_folder}")

def main() -> None:
    logging.info("Starting frame extraction...")
    extract_every_nth_frame_from_videos_in_folder(
        DataPaths.VIDEOS_INPUT_DIR,
        DetectionPaths.images_input_dir,
        DataConfig.FRAME_STEP_INTERVAL,
        DataPaths.RAWFRAMES_EXTRACTION_ERROR_LOG,
        DataPaths.PROCESSED_VIDEOS_LOG,
    )
    logging.info("Frame extraction complete.")

if __name__ == "__main__":
    main()