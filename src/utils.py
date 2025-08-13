import cv2
import os
import logging
import subprocess
import sqlite3
import random
import shutil
import gc  # Garbage collection
import concurrent.futures
from pathlib import Path
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
import glob
import tempfile
from moviepy.editor import VideoFileClip
from config import DataConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("extract_frames.log"),
        logging.StreamHandler()
    ]
)

def split_videos_into_train_val(
    input_folder: Path,
    ) -> Union[list, list]:
    """
    This function splits the videos in the input folder into train and validation sets
    It returns the list of video names in the train and validation sets.
    """
    # Get all video files in the input folder
    input_folder = DataPaths.VIDEOS_INPUT_DIR
    train_ratio = DataConfig.TRAIN_SPLIT_RATIO
    video_files = [f for f in input_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']]

    # Calculate total duration of all videos
    total_duration = sum(get_video_length(video) for video in video_files)
    train_duration_target = total_duration * train_ratio

    # Shuffle videos to randomize assignment
    random.shuffle(video_files)

    # Initialize durations
    current_train_duration = 0
    train_videos = []
    val_videos = []

    # Assign videos to train and val
    for video in video_files:
        video_duration = get_video_length(video)
        if current_train_duration + video_duration <= train_duration_target:
            train_videos.append(video.name)
            current_train_duration += video_duration
        else:
            val_videos.append(video.name)

    logging.info(f"Total duration: {total_duration:.2f} seconds")
    logging.info(f"Train set duration: {current_train_duration:.2f} seconds")
    logging.info(f"Validation set duration: {total_duration - current_train_duration:.2f} seconds")
    logging.info(f"Train videos: {len(train_videos)}")
    logging.info(f"Validation videos: {len(val_videos)}")

    return train_videos, val_videos


def get_frame_width_height(video_file_path: Path) -> tuple:
    """
    This function gets the frame width and height of a video file.

    Parameters
    ----------
    video_file_path : Path
        the path to the video file

    Returns
    -------
    tuple
        the frame width and height
    """

    # Open the video file
    cap = cv2.VideoCapture(video_file_path)

    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Always release the VideoCapture object
    cap.release()
    return frame_width, frame_height


def get_video_length(
    file_path: Path
) -> float:
    """
    This function returns the length of a video file in seconds.
    
    Parameters
    ----------
    file_path : Path
        the path to the video file
    
    Returns
    -------
    float
        the length of the video in seconds
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


def get_video_frame_count(
    video_path: Path
) -> int:
    """
    Get the total number of frames in a video.
    
    Parameters
    ----------
    video_path : str
        The path to the video file.
        
    Returns
    -------
    int
        The total number of frames in the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def split_videos(
    video_files: List[Path], 
    split_ratio: float
) -> Tuple[List[Path], List[Path]]:
    """
    Splits the list of videos into training and validation sets, balancing the split by video length.

    Parameters:
    ----------
    video_files: list
        The list of video files to split.
    split_ratio: float
        The ratio of training videos to validation videos.

    Returns:
    -------
    tuple of lists
        A tuple containing two lists: training videos and validation videos.
    """
    # Get the duration of each video
    video_durations = [(video_file, get_video_length(video_file)) for video_file in video_files]
    # Calculate the total duration of all videos
    total_duration = sum(duration for _, duration in video_durations)
    
    train_videos = []
    val_videos = []
    train_duration = 0
    val_duration = 0
    
    # Shuffle the videos to randomize the assignment
    random.shuffle(video_durations)

    # Assign videos to train and val sets
    for video_file, duration in video_durations:
        if train_duration / total_duration < split_ratio:
            train_videos.append(video_file)
            train_duration += duration
        else:
            val_videos.append(video_file)
            val_duration += duration

    logging.info(f"Total training duration: {train_duration} seconds")
    logging.info(f"Total validation duration: {val_duration} seconds")

    return train_videos, val_videos


def prepare_video_dataset(
    output_dir: Path,
    model: str,
    fps: int = None,
    batch_size: int = DataConfig.VIDEO_BATCH_SIZE,
) -> None:
    """
    Extracts frames from all videos in the given directory, splits them into training and validation sets,
    and saves the frames in corresponding folders.
    
    Parameters:
    ----------
    output_dir: Path
        The directory to save the extracted frames.
    split_ratio: float
        The ratio of training videos to validation videos.
    fps: int
        The frames per second to extract.
    model: str
        The model for which the frames are extracted.
    batch_size: int
        The number of videos to process concurrently.
    """
    logging.info(f"Starting frame extraction from videos in {video_dir} to {routput_dir} at {fps} FPS with split ratio {split_ratio}.")

    video_dir = DataPaths.VIDEOS_INPUT_DIR
    split_ratio = DataConfig.TRAIN_SPLIT_RATIO
    
    # Ensure output directories exist
    train_output_dir = output_dir / 'train'
    val_output_dir = output_dir / 'val'
    for dir_path in [train_output_dir, val_output_dir]:
        shutil.rmtree(dir_path, ignore_errors=True)
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get all video files
    video_files = [f for f in video_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.mp4', '.MP4']]
    logging.info(f"Found {len(video_files)} video files to process.")

    # Split videos into train and val sets
    train_videos, val_videos = split_videos(video_files, split_ratio)

    # Process videos in batches
    for i in range(0, len(train_videos), batch_size):
        batch = train_videos[i:i + batch_size]
        logging.info(f"Processing training batch {i // batch_size + 1}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(lambda video: process_video_yolo(video, train_output_dir), batch)
                
    for i in range(0, len(val_videos), batch_size):
        batch = val_videos[i:i + batch_size]
        logging.info(f"Processing validation batch {i // batch_size + 1}")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(lambda video: process_video_yolo(video, train_output_dir), batch)
    logging.info("Completed frame extraction for all videos.")     
        
def create_video_writer(
    output_path: str,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
) -> cv2.VideoWriter:
    """
    This function creates a VideoWriter object to write the output video.

    Parameters
    ----------
    output_path : str
        the path to the output video file
    frames_per_second : int
        the frames per second of the video
    frame_width : int
        the width of the frame
    frame_height : int
        the height of the frame

    Returns
    -------
    cv2.VideoWriter
        the video writer object
    """

    # Create a VideoWriter object to write the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_path, fourcc, frames_per_second, (frame_width, frame_height)
    )

    return out

def extract_audio_from_video(video_path: Path, audio_output_path: Path) -> None:
    """
    Extracts the audio from a video file and saves it directly as a 16kHz WAV file.
    
    Parameters
    ----------
    video_path : Path
        Path to the input video file.
    audio_output_path : Path
        Path where the 16kHz WAV file should be saved.
    """
    parent_dir = audio_output_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use ffmpeg directly to extract and convert audio in one step
        process = subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(audio_output_path)
        ], check=True, capture_output=True, text=True)

        if process.returncode == 0:
            logging.info(f"Successfully stored 16kHz audio at {audio_output_path}")
        else:
            logging.error(f"Error extracting audio: {process.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")

def extract_audio_from_videos_in_folder(videos_input_dir: Path, output_dir: Path):
    """
    Extracts audio from all video files in the specified folder, if not already done.
    """
    logging.info(f"Scanning folder: {videos_input_dir}")  # Debugging step

    # load file with problematic videos
    problematic_videos = []
    with open("/home/nele_pauline_suffo/outputs/audio_extraction/problematic_audio_files.txt", 'r') as f:
        problematic_videos = [line.strip() for line in f.readlines()]
    for video_file in videos_input_dir.iterdir():
        
        if video_file.suffix.lower() not in ['.mp4', '.MP4'] or video_file.name in problematic_videos:
            logging.info(f"Skipping problematic video file: {video_file}")
            continue
        
        # skip if video file is already in the output directory
        if (output_dir / f"{video_file.stem}.wav").exists():
            logging.info(f"Audio already exists for: {video_file}")
            continue
        
        # create output directory if it doesn't exist
        audio_output_path = output_dir / f"{video_file.stem}.wav"

        if not audio_output_path.exists():
            logging.info(f"Extracting audio for: {video_file}")  # Debugging step
            extract_audio_from_video(video_file, audio_output_path)
            logging.info(f"Finished processing: {video_file}")  # Debugging step
        else:
            logging.info(f"Audio already exists for: {video_file}")  # Debugging step