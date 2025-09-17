import logging
import sqlite3
import re
from pathlib import Path
from config import DataConfig

def get_video_id(video_name: str, cursor: sqlite3.Cursor) -> int:
    """Get video_id from Videos table using video name"""
    cursor.execute('SELECT video_id FROM Videos WHERE video_name = ?', (video_name,))
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        logging.error(f"Video {video_name} not found in Videos table")
        return None
    
def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename using regex"""
    # Try to extract numbers from filename (e.g., frame_001.jpg -> 1, video_name_0042.jpg -> 42)
    numbers = re.findall(r'\d+', filename)
    if numbers:
        return int(numbers[-1])  # Use last number found
    else:
        return 0    

def get_frame_paths(video_frame_dir: Path):
    """
    Returns a sorted list of frame paths in video_frame_dir
    for all valid extensions defined in DataConfig.VALID_EXTENSIONS.
    """
    frame_paths = []
    for ext in DataConfig.VALID_EXTENSIONS:
        frame_paths.extend(video_frame_dir.glob(f"*{ext}"))

    # Sort by the numeric part at the end of the filename
    frame_paths = sorted(
        frame_paths,
        key=lambda x: int(x.stem.split('_')[-1])
    )
    return frame_paths