import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple
from constants import AudioClassification, Analysis

def load_ground_truth_segments(
    gt_path: Path = Analysis.GROUND_TRUTH_SEGMENTS_CSV,
    meta_path: Path = Analysis.INTERACTION_SEGMENTS_CSV,
) -> pd.DataFrame:
  """Loads and standardizes ground-truth segments to match pipeline schema."""
  df_gt = pd.read_csv(gt_path, sep=";")
  df_gt = df_gt.loc[:, ~df_gt.columns.str.contains("^Unnamed")].dropna(
      subset=["video_name", "interaction_type", "start_time_min", "end_time_min"]
  )
  df_gt["interaction_type"] = (
      df_gt["interaction_type"].astype(str).str.strip().str.title()
  )
  df_gt["start_time_sec"] = df_gt["start_time_min"].apply(time_to_seconds)
  df_gt["end_time_sec"] = df_gt["end_time_min"].apply(time_to_seconds)
  df_gt["duration_sec"] = df_gt["end_time_sec"] - df_gt["start_time_sec"]
  df_gt = df_gt[df_gt["duration_sec"] > 0].copy()
  df_gt["child_id"] = df_gt["video_name"].apply(
      lambda x: extract_child_id(x) or "unknown"
  )

  if meta_path.exists():
    meta_df = pd.read_csv(meta_path)
    if (
        "video_name" in meta_df.columns
        and "age_at_recording" in meta_df.columns
    ):
      lookup = meta_df[["video_name", "age_at_recording"]].drop_duplicates(
          subset=["video_name"]
      )
      df_gt = df_gt.merge(lookup, on="video_name", how="left")
    else:
      df_gt["age_at_recording"] = np.nan
  else:
    df_gt["age_at_recording"] = np.nan

  return df_gt

def create_second_level_labels(segments_df: pd.DataFrame, 
                               video_duration_seconds: int) -> np.ndarray:
    """
    Creates a second-by-second label array for a video based on segments. 
    
    Parameters:
    ----------
    segments_df: pd.DataFrame
        DataFrame containing 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    video_duration_seconds: int
        Total duration of the video in seconds.
    
    Returns:
    -------
    np.ndarray
        An array of shape (video_duration_seconds,) where each index corresponds to a second in the
    """
    labels = np.full(video_duration_seconds, None, dtype=object)
    
    # Ensure time columns exist and are numeric
    if 'start_time_sec' not in segments_df.columns or 'end_time_sec' not in segments_df.columns:
        if 'start_time_min' in segments_df.columns and 'end_time_min' in segments_df.columns:
             segments_df['start_time_sec'] = segments_df['start_time_min'].apply(time_to_seconds)
             segments_df['end_time_sec'] = segments_df['end_time_min'].apply(time_to_seconds)
        else:
            # Cannot process without time in seconds
            return labels

    for _, segment in segments_df.iterrows():
        try:
            # Use floating point conversion then rounding for robustness
            start_sec = int(np.round(float(segment['start_time_sec'])))
        except Exception:
            start_sec = 0
            
        try:
            end_sec = int(np.round(float(segment['end_time_sec'])))
            # Clip end_sec to prevent out-of-bounds indexing
            # Note: We are using [start, end) second interval in the IRR script, 
            # but the original script logic used [start, end] seconds, 
            # so we maintain that original logic here for compatibility: labels[start:end + 1]
            end_sec = min(end_sec, video_duration_seconds - 1)
        except Exception:
            # If end_sec is invalid, the segment is effectively 0 duration
            end_sec = start_sec

        interaction_type = str(segment['interaction_type']).lower()
        start_sec = max(0, start_sec)

        # Assign interaction type to the range of seconds (inclusive start and end second index)
        if start_sec <= end_sec: 
            labels[start_sec:end_sec + 1] = interaction_type
           
    return labels

def time_to_seconds(time_str: str) -> float:
    """
    Converts MM:SS or float seconds string to float seconds.
    
    Parameters:
    ----------
    time_str : str
        Time in MM:SS format or as a float string.
        
    Returns:
    -------
    float or None
        Time in seconds as a float, or None if conversion fails.
    """
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            # MM:SS format
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            # HH:MM:SS format
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return None
    
def extract_child_id(video_name: str) -> str:
    """
    Extracts the 6-digit child ID from a video name string.
    Example: 'id123456_video.mp4' -> '123456'
    
    Parameters:
    ----------
    video_name : str
        The name of the video file.
        
    Returns:
    -------
    str or None
        The extracted child ID, or None if not found.
    """
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def merge_overlapping_vocalizations(vocs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping vocalizations from the same speaker in the same video.
    
    Parameters:
    ----------
    vocs_df : pd.DataFrame
        DataFrame containing vocalization information with columns: 'video_name', 'speaker', 'start
        _time_seconds', 'end_time_seconds', and 'speech_type'.
    
    Returns:
    -------
    pd.DataFrame
        A DataFrame with merged vocalizations, where overlapping vocalizations from the same speaker
        in the same video are merged into a single vocalization with the earliest start time and latest
    """
    merged_vocs = []
    
    # Group by video and speaker
    for (video_name, speaker), group in vocs_df.groupby(['video_name', 'speaker']):
        # Sort by start time
        group = group.sort_values('start_time_seconds').reset_index(drop=True)
        
        if len(group) == 0:
            continue
            
        current_vocalization = group.iloc[0].copy()
        
        for i in range(1, len(group)):
            next_voc = group.iloc[i]
            
            # Check if current and next vocalization overlap or are adjacent
            if next_voc['start_time_seconds'] <= current_vocalization['end_time_seconds']:
                # Merge: extend end time to the maximum of both
                current_vocalization['end_time_seconds'] = max(
                    current_vocalization['end_time_seconds'],
                    next_voc['end_time_seconds']
                )
            else:
                # No overlap, save current and start new one
                merged_vocs.append(current_vocalization)
                current_vocalization = next_voc.copy()
        
        # Add the last vocalization
        merged_vocs.append(current_vocalization)
    
    return pd.DataFrame(merged_vocs).reset_index(drop=True)

def parse_rttm(file_path: Path = AudioClassification.VTC_RTTM_FILE, target_speech_types: list = None) -> pd.DataFrame:
    """
    Parse a RTTM file and return a DataFrame with vocalization information.

    Parameters
    ----------
    file_path : Path, optional
        Path to the RTTM file, by default AudioClassification.VTC_RTTM_FILE
    target_speech_types : list, optional
        List of speech types to include (e.g., ['KCDS', 'OHS']), by default None (include all)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with vocalization information
    """
    data = []
    if not file_path.exists(): return pd.DataFrame()
    with open(file_path, 'r') as f:
        # We expect lines in the format: SPEAKER <video_name> <channel> <start_time> <duration> <ortho> <stype> <speaker_type> <speaker_name>
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                speech_type = parts[7]
                # We are only interested in KCDS and OHS vocalizations for this analysis
                if target_speech_types is not None and speech_type not in target_speech_types:
                    continue
                data.append({
                    'video_name': parts[1],
                    'start_time_seconds': float(parts[3]),
                    'end_time_seconds': float(parts[3]) + float(parts[4]),
                    'duration_seconds': float(parts[4]),
                    'speech_type': speech_type
                })
    return pd.DataFrame(data)

def merge_overlapping_intervals(intervals: List[Tuple[float, float]]):
    """
    Merges overlapping or adjacent intervals and calculates total duration.

    Parameters
    ----------
    intervals : list of tuples
        List of (start_time, end_time) tuples representing intervals.

    Returns
    -------
    merged : list of tuples
        List of merged (start_time, end_time) intervals.
    total_duration : float
        Total duration covered by the merged intervals.
    """
    if not intervals: return [], 0.0
    intervals = sorted(intervals)
    merged = [intervals[0]]
    # Iterate through sorted intervals and merge as needed
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    total_duration = sum(end - start for start, end in merged)
    return merged, total_duration

def get_child_fold_boundaries(segments_df, num_folds=5):
    """
    Returns a dictionary mapping child_id to a list of (start, end) 
    timestamps for each of the 5 folds.
    """
    # Calculate global duration per child
    child_durations = segments_df.groupby('child_id')['duration_sec'].sum().to_dict()
    
    boundaries = {}
    for child_id, total_dur in child_durations.items():
        fold_size = total_dur / num_folds
        # Generate boundaries: [0, size, 2*size, 3*size, 4*size, total]
        points = [i * fold_size for i in range(num_folds + 1)]
        # Create pairs: (0, 20), (20, 40), etc.
        boundaries[child_id] = [(points[i], points[i+1]) for i in range(num_folds)]
        
    return boundaries