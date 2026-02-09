import re
import pandas as pd
import numpy as np

def create_second_level_labels(segments_df: pd.DataFrame, video_duration_seconds: int) -> np.ndarray:
    """
    Creates a second-by-second label array for a video based on segments. 
    
    Parameters:
    - segments_df: DataFrame with 'start_time_sec', 'end_time_sec', and 'interaction_type'.
    - video_duration_seconds: The total length of the video in seconds.
    
    Returns:
    - A numpy array where each index corresponds to a second and the value is the label, 
      or None if unclassified.
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

def time_to_seconds(time_str):
    """Converts MM:SS or float seconds string to float seconds.
    
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
    
def extract_child_id(video_name):
    """
    Extracts the 6-digit child ID from a video name string.
    Example: 'id123456_video.mp4' -> '123456'
    """
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def merge_overlapping_vocalizations(vocs_df):
    """Merge overlapping vocalizations from the same speaker in the same video."""
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