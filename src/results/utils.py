import re
import pandas as pd

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