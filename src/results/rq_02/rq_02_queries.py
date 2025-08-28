# Research Question: 2. How much language does the key child produce (utterances and words)?

import sqlite3
import pandas as pd
import re
from pathlib import Path
from constants import DataPaths, ResearchQuestions

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def row_wise_mapping(voc_row, video_segments_df):
    """
    Map a single vocalization to overlapping interaction segment in a video file.
    
    Parameters:
    ----------
    voc_row : pd.Series
        A row from the vocalizations DataFrame with columns:
        ['vocalization_id', 'video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'words']
    video_segments_df : pd.DataFrame
        DataFrame containing interaction segments for one video_id with columns:
        ['start_time_sec', 'end_time_sec', 'category']  
    
    Returns:
    -------
    list of dict
        List of dictionaries with mapped vocalizations including:
        ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'words', 'interaction_type']
    """
    # Find overlapping segments (already filtered to same video)
    overlaps = video_segments_df[
        (video_segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (video_segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ]
    results = []
    # For each overlapping segment, calculate the overlap duration and proportion of words
    for _, seg in overlaps.iterrows():
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        overlap_end = min(voc_row['end_time_seconds'], seg['end_time_sec'])
        overlap_seconds = overlap_end - overlap_start + 1
        total_seconds = voc_row['end_time_seconds'] - voc_row['start_time_seconds'] + 1
        prop = overlap_seconds / total_seconds
        results.append({
            'video_id': voc_row['video_id'],
            'child_id': voc_row['child_id'],
            'age_at_recording': voc_row['age_at_recording'],
            'start_time_seconds': voc_row['start_time_seconds'],
            'end_time_seconds': voc_row['end_time_seconds'],
            'seconds': overlap_seconds,
            'words': voc_row['words'] * prop,
            'interaction_type': seg['category']
        })
    return results

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = ResearchQuestions.INTERACTION_SEGMENTS_CSV):
    """
    Retrieve KCHI vocalizations from the SQLite database and map them to interaction segments.
    
    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'category']
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with mapped vocalizations including:
        ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'words', 'interaction_type']
    """
    # Load segments DataFrame
    segments_df = pd.read_csv(segments_csv_path)
    
    # Load age data
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)

    # Connect to the SQLite database and query KCHI vocalizations with video names
    conn = sqlite3.connect(db_path)
    vocalizations_query = """
    SELECT v.vocalization_id, v.video_id, vid.video_name, v.start_time_seconds, v.end_time_seconds, v.words
    FROM Vocalizations v
    JOIN Videos vid ON v.video_id = vid.video_id
    WHERE v.speaker = 'KCHI'
    """
    kchi_vocs = pd.read_sql_query(vocalizations_query, conn)
    conn.close()
    
    # Extract child_id from video_name
    kchi_vocs['child_id'] = kchi_vocs['video_name'].apply(extract_child_id)
    
    # Merge with age data to get age at recording
    kchi_vocs = kchi_vocs.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')

    print(f"Found {len(kchi_vocs)} KCHI vocalizations across {kchi_vocs['video_id'].nunique()} videos")
    print(f"Found {len(segments_df)} interaction segments across {segments_df['video_id'].nunique()} videos")
    
    # Check video alignment
    voc_videos = set(kchi_vocs['video_id'].unique())
    seg_videos = set(segments_df['video_id'].unique())
    common_videos = voc_videos & seg_videos
    print(f"Common videos between vocalizations and segments: {len(common_videos)}")
    
    mapped_rows = []
    
    # Process each video separately
    for video_id in common_videos:       
        # Filter data to this specific video
        video_vocalizations = kchi_vocs[kchi_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]
                
        print(f"  Video {video_id}: {len(video_vocalizations)} vocalizations, {len(video_segments)} segments")
        # Map each vocalization in this video to segments in this video
        for _, voc_row in video_vocalizations.iterrows():
            mapped_rows.extend(row_wise_mapping(voc_row, video_segments))
    
    return pd.DataFrame(mapped_rows)