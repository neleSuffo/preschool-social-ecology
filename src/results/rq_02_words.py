# Research Question: 2. How much language does the key child produce (utterances and words)?

import sqlite3
import pandas as pd
import re
from pathlib import Path
from constants import DataPaths, Inference

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

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV):
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
    
    # Check video alignment
    voc_videos = set(kchi_vocs['video_id'].unique())
    seg_videos = set(segments_df['video_id'].unique())
    common_videos = voc_videos & seg_videos    
    mapped_rows = []
    
    # Process each video separately
    for video_id in common_videos:       
        # Filter data to this specific video
        video_vocalizations = kchi_vocs[kchi_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]
                
        # Map each vocalization in this video to segments in this video
        for _, voc_row in video_vocalizations.iterrows():
            mapped_rows.extend(row_wise_mapping(voc_row, video_segments))
    
    return pd.DataFrame(mapped_rows)

def main(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV):
    """
    Main function to analyze child language production by interaction context.
    
    Research Question 2: How much language does the key child produce (utterances and words)?
    
    This script:
    1. Extracts KCHI (key child) vocalizations from the database
    2. Maps them to interaction segments (Alone, Co-present Silent, Interacting)
    3. Aggregates by child_id, age_at_recording, and interaction_type
    4. Calculates word counts and durations by interaction type
    5. Saves aggregated results to CSV
    
    Parameters
    ----------
    db_path : Path, optional
        Path to the SQLite database. Defaults to DataPaths.INFERENCE_DB_PATH
    segments_csv_path : Path, optional
        Path to interaction segments CSV. Defaults to Inference.INTERACTION_SEGMENTS_CSV
    """   
    print("🗣️ RESEARCH QUESTION 2: CHILD LANGUAGE PRODUCTION ANALYSIS")
    print("=" * 70)
    print("Analyzing key child vocalizations by interaction context...")
    
    # Step 1: Map vocalizations to interaction segments
    print("\n🔄 Step 1: Mapping vocalizations to interaction segments...")
    mapped_vocalizations = map_vocalizations_to_segments(db_path, segments_csv_path)
        
    # Step 2: Aggregate by child_id, age_at_recording, and interaction_type
    print("\n📋 Step 2: Aggregating by child, age, and interaction type...")
    
    if len(mapped_vocalizations) > 0:
        aggregated_data = mapped_vocalizations.groupby(['child_id', 'age_at_recording', 'interaction_type']).agg({
            'words': 'sum',           # Total words
            'seconds': 'sum',         # Total duration in seconds
        }).reset_index()
        
        # Calculate words per minute for each segment
        aggregated_data['minutes'] = (aggregated_data['seconds'] / 60).round(2)
        aggregated_data['words_per_minute'] = aggregated_data['words'] / (aggregated_data['minutes']).replace([float('inf'), -float('inf')], 0).round(2)

        aggregated_data = aggregated_data.drop(columns=['seconds'])
        # Round numerical values for cleaner output
        aggregated_data['words'] = aggregated_data['words'].round(1)
    else:
        print("   ⚠️ No vocalizations found to aggregate")
        # Create empty DataFrame with expected columns
        aggregated_data = pd.DataFrame(columns=[
            'child_id', 'age_at_recording', 'interaction_type', 
            'words', 'seconds', 'minutes', 'words_per_minute'
        ])
    
    # Step 3: Save aggregated results
    print(f"\n💾 Step 3: Saving aggregated results...")
    aggregated_data.to_csv(Inference.WORD_SUMMARY_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Aggregated results saved to: {Inference.WORD_SUMMARY_CSV}")
    print("=" * 70)
    
    return aggregated_data

if __name__ == "__main__":
    # Run the analysis with default parameters
    main()