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
        ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'words', 'interaction_type', 'segment_start_time', 'segment_end_time']
    """
    # Find overlapping segments (already filtered to same video)
    overlaps = video_segments_df[
        (video_segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (video_segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ]
    results = []
    
    # Calculate words per second rate for this vocalization
    total_seconds = voc_row['end_time_seconds'] - voc_row['start_time_seconds']
    words_per_second = voc_row['words'] / total_seconds if total_seconds > 0 else 0
    
    # Sort segments by start time to handle gaps properly
    overlaps = overlaps.sort_values('start_time_sec').reset_index(drop=True)
    
    # For each overlapping segment, calculate overlap duration and allocate words
    for i, (_, seg) in enumerate(overlaps.iterrows()):
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        
        # For overlap_end: use segment end, but extend to next segment start if there's a gap
        if i < len(overlaps) - 1:  # Not the last segment
            next_seg_start = overlaps.iloc[i + 1]['start_time_sec']
            # Extend current segment to next segment start (fills gaps)
            extended_end = next_seg_start
            # Calculate the filled gap duration
            gap_seconds = next_seg_start - seg['end_time_sec']
            total_segment_duration = (seg['end_time_sec'] - seg['start_time_sec']) + gap_seconds
        else:
            # Last segment, use its actual end
            extended_end = seg['end_time_sec']
            total_segment_duration = seg['end_time_sec'] - seg['start_time_sec']
        
        overlap_end = min(voc_row['end_time_seconds'], extended_end)
        overlap_seconds = overlap_end - overlap_start
        
        # Allocate words based on overlap duration and words-per-second rate
        allocated_words = words_per_second * overlap_seconds
        
        results.append({
            'video_id': voc_row['video_id'],
            'child_id': voc_row['child_id'],
            'age_at_recording': voc_row['age_at_recording'],
            'start_time_seconds': voc_row['start_time_seconds'],
            'end_time_seconds': voc_row['end_time_seconds'],
            'seconds': overlap_seconds,
            'words': allocated_words,
            'interaction_type': seg['category'],
            'total_segment_duration': total_segment_duration,
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec']
        })
    return results

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV):
    """
    Retrieve KCHI vocalizations from the SQLite database and map them to interaction segments.
    The mapped_vocalizations DataFrame contains the following columns:
    ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'words', 'interaction_type']

    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'category']
        
    Returns:
    -------
    tuple
        (mapped_vocalizations_df, segments_df) - DataFrame with mapped vocalizations and original segments
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

def main():
    """
    Main function to analyze child language production by interaction context.
    
    Research Question 2: How much language does the key child produce (utterances and words)?
    
    This script:
    1. Extracts KCHI (key child) vocalizations from the database
    2. Maps them to interaction segments (Alone, Co-present Silent, Interacting)  
    3. Outputs mapped vocalization data with overlap durations
    4. Saves results to CSV
    
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
    mapped_vocalizations = map_vocalizations_to_segments()

    # Step 2: Use mapped vocalizations directly
    print("\n📋 Step 2: Using mapped vocalizations data...")
    
    # Convert seconds to minutes
    mapped_vocalizations['kchi_speech_minutes'] = mapped_vocalizations['seconds'] / 60
    
    # Select final columns
    final_data = mapped_vocalizations[[
        'child_id', 'age_at_recording', 'interaction_type',
        'segment_start_time', 'segment_end_time',
        'start_time_seconds', 'end_time_seconds', 'seconds', 'kchi_speech_minutes', 'total_segment_duration'
    ]].copy()

    final_data["segment_duration_minutes"] = (final_data["total_segment_duration"] / 60)
    final_data["speech_activity_percent"] = (final_data["kchi_speech_minutes"] / final_data["segment_duration_minutes"]).fillna(0)

    # Sort by child, age, and start time
    final_data = final_data.drop(columns=['seconds', 'total_segment_duration'])
    final_data = final_data.sort_values(['child_id', 'age_at_recording', 'start_time_seconds']).reset_index(drop=True)    
    # No filtering needed - we want to keep all segments, even those with 0 KCHI minutes
    
    # Step 2.5: Create segment-level aggregation
    print("\n📊 Step 2.5: Aggregating KCHI speech by segment...")
    segment_totals = final_data.groupby([
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time'
    ]).agg({
        'kchi_speech_minutes': 'sum',
        'segment_duration_minutes': 'first',  # Same for all rows in a segment
        'speech_activity_percent': 'first'    # Will be recalculated
    }).reset_index()
    
    # Recalculate speech activity percentage for aggregated data
    segment_totals['speech_activity_percent'] = (
        segment_totals['kchi_speech_minutes'] / segment_totals['segment_duration_minutes']
    ).fillna(0)
    
    # Sort segment totals
    segment_totals = segment_totals.sort_values([
        'child_id', 'age_at_recording', 'segment_start_time'
    ]).reset_index(drop=True)
    
    # Step 3: Save results
    print(f"\n💾 Step 3: Saving segment totals...")
    
    # Save aggregated segment totals only
    segment_totals.to_csv(Inference.WORD_SUMMARY_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Aggregated segment totals saved to: {Inference.WORD_SUMMARY_CSV}")
    print(f"📊 Total unique segments: {len(segment_totals)}")
    print("=" * 70)
    
    return segment_totals

if __name__ == "__main__":
    # Run the analysis with default parameters
    main()