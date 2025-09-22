# Research Question: 2. How much language does the key child produce (utterances and words)?

import sqlite3
import pandas as pd
import re
from pathlib import Path
from constants import DataPaths, Inference
from config import DataConfig
from utils import extract_child_id

def merge_overlapping_intervals(intervals):
    """
    Merge overlapping time intervals to avoid double-counting speech time.
    
    Parameters:
    ----------
    intervals : list of tuple
        List of (start_time, end_time) tuples representing speech intervals
    
    Returns:
    -------
    list of tuple
        List of merged, non-overlapping intervals
    float
        Total duration of merged intervals
    """
    if not intervals:
        return [], 0.0
    
    # Sort intervals by start time
    intervals = sorted(intervals)
    merged = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        # If current interval overlaps with the last merged interval
        if current_start <= last_end:
            # Merge by extending the end time
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            # No overlap, add as new interval
            merged.append((current_start, current_end))
    
    # Calculate total duration
    total_duration = sum(end - start for start, end in merged)
    
    return merged, total_duration

def row_wise_mapping(voc_row, video_segments_df):
    """
    Map a single vocalization to overlapping interaction segment in a video file.
    
    Parameters:
    ----------
    voc_row : pd.Series
        A row from the vocalizations DataFrame with columns:
        ['vocalization_id', 'video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds']
    video_segments_df : pd.DataFrame
        DataFrame containing interaction segments for one video_id with columns:
        ['start_time_sec', 'end_time_sec', 'interaction_type']  
    
    Returns:
    -------
    list of dict
        List of dictionaries with mapped vocalizations including:
        ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'interaction_type', 'segment_start_time', 'segment_end_time']
    """
    # Find overlapping segments (already filtered to same video)
    overlaps = video_segments_df[
        (video_segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (video_segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ]
    results = []
    
    # Sort segments by start time
    overlaps = overlaps.sort_values('start_time_sec').reset_index(drop=True)
    
    # For each overlapping segment, calculate overlap duration
    for _, seg in overlaps.iterrows():
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        overlap_end = min(voc_row['end_time_seconds'], seg['end_time_sec'])
        overlap_seconds = max(0, overlap_end - overlap_start)
        
        # Total segment duration should be the segment's own length
        total_segment_duration = seg['end_time_sec'] - seg['start_time_sec']
        
        results.append({
            'video_id': voc_row['video_id'],
            'child_id': voc_row['child_id'],
            'age_at_recording': voc_row['age_at_recording'],
            'start_time_seconds': voc_row['start_time_seconds'],
            'end_time_seconds': voc_row['end_time_seconds'],
            'seconds': overlap_seconds,
            'interaction_type': seg['interaction_type'],
            'total_segment_duration': total_segment_duration,
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec']
        })
    return results

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV):
    """
    Retrieve KCHI audio classifications from the SQLite database and map them to interaction segments.
    
    This function queries frame-level KCHI audio detections and converts each frame to a time interval,
    then maps these frame-level detections to interaction contexts.
    
    The mapped_vocalizations DataFrame contains the following columns:
    ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'interaction_type']

    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database containing AudioClassifications table.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'interaction_type']
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with mapped frame-level KCHI detections to interaction segments
    """
    # Load segments DataFrame
    segments_df = pd.read_csv(segments_csv_path)
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)

    # Connect to the SQLite database and query KCHI audio classifications with video names
    conn = sqlite3.connect(db_path)
    audio_query = """
    SELECT ac.video_id, vid.video_name, ac.frame_number, ac.has_kchi
    FROM AudioClassifications ac
    JOIN Videos vid ON ac.video_id = vid.video_id
    WHERE ac.has_kchi = 1
    ORDER BY ac.video_id, ac.frame_number
    """
    kchi_frames = pd.read_sql_query(audio_query, conn)
    conn.close()
    
    # Handle case where no KCHI speech is detected
    if len(kchi_frames) == 0:
        print("⚠️ Warning: No KCHI speech frames found in audio classifications")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Convert frame numbers to time using fps
    kchi_frames['start_time_seconds'] = kchi_frames['frame_number'] / DataConfig.FPS
    kchi_frames['end_time_seconds'] = (kchi_frames['frame_number'] + 1) / DataConfig.FPS  # Each frame represents 1/fps seconds
    
    # Add vocalization_id for compatibility (just use index)
    kchi_frames['vocalization_id'] = range(len(kchi_frames))
    
    # Rename to kchi_vocs to maintain compatibility with existing code
    kchi_vocs = kchi_frames.copy()
    
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
    1. Extracts KCHI (key child) audio classifications from the database
    2. Converts frame-level detections to frame-level time intervals  
    3. Maps frame-level KCHI detections to interaction contexts (Alone, Co-present Silent, Interacting)  
    4. Outputs mapped frame-level data with overlap durations
    5. Saves results to CSV
    
    Parameters
    ----------
    db_path : Path, optional
        Path to the SQLite database. Defaults to DataPaths.INFERENCE_DB_PATH
    segments_csv_path : Path, optional
        Path to interaction segments CSV. Defaults to Inference.INTERACTION_SEGMENTS_CSV
    """   
    print("🗣️ RESEARCH QUESTION 2: CHILD LANGUAGE PRODUCTION ANALYSIS")
    print("=" * 70)
    print("Analyzing key child audio classifications by interaction context...")
    
    # Step 1: Map audio classifications to interaction segments
    mapped_vocalizations = map_vocalizations_to_segments()
    
    # Handle empty result
    if len(mapped_vocalizations) == 0:
        print("❌ No KCHI speech data found. Exiting analysis.")
        return pd.DataFrame()

    # Step 2: Use mapped vocalizations directly    
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
    
    # Step 3: Remove overlapping vocalizations and calculate true speech time per segment
    segment_results = []
    
    # Process each unique segment separately to handle overlaps
    for segment_key, segment_data in mapped_vocalizations.groupby([
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time'
    ]):
        child_id, age_at_recording, interaction_type, seg_start, seg_end = segment_key
        
        # Get all frame intervals within this interaction segment
        intervals = []
        for _, row in segment_data.iterrows():
            # Use the actual frame times within the interaction segment
            voc_start = max(row['start_time_seconds'], row['segment_start_time'])
            voc_end = min(row['end_time_seconds'], row['segment_end_time'])
            if voc_end > voc_start:  # Valid interval
                intervals.append((voc_start, voc_end))
        
        # Merge overlapping intervals to get true speech time
        merged_intervals, total_speech_seconds = merge_overlapping_intervals(intervals)
        
        # Get segment duration (should be consistent for all rows in this segment)
        segment_duration = segment_data['total_segment_duration'].iloc[0]
        
        segment_results.append({
            'child_id': child_id,
            'age_at_recording': age_at_recording,
            'interaction_type': interaction_type,
            'segment_start_time': seg_start,
            'segment_end_time': seg_end,
            'total_speech_seconds': total_speech_seconds,
            'total_segment_duration': segment_duration,
            'num_merged_intervals': len(merged_intervals)
        })
    
    segment_totals = pd.DataFrame(segment_results)
    
    # Calculate minutes and percentages using overlap-corrected speech time
    segment_totals['kchi_speech_minutes'] = segment_totals['total_speech_seconds'] / 60
    segment_totals['segment_duration_minutes'] = segment_totals['total_segment_duration'] / 60
    
    # Recalculate speech activity percentage for aggregated data (overlap-corrected)
    segment_totals['speech_activity_percent'] = (
        segment_totals['total_speech_seconds'] / segment_totals['total_segment_duration']
    ).fillna(0)
    
    # Verify no percentages exceed 100%
    max_percentage = segment_totals['speech_activity_percent'].max()
    if max_percentage > 1.0:
        print(f"⚠️  Warning: Maximum percentage is {max_percentage:.1%}, which exceeds 100%")
    
    # Sort segment totals
    segment_totals = segment_totals.sort_values([
        'child_id', 'age_at_recording', 'segment_start_time'
    ]).reset_index(drop=True)
    
    # Step 4: Save results
    # Select final columns for output
    final_output = segment_totals[[
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time', 
        'kchi_speech_minutes', 'segment_duration_minutes', 'speech_activity_percent'
    ]].copy()
    
    # Save aggregated segment totals only
    final_output.to_csv(Inference.KCS_SUMMARY_CSV, index=False)
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📄 Output saved to: {Inference.KCS_SUMMARY_CSV}")
    print("=" * 70)
    
    return segment_totals

if __name__ == "__main__":
    main()