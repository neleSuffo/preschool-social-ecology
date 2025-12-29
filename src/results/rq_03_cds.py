import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths, Inference
from utils import extract_child_id  
from config import DataConfig

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
    
    # Iterate through each overlapping segment to calculate time-based overlap
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
            'speech_type': voc_row['speech_type'],  # Add speech type (CDS or OHS)
            'interaction_type': seg['interaction_type'],
            'total_segment_duration': total_segment_duration,
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec']
        })
    return results

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV, frame_metadata_csv: Path = Inference.FRAME_LEVEL_INTERACTIONS_CSV):
    """
    Retrieve both CDS and OHS classifications from the SQLite database and map them to interaction segments.
    
    This function queries frame-level CDS and OHS audio detections, converts each frame to a time interval,
    and creates separate records for each speech type, then maps these to interaction contexts.
    
    The mapped DataFrame contains the following columns:
    ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'speech_type', 'interaction_type']

    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database containing AudioClassifications table.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'category']
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with mapped frame-level CDS and OHS detections to interaction segments,
        with speech_type column indicating 'CDS' or 'OHS'
    """
    # 1. Load data from CSVs
    segments_df = pd.read_csv(segments_csv_path)
    # This CSV contains the 'is_media_interaction' column
    metadata_df = pd.read_csv(frame_metadata_csv) 
    
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=';')

    # 2. Get Audio Classifications from SQL
    conn = sqlite3.connect(db_path)
    audio_query = """
    SELECT ac.video_id, vid.video_name, ac.frame_number, ac.has_cds, ac.has_ohs
    FROM AudioClassifications ac
    JOIN Videos vid ON ac.video_id = vid.video_id
    WHERE ac.has_cds = 1 OR ac.has_ohs = 1
    """
    exposure_frames = pd.read_sql_query(audio_query, conn)
    conn.close()

    # 3. Merge SQL data with CSV metadata to get 'is_media_interaction'
    # We merge on video_name and frame_number to ensure a perfect match
    exposure_frames = exposure_frames.merge(
        metadata_df[['video_name', 'frame_number', 'is_media_interaction']], 
        on=['video_name', 'frame_number'], 
        how='left'
    )

    # 4. Filter out Media Interaction frames
    # We drop any frames where is_media_interaction is 1
    exposure_frames = exposure_frames[exposure_frames['is_media_interaction'] != 1].copy()

    # 5. Handle empty data after filtering
    if exposure_frames.empty:
        print("⚠️ All speech frames were filtered out (media interaction).")
        return pd.DataFrame()

    # Convert frame numbers to time using fps
    exposure_frames['start_time_seconds'] = exposure_frames['frame_number'] / DataConfig.FPS
    exposure_frames['end_time_seconds'] = (exposure_frames['frame_number'] + 1) / DataConfig.FPS
    
    exposure_frames['child_id'] = exposure_frames['video_name'].apply(extract_child_id)
    exposure_frames = exposure_frames.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')
    
    all_speech_records = []
    
    for _, frame_row in exposure_frames.iterrows():
        base_record = {
            'video_id': frame_row['video_id'],
            'child_id': frame_row['child_id'],
            'age_at_recording': frame_row['age_at_recording'],
            'start_time_seconds': frame_row['start_time_seconds'],
            'end_time_seconds': frame_row['end_time_seconds'],
            'vocalization_id': len(all_speech_records)
        }
        
        if frame_row['has_cds'] == 1:
            record = base_record.copy()
            record['speech_type'] = 'CDS'
            all_speech_records.append(record)
            
        if frame_row['has_ohs'] == 1:
            record = base_record.copy()
            record['speech_type'] = 'OHS'
            all_speech_records.append(record)
    
    other_vocs = pd.DataFrame(all_speech_records)
    
    # [Rest of the mapping logic remains the same] ...
    voc_videos = set(other_vocs['video_id'].unique())
    seg_videos = set(segments_df['video_id'].unique())
    common_videos = voc_videos & seg_videos    
    mapped_rows = []
    
    for video_id in common_videos:       
        video_vocalizations = other_vocs[other_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]

        for _, voc_row in video_vocalizations.iterrows():
            mapped_rows.extend(row_wise_mapping(voc_row, video_segments))
    
    return pd.DataFrame(mapped_rows)

def main():
    """
    Main function to analyze speech exposure (CDS + OHS) and CDS-specific exposure by interaction context.
    
    Research Question 4: How much speech is the key child exposed to (total and CDS-specific)?
    
    This script:
    1. Extracts both CDS and OHS frame-level detections from the AudioClassifications table
    2. Converts frame detections to time intervals with speech type labels
    3. Maps them to interaction segments (Alone, Co-present, Interacting)  
    4. Calculates both total exposure (CDS + OHS) and CDS-only exposure
    5. Saves results to CSV
    """   
    print("🗣️ RESEARCH QUESTION 4: SPEECH EXPOSURE ANALYSIS")
    print("=" * 70)
    print("Analyzing speech exposure (CDS + OHS) and CDS-specific exposure by interaction context...")
    
    # Step 1: Map speech to interaction segments
    mapped_speech = map_vocalizations_to_segments()

    if mapped_speech.empty:
        print("\n⚠️ No CDS or OHS speech found. Exiting analysis.")
        return
            
    exposure_categories = ['TOTAL', 'CDS_ONLY', 'OHS_ONLY']
    total_exposure_results = []

    # Get the unique segments to iterate through
    unique_segments = mapped_speech.groupby([
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time'
    ])
    
    for segment_key, segment_data in unique_segments:
        child_id, age_at_recording, interaction_type, seg_start, seg_end = segment_key
        segment_duration = segment_data['total_segment_duration'].iloc[0]

        for exp_type in exposure_categories:
            intervals = []
            
            # Filter intervals based on type
            if exp_type == 'TOTAL':
                filtered_data = segment_data
            elif exp_type == 'CDS_ONLY':
                filtered_data = segment_data[segment_data['speech_type'] == 'CDS']
            else: # OHS_ONLY
                filtered_data = segment_data[segment_data['speech_type'] == 'OHS']

            for _, row in filtered_data.iterrows():
                frame_start = max(row['start_time_seconds'], row['segment_start_time'])
                frame_end = min(row['end_time_seconds'], row['segment_end_time'])
                if frame_end > frame_start:
                    intervals.append((frame_start, frame_end))
            
            # Merge to get duration for this specific category
            merged, speech_seconds = merge_overlapping_intervals(intervals)
            
            total_exposure_results.append({
                'child_id': child_id,
                'age_at_recording': age_at_recording,
                'interaction_type': interaction_type,
                'segment_start_time': seg_start,
                'segment_end_time': seg_end,
                'exposure_type': exp_type,
                'total_speech_seconds': speech_seconds,
                'total_segment_duration': segment_duration,
                'num_merged_intervals': len(merged)
            })
    
    segment_totals = pd.DataFrame(total_exposure_results)
    
    # Step 3: Calculate exposure percentages
    segment_totals['speech_minutes'] = segment_totals['total_speech_seconds'] / 60
    segment_totals['segment_duration_minutes'] = segment_totals['total_segment_duration'] / 60
    
    # Calculate exposure percentage
    segment_totals['exposure_percent'] = (
        segment_totals['total_speech_seconds'] / segment_totals['total_segment_duration']
    ).fillna(0)
    
    # Verify no percentages exceed 100%
    max_total_exposure = segment_totals[segment_totals['exposure_type'] == 'TOTAL']['exposure_percent'].max()
    max_cds_exposure = segment_totals[segment_totals['exposure_type'] == 'CDS_ONLY']['exposure_percent'].max()
    max_ohs_exposure = segment_totals[segment_totals['exposure_type'] == 'OHS_ONLY']['exposure_percent'].max()
    
    for exposure_check, max_value in {
        'TOTAL': max_total_exposure,
        'CDS_ONLY': max_cds_exposure,
        'OHS_ONLY': max_ohs_exposure
    }.items():
        if max_value > 1.0:
            print(f"⚠️ Warning: {exposure_check} exposure percentage exceeds 100% ({max_value*100:.2f}%)")
    
    # Sort and prepare final output
    segment_totals = segment_totals.sort_values([
        'child_id', 'age_at_recording', 'segment_start_time', 'exposure_type'
    ]).reset_index(drop=True)

    # Select final columns for output
    final_output = segment_totals[[
        'child_id', 'age_at_recording', 'interaction_type', 'exposure_type',
        'segment_start_time', 'segment_end_time', 
        'speech_minutes', 'segment_duration_minutes', 'exposure_percent'
    ]].copy()
        
    # Clean and convert age_at_recording to float
    final_output['age_at_recording'] = (
    final_output['age_at_recording']
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.replace(',', '.', regex=False)
    .str.strip()
    )

    final_output['age_at_recording'] = pd.to_numeric(
        final_output['age_at_recording'],
        errors='coerce'
    )
    # Save results with both total and CDS-only exposure
    final_output.to_csv(Inference.CDS_SUMMARY_CSV, index=False)
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📄 Output saved to: {Inference.CDS_SUMMARY_CSV}")
    print("=" * 70)
    
    return final_output

if __name__ == "__main__":
    main()