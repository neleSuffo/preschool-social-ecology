import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from constants import DataPaths, Inference
from utils import extract_child_id  
from config import DataConfig

def merge_overlapping_intervals(intervals):
    """Merge overlapping time intervals to avoid double-counting speech time."""
    if not intervals:
        return [], 0.0
    
    intervals = sorted(intervals)
    merged = [intervals[0]]
    
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    
    total_duration = sum(end - start for start, end in merged)
    return merged, total_duration

def row_wise_mapping(voc_row, video_segments_df):
    """Map speech intervals to interaction segments."""
    overlaps = video_segments_df[
        (video_segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (video_segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ].copy()
    
    results = []
    for _, seg in overlaps.iterrows():
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        overlap_end = min(voc_row['end_time_seconds'], seg['end_time_sec'])
        overlap_seconds = max(0, overlap_end - overlap_start)
        
        results.append({
            'video_id': voc_row['video_id'],
            'child_id': voc_row['child_id'],
            'age_at_recording': voc_row['age_at_recording'],
            'start_time_seconds': voc_row['start_time_seconds'],
            'end_time_seconds': voc_row['end_time_seconds'],
            'seconds': overlap_seconds,
            'speech_type': voc_row['speech_type'],
            'interaction_type': seg['interaction_type'],
            'total_segment_duration': seg['end_time_sec'] - seg['start_time_sec'],
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec']
        })
    return results

def map_vocalizations_to_segments():
    # 1. Load data
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    metadata_df = pd.read_csv(Inference.FRAME_LEVEL_INTERACTIONS_CSV) 
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=';')

    # 2. SQL Query
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    audio_query = """
    SELECT ac.video_id, vid.video_name, ac.frame_number, ac.has_cds, ac.has_ohs
    FROM AudioClassifications ac
    JOIN Videos vid ON ac.video_id = vid.video_id
    WHERE ac.has_cds = 1 OR ac.has_ohs = 1
    """
    exposure_frames = pd.read_sql_query(audio_query, conn)
    conn.close()

    if exposure_frames.empty:
        return pd.DataFrame(), segments_df

    # 3. Merge Metadata and Filter Media
    exposure_frames = exposure_frames.merge(
        metadata_df[['video_name', 'frame_number', 'is_media_interaction']], 
        on=['video_name', 'frame_number'], 
        how='left'
    )
    exposure_frames['is_media_interaction'] = exposure_frames['is_media_interaction'].fillna(0)
    exposure_frames = exposure_frames[exposure_frames['is_media_interaction'] != 1].copy()

    # 4. Prepare Intervals (Vectorized)
    exposure_frames['start_time_seconds'] = exposure_frames['frame_number'] / DataConfig.FPS
    exposure_frames['end_time_seconds'] = (exposure_frames['frame_number'] + 1) / DataConfig.FPS
    exposure_frames['child_id'] = exposure_frames['video_name'].apply(extract_child_id)
    exposure_frames = exposure_frames.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')

    # Separate CDS and OHS records
    cds_recs = exposure_frames[exposure_frames['has_cds'] == 1].copy()
    cds_recs['speech_type'] = 'CDS'
    ohs_recs = exposure_frames[exposure_frames['has_ohs'] == 1].copy()
    ohs_recs['speech_type'] = 'OHS'
    
    other_vocs = pd.concat([cds_recs, ohs_recs], ignore_index=True)

    # 5. Mapping logic
    voc_videos = set(other_vocs['video_id'].unique())
    common_videos = voc_videos & set(segments_df['video_id'].unique())
    mapped_rows = []
    
    for video_id in common_videos:       
        video_vocalizations = other_vocs[other_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]
        for _, voc_row in video_vocalizations.iterrows():
            mapped_rows.extend(row_wise_mapping(voc_row, video_segments))
    
    return pd.DataFrame(mapped_rows), segments_df

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
    print("🗣️ RQ 04: SPEECH EXPOSURE ANALYSIS (Excl. Media)")
    print("=" * 70)
    
    mapped_speech, original_segments = map_vocalizations_to_segments()

    if mapped_speech.empty:
        print("⚠️ No valid speech detected after media filtering.")
        return

    exposure_categories = ['TOTAL', 'CDS_ONLY', 'OHS_ONLY']
    results = []

    # Grouping key for all segments
    group_cols = ['child_id', 'age_at_recording', 'interaction_type', 'segment_start_time', 'segment_end_time']

    # Process segments that have speech
    grouped = mapped_speech.groupby(group_cols)
    for segment_key, segment_data in grouped:
        duration = segment_data['total_segment_duration'].iloc[0]
        
        for exp_type in exposure_categories:
            if exp_type == 'TOTAL':
                data = segment_data
            elif exp_type == 'CDS_ONLY':
                data = segment_data[segment_data['speech_type'] == 'CDS']
            else:
                data = segment_data[segment_data['speech_type'] == 'OHS']
            
            intervals = list(zip(data['start_time_seconds'], data['end_time_seconds']))
            _, speech_seconds = merge_overlapping_intervals(intervals)
            
            results.append({
                **dict(zip(group_cols, segment_key)),
                'exposure_type': exp_type,
                'total_speech_seconds': speech_seconds,
                'total_segment_duration': duration
            })

    # --- ADD MISSING ZERO RECORDS ---
    # Convert original segments to same format as results
    res_df = pd.DataFrame(results)
    
    # Ensure age is numeric for matching
    original_segments['age_at_recording'] = pd.to_numeric(original_segments['age_at_recording'].astype(str).str.replace(',', '.'), errors='coerce')
    
    all_segments_list = []
    for _, seg in original_segments.iterrows():
        for exp_type in exposure_categories:
            all_segments_list.append({
                'child_id': extract_child_id(seg['video_name']),
                'age_at_recording': seg['age_at_recording'],
                'interaction_type': seg['interaction_type'],
                'segment_start_time': seg['start_time_sec'],
                'segment_end_time': seg['end_time_sec'],
                'exposure_type': exp_type,
                'total_speech_seconds': 0.0,
                'total_segment_duration': seg['end_time_sec'] - seg['start_time_sec']
            })
    
    full_template = pd.DataFrame(all_segments_list)
    
    # Combine and keep the one with the most speech
    final_df = pd.concat([res_df, full_template]).sort_values('total_speech_seconds', ascending=False)
    final_df = final_df.drop_duplicates(subset=group_cols + ['exposure_type'])

    # Final Metrics
    final_df['speech_minutes'] = final_df['total_speech_seconds'] / 60
    final_df['segment_duration_minutes'] = final_df['total_segment_duration'] / 60
    final_df['exposure_percent'] = (final_df['total_speech_seconds'] / final_df['total_segment_duration']).fillna(0)

    # Save
    final_df.to_csv(Inference.CDS_SUMMARY_CSV, index=False)
    print(f"✅ Success! Saved to {Inference.CDS_SUMMARY_CSV}")

if __name__ == "__main__":
    main()