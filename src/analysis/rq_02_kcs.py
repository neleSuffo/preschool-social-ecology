import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths, Inference
from config import DataConfig

def merge_overlapping_intervals(intervals):
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

def main():
    print("🗣️ RESEARCH QUESTION 2: CHILD LANGUAGE PRODUCTION ANALYSIS")
    print("=" * 70)
    
    # 1. Load segments and log initial count
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    initial_child_ids = set(segments_df['child_id'].unique())
    print(f"📊 [STAGE 1] Unique children in segments CSV: {len(initial_child_ids)}")
    
    # 2. Extract ALL speech frames
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    audio_query = "SELECT video_id, frame_number FROM AudioClassifications WHERE has_kchi = 1"
    kchi_frames = pd.read_sql_query(audio_query, conn)
    conn.close()
    
    db_video_ids = set(kchi_frames['video_id'].unique())
    print(f"📊 [STAGE 2] Unique videos with speech in DB: {len(db_video_ids)}")

    # Time conversion
    kchi_frames['start_time_seconds'] = kchi_frames['frame_number'] / DataConfig.FPS
    kchi_frames['end_time_seconds'] = (kchi_frames['frame_number'] + DataConfig.FRAME_STEP_INTERVAL) / DataConfig.FPS

    segment_results = []
    processed_videos = set()
    
    # 3. Iterate through segments
    for video_id, video_segments in segments_df.groupby('video_id'):
        processed_videos.add(video_id)
        vid_speech = kchi_frames[kchi_frames['video_id'] == video_id]
        
        for _, seg in video_segments.iterrows():
            overlap_frames = vid_speech[
                (vid_speech['start_time_seconds'] < seg['end_time_sec']) &
                (vid_speech['end_time_seconds'] > seg['start_time_sec'])
            ]
            
            if len(overlap_frames) == 0:
                total_speech_seconds = 0.0
            else:
                intervals = [(max(f['start_time_seconds'], seg['start_time_sec']), 
                              min(f['end_time_seconds'], seg['end_time_sec'])) 
                             for _, f in overlap_frames.iterrows()]
                _, total_speech_seconds = merge_overlapping_intervals(intervals)
            
            segment_results.append({
                'child_id': seg['child_id'],
                'video_id': video_id, # Added for debugging
                'total_speech_seconds': total_speech_seconds,
                'total_segment_duration': seg['duration_sec'],
                'age_raw': seg['age_at_recording'] # Keep raw for check
            })
    
    final_output = pd.DataFrame(segment_results)
    
    # 4. Age Cleaning & Final Checks
    final_output['age_at_recording'] = pd.to_numeric(
        final_output['age_raw'].astype(str).str.replace('"', '').str.replace(',', '.'), 
        errors='coerce'
    )
    
    # Check for children lost during age conversion
    invalid_age_children = final_output[final_output['age_at_recording'].isna()]['child_id'].unique()
    
    # Aggregation
    child_level = final_output.dropna(subset=['age_at_recording']).groupby('child_id').agg({
        'total_speech_seconds': 'sum',
        'total_segment_duration': 'sum',
        'age_at_recording': 'min'
    }).reset_index()

    # --- THE AUDIT REPORT ---
    print("\n" + "🔍 DATA FLOW AUDIT REPORT" + "\n" + "-"*30)
    print(f"1. Children started with (CSV):       {len(initial_child_ids)}")
    print(f"2. Children after segment processing:  {len(final_output['child_id'].unique())}")
    print(f"3. Children with invalid Age values:   {len(invalid_age_children)}")
    print(f"4. Children in final summary:          {len(child_level)}")
    
    if len(initial_child_ids) > len(child_level):
        lost_ids = initial_child_ids - set(child_level['child_id'])
        print(f"⚠️ LOST CHILD IDs (first 5): {list(lost_ids)[:5]}")
    
    child_level.to_csv(Inference.GLOBAL_KCS_SUMMARY_CSV, index=False)
    return final_output

if __name__ == "__main__":
    main()