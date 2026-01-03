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
    
    # 1. Load the segments file (The "Master List" of intervals)
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    
    # 2. Extract ALL speech frames from the database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    audio_query = """
    SELECT ac.video_id, ac.frame_number
    FROM AudioClassifications ac
    WHERE ac.has_kchi = 1
    """
    kchi_frames = pd.read_sql_query(audio_query, conn)
    conn.close()
    
    # We use DataConfig.AUDIO_STEP to ensure one detection covers the whole window
    kchi_frames['start_time_seconds'] = kchi_frames['frame_number'] / DataConfig.FPS
    kchi_frames['end_time_seconds'] = (kchi_frames['frame_number'] + DataConfig.FRAME_STEP_INTERVAL) / DataConfig.FPS

    segment_results = []
    
    # 3. Iterate through EVERY segment from the segments CSV
    for video_id, video_segments in segments_df.groupby('video_id'):
        vid_speech = kchi_frames[kchi_frames['video_id'] == video_id]
        
        for _, seg in video_segments.iterrows():
            # Find detections that overlap with this segment
            overlap_frames = vid_speech[
                (vid_speech['start_time_seconds'] < seg['end_time_sec']) &
                (vid_speech['end_time_seconds'] > seg['start_time_sec'])
            ]
            
            if len(overlap_frames) == 0:
                total_speech_seconds = 0.0
            else:
                # Clip frame intervals to segment boundaries
                intervals = []
                for _, frame in overlap_frames.iterrows():
                    start = max(frame['start_time_seconds'], seg['start_time_sec'])
                    end = min(frame['end_time_seconds'], seg['end_time_sec'])
                    intervals.append((start, end))
                
                # Merge overlapping/adjacent intervals
                _, total_speech_seconds = merge_overlapping_intervals(intervals)
            
            segment_results.append({
                'child_id': seg['child_id'],
                'video_name': seg['video_name'],
                'age_at_recording': seg['age_at_recording'],
                'interaction_type': seg['interaction_type'],
                'segment_start_time': seg['start_time_sec'],
                'segment_end_time': seg['end_time_sec'],
                'total_speech_seconds': total_speech_seconds,
                'total_segment_duration': seg['duration_sec']
            })
    
    # 4. Final Aggregation and Derived Columns
    final_output = pd.DataFrame(segment_results)
    final_output['kchi_speech_minutes'] = final_output['total_speech_seconds'] / 60
    final_output['segment_duration_minutes'] = final_output['total_segment_duration'] / 60
    final_output['speech_activity_percent'] = (
        final_output['total_speech_seconds'] / final_output['total_segment_duration']
    ).fillna(0)
    
    # Age Cleaning
    final_output['age_at_recording'] = (
        final_output['age_at_recording']
        .astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_output['age_at_recording'] = pd.to_numeric(final_output['age_at_recording'], errors='coerce')

    # Save
    final_output.to_csv(Inference.KCS_SUMMARY_CSV, index=False)
    print(f"✅ KCS summary saved to: {Inference.KCS_SUMMARY_CSV}")
    
    return final_output

if __name__ == "__main__":
    main()