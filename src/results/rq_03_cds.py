import pandas as pd
import numpy as np
from pathlib import Path
from constants import DataPaths, Inference, AudioClassification
from utils import extract_child_id  

def parse_rttm(file_path: Path = AudioClassification.VTC_RTTM_FILE) -> pd.DataFrame:
    data = []
    if not file_path.exists(): return pd.DataFrame()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                speech_type = parts[7]
                if speech_type not in ['KCDS', 'OHS']: continue 
                data.append({
                    'video_name': parts[1],
                    'start_time_seconds': float(parts[3]),
                    'end_time_seconds': float(parts[3]) + float(parts[4]),
                    'speech_type': speech_type
                })
    return pd.DataFrame(data)

def merge_overlapping_intervals(intervals):
    if not intervals: return [], 0.0
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged, sum(end - start for start, end in merged)

def main():
    print("🗣️ RQ 04: SPEECH EXPOSURE ANALYSIS")
    print("="*70)
    
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    all_vocalizations = parse_rttm()
    
    # Standardize segments_df to match RTTM video names
    segments_df['child_id'] = segments_df['video_name'].apply(extract_child_id)
    
    exposure_categories = ['TOTAL', 'KCDS_ONLY', 'OHS_ONLY']
    final_rows = []

    for _, seg in segments_df.iterrows():
        # Get vocalizations for this segment
        group = all_vocalizations[
            (all_vocalizations['video_name'] == seg['video_name']) & 
            (all_vocalizations['start_time_seconds'] < seg['end_time_sec']) & 
            (all_vocalizations['end_time_seconds'] > seg['start_time_sec'])
        ].copy()

        # Calculate overlap with the segment boundaries
        if not group.empty:
            group['ov_start'] = np.maximum(group['start_time_seconds'], seg['start_time_sec'])
            group['ov_end'] = np.minimum(group['end_time_seconds'], seg['end_time_sec'])

        duration = seg['end_time_sec'] - seg['start_time_sec']

        for exp_type in exposure_categories:
            if group.empty:
                speech_seconds = 0.0
            else:
                if exp_type == 'TOTAL':
                    data = group
                elif exp_type == 'KCDS_ONLY':
                    data = group[group['speech_type'] == 'KCDS']
                else:
                    data = group[group['speech_type'] == 'OHS']
                
                intervals = list(zip(data['ov_start'], data['ov_end']))
                _, speech_seconds = merge_overlapping_intervals(intervals)

            final_rows.append({
                'child_id': seg['child_id'],
                'video_name': seg['video_name'],
                'age_at_recording': seg['age_at_recording'],
                'interaction_type': seg['interaction_type'],
                'segment_start_time': seg['start_time_sec'],
                'segment_end_time': seg['end_time_sec'],
                'exposure_type': exp_type,
                'total_speech_seconds': speech_seconds,
                'total_segment_duration': duration,
                'segment_duration_minutes': duration/60
            })

    final_df = pd.DataFrame(final_rows)
    final_df['exposure_percent'] = (final_df['total_speech_seconds'] / final_df['total_segment_duration']).fillna(0)
    
    # Final cleanup and sort
    final_df = final_df.sort_values(['video_name', 'segment_start_time', 'exposure_type'])
    final_df.to_csv(Inference.CDS_SUMMARY_CSV, index=False)
    print(f"✅ Clean results saved to {Inference.CDS_SUMMARY_CSV}")

if __name__ == "__main__":
    main()