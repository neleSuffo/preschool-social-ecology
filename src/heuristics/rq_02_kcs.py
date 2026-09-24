import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from constants import Analysis
from src.heuristics.utils import parse_rttm, merge_overlapping_intervals, get_child_fold_boundaries

def main(output_folder: Path = None,
         use_folds: bool = True,
         use_ground_truth: bool = False):
    """
    Generates segment-level and child-level summaries of Key Child Directed Speech (KCDS) exposure by merging segment-level interactions with vocalization data extracted from RTTM files.

    Parameters
    ----------
    output_folder : Path, optional
        Optional output folder path to save the interaction composition CSV. If not provided, saves to default location defined in constants, by default None
    use_folds : bool, optional
        If True, calculates per-fold metrics (5-fold cross-validation style). If False, calculates metrics across entire recording, by default True
    use_ground_truth: bool. optional
        If true, match the audio detections on the ground truth segments only, by default False
    """
    fold_mode = "WITH per-fold metrics" if use_folds else "WITHOUT per-fold metrics (overall only)"
    gt_addition = "for ground truth dataset" if use_ground_truth else "''"
    print("🗣️ RESEARCH QUESTION 2: CHILD LANGUAGE PRODUCTION ANALYSIS")
    print(f"   Mode: {fold_mode} {gt_addition}")
    print("=" * 70)
    
    # 1. Load segments file
    if use_ground_truth:
        segments_path = Analysis.GROUND_TRUTH_SEGMENTS_CSV
    else:
        segments_path = (
            output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name
            if output_folder
            else Analysis.INTERACTION_SEGMENTS_CSV
        )
    segments_df = pd.read_csv(segments_path)
    
    # 2. Extract Key Child (KCHI) speech from RTTM file
    kchi_vocalizations = parse_rttm(target_speech_types=['KCHI'])

    if kchi_vocalizations.empty:
        print("⚠️ Warning: No KCHI vocalizations found in RTTM file.")
        
    # 3. Pre-calculate boundaries and create global timeline offsets
    # Use max end timestamp per video to avoid overlap when segments have gaps (especially in ground truth)
    video_stats = (
        segments_df.groupby(['child_id', 'video_name'])['end_time_sec']
        .max()
        .reset_index(name='video_span_sec')
    )
    video_stats = video_stats.sort_values(['child_id', 'video_name'])   
    
    # Reset cumsum for every child
    video_stats['offset_raw'] = video_stats.groupby('child_id')['video_span_sec'].shift(1).fillna(0)
    video_stats['offset'] = video_stats.groupby('child_id')['offset_raw'].transform('cumsum')
    
    # Merge back using both keys to be safe
    segments_df = segments_df.merge(video_stats[['child_id', 'video_name', 'offset']], on=['child_id', 'video_name'])
    
    # Create local-to-global timestamps
    segments_df['global_start'] = segments_df['start_time_sec'] + segments_df['offset']
    segments_df['global_end'] = segments_df['end_time_sec'] + segments_df['offset']

    # 4. Pre-calculate fold boundaries (either 5-fold or single fold across entire duration)
    if use_folds:
        fold_map = get_child_fold_boundaries(segments_df)
    else:
        # Create a single fold covering the entire duration for each child
        fold_map = {}
        for child_id in segments_df['child_id'].unique():
            child_data = segments_df[segments_df['child_id'] == child_id]
            f_start = child_data['global_start'].min()
            f_end = child_data['global_end'].max()
            fold_map[child_id] = [(f_start, f_end)]
    
    final_rows = []

    # Iterate by Child -> Fold -> Segment
    for child_id, folds in fold_map.items():
        # Get all segments for this child
        child_segs = segments_df[segments_df['child_id'] == child_id]
        
        for fold_idx, (f_start, f_end) in enumerate(folds):
            fold_num = fold_idx + 1
        
            for _, seg in child_segs.iterrows():
                # --- STEP A: Clip the segment to the fold boundaries ---
                overlap_start = max(seg['global_start'], f_start)
                overlap_end = min(seg['global_end'], f_end)
                
                if overlap_start < overlap_end:
                    # This piece of the segment belongs to this fold!
                    current_duration = overlap_end - overlap_start
                    
                    # STEP B: Convert global overlap back to local time for RTTM filtering
                    local_overlap_start = overlap_start - seg['offset']
                    local_overlap_end = overlap_end - seg['offset']
        
                    group = kchi_vocalizations[
                        (kchi_vocalizations['video_name'] == seg['video_name']) & 
                        (kchi_vocalizations['start_time_seconds'] < local_overlap_end) & 
                        (kchi_vocalizations['end_time_seconds'] > local_overlap_start)
                    ].copy()
                                
                    if not group.empty:
                        # STEP C: TRIPLE CLIP (Vocal | Segment | Fold) using local times
                        group['clipped_start'] = np.maximum(group['start_time_seconds'], local_overlap_start)
                        group['clipped_end'] = np.minimum(group['end_time_seconds'], local_overlap_end)
                    
                        intervals = list(zip(group['clipped_start'], group['clipped_end']))
                        _, total_speech_seconds = merge_overlapping_intervals(intervals)
                    else:
                        total_speech_seconds = 0.0
                    
                    final_rows.append({
                        'child_id': seg['child_id'],
                        'fold': fold_num,
                        'video_name': seg['video_name'],
                        'age_at_recording': seg['age_at_recording'],
                        'interaction_type': seg['interaction_type'],
                        'segment_start_time': seg['start_time_sec'],
                        'segment_end_time': seg['end_time_sec'],
                        'total_speech_seconds': total_speech_seconds,
                        'total_segment_duration': current_duration,
                        'segment_duration_minutes': current_duration / 60
                    })
    
    # 4. Final Aggregation and Derived Columns
    final_df = pd.DataFrame(final_rows)
    final_df['speech_activity_percent'] = (
        final_df['total_speech_seconds'] / final_df['total_segment_duration']
    ).fillna(0)
    final_df['kchi_speech_minutes'] = final_df['total_speech_seconds'] / 60
    final_df['segment_duration_minutes'] = final_df['total_segment_duration'] / 60
    
    # Age Cleaning
    final_df['age_at_recording'] = (
        final_df['age_at_recording']
        .astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_df['age_at_recording'] = pd.to_numeric(final_df['age_at_recording'], errors='coerce')
    
    # Save segment-level results
    suffix = "_gt" if use_ground_truth else ""
    base_folder = (
        output_folder if output_folder else Analysis.KCS_SUMMARY_CSV.parent
    )
    output_path_kcs = base_folder / f"{Analysis.KCS_SUMMARY_CSV.stem}{suffix}.csv"
    final_df.to_csv(output_path_kcs, index=False)
    print(f"✅ KCS state summary saved to: {output_path_kcs}")
    
    # ----- PART 2A: Child-Level Aggregation ------
    # Aggregation strategy depends on use_folds flag
    if use_folds:
        # Group by child AND fold to preserve the 5 data points per child
        child_level_summary = final_df.groupby(['child_id', 'fold']).agg({
            'total_speech_seconds': 'sum',
            'total_segment_duration': 'sum',
            'age_at_recording': 'min'
        }).reset_index()
    else:
        # Group by child only (single fold, so fold column will be 1)
        child_level_summary = final_df.groupby('child_id').agg({
            'total_speech_seconds': 'sum',
            'total_segment_duration': 'sum',
            'age_at_recording': 'min'
        }).reset_index()

    # Calculate child-level percentage
    child_level_summary['speech_activity_percent'] = (
        child_level_summary['total_speech_seconds'] / 
        child_level_summary['total_segment_duration']
    ).fillna(0)

    # Convert to minutes for easier reading
    child_level_summary['total_speech_minutes'] = child_level_summary['total_speech_seconds'] / 60
    child_level_summary['total_recording_minutes'] = child_level_summary['total_segment_duration'] / 60

    # Save child-level results
    suffix = "_gt" if use_ground_truth else ""
    base_folder = (
        output_folder if output_folder else Analysis.GLOBAL_KCS_SUMMARY_CSV.parent
    )
    output_path_gkcs = base_folder / f"{Analysis.GLOBAL_KCS_SUMMARY_CSV.stem}{suffix}.csv"
    child_level_summary.to_csv(output_path_gkcs, index=False)
    print(f"✅ Child-level summary saved to: {output_path_gkcs}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KCS (Key Child Speech) analysis summaries")
    parser.add_argument('--output_folder', type=str, default=None, help="Optional output folder path to save the interaction composition CSV")
    parser.add_argument('--use_folds', action='store_true', default=False, help="Calculate per-fold metrics (5-fold cross-validation). If not set, calculates overall metrics only.")
    parser.add_argument('--use_gt', action='store_true', default=False, help="Map vocalizations onto manual ground-truth segments instead of pipeline segments")
    args = parser.parse_args()
        
    main(output_folder=Path(args.output_folder) if args.output_folder else None,
         use_folds=args.use_folds,
         use_ground_truth=args.use_gt,
         )