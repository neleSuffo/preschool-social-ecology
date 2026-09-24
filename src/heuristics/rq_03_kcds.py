import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from constants import Analysis
from src.heuristics.utils import parse_rttm, merge_overlapping_intervals, get_child_fold_boundaries

def main(
    output_folder: Path = None,
    use_folds: bool = True,
    use_ground_truth: bool = False,
):
    """
    Generates frame-level interaction composition by merging segment-level interactions with frame-level metadata.
    
    Parameters
    ----------
    output_folder : Path, optional
        Optional output folder path to save the interaction composition CSV. If not provided, saves to default location defined in constants, by default None
    use_folds : bool, optional
        If True, calculates per-fold metrics (5-fold cross-validation style). If False, calculates metrics across entire recording, by default True
    """
    fold_mode = "WITH per-fold metrics" if use_folds else "WITHOUT per-fold metrics (overall only)"
    print("🗣️ RESEARCH QUESTION 03: SPEECH EXPOSURE ANALYSIS")
    print(f"   Mode: {fold_mode}")
    print("="*70)
    
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
        
    # 2. Extract both KCDS and OHS vocalizations from RTTM file
    all_vocalizations = parse_rttm(target_speech_types=['KCDS', 'OHS'])

    if all_vocalizations.empty:
        print("⚠️ Warning: No OHS or KCDS vocalizations found in RTTM file.")
        
    # 3. GLOBAL TIMELINE LOGIC
    # Calculate video span using max timestamp to avoid overlap with sparse annotations
    video_stats = (segments_df.groupby(['child_id', 'video_name'])['end_time_sec'].max().reset_index(name='video_span_sec'))
    video_stats = video_stats.sort_values(['child_id', 'video_name'])   
    
    # Reset cumulative sum for every child
    video_stats['offset_raw'] = video_stats.groupby('child_id')['video_span_sec'].shift(1).fillna(0)
    video_stats['offset'] = video_stats.groupby('child_id')['offset_raw'].transform('cumsum')
    
    # Merge offsets back to segments
    segments_df = segments_df.merge(video_stats[['child_id', 'video_name', 'offset']], on=['child_id', 'video_name'])
    
    # Create global timestamps for accurate fold matching
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
        
    exposure_categories = ['TOTAL', 'KCDS_ONLY', 'OHS_ONLY']
    final_rows = []

    # 4. Iterate by Child -> Fold -> Segment
    for child_id, folds in fold_map.items():
        child_segs = segments_df[segments_df['child_id'] == child_id]
    
        for fold_idx, (f_start, f_end) in enumerate(folds):
            fold_num = fold_idx + 1
        
            for _, seg in child_segs.iterrows():
                # --- STEP A: Clip the segment to the fold boundaries using GLOBAL timeline ---
                overlap_start = max(seg['global_start'], f_start)
                overlap_end = min(seg['global_end'], f_end)
                
                if overlap_start < overlap_end:
                    current_duration = overlap_end - overlap_start
                    
                    # --- STEP B: Convert global overlap back to local time for RTTM filtering ---
                    local_overlap_start = overlap_start - seg['offset']
                    local_overlap_end = overlap_end - seg['offset']
                    
                    group = all_vocalizations[
                        (all_vocalizations['video_name'] == seg['video_name']) & 
                        (all_vocalizations['start_time_seconds'] < local_overlap_end) & 
                        (all_vocalizations['end_time_seconds'] > local_overlap_start)
                    ].copy()

                    if group.empty:
                        # No vocalizations in this window: set all exposure categories to 0
                        for exp_type in exposure_categories:
                            final_rows.append({
                                'child_id': seg['child_id'],
                                'fold': fold_num,
                                'video_name': seg['video_name'],
                                'age_at_recording': seg['age_at_recording'],
                                'interaction_type': seg['interaction_type'],
                                'segment_start_time': seg['start_time_sec'],
                                'segment_end_time': seg['end_time_sec'],
                                'exposure_type': exp_type,
                                'total_speech_seconds': 0.0,
                                'total_segment_duration': current_duration,
                                'segment_duration_minutes': current_duration / 60
                            })
                    else:
                        # Clip timestamps once
                        group['clipped_start'] = np.maximum(group['start_time_seconds'], local_overlap_start)
                        group['clipped_end'] = np.minimum(group['end_time_seconds'], local_overlap_end)
            
                        for exp_type in exposure_categories:
                            if exp_type == 'TOTAL':
                                data = group
                            elif exp_type == 'KCDS_ONLY':
                                data = group[group['speech_type'] == 'KCDS']
                            else:
                                data = group[group['speech_type'] == 'OHS']
                            
                            if data.empty:
                                speech_seconds = 0.0
                            else:
                                intervals = list(zip(data['clipped_start'], data['clipped_end']))
                                _, speech_seconds = merge_overlapping_intervals(intervals)

                            final_rows.append({
                                'child_id': seg['child_id'],
                                'fold': fold_num,
                                'video_name': seg['video_name'],
                                'age_at_recording': seg['age_at_recording'],
                                'interaction_type': seg['interaction_type'],
                                'segment_start_time': seg['start_time_sec'],
                                'segment_end_time': seg['end_time_sec'],
                                'exposure_type': exp_type,
                                'total_speech_seconds': speech_seconds,
                                'total_segment_duration': current_duration,
                                'segment_duration_minutes': current_duration / 60
                            })

    final_df = pd.DataFrame(final_rows)
    final_df['exposure_percent'] = (final_df['total_speech_seconds'] / final_df['total_segment_duration']).fillna(0)
        
    # Final cleanup and sort
    final_df = final_df.sort_values(['video_name', 'segment_start_time', 'exposure_type'])
    
    # Save Segment-Level Summary
    suffix = "_gt" if use_ground_truth else ""
    base_folder = (
        output_folder if output_folder else Analysis.CDS_SUMMARY_CSV.parent
    )
    output_path_cds = (
        base_folder / f"{Analysis.CDS_SUMMARY_CSV.stem}{suffix}.csv"
    )
    final_df.to_csv(output_path_cds, index=False)
    print(f"✅ Clean results saved to {output_path_cds}")
        
    # ----- PART 3A: Child-Level Aggregation ------
    # Aggregation strategy depends on use_folds flag
    if use_folds:
        # Group by child, fold, and exposure_type to preserve the stability data points
        child_exposure_summary = final_df.groupby(['child_id', 'fold', 'exposure_type']).agg({
            'total_speech_seconds': 'sum',
            'total_segment_duration': 'sum',
            'age_at_recording': 'min'
        }).reset_index()
    else:
        # Group by child and exposure_type only (single fold, so fold column will be 1)
        child_exposure_summary = final_df.groupby(['child_id', 'exposure_type']).agg({
            'total_speech_seconds': 'sum',
            'total_segment_duration': 'sum',
            'age_at_recording': 'min'
        }).reset_index()

    # Calculate global percentage for that specific exposure type
    child_exposure_summary['exposure_percent'] = (
        child_exposure_summary['total_speech_seconds'] / 
        child_exposure_summary['total_segment_duration']
    ).fillna(0)

    # Convert to minutes for easier reporting
    child_exposure_summary['total_speech_minutes'] = child_exposure_summary['total_speech_seconds'] / 60
    child_exposure_summary['total_recording_minutes'] = child_exposure_summary['total_segment_duration'] / 60

    output_path_gcds = (
        base_folder / f"{Analysis.GLOBAL_CDS_SUMMARY_CSV.stem}{suffix}.csv"
    )
    child_exposure_summary.to_csv(output_path_gcds, index=False)
    print(f"✅ Child-level exposure summary saved to: {output_path_gcds}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CDS (Child Directed Speech) and OHS exposure analysis summaries")
    parser.add_argument('--output_folder', type=str, default=None, help="Optional output folder path to save the interaction composition CSV")
    parser.add_argument('--use_folds', action='store_true', default=False, help="Calculate per-fold metrics (5-fold cross-validation). If not set, calculates overall metrics only.")
    parser.add_argument('--use_gt', action='store_true', default=False, help="Map vocalizations onto manual ground-truth segments instead of pipeline segments")
    args = parser.parse_args()
        
    main(
      output_folder=Path(args.output_folder) if args.output_folder else None,
      use_folds=args.use_folds,
      use_ground_truth=args.use_gt,
  )