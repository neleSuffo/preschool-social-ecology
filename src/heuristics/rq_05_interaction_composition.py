import pandas as pd
import argparse
import numpy as np
from pathlib import Path
from constants import Analysis
from config import DataConfig
from src.heuristics.utils import get_child_fold_boundaries, extract_child_id

def add_interaction_columns(frames_df: pd.DataFrame, 
                            segments_df: pd.DataFrame, 
                            use_ground_truth: bool,
                            social_state_mode="tertiary") -> pd.DataFrame:
    """
    Adds binary columns to frames_df indicating whether each frame falls within an interaction segment, and if so, what type of interaction.
    
    Parameters:
    ----------
    frames_df: pd.DataFrame 
        DataFrame with columns ['video_name', 'frame_number', 'proximity', ...]
    segments_df: pd.DataFrame
        DataFrame with columns ['video_name', 'segment_start', 'segment_end', 'interaction_type']
    use_ground_truth: bool
        If True, use ground truth segments instead of pipeline segments for interaction mapping.
    social_state_mode: str
        "binary" for Interacting vs Not Interacting, "tertiary" for Interacting vs Alone vs Available
    """
    # 1. Setup column names
    target_cols = ['is_interaction']
    type_to_col = {'Interacting': 'is_interaction'}
    
    if social_state_mode == "binary":
        target_cols.append('is_not_interaction') 
        type_to_col['Not_Interacting'] = 'is_not_interaction'
    else:
        target_cols.extend(['is_alone', 'is_available'])
        type_to_col.update({'Alone': 'is_alone', 'Available': 'is_available'})

    # Initialize columns at once with False (more memory efficient than one-by-one)
    for col in target_cols:
        frames_df[col] = False

    # 2. Process by video to reduce search space
    # Convert frames_df to a dictionary of dataframes for O(1) access
    video_groups = dict(list(frames_df.groupby('video_name')))
    processed_fragments = []

    for video_name, segments in segments_df.groupby('video_name'):
        if video_name not in video_groups:
            continue
        
        v_frames = video_groups[video_name].copy()
        
        # For each segment in this specific video
        for _, seg in segments.iterrows():
            col = type_to_col.get(seg['interaction_type'])
            if col:
                if use_ground_truth:
                    # Load ground truth segments for this video
                    # Check if the segment overlaps with any ground truth segment
                    start_frame = seg['start_time_sec'] * DataConfig.FPS
                    end_frame = seg['end_time_sec'] * DataConfig.FPS
                else:
                    # Use pipeline segments directly
                    start_frame = seg['segment_start']
                    end_frame = seg['segment_end']

                # Use boolean indexing to set the column for the relevant frames
                mask = (v_frames['frame_number'] >= start_frame) & \
                       (v_frames['frame_number'] <= end_frame)
                v_frames.loc[mask, col] = True
        
        processed_fragments.append(v_frames)
        # Remove from dict to track which videos had segments
        del video_groups[video_name]

    # 3. Reconstruct the dataframe
    if use_ground_truth:
        # Only keep fragments for videos present in GT
        final_df = pd.concat(processed_fragments).sort_index()
    else:
        # Add back videos that had no segments (they stay all False)
        remaining_frames = list(video_groups.values())
        final_df = pd.concat(processed_fragments + remaining_frames).sort_index()
        
    return final_df

def main(social_state_mode: str = 'tertiary',
         output_folder: Path = None,
         use_folds: bool = True,
         use_ground_truth: bool = True):
         
    """
    Generate frame-level interaction composition by merging segment-level interactions with frame-level metadata.

    Parameters
    ----------
    social_state_mode : str, optional
        Whether to use "binary" (Interacting vs Not Interacting) or "tertiary" (Interacting vs Alone vs Available) classification, by default "tertiary"
    output_folder : Path, optional
        Optional output folder path to save the interaction composition CSV. If not provided, saves to default location defined in constants, by default None
    use_folds : bool, optional
        If True, calculates per-fold metrics (5-fold cross-validation style). If False, calculates metrics across entire recording, by default True
        
    Raises
    ------
    FileNotFoundError
        _description_
    """
    fold_mode = "WITH per-fold metrics" if use_folds else "WITHOUT per-fold metrics (overall only)"
    print(f"🚀 INTERACTION PROCESSING (Mode: {social_state_mode.upper()}, Folds: {fold_mode})")
    print("=" * 70)

    # Step 1: Load data
    if output_folder:
        frames_path = output_folder / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        segments_path = output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name
    else:
        frames_path = Analysis.FRAME_LEVEL_INTERACTIONS_CSV
        segments_path = Analysis.INTERACTION_SEGMENTS_CSV

    frames_df = pd.read_csv(frames_path)

    if use_ground_truth:
        segments_path = Analysis.GROUND_TRUTH_SEGMENTS_CSV
        segments_df = pd.read_csv(segments_path)
        # Filter frames to only keep videos present in the ground truth
        gt_videos = segments_df['video_name'].unique()
        frames_df = frames_df[frames_df['video_name'].isin(gt_videos)].copy()
    else:
        segments_df = pd.read_csv(segments_path)
    
    # Standardize types
    segments_df['start_time_sec'] = pd.to_numeric(segments_df['start_time_sec'], errors='coerce')
    segments_df['end_time_sec'] = pd.to_numeric(segments_df['end_time_sec'], errors='coerce')
    segments_df['duration_sec'] = pd.to_numeric(segments_df['duration_sec'], errors='coerce')
    segments_df['child_id'] = segments_df['child_id'].astype(str)
    frames_df['child_id'] = frames_df['video_name'].apply(extract_child_id).astype(str)
    
    # Step 2: Global Timeline Logic for Frames
    video_stats = (
        segments_df.groupby(['child_id', 'video_name'])['end_time_sec']
        .max()
        .reset_index(name='video_span_sec')
    )
    video_stats = video_stats.sort_values(['child_id', 'video_name'])
    video_stats['offset_raw'] = video_stats.groupby('child_id')['video_span_sec'].shift(1).fillna(0)
    video_stats['offset_sec'] = video_stats.groupby('child_id')['offset_raw'].transform('cumsum')
    
    # Merge offsets back to frames and segments
    frames_df = frames_df.merge(video_stats[['child_id', 'video_name', 'offset_sec']], on=['child_id', 'video_name'])
    segments_df = segments_df.merge(video_stats[['child_id', 'video_name', 'offset_sec']], on=['child_id', 'video_name'])
    
    # Segment global boundaries for folds
    segments_df['global_start'] = segments_df['start_time_sec'] + segments_df['offset_sec']
    segments_df['global_end'] = segments_df['end_time_sec'] + segments_df['offset_sec']

    # Local frame time is simply frame_number / FPS
    frames_df['global_sec_pos'] = (frames_df['frame_number'] / DataConfig.FPS) + frames_df['offset_sec']
    
    # Step 3: Assign Folds (conditional)
    if use_folds:
        fold_map = get_child_fold_boundaries(segments_df)
        frames_df['fold'] = 0
        
        for child_id, folds in fold_map.items():
            child_mask = frames_df['child_id'] == child_id
            for fold_idx, (f_start, f_end) in enumerate(folds):
                # Check which frames fall within the time-boundary of this fold
                fold_mask = (frames_df['global_sec_pos'] >= f_start) & (frames_df['global_sec_pos'] < f_end)
                if fold_idx == 4: # Last fold
                    fold_mask = (frames_df['global_sec_pos'] >= f_start) & (frames_df['global_sec_pos'] <= f_end + 0.1)
                
                frames_df.loc[child_mask & fold_mask, 'fold'] = fold_idx + 1
    else:
        # Single fold covering entire duration for each child
        frames_df['fold'] = 1
            
    # Step 2: Optimized Processing
    frames_df = add_interaction_columns(frames_df, segments_df, use_ground_truth, social_state_mode=social_state_mode)
    
    # # 3. Fast Metadata Merge
    print("📝 Syncing metadata from segments...")
    metadata_map = segments_df[['video_name', 'age_at_recording', 'child_id']].drop_duplicates()

    # Ensure age is numeric
    metadata_map['age_at_recording'] = pd.to_numeric(
        metadata_map['age_at_recording'].astype(str)
        .str.replace('"', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip(), 
        errors='coerce'
    )
    # Vectorized operations
    frames_df = frames_df.merge(metadata_map, on=['video_name', 'child_id'], how='left')
    frames_df['proximity_filled'] = frames_df['proximity'].fillna(-1)

    # Step 4: Save
    suffix = "_gt" if use_ground_truth else ""
    base_folder = output_folder if output_folder else Analysis.INTERACTION_COMPOSITION_CSV.parent
    output_path = base_folder / f"{Analysis.INTERACTION_COMPOSITION_CSV.stem}{suffix}.csv"
    frames_df.to_csv(output_path, index=False)

    # adjust print message based on social_state_mode
    if social_state_mode == "binary":
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Not Interacting: {frames_df['is_not_interaction'].sum()} frames.")
    else:         
        print(f"\n✅ Done! Interacting: {frames_df['is_interaction'].sum()} frames, Alone: {frames_df['is_alone'].sum()} frames, Available: {frames_df['is_available'].sum()} frames.")
    print(f"📄 Saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-level interaction composition analysis")
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--output_folder', type=str, default=None, help="Optional output folder path to save the interaction composition CSV")
    parser.add_argument('--use_folds', action='store_true', default=False, help="Calculate per-fold metrics (5-fold cross-validation). If not set, calculates overall metrics only.")
    parser.add_argument('--use_gt', action='store_true', default=False, help="Map vocalizations onto manual ground-truth segments instead of pipeline segments")
    args = parser.parse_args()
        
    main(social_state_mode=args.social_state_mode, 
         output_folder=Path(args.output_folder) if args.output_folder else None,
         use_folds=args.use_folds,
         use_ground_truth=args.use_gt)