import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from constants import Analysis
from config import AnalysisConfig
from src.heuristics.utils import parse_rttm, get_child_fold_boundaries

# Configure logging to show thresholds in the terminal
logging.basicConfig(level=logging.INFO, format='%(message)s')

def process_block(block):
    """
    Categorizes the block into one of four types:
    1. Successful Initiation (Child starts, turn occurs)
    2. Successful Response (Adult starts, turn occurs)
    3. Unanswered Child Bid (Child starts, no turn)
    4. Unanswered Adult Prompt (Adult starts, no turn)
    
    A block is defined as a sequence of vocalizations from the same speech_type with gaps less than the defined thresholds.
    
    Parameters:
    ----------
    block: list of dicts
        Each dict contains:
        - speech_type: 'KCHI' or 'KCDS'
        - start_time_seconds
        - end_time_seconds  
        
    Returns:
    --------
    turns: int
        Total number of turns in the block
    succ_init: int
        1 if block is a successful initiation, else 0
    succ_resp: int
        1 if block is a successful response, else 0
    fail_init: int
        1 if block is an unanswered child bid, else 0
    fail_resp: int
        1 if block is an unanswered adult prompt, else 0
    block_duration: float
        Duration of the block in seconds
    """
    if not block:
        return 0, 0, 0, 0, 0, 0
    
    # Calculate duration: End of last vocalization - Start of first vocalization
    block_duration = block[-1]['end_time_seconds'] - block[0]['start_time_seconds']
    
    turns = 0
    for i in range(1, len(block)):
        if block[i]['speech_type'] != block[i-1]['speech_type']:
            turns += 1
            
    # Initialize all counters
    succ_init, succ_resp, fail_init, fail_resp  = 0, 0, 0, 0
    first_speaker = block[0]['speech_type']
    
    if turns > 0:
        if first_speaker == 'KCHI':
            succ_init = 1  # Successful exchange led by child
        else:
            succ_resp = 1  # Successful exchange led by adult
    else:
        if first_speaker == 'KCHI':
            fail_init = 1  # Child spoke, but no one replied
        else:
            fail_resp = 1  # Adult spoke, but child didn't reply
            
    return turns, succ_init, succ_resp, fail_init, fail_resp, block_duration

def count_directional_turns(vocalizations: pd.DataFrame, 
                            segments_df: pd.DataFrame,
                            use_folds: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Counts directional turn-taking within each interacting segment and measures individual child turn durations.
    
    Parameters:
    ----------
    vocalizations: DataFrame
        containing vocalization data with columns:
        - video_name
        - start_time_seconds
        - end_time_seconds
        - speech_type (KCHI or KCDS)
    segments_df: DataFrame
        containing interaction segment data with columns:
        - child_id
        - video_name
        - age_at_recording
        - start_time_sec
        - end_time_sec
        - duration_sec
        - interaction_type
    
    Returns:
    --------
    results_df: DataFrame
        containing turn-taking counts and proportions for each segment
    raw_turns_df: DataFrame
        containing individual child turn durations and metadata for further analysis
    """
    # 1. GLOBAL TIMELINE OFFSETS
    video_stats = (
        segments_df.groupby(['child_id', 'video_name'])['end_time_sec']
        .max()
        .reset_index(name='video_span_sec')
    )
    video_stats = video_stats.sort_values(['child_id', 'video_name'])   
    video_stats['offset_raw'] = video_stats.groupby('child_id')['video_span_sec'].shift(1).fillna(0)
    video_stats['offset'] = video_stats.groupby('child_id')['offset_raw'].transform('cumsum')
    
    segments_df = segments_df.merge(video_stats[['child_id', 'video_name', 'offset']], on=['child_id', 'video_name'])
    segments_df['global_start'] = segments_df['start_time_sec'] + segments_df['offset']
    segments_df['global_end'] = segments_df['end_time_sec'] + segments_df['offset']

    # 2. FOLD BOUNDARIES (conditional)
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
    
    results = []
    raw_turns_df = []
    MAX_TURN_GAP = AnalysisConfig.MAX_TURN_TAKING_GAP_SEC
    MAX_SAME_GAP = AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC

    # 3. Iterate by Child -> Fold -> Segment
    for child_id, folds in fold_map.items():
        child_segs = segments_df[segments_df['child_id'] == child_id]
        
        for fold_idx, (f_start, f_end) in enumerate(folds):
            fold_num = fold_idx + 1
            
            for _, seg in child_segs.iterrows():
                if seg['interaction_type'] != 'Interacting':
                    continue
                # STEP A: Global Clipping
                overlap_start = max(seg['global_start'], f_start)
                overlap_end = min(seg['global_end'], f_end)
                
                if overlap_start < overlap_end:
                    current_duration = overlap_end - overlap_start
                    local_overlap_start = overlap_start - seg['offset']
                    local_overlap_end = overlap_end - seg['offset']

                    # STEP B: Local Filter
                    seg_vocs = vocalizations[
                        (vocalizations['video_name'] == seg['video_name']) & 
                        (vocalizations['start_time_seconds'] < local_overlap_end) & 
                        (vocalizations['end_time_seconds'] > local_overlap_start)
                    ].copy().sort_values('start_time_seconds')

                    # STEP C: TRIPLE CLIP
                    seg_vocs['clipped_start'] = np.maximum(seg_vocs['start_time_seconds'], local_overlap_start)
                    seg_vocs['clipped_end'] = np.minimum(seg_vocs['end_time_seconds'], local_overlap_end)
                    seg_vocs['duration_seconds'] = seg_vocs['clipped_end'] - seg_vocs['clipped_start']
                    
                    # Update the time columns for the process_block logic
                    seg_vocs['start_time_seconds'] = seg_vocs['clipped_start']
                    seg_vocs['end_time_seconds'] = seg_vocs['clipped_end']
                    
                    seg_vocs = seg_vocs[seg_vocs['speech_type'].isin(['KCHI', 'KCDS'])].reset_index(drop=True)

                    # Capture Individual Turn Durations for the fold
                    child_only = seg_vocs[seg_vocs['speech_type'] == 'KCHI']
                    for _, voc in child_only.iterrows():
                        raw_turns_df.append({
                            'child_id': child_id,
                            'fold': fold_num,
                            'age_at_recording': seg['age_at_recording'],
                            'video_name': seg['video_name'],
                            'vocalization_duration_sec': voc['duration_seconds'],
                            'start_time': voc['start_time_seconds']
                        })
                    
                    # Process Turn Blocks
                    total_turns, s_init, s_resp, f_init, f_resp, total_block_duration = 0, 0, 0, 0, 0, 0
                    
                    if not seg_vocs.empty:
                        current_block = [seg_vocs.iloc[0].to_dict()]
                        for i in range(1, len(seg_vocs)):
                            prev = seg_vocs.iloc[i-1]
                            curr = seg_vocs.iloc[i]
                            gap = curr['start_time_seconds'] - prev['end_time_seconds']
                            threshold = MAX_SAME_GAP if curr['speech_type'] == prev['speech_type'] else MAX_TURN_GAP
                            
                            if gap <= threshold:
                                current_block.append(curr.to_dict())
                            else:
                                t, si, sr, fi, fr, bd = process_block(current_block)
                                total_turns += t; s_init += si; s_resp += sr; f_init += fi; f_resp += fr
                                total_block_duration += bd
                                current_block = [curr.to_dict()]
                        
                        t, si, sr, fi, fr, bd = process_block(current_block)
                        total_turns += t; s_init += si; s_resp += sr; f_init += fi; f_resp += fr
                        total_block_duration += bd

                    results.append({
                        'child_id': child_id,
                        'fold': fold_num,
                        'block_duration_sec': total_block_duration,
                        'video_name': seg['video_name'],
                        'age_at_recording': seg['age_at_recording'],
                        'segment_start': seg['start_time_sec'],
                        'segment_end': seg['end_time_sec'],
                        'duration_sec': current_duration,
                        'total_turns': total_turns,
                        'successful_initiations': s_init,
                        'successful_responses': s_resp,
                        'unanswered_child_bids': f_init,
                        'unanswered_adult_prompts': f_resp,
                        'segment_duration_minutes': current_duration / 60
                    })
        
    return pd.DataFrame(results), pd.DataFrame(raw_turns_df)

def main(social_state_mode: str = 'tertiary',
         output_folder: Path = None,
         use_folds: bool = True,
         use_ground_truth: bool = False):
    """
    Executes the turn-taking analysis for naturalistic social interactions, categorizing blocks of vocalizations into successful initiations, successful responses, unanswered child bids, and unanswered adult prompts. It also calculates the total number of turns and their density per minute for each segment, as well as individual child turn durations for further analysis.

    Parameters
    ----------
    social_state_mode : str, optional
        The mode for categorizing social states, either 'binary' (Interacting vs. Not Interacting) or 'tertiary' (Alone, Available, Interacting). Default is 'tertiary'.
    output_folder : Path, optional
        Optional output folder path to save the turn-taking analysis CSV. If not provided, saves to default location defined in constants, by default None
    use_folds : bool, optional
        If True, calculates per-fold metrics (5-fold cross-validation style). If False, calculates metrics across entire recording, by default True
    """
    fold_mode = ('WITH per-fold metrics' if use_folds else 'WITHOUT per-fold metrics (overall only)')
    gt_addition = 'for ground truth dataset' if use_ground_truth else ''
    print('🗣️ RESEARCH QUESTION 4: TURN-TAKING ANALYSIS')
    print(f'   Mode: {fold_mode} {gt_addition}'.strip())
    print('=' * 70)
        
    # 1. APPLY MODE AND LOG PARAMETERS
    AnalysisConfig.apply_mode(social_state_mode)
    
    logging.info(f"⚙️  MODE DETECTED: {social_state_mode.upper()}")
    logging.info(f"📏 Turn-Taking Gap: {AnalysisConfig.MAX_TURN_TAKING_GAP_SEC}s")
    logging.info(f"📏 Same-Speaker Gap: {AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC}s")
    logging.info("-" * 70)
    
    # 2. Load Segments and Vocalizations
    if use_ground_truth:
        segments_path = Analysis.GROUND_TRUTH_SEGMENTS_CSV
    else:
        segments_path = (
            output_folder / Analysis.INTERACTION_SEGMENTS_CSV.name
            if output_folder
            else Analysis.INTERACTION_SEGMENTS_CSV
        )
    segments_df = pd.read_csv(segments_path)
    all_vocalizations = parse_rttm(target_speech_types=['KCHI', 'KCDS'])
    
    # 3. Categorize Social Blocks
    final_df, raw_turns_df = count_directional_turns(all_vocalizations, segments_df, use_folds=use_folds)
    
    # 4. Calculate Global Totals and Proportions
    final_df['total_attempts'] = (
        final_df['successful_initiations'] + final_df['successful_responses'] +
        final_df['unanswered_child_bids'] + final_df['unanswered_adult_prompts']
    )
    
    # Calculate percentages of the total social landscape
    for col in ['successful_initiations', 'successful_responses', 'unanswered_child_bids', 'unanswered_adult_prompts']:
        final_df[f'pct_{col}'] = (final_df[col] / final_df['total_attempts']).fillna(0)
    
    # Standard volume metric
    final_df['turns_per_minute'] = (final_df['total_turns'] / final_df['segment_duration_minutes']).fillna(0)
    
    # 5. Metadata Cleanup and Sort
    final_df['age_at_recording'] = (
        final_df['age_at_recording'].astype(str).str.replace('"', '').str.replace(',', '.').str.strip()
    )
    final_output = final_df.sort_values(['video_name', 'segment_start'])
    
    # 6. Save Results
    suffix = '_gt' if use_ground_truth else ''
    base_folder = (
        output_folder if output_folder else Analysis.TURN_TAKING_CSV.parent
    )
    output_path_tt = (
        base_folder / f'{Analysis.TURN_TAKING_CSV.stem}{suffix}.csv'
    )
    final_output.to_csv(output_path_tt, index=False)
    print(f"✅ Full four-category analysis saved to {output_path_tt}")

    # ----- Part 2: Child-Level Aggregation (Relative to Total Recording) -----
    # 1. True total recording duration for every child across ALL segments (Alone + Available + Interacting)    # (This includes Alone, Available, and Interacting time)
    child_total_durations = segments_df.groupby('child_id')['duration_sec'].sum().reset_index()
    child_total_durations.rename(columns={'duration_sec': 'total_recording_duration_sec'}, inplace=True)
    
    # ----- Child-Level Aggregation (Fold-Aware or Overall) -----
    if use_folds:
        # Step A: Compute total recording duration per fold across ALL states
        fold_durations = []
        for child_id, folds in fold_map.items():
            child_segs = segments_df[segments_df['child_id'] == child_id]
            for fold_idx, (f_start, f_end) in enumerate(folds):
                fold_num = fold_idx + 1
                f_dur = 0.0
                for _, seg in child_segs.iterrows():
                    o_start = max(seg['global_start'], f_start)
                    o_end = min(seg['global_end'], f_end)
                    if o_start < o_end:
                        f_dur += (o_end - o_start)
                fold_durations.append({
                    'child_id': child_id,
                    'fold': fold_num,
                    'total_recording_duration_sec': f_dur
                })
        df_fold_durations = pd.DataFrame(fold_durations)

        # Step B: Sum turns within each fold
        child_level_turns = final_df.groupby(['child_id', 'fold']).agg({
            'total_turns': 'sum',
            'duration_sec': 'sum',
            'age_at_recording': 'min'
        }).reset_index()
        child_level_turns.rename(columns={'duration_sec': 'total_interacting_sec'}, inplace=True)

        # Step C: Merge total recording duration into the fold summary
        child_level_turns = child_level_turns.merge(df_fold_durations, on=['child_id', 'fold'], how='left')
    else:
        # Group by child only
        child_level_turns = final_df.groupby('child_id').agg({
            'total_turns': 'sum',
            'duration_sec': 'sum',
            'age_at_recording': 'min'
        }).reset_index()
        child_level_turns.rename(columns={'duration_sec': 'total_interacting_sec'}, inplace=True)
        
        # Merge true total recording duration across all states
        child_level_turns = child_level_turns.merge(child_total_durations, on='child_id', how='left')

    # Calculate global density (turns per minute of OVERALL recording time)
    child_level_turns['total_recording_minutes'] = child_level_turns['total_recording_duration_sec'] / 60
    child_level_turns['turns_per_minute'] = (
        child_level_turns['total_turns'] / child_level_turns['total_recording_minutes']
    ).fillna(0)     

    # Save Child-Level Results
    output_path_gtt = (
        base_folder / f'{Analysis.GLOBAL_TURN_TAKING_CSV.stem}{suffix}.csv'
    )
    child_level_turns.to_csv(output_path_gtt, index=False)
    print(f"✅ Child-level turn-taking (relative to total duration) saved to {output_path_gtt}")
    
    # ----- Part 3: SAVE INDIVIDUAL TURN DURATIONS -----
    if not raw_turns_df.empty:
        # Cleanup age strings in the raw durations dataframe
        raw_turns_df['age_at_recording'] = (
            raw_turns_df['age_at_recording'].astype(str)
            .str.replace('"', '').str.replace(',', '.')
            .str.strip()
        )
        
        # Save to the granular output path
        output_path_td = (base_folder / f'{Analysis.TURN_DURATION_CSV.stem}{suffix}.csv')
        raw_turns_df.to_csv(output_path_td, index=False)
        print(f'✅ Individual child turn durations saved to {output_path_td}')
    else:
        print('⚠️ No child vocalizations found to save for turn duration analysis.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn-Taking Analysis for Naturalistic Social Interactions")
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--output_folder', type=str, default=None, help="Optional output folder path to save the interaction composition CSV")
    parser.add_argument('--use_folds', action='store_true', default=False, help="Calculate per-fold metrics (5-fold cross-validation). If not set, calculates overall metrics only.")
    parser.add_argument('--use_gt', action='store_true', default=False, help="Map vocalizations onto manual ground-truth segments instead of pipeline segments")
    args = parser.parse_args()
    main(social_state_mode=args.social_state_mode,
         output_folder=Path(args.output_folder) if args.output_folder else None,
         use_folds=args.use_folds,
         use_ground_truth=args.use_gt)