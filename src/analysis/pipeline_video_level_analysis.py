import re
import argparse
import sys
import sqlite3
import pandas as pd
import shutil
import numpy as np
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Inference, Evaluation, DataPaths
from config import DataConfig, InferenceConfig

# Constants
FPS = DataConfig.FPS # frames per second

# def get_min_segment_duration(interaction_type: str) -> float:
#     """
#     Returns the minimum required segment duration (in seconds) based on the interaction type.
    
#     This function relies on the assumption that the following constants are defined
#     in InferenceConfig:
#     - MIN_INTERACTING_SEGMENT_DURATION_SEC
#     - MIN_ALONE_SEGMENT_DURATION_SEC
#     - MIN_AVAILABLE_SEGMENT_DURATION_SEC
#     """
#     # Assuming the interaction_type is properly capitalized (e.g., 'Interacting', 'Alone')
#     type_map = {
#         'Interacting': InferenceConfig.MIN_INTERACTING_SEGMENT_DURATION_SEC,
#         'Alone': InferenceConfig.MIN_ALONE_SEGMENT_DURATION_SEC,
#         'Available': InferenceConfig.MIN_AVAILABLE_SEGMENT_DURATION_SEC,
#     }
#     # Use a default minimum (e.g., the largest) if the type is unexpected
#     default_min = getattr(InferenceConfig, 'MIN_ALONE_SEGMENT_DURATION_SEC', 15.0) 
#     return type_map.get(interaction_type, default_min)

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def create_segments_for_video(video_id, video_df):
    """
    Create segments for a single video. Buffers short state changes and enforces minimum segment durations.
    
    Parameters
    ----------
    video_id : int
        Video identifier
    video_df : pd.DataFrame
        Frame-level data for this video
        
    Returns
    -------
    list
        List of segment dictionaries
    """
    video_df = video_df.sort_values('frame_number').reset_index(drop=True)
    
    if len(video_df) == 0:
        return []
                
    states = video_df['interaction_type'].values
    frame_numbers = video_df['frame_number'].values
    video_name = video_df['video_name'].iloc[0]
    
    segments = []
    current_state = states[0]
    segment_start_frame = frame_numbers[0]
    
    for i in range(1, len(states)):
        if states[i] != current_state:
            segment_end_frame = frame_numbers[i-1]
            segment_duration = (segment_end_frame - segment_start_frame) / FPS
            
            segments.append({
                'video_id': video_id,
                'video_name': video_name,
                'interaction_type': current_state,
                'segment_start': segment_start_frame,
                'segment_end': segment_end_frame,
                'start_time_sec': segment_start_frame / FPS,
                'end_time_sec': segment_end_frame / FPS,
                'duration_sec': segment_duration
            })
            
            current_state = states[i]
            segment_start_frame = frame_numbers[i]
    
    # Final segment
    segment_end_frame = frame_numbers[-1]
    segment_duration = (segment_end_frame - segment_start_frame) / FPS
    segments.append({
        'video_id': video_id,
        'video_name': video_name,
        'interaction_type': current_state,
        'segment_start': segment_start_frame,
        'segment_end': segment_end_frame,
        'start_time_sec': segment_start_frame / FPS,
        'end_time_sec': segment_end_frame / FPS,
        'duration_sec': segment_duration
    })
    
    return segments

def merge_same_segments(segments_df):
    """
    Merge segments of the same category that have small gaps between them.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments
        
    Returns
    -------
    pd.DataFrame
        DataFrame with merged segments
    """
    merged_segments = []
    merge_count = 0
    
    for video_id, video_segments in segments_df.groupby('video_id'):
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
        
        if len(video_segments) == 0:
            continue
        
        current_segment = video_segments.iloc[0].copy()
        
        for i in range(1, len(video_segments)):
            next_segment = video_segments.iloc[i]
            
            # --- 1. Calculate the Gap Duration ---
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            
            # Check 1: Must be the same interaction type
            # Check 2: The gap must be positive (i.e., not overlapping) AND small
            if (current_segment['interaction_type'] == next_segment['interaction_type']):
                
                # If conditions met, merge them by extending the end time                
                current_segment['segment_end'] = next_segment['segment_end']
                current_segment['end_time_sec'] = next_segment['end_time_sec']
                current_segment['duration_sec'] = (
                    current_segment['end_time_sec'] - current_segment['start_time_sec']
                )
                merge_count += 1
                
            else:
                # Save current segment and start a new one
                merged_segments.append(current_segment.to_dict())
                current_segment = next_segment.copy()
        
        # Add the last segment
        merged_segments.append(current_segment.to_dict())
    
    if merged_segments:
        result_df = pd.DataFrame(merged_segments)
        result_df = result_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
        return result_df
    else:
        return segments_df

def fill_gaps_with_default(segments_df, default_type="Alone"):
    filled_segments = []

    for video_id, video_df in segments_df.groupby('video_id'):
        v_segs = video_df.sort_values('start_time_sec').to_dict('records')
        
        for i in range(len(v_segs)):
            filled_segments.append(v_segs[i])
            
            if i < len(v_segs) - 1:
                gap = v_segs[i+1]['start_time_sec'] - v_segs[i]['end_time_sec']
                
                if 0 < gap <= InferenceConfig.GAP_STRETCH_THRESHOLD:
                    # Small gap: Stretch previous segment
                    filled_segments[-1]['end_time_sec'] = v_segs[i+1]['start_time_sec']
                    filled_segments[-1]['segment_end'] = v_segs[i+1]['segment_start'] - 1
                elif gap > InferenceConfig.GAP_STRETCH_THRESHOLD:
                    # Large gap: Insert Default
                    filled_segments.append({
                        'video_id': video_id,
                        'video_name': v_segs[i]['video_name'],
                        'interaction_type': default_type,
                        'start_time_sec': v_segs[i]['end_time_sec'],
                        'end_time_sec': v_segs[i+1]['start_time_sec'],
                        'segment_start': v_segs[i]['segment_end'] + 1,
                        'segment_end': v_segs[i+1]['segment_start'] - 1
                    })
    return pd.DataFrame(filled_segments)
    
def print_segment_summary(segments_df):
    """
    Print summary statistics for the created segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments
    """
    if len(segments_df) > 0:
        total_segments = len(segments_df)
        # Recalculate duration for all segments after reclassification
        segments_df['duration_sec'] = segments_df['end_time_sec'] - segments_df['start_time_sec']

        interacting_segments = len(segments_df[segments_df['interaction_type'] == 'Interacting'])
        alone_segments = len(segments_df[segments_df['interaction_type'] == 'Alone'])
        available_segments = len(segments_df[segments_df['interaction_type'] == 'Available'])

        # Calculate total duration for each segment interaction_type in minutes (with two decimals)
        total_duration = round(segments_df['duration_sec'].sum() / 60, 2)
        interacting_duration = round(segments_df[segments_df['interaction_type'] == 'Interacting']['duration_sec'].sum() / 60, 2)
        alone_duration = round(segments_df[segments_df['interaction_type'] == 'Alone']['duration_sec'].sum() / 60, 2)
        copresent_duration = round(segments_df[segments_df['interaction_type'] == 'Available']['duration_sec'].sum() / 60, 2)

        print(f"\n📊 Final segment summary:")
        print(f"   Total segments: {total_segments} ({total_duration} minutes)")
        print(f"   Interacting: {interacting_segments} ({interacting_duration} minutes - {interacting_duration/total_duration*100:.1f}%)")
        print(f"   Alone: {alone_segments} ({alone_duration} minutes - {alone_duration/total_duration*100:.1f}%)")
        if available_segments > 0:
            print(f"   Available: {available_segments} ({copresent_duration} minutes - {copresent_duration/total_duration*100:.1f}%)")
    else:
        print("\n📊 No segments created")

def biased_conservative_mode(window):
    """
    1 = Interacting, 2 = Available, 3 = Alone
    Goal: Protect Interacting recall while forcing Available to prove itself.
    """
    counts = pd.Series(window).value_counts(normalize=True)
    
    # Keep Interacting protection at 20%
    if 1 in counts and counts[1] >= 0.20:
        return 1
    
    # Lower the Available bar from 65% to 45%
    # This helps recover the 28.6% leak to Alone
    if 2 in counts and counts[2] > 0.45:
        return 2
    
    return 3

def main(output_file_path: Path, frame_data_path: Path, hyperparameter_tuning: bool = False):
    """
    Main entry point for video-level segment analysis.
    Creates mutually exclusive interaction segments from frame-level data.
    
    Adjusted Workflow:
    1. Load frame-level data.
    2. Apply Rolling Mode Smoothing (3-second window) to remove jitter.
    3. Create initial segments (without hard-drop duration filters).
    4. Fill gaps with "Alone" as the default conservative state.
    5. Final consolidation merge.
    """     
    if hyperparameter_tuning:
        run_dir = output_file_path.parent
        try:
            script_path = Path(__file__)
            shutil.copy(script_path, run_dir / script_path.name)
        except NameError:
            print("⚠️ __file__ not defined, skipping script copy.")
        
    # Step 1: Load frame-level data
    frame_data = pd.read_csv(frame_data_path)

    # --- ROLLING MODE SMOOTHING ---
    # Center=True ensures transitions aren't shifted in time.
    window_size = int(InferenceConfig.ROLLING_SMOOTH_WINDO_SEC * (DataConfig.FPS / InferenceConfig.SAMPLE_RATE)) 
    
    # We use a custom lambda to find the mode of the numeric window
    # frame_data['interaction_type'] = frame_data.groupby('video_id')['interaction_type'].transform(
    #     lambda x: x.rolling(window=window_size, center=True, min_periods=1).apply(
    #         lambda y: pd.Series(y).mode()[0]))

    # Apply the weighted transformation
    frame_data['interaction_type'] = frame_data.groupby('video_id')['interaction_type'].transform(
        lambda x: x.rolling(window=window_size, center=True, min_periods=1).apply(biased_conservative_mode))

    # Step 2: Create segments for each video
    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        video_segments = create_segments_for_video(video_id, video_df)
        all_segments.extend(video_segments)
    
    # Step 3: Convert to DataFrame
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
    else:
        segments_df = pd.DataFrame(columns=['video_id', 'video_name', 'interaction_type',
                                        'segment_start', 'segment_end', 
                                        'start_time_sec', 'end_time_sec', 'duration_sec'])

    # Step 4: Fill all remaining gaps to create a continuous timeline
    print("⏳ Filling gaps with default type: Alone")
    segments_df = fill_gaps_with_default(segments_df, default_type="Alone")

    # Step 5: Final consolidation merge
    print("🧹 Final consolidation: Merging reclassified segments of the same type...")
    segments_df = merge_same_segments(segments_df)

    # Step 6: Generate and print summary
    print_segment_summary(segments_df)
    
    # Merge age and child metadata
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")
    age_df = age_df[["video_name", "age_at_recording", "child_id"]]
    segments_df = segments_df.merge(age_df, on="video_name", how="left")
    
    # Map back to strings only for the final output report
    int_to_state = {1: 'Interacting', 2: 'Available', 3: 'Alone'}
    segments_df['interaction_type'] = segments_df['interaction_type'].map(int_to_state)
    
    # Step 7: Save final results
    segments_df.to_csv(output_file_path, index=False)
    segments_df.to_csv(Evaluation.PRED_SECONDWISE_FILE_PATH, index=False)
    
    print(f"✅ Saved {len(segments_df)} interaction segments to {output_file_path}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Video-level social interaction segment analysis')
    parser.add_argument('--folder_path', type=str, required=True, help='Folder path containing input CSV and where outputs will be saved')

    args = parser.parse_args()
    folder_path = Path(args.folder_path)

    # Ensure folder exists
    folder_path.mkdir(parents=True, exist_ok=True)

    # Construct paths automatically
    output_path = folder_path / Inference.INTERACTION_SEGMENTS_CSV.name
    prefix = Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem
    matching_files = list(folder_path.glob(f"{prefix}*.csv"))

    if not matching_files:
        raise FileNotFoundError(
            f"No file starting with '{prefix}' found in {folder_path}"
        )
    elif len(matching_files) > 1:
        print(f"⚠️ Multiple files found starting with '{prefix}', using the first one:")
        for f in matching_files:
            print(f"   - {f.name}")

    input_path = matching_files[0]
    print(f"Using input frame-level data from: {input_path}")

    # Run main analysis
    main(output_file_path=output_path, frame_data_path=input_path, hyperparameter_tuning=False)

    # Copy current script into folder for reproducibility
    try:
        current_script = Path(__file__)
        destination_script = folder_path / current_script.name
        shutil.copy(current_script, destination_script)
        print(f"🧾 Copied script to {destination_script}")
    except Exception as e:
        print(f"⚠️ Could not copy script to folder: {e}")