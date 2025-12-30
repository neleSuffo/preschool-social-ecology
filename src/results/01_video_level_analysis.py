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

def get_min_segment_duration(interaction_type: str) -> float:
    """
    Returns the minimum required segment duration (in seconds) based on the interaction type.
    
    This function relies on the assumption that the following constants are defined
    in InferenceConfig:
    - MIN_INTERACTING_SEGMENT_DURATION_SEC
    - MIN_ALONE_SEGMENT_DURATION_SEC
    - MIN_AVAILABLE_SEGMENT_DURATION_SEC
    """
    # Assuming the interaction_type is properly capitalized (e.g., 'Interacting', 'Alone')
    type_map = {
        'Interacting': InferenceConfig.MIN_INTERACTING_SEGMENT_DURATION_SEC,
        'Alone': InferenceConfig.MIN_ALONE_SEGMENT_DURATION_SEC,
        'Available': InferenceConfig.MIN_AVAILABLE_SEGMENT_DURATION_SEC,
    }
    # Use a default minimum (e.g., the largest) if the type is unexpected
    default_min = getattr(InferenceConfig, 'MIN_ALONE_SEGMENT_DURATION_SEC', 15.0) 
    return type_map.get(interaction_type, default_min)

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
                
    # Get interaction states and frame numbers
    states = video_df['interaction_type'].values
    frame_numbers = video_df['frame_number'].values
    video_name = video_df['video_name'].iloc[0]
    
    # Buffer short state changes
    buffered_states = states.copy()  # Disable buffering for now
    
    segments = []
    current_state = buffered_states[0]
    segment_start_frame = frame_numbers[0]
    
    # Process state changes
    for i in range(1, len(buffered_states)):
        if buffered_states[i] != current_state:
            segment_end_frame = frame_numbers[i-1]
            
            required_min_duration = get_min_segment_duration(current_state)
            
            segment_duration = (segment_end_frame - segment_start_frame) / FPS
            if segment_duration >= required_min_duration:
                
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
            
            current_state = buffered_states[i]
            segment_start_frame = frame_numbers[i]
    
    # Handle the final segment
    segment_end_frame = frame_numbers[-1]
    
    required_min_duration = get_min_segment_duration(current_state)
    
    segment_duration = (segment_end_frame - segment_start_frame) / FPS
    if segment_duration >= required_min_duration:
        
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

def reclassify_implicit_turn_taking(segments_df, frame_data):
    """
    Reclassify 'Available' or 'Alone' segments to 'Interacting' if they contain 
    sufficient evidence of implicit, KCHI-based turn-taking.

    Criteria (Implicit Turn-Taking Character):
    1. Segment type is 'Available' or 'Alone'.
    2. At least 20% of the segment's sampled frames are KCHI-only (KCHI=1, CDS=0, OHS=0).
    3. Person/Face (person_present) is detected for at least 5% of the segment's frames.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments (after buffering and merging).
    frame_data : pd.DataFrame
        Original frame-level data containing multimodal flags.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with reclassified segments.
    """    
    reclassified_count = 0
    # Use copy for safe modification
    updated_segments_df = segments_df.copy()

    # Iterate over the segments to check for reclassification
    for index, segment in segments_df.iterrows():
        
        # --- Condition 1: Target Segments ---
        if segment['interaction_type'] not in ['Available', 'Alone']:
            continue
        
        video_name = segment['video_name']
        start_frame = segment['segment_start']
        end_frame = segment['segment_end']
        
        # Filter frame data for the current segment's video and frame range
        current_video_frames = frame_data[frame_data['video_name'] == video_name]
        segment_frames = current_video_frames[
            (current_video_frames['frame_number'] >= start_frame) & 
            (current_video_frames['frame_number'] <= end_frame)
        ].copy() # Work on a copy of segment frames
        
        total_segment_frames = len(segment_frames)
        if total_segment_frames == 0:
            continue
            
        # --- Condition 2: KCHI-Only Density Check ---
        # Frames must have KCHI but absolutely no CDS
        kchi_only_count = segment_frames[
            (segment_frames['has_kchi'] == 1) & 
            (segment_frames['has_cds'] == 0)
        ].shape[0]
        
        kchi_only_fraction = kchi_only_count / total_segment_frames

        if kchi_only_fraction < InferenceConfig.KCHI_ONLY_FRACTION_THRESHOLD:
            continue
            
        # --- Condition 3: Person Presence Check ---
        # The 'person_or_face_present' column fuses face and person detection
        person_presence_count = (segment_frames['person_or_face_present'] == 1).sum()
        person_presence_fraction = person_presence_count / total_segment_frames
        
        if person_presence_fraction >= InferenceConfig.MIN_PERSON_PRESENCE_FRACTION:
            # All conditions met: Reclassify to 'Interacting'
            
            # Use .loc on the updated_segments_df copy
            updated_segments_df.loc[index, 'interaction_type'] = 'Interacting'
            reclassified_count += 1
            
    print(f"   Reclassified {reclassified_count} 'Available'/'Alone' segments to 'Interacting' (Implicit Turn-Taking).")
    return updated_segments_df

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

def reclassify_alone_segments(segments_df, frame_data, detection_col='person_or_face_present'):
    """
    Reclassify 'Alone' segments to 'Available' if they contain sufficient evidence
    of partner presence (visual or audio) that exceeds defined thresholds.
    
    (EXCLUDES segments where the frame-level classification was ALONE due to MEDIA.)
    
    CRITERIA FOR RECLASSIFICATION (Alone -> Available):
    1. Segment type must be 'Alone'.
    2. Segment length must be longer than MIN_RECLASSIFY_DURATION_SEC (to avoid short noise).
    3. Person/Face detection (person_or_face_present) must occur for > 5% of segment frames.
    4. OR Partner audio (has_cds OR has_ohs) must occur for > 5% of segment frames.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with final interaction segments.
    frame_data : pd.DataFrame
        Original frame-level data.
    detection_col : str
        The name of the fused detection column.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with reclassified segments.
    """        
    # Ensure necessary columns exist (including the new media flag for exclusion)
    required_cols = [detection_col, 'has_cds', 'has_ohs', 'is_media_interaction'] 
    if not all(col in frame_data.columns for col in required_cols):
        print(f"⚠️ Warning: Required frame columns {required_cols} not found. Skipping alone reclassification.")
        return segments_df
        
    reclassified_count = 0
    updated_segments_df = segments_df.copy()

    for video_id, video_segments in updated_segments_df.groupby('video_id'):
        
        video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=False)
        original_indices = video_segments['index'] # Map back to original segments_df index
        
        for i in range(len(video_segments)):
            segment = video_segments.iloc[i]
            
            # --- Condition 1: Must be 'Alone' ---
            if segment['interaction_type'] != 'Alone':
                continue
            
            #--- Condition 2: Duration Check (must be > MIN_RECLASSIFY_DURATION_SEC seconds) ---
            if segment['duration_sec'] <= InferenceConfig.MIN_RECLASSIFY_DURATION_SEC:
               continue

            # --- Condition 3: Exclude Media Interaction Segments ---
            video_name = segment['video_name']
            start_frame = segment['segment_start']
            end_frame = segment['segment_end']
            
            current_video_frames = frame_data[frame_data['video_name'] == video_name]
            segment_frames = current_video_frames[
                (current_video_frames['frame_number'] >= start_frame) & 
                (current_video_frames['frame_number'] <= end_frame)
            ]
            
            # Check if *any* frame in the segment was flagged as media interaction
            # If any frame was flagged as media interaction, we assume the whole segment
            # was derived from that state and should not be reclassified.
            if segment_frames['is_media_interaction'].any():
                 print(f"   [Alone->Available] Segment {segment['start_time_sec']:.1f}s skipped due to MEDIA flag.")
                 continue
            
            total_frames = len(segment_frames)
            if total_frames == 0:
                continue
                
            # --- Condition 4: Visual Presence Check (> 5%) ---
            person_count = (segment_frames[detection_col] == 1).sum()
            visual_fraction = person_count / total_frames
            
            # --- Condition 5: Partner Audio Check (> 5%) ---
            partner_audio_count = (
                (segment_frames['has_cds'] == 1) | 
                (segment_frames['has_ohs'] == 1)
            ).sum()
            audio_fraction = partner_audio_count / total_frames
            
            # --- Reclassification Logic (OR condition) ---
            
            should_reclassify = False
            
            if visual_fraction > InferenceConfig.ALONE_RECLASSIFY_VISUAL_THRESHOLD:
                # print(f"   [Alone->Available] Segment {segment['start_time_sec']:.1f}s: Visual frac {visual_fraction:.2f} > {InferenceConfig.ALONE_RECLASSIFY_VISUAL_THRESHOLD}")
                should_reclassify = True

            if audio_fraction > InferenceConfig.ALONE_RECLASSIFY_AUDIO_THRESHOLD:
                # print(f"   [Alone->Available] Segment {segment['start_time_sec']:.1f}s: Audio frac {audio_fraction:.2f} > {InferenceConfig.ALONE_RECLASSIFY_AUDIO_THRESHOLD}")
                should_reclassify = True

            if should_reclassify:
                # Get the original index and apply reclassification
                original_idx = original_indices.iloc[i]
                updated_segments_df.loc[original_idx, 'interaction_type'] = 'Available'
                reclassified_count += 1
            
    print(f"   Reclassified {reclassified_count} 'Alone' segments to 'Available' (High Presence Rule).")
    return updated_segments_df

def reclassify_ghost_segments(segments_df, frame_data):
    """
    Reclassifies 'Interacting' or 'Available' segments to 'Alone' if they 
    lack sufficient visual human presence, using type-specific thresholds.
    """
    reclassified_count = 0
    updated_segments_df = segments_df.copy()

    for index, segment in segments_df.iterrows():
        # Skip segments already classified as Alone
        if segment['interaction_type'] == 'Alone':
            continue
        
        # 1. Select type-specific thresholds
        if segment['interaction_type'] == 'Interacting':
            min_duration = InferenceConfig.MIN_GHOST_CHECK_DURATION_INTERACTING
            visual_threshold = InferenceConfig.GHOST_VISUAL_THRESHOLD_INTERACTING
        else: # Available
            min_duration = InferenceConfig.MIN_GHOST_CHECK_DURATION_AVAILABLE
            visual_threshold = InferenceConfig.GHOST_VISUAL_THRESHOLD_AVAILABLE

        # Skip if the segment is shorter than the type-specific minimum
        if segment['duration_sec'] < min_duration:
            continue
        
        video_name = segment['video_name']
        start_frame = segment['segment_start']
        end_frame = segment['segment_end']
        
        # 2. Filter frame data and calculate presence
        current_video_frames = frame_data[frame_data['video_name'] == video_name]
        segment_frames = current_video_frames[
            (current_video_frames['frame_number'] >= start_frame) & 
            (current_video_frames['frame_number'] <= end_frame)
        ]
        
        if segment_frames.empty:
            continue
            
        human_presence_mask = (segment_frames['person_or_face_present'] == 1)        
        human_presence_frac = human_presence_mask.sum() / len(segment_frames)
        
        # 3. Apply Reclassification
        if human_presence_frac < visual_threshold:
            updated_segments_df.loc[index, 'interaction_type'] = 'Alone'
            reclassified_count += 1

    print(f"   Reclassified {reclassified_count} 'Ghost' segments to 'Alone' (Dual-Threshold Gate).")
    return updated_segments_df

def main(output_file_path: Path, frame_data_path: Path, hyperparameter_tuning: False):
    """
    Main entry point for video-level segment analysis.
    Creates mutually exclusive interaction segments from frame-level data.
    
    This function processes frame-level interaction data through several steps:
    1. Load and process frame-level data
    2. Create initial segments for each video
    3. Merge segments with small gaps
    4. Reclassify 'Available' segments if no detection occurred (NEW STEP)
    5. Add metadata (child_id, age)
    6. Generate summary statistics
    7. Save results to CSV
    
    Parameters
    ----------
    output_file_path : Path
        Path to the output CSV file for saving the segments.
    frame_data_path : Path
        Path to the CSV file containing frame-level interaction data.
    hyperparameter_tuning: Bool
        Whether script runs in hyperparmeter mode or not 
    """     
    if hyperparameter_tuning:
        run_dir = output_file_path.parent
           
        try:
            script_path = Path(__file__)
            shutil.copy(script_path, run_dir / script_path.name)
        except NameError:
            print("⚠️ __file__ not defined (likely running in notebook), skipping script copy.")
        
    # Step 1: Load frame-level data
    frame_data = pd.read_csv(frame_data_path)

    # Step 2: Create segments for each video
    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        video_segments = create_segments_for_video(video_id, video_df)
        all_segments.extend(video_segments)
    
    # save intermediate segments for debugging
    intermediate_segments_path = output_file_path.parent / f"intermediate_segments_{output_file_path.name}"
    if all_segments:
        intermediate_df = pd.DataFrame(all_segments)
        intermediate_df.to_csv(intermediate_segments_path, index=False)
        print(f"✅ Saved intermediate segments to {intermediate_segments_path}")
    # Step 3: Convert to DataFrame and merge segments with small gaps
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
        
        segments_df = merge_same_segments(segments_df)
        
    else:
        segments_df = pd.DataFrame(columns=['video_id', 'video_name', 'interaction_type',
                                        'segment_start', 'segment_end', 
                                        'start_time_sec', 'end_time_sec', 'duration_sec'])

    # Step 4a: Reclassify short, sandwiched 'Alone' segments between Alone segments to 'Available'
    segments_df = reclassify_alone_segments(segments_df, frame_data, detection_col='person_or_face_present')
    segments_df = reclassify_implicit_turn_taking(segments_df, frame_data)

    # Rerun merge AFTER all types have been finalized to clean up fragmentation
    print("🧹 Final consolidation: Merging reclassified segments of the same type...")
    segments_df = merge_same_segments(segments_df)
    segments_df = reclassify_ghost_segments(segments_df, frame_data)

    # Step 6: Final Step: Fill all remaining gaps to create a continuous timeline
    segments_df = fill_gaps_with_default(segments_df, default_type="Available")
    segments_df = merge_same_segments(segments_df)

    # Step 7: Generate and print summary
    print_segment_summary(segments_df)
    
    # Read age csv    
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH, sep=";", decimal=",")
    age_df = age_df[["video_name", "age_at_recording", "child_id"]]
    # Merge on video_name to get age information
    segments_df = segments_df.merge(age_df, on="video_name", how="left")
    
    # Step 8: Save results
    segments_df.to_csv(output_file_path, index=False)
    segments_df.to_csv(Evaluation.PRED_SECONDWISE_FILE_PATH, index=False)
    print(f"✅ Saved {len(segments_df)} interaction segments to {output_file_path}")
    print(f" Saved interaction segments gt to {Evaluation.PRED_SECONDWISE_FILE_PATH}")

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