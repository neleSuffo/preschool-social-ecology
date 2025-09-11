# Research Question 1. How much time do children spend alone?
#
# This script generates video-level segments to analyze the amount of time children spend alone.
# It applies post-processing to the frame-level data to create these segments.

import re
import argparse
import sys
import pandas as pd
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Inference
from config import DataConfig, InferenceConfig

# Constants
FPS = DataConfig.FPS # frames per second

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def validate_interaction_segment(interaction_type, duration, video_id, start_time_sec, end_time_sec, video_name):
    """
    Validate "Interacting" segments for sufficient turn-taking evidence.
    
    Checks if KCHI and other speakers alternate within 3-second windows 
    at least 3 times in segments longer than 30 seconds.
    
    Returns the final interaction_type (may downgrade "Interacting" to "Alone")
    """
    if interaction_type == "Interacting" and duration > InferenceConfig.VALIDATION_SEGMENT_DURATION_SEC:
        # Import here to avoid circular imports
        import sqlite3
        from constants import DataPaths
        
        # Connect to database and get vocalizations for this video and time window
        conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
        vocalizations_query = """
        SELECT speaker, start_time_seconds, end_time_seconds
        FROM Vocalizations v
        JOIN Videos vid ON v.video_id = vid.video_id
        WHERE vid.video_id = ? 
        AND start_time_seconds < ? 
        AND end_time_seconds > ?
        ORDER BY start_time_seconds
        """
        
        vocs_df = pd.read_sql_query(vocalizations_query, conn, 
                                params=[video_id, end_time_sec, start_time_sec])
        conn.close()

        if len(vocs_df) < 1:  # Need at least 1 vocalization for turn-taking
            return "Alone"

        # Count turn-taking instances within 5-second windows
        turn_taking_count = 0
        
        for i in range(len(vocs_df) - 1):
            current_voc = vocs_df.iloc[i]
            next_voc = vocs_df.iloc[i + 1]
            
            # Check if speakers are different
            if current_voc['speaker'] != next_voc['speaker']:
                # Check if one is KCHI and the other is not
                speakers = {current_voc['speaker'], next_voc['speaker']}
                if 'KCHI' in speakers and len(speakers) == 2:
                    # Calculate time gap between vocalizations
                    time_gap = next_voc['start_time_seconds'] - current_voc['end_time_seconds']

                    # If within 5-second window, count as turn-taking
                    if 0 <= time_gap <= 5.0:
                        turn_taking_count += 1
        
        # Require at least 3 turn-taking instances for validation
        if turn_taking_count < 2:
            return "Alone"
    
    return interaction_type

def buffer_short_state_changes(states, frame_numbers):
    """
    Buffer short state changes to avoid rapid transitions.
    
    Parameters
    ----------
    states : np.array
        Array of interaction states
    frame_numbers : np.array
        Array of frame numbers
        
    Returns
    -------
    np.array
        Buffered states with short changes smoothed out
    """
    buffered_states = states.copy()
    i = 0
    while i < len(buffered_states) - 1:
        current_state = buffered_states[i]
        j = i + 1
        # Find the end of the current run of states
        while j < len(buffered_states) and buffered_states[j] == current_state:
            j += 1
        
        # The length of the current run of states
        run_duration = (frame_numbers[j-1] - frame_numbers[i]) / FPS
        
        # If the run is short and not at the beginning/end, merge it
        if run_duration < InferenceConfig.MIN_CHANGE_DURATION_SEC and i > 0 and j < len(buffered_states):
            # Replace the short run with the previous state
            buffered_states[i:j] = buffered_states[i-1]
            # Reset i to re-evaluate from the previous point
            i = 0
        else:
            i = j
    
    return buffered_states

def create_segments_for_video(video_id, video_df):
    """
    Create segments for a single video.
    
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
    states = video_df['interaction_category'].values
    frame_numbers = video_df['frame_number'].values
    video_name = video_df['video_name'].iloc[0]
    
    # Buffer short state changes
    buffered_states = buffer_short_state_changes(states, frame_numbers)
    
    segments = []
    current_state = buffered_states[0]
    segment_start_frame = frame_numbers[0]
    
    # Process state changes
    for i in range(1, len(buffered_states)):
        if buffered_states[i] != current_state:
            segment_end_frame = frame_numbers[i-1]
            
            # Only keep segments longer than minimum duration
            segment_duration = (segment_end_frame - segment_start_frame) / FPS
            if segment_duration >= InferenceConfig.MIN_SEGMENT_DURATION_SEC:

                # Validate "Interacting" segments for turn-taking evidence
                # final_interaction_type = validate_interaction_segment(
                #     current_state, segment_duration, video_id, 
                #     segment_start_frame / FPS, segment_end_frame / FPS, video_name
                # )
                
                segments.append({
                    'video_id': video_id,
                    'video_name': video_name,
                    #'interaction_type': final_interaction_type,
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
    segment_duration = (segment_end_frame - segment_start_frame) / FPS
    if segment_duration >= InferenceConfig.MIN_SEGMENT_DURATION_SEC:
        
        # # Validate "Interacting" segments for turn-taking evidence
        # final_interaction_type = validate_interaction_segment(
        #     current_state, segment_duration, video_id, 
        #     segment_start_frame / FPS, segment_end_frame / FPS, video_name
        # )
        
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

def merge_segments_with_small_gaps(segments_df):
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
            
            # Calculate gap between current segment end and next segment start
            gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
            # If same interaction type and gap is less than the configured duration, merge
            if (current_segment['interaction_type'] == next_segment['interaction_type'] and 
                gap_duration < InferenceConfig.GAP_MERGE_DURATION_SEC):
                
                # Extend current segment to include the next one
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
        
        # Don't forget the last segment
        merged_segments.append(current_segment.to_dict())
    
    if merged_segments:
        result_df = pd.DataFrame(merged_segments)
        result_df = result_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
        return result_df
    else:
        return segments_df

def add_metadata_to_segments(segments_df, frame_data):
    """
    Add child_id and age information to segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments
    frame_data : pd.DataFrame
        Frame-level data containing age information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added metadata
    """    
    # Add child_id to segments_df using extract_child_id
    segments_df['child_id'] = segments_df['video_name'].apply(extract_child_id)
    
    # change segment name "co-present silent" to "Co-present"
    segments_df['interaction_type'] = segments_df['interaction_type'].replace("Co-present Silent", "Co-present")
    # Extract age information from frame_data (get unique video_name to age_at_recording mapping)
    try:
        # Get unique video_name to age_at_recording mapping from frame_data
        age_mapping = frame_data[['video_name', 'age_at_recording']].drop_duplicates()
        
        # Merge age information based on video_name
        segments_df = segments_df.merge(
            age_mapping, 
            on='video_name', 
            how='left'
        )
        
        # Check merge success
        missing_age_count = segments_df['age_at_recording'].isna().sum()
        if missing_age_count > 0:
            print(f"⚠️ Warning: {missing_age_count} segments missing age data ({missing_age_count/len(segments_df)*100:.1f}%)")
            
            # Show some examples of unmatched video names
            unmatched_videos = segments_df[segments_df['age_at_recording'].isna()]['video_name'].unique()[:5]
            print(f"Examples of unmatched video names: {list(unmatched_videos)}")
            
    except KeyError:
        print(f"⚠️ Warning: 'age_at_recording' column not found in frame data")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None
    except Exception as e:
        print(f"⚠️ Warning: Error extracting age data from frame data: {e}")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None

    # Reorder columns so that child_id and age_at_recording come after video_name
    cols = ['video_name', 'child_id', 'age_at_recording'] + [col for col in segments_df.columns if col not in ['video_name', 'child_id', 'age_at_recording']]
    segments_df = segments_df.loc[:, cols]
    
    return segments_df

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
        interacting_segments = len(segments_df[segments_df['interaction_type'] == 'Interacting'])
        alone_segments = len(segments_df[segments_df['interaction_type'] == 'Alone'])
        copresent_segments = len(segments_df[segments_df['interaction_type'] == 'Co-present'])

        # Calculate total duration for each segment interaction_type in minutes (with two decimals)
        total_duration = round(segments_df['duration_sec'].sum() / 60, 2)
        interacting_duration = round(segments_df[segments_df['interaction_type'] == 'Interacting']['duration_sec'].sum() / 60, 2)
        alone_duration = round(segments_df[segments_df['interaction_type'] == 'Alone']['duration_sec'].sum() / 60, 2)
        copresent_duration = round(segments_df[segments_df['interaction_type'] == 'Co-present']['duration_sec'].sum() / 60, 2)

        print(f"\n📊 Final segment summary:")
        print(f"   Total segments: {total_segments} ({total_duration} minutes)")
        print(f"   Interacting: {interacting_segments} ({interacting_duration} minutes - {interacting_duration/total_duration*100:.1f}%)")
        print(f"   Alone: {alone_segments} ({alone_duration} minutes - {alone_duration/total_duration*100:.1f}%)")
        if copresent_segments > 0:
            print(f"   Co-present: {copresent_segments} ({copresent_duration} minutes - {copresent_duration/total_duration*100:.1f}%)")
    else:
        print("\n📊 No segments created")

def main(output_dir: Path, frame_data_path: Path):
    """
    Main entry point for video-level segment analysis.
    Creates mutually exclusive interaction segments from frame-level data.
    
    This function processes frame-level interaction data through several steps:
    1. Load and process frame-level data
    2. Create initial segments for each video
    3. Merge segments with small gaps
    4. Add metadata (child_id, age)
    5. Generate summary statistics
    6. Save results to CSV
    
    Parameters
    ----------
    output_dir : Path
        Directory to save the output segments.
    frame_data_path : Path
        Path to the CSV file containing frame-level interaction data.
    """        
    # Step 1: Load frame-level data
    frame_data = pd.read_csv(frame_data_path)

    # Step 2: Create segments for each video
    all_segments = []
    for video_id, video_df in frame_data.groupby('video_id'):
        video_segments = create_segments_for_video(video_id, video_df)
        all_segments.extend(video_segments)
    
    # Step 3: Convert to DataFrame and merge segments with small gaps
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
        
        segments_df = merge_segments_with_small_gaps(segments_df)
        
    else:
        segments_df = pd.DataFrame(columns=['video_id', 'video_name', 'interaction_type',
                                        'segment_start', 'segment_end', 
                                        'start_time_sec', 'end_time_sec', 'duration_sec'])
        
    # Step 4: Add metadata
    segments_df = add_metadata_to_segments(segments_df, frame_data)
    
    # Step 5: Generate and print summary
    print_segment_summary(segments_df)
    
    # Step 6: Save results
    file_name = Inference.INTERACTION_SEGMENTS_CSV.name
    segments_df.to_csv(output_dir / file_name, index=False)
    print(f"✅ Saved {len(segments_df)} interaction segments to {output_dir / file_name}")

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Video-level social interaction segment analysis')
    parser.add_argument('--input', type=str, default=str(Inference.FRAME_LEVEL_INTERACTIONS_CSV),
                    help='Path to the frame-level interactions CSV file')
    parser.add_argument('--output', type=str,
                    help='Output CSV file path (if not specified, uses default output directory)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis and save to specific file
        main(output_dir=output_dir, frame_data_path=Path(args.input))
        
        # If the generated file has a different name, rename it
        default_output = output_dir / Inference.INTERACTION_SEGMENTS_CSV.name
        if default_output.exists() and default_output != output_path:
            default_output.rename(output_path)
            print(f"✅ Renamed output to {output_path}")
    else:
        # Use default behavior
        main(output_dir=Inference.BASE_OUTPUT_DIR, frame_data_path=Path(args.input))