# Research Question 1. How much time do children spend alone?
#
# This script analyzes multimodal social interaction patterns by combining:
# - Visual person detection (child/adult bodies)
# - Visual face detection (child/adult faces with proximity measures)  
# - Audio vocalization detection (child/other speaker identification)
#
# The analysis produces frame-level classifications of social contexts to understand
# when children are alone vs. in various types of social interactions.

import sqlite3
import re
import sys
import pandas as pd
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, ResearchQuestions
from config import DataConfig, Research_QuestionConfig

# Constants
FPS = DataConfig.FPS # frames per second

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def create_interaction_segments():
    """
    Creates mutually exclusive segments, buffering small state changes.

    Args:
        results_df (pd.DataFrame): DataFrame with 'video_id', 'frame_number', 'interaction_category'.

    Returns:
        pd.DataFrame: A DataFrame of the extracted, buffered segments.
    """
    print("Creating segments...")
    frame_data = pd.read_csv(ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV)

    all_segments = []

    for video_id, video_df in frame_data.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').reset_index(drop=True)
        
        if len(video_df) == 0:
            continue
                    
        # Get interaction states and frame numbers
        states = video_df['interaction_category'].values
        frame_numbers = video_df['frame_number'].values
        video_name = video_df['video_name'].iloc[0]
        
        # Buffer short state changes
        buffered_states = states.copy()
        i = 0
        while i < len(buffered_states) - 1:
            current_state = buffered_states[i]
            j = i + 1
            # Find the end of the current run of states
            while j < len(buffered_states) and buffered_states[j] == current_state:
                j += 1
            
            # The length of the current run of states
            run_length = j - i
            run_duration = (frame_numbers[j-1] - frame_numbers[i]) / FPS
            
            # If the run is short and not at the beginning/end, merge it
            if run_duration < Research_QuestionConfig.RQ1_MIN_CHANGE_DURATION_SEC and i > 0 and j < len(buffered_states):
                # Replace the short run with the previous state
                buffered_states[i:j] = buffered_states[i-1]
                # Reset i to re-evaluate from the previous point
                i = 0
            else:
                i = j
        
        # Now, find state changes in the buffered states
        current_state = buffered_states[0]
        segment_start_frame = frame_numbers[0]
        
        for i in range(1, len(buffered_states)):
            if buffered_states[i] != current_state:
                segment_end_frame = frame_numbers[i-1]
                
                # Only keep segments longer than minimum duration
                segment_duration = (segment_end_frame - segment_start_frame) / FPS
                if segment_duration >= Research_QuestionConfig.RQ1_MIN_SEGMENT_DURATION_SEC:
                    all_segments.append({
                        'video_id': video_id,
                        'video_name': video_name,
                        'category': current_state,
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
        if segment_duration >= Research_QuestionConfig.RQ1_MIN_SEGMENT_DURATION_SEC:
            all_segments.append({
                'video_id': video_id,
                'video_name': video_name,
                'category': current_state,
                'segment_start': segment_start_frame,
                'segment_end': segment_end_frame,
                'start_time_sec': segment_start_frame / FPS,
                'end_time_sec': segment_end_frame / FPS,
                'duration_sec': segment_duration
            })
    
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
    else:
        segments_df = pd.DataFrame(columns=['video_id', 'video_name', 'category',
                                        'segment_start', 'segment_end', 
                                        'start_time_sec', 'end_time_sec', 'duration_sec'])
        
    # Add child_id to segments_df using extract_child_id
    segments_df['child_id'] = segments_df['video_name'].apply(extract_child_id)
    
    # Load subjects CSV to get age information
    try:
        subjects_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
        print(f"📋 Loaded subjects data with {len(subjects_df)} records for age mapping")
        
        # Merge age information based on video_name
        segments_df = segments_df.merge(
            subjects_df[['video_name', 'age_at_recording']], 
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
        else:
            print("✅ All segments successfully matched with age data")
            
    except FileNotFoundError:
        print(f"⚠️ Warning: Subjects CSV not found at {DataPaths.SUBJECTS_CSV_PATH}")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None
    except Exception as e:
        print(f"⚠️ Warning: Error loading subjects data: {e}")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None

    # Reorder columns so that child_id and age_at_recording come after video_name
    cols = ['video_name', 'child_id', 'age_at_recording'] + [col for col in segments_df.columns if col not in ['video_name', 'child_id', 'age_at_recording']]
    segments_df = segments_df.loc[:, cols]

    print(f"Created {len(segments_df)} segments after buffering.")
    segments_df.to_csv(ResearchQuestions.INTERACTION_SEGMENTS_CSV, index=False)

if __name__ == "__main__":
    # Just extract segments from existing frame data
    create_interaction_segments()