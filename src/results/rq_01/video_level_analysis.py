# Research Question 1. How much time do children spend alone?
#
# This script generates video-level segments to analyze the amount of time children spend alone.
# It applies post-processing to the frame-level data to create these segments.

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

def validate_interaction_segment(category, duration, video_df, start_frame, end_frame, video_name):
        """
        Validate "Interacting" segments for sufficient visual evidence of people.
        
        Returns the final category (may downgrade "Interacting" to "Alone")
        """
        if category == "Interacting" and duration > Research_QuestionConfig.VALIDATION_SEGMENT_DURATION_SEC:
            # Get frames within this segment
            segment_frames = video_df[
                (video_df['frame_number'] >= start_frame) & 
                (video_df['frame_number'] <= end_frame)
            ]
            
            # Check if at least 5% of frames have adult or child present
            if len(segment_frames) > 0:
                frames_with_people = segment_frames[
                    (segment_frames.get('child_present', 0) == 1) | 
                    (segment_frames.get('adult_present', 0) == 1)
                ]
                
                people_presence_ratio = len(frames_with_people) / len(segment_frames)

                if people_presence_ratio < Research_QuestionConfig.PERSON_PRESENT_THRESHOLD:  # Less than 5%
                    return "Alone"
        
        return category
    
def create_interaction_segments(output_dir: Path, frame_data_path: Path):
    """
    Creates mutually exclusive segments, buffering small state changes and stores them.

    Parameters
    ----------
    output_dir : Path
        Directory to save the output segments.
    frame_data_path : Path
        Path to the CSV file containing frame-level interaction data.
    """    
    print("Creating segments...")
    frame_data = pd.read_csv(frame_data_path)

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
                    
                    # Validate "Interacting" segments for visual evidence
                    final_category = validate_interaction_segment(
                        current_state, segment_duration, video_df, 
                        segment_start_frame, segment_end_frame, video_name
                    )
                    
                    all_segments.append({
                        'video_id': video_id,
                        'video_name': video_name,
                        'category': final_category,
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
            
            # Validate "Interacting" segments for visual evidence
            final_category = validate_interaction_segment(
                current_state, segment_duration, video_df, 
                segment_start_frame, segment_end_frame, video_name
            )
            
            all_segments.append({
                'video_id': video_id,
                'video_name': video_name,
                'category': final_category,
                'segment_start': segment_start_frame,
                'segment_end': segment_end_frame,
                'start_time_sec': segment_start_frame / FPS,
                'end_time_sec': segment_end_frame / FPS,
                'duration_sec': segment_duration
            })
    
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
        
        # Post-processing: Merge segments of the same category with small gaps
        print("Post-processing: Merging segments with small gaps...")
        merged_segments = []
        
        for video_id, video_segments in segments_df.groupby('video_id'):
            video_segments = video_segments.sort_values('start_time_sec').reset_index(drop=True)
            
            if len(video_segments) == 0:
                continue
            
            current_segment = video_segments.iloc[0].copy()
            merge_count = 0
            
            for i in range(1, len(video_segments)):
                next_segment = video_segments.iloc[i]
                
                # Calculate gap between current segment end and next segment start
                gap_duration = next_segment['start_time_sec'] - current_segment['end_time_sec']
                
                # If same category and gap is less than 3 seconds, merge
                if (current_segment['category'] == next_segment['category'] and 
                    gap_duration < 5.0):
                    
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
            segments_df = pd.DataFrame(merged_segments)
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

    # Final summary statistics
    if len(segments_df) > 0:
        total_segments = len(segments_df)
        interacting_segments = len(segments_df[segments_df['category'] == 'Interacting'])
        alone_segments = len(segments_df[segments_df['category'] == 'Alone'])
        copresent_segments = len(segments_df[segments_df['category'] == 'Co-present Silent'])
        
        print(f"\n📊 Final segment summary:")
        print(f"   Total segments: {total_segments}")
        print(f"   Interacting: {interacting_segments} ({interacting_segments/total_segments*100:.1f}%)")
        print(f"   Alone: {alone_segments} ({alone_segments/total_segments*100:.1f}%)")
        if copresent_segments > 0:
            print(f"   Co-present Silent: {copresent_segments} ({copresent_segments/total_segments*100:.1f}%)")

    print(f"Created {len(segments_df)} segments after buffering and merging.")
    file_name = ResearchQuestions.INTERACTION_SEGMENTS_CSV.name
    segments_df.to_csv(output_dir / file_name, index=False)
    print(f"Saved interaction segments to {output_dir / file_name}")

if __name__ == "__main__":
    # Just extract segments from existing frame data
    create_interaction_segments(output_dir=ResearchQuestions.RQ1_OUTPUT_DIR, frame_data_path=ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV)