# Research Question: 6. Which aspects contribute to detected interactions?

import sqlite3
import pandas as pd
import re
from pathlib import Path
from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig

def extract_child_id(video_name):
    """
    Extracts the 6-digit child ID from a video name string.
    Example: 'id123456_video.mp4' -> '123456'
    """
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def analyze_audio_turn_taking(segment_row, vocalizations_df):
    """
    Analyze turn-taking characteristics within a segment.
    
    Parameters:
    ----------
    segment_row : pd.Series
        Row from segments DataFrame with segment boundaries
    vocalizations_df : pd.DataFrame
        DataFrame with all vocalizations including speaker info
        
    Returns:
    -------
    dict
        Dictionary with turn-taking metrics
    """
    # Filter vocalizations to this segment
    segment_vocs = vocalizations_df[
        (vocalizations_df['child_id'] == segment_row['child_id']) &
        (vocalizations_df['start_time_seconds'] >= segment_row['segment_start_time']) &
        (vocalizations_df['end_time_seconds'] <= segment_row['segment_end_time'])
    ].sort_values('start_time_seconds').reset_index(drop=True)
    
    if len(segment_vocs) < 2:
        return {
            'has_turn_taking': False,
            'turn_count': 0,
            'kchi_vocalizations': len(segment_vocs[segment_vocs['speaker'] == 'KCHI']),
            'other_vocalizations': len(segment_vocs[segment_vocs['speaker'] != 'KCHI']),
            'total_vocalizations': len(segment_vocs)
        }
    
    # Count turns between KCHI and others
    turn_count = 0
    for i in range(len(segment_vocs) - 1):
        curr = segment_vocs.iloc[i]
        nxt = segment_vocs.iloc[i + 1]
        
        # Check for speaker alternation
        curr_is_kchi = curr['speaker'] == 'KCHI'
        next_is_kchi = nxt['speaker'] == 'KCHI'
        
        if curr_is_kchi != next_is_kchi:  # Speaker change
            # Check gap between vocalizations (≤5 seconds)
            gap = nxt['start_time_seconds'] - curr['end_time_seconds']
            if gap <= 5.0:  # Valid turn
                turn_count += 1
    
    return {
        'has_turn_taking': turn_count > 0,
        'turn_count': turn_count,
        'kchi_vocalizations': len(segment_vocs[segment_vocs['speaker'] == 'KCHI']),
        'other_vocalizations': len(segment_vocs[segment_vocs['speaker'] != 'KCHI']),
        'total_vocalizations': len(segment_vocs)
    }

def analyze_proximity_frames(segment_row, frames_df):
    """
    Analyze proximity characteristics within a segment.
    
    Parameters:
    ----------
    segment_row : pd.Series
        Row from segments DataFrame with segment boundaries
    frames_df : pd.DataFrame
        DataFrame with frame-level proximity data
        
    Returns:
    -------
    dict
        Dictionary with proximity metrics
    """
    # Convert segment times to frame numbers
    start_frame = int(segment_row['segment_start_time'] * DataConfig.FPS)
    end_frame = int(segment_row['segment_end_time'] * DataConfig.FPS)
    
    # Filter frames to this segment
    segment_frames = frames_df[
        (frames_df['child_id'] == segment_row['child_id']) &
        (frames_df['frame_number'] >= start_frame) &
        (frames_df['frame_number'] <= end_frame)
    ]
    
    if len(segment_frames) == 0:
        return {
            'total_frames': 0,
            'proximity_frames': 0,
            'proximity_percentage': 0.0,
            'avg_proximity_score': 0.0
        }
    
    # Count close proximity frames (proximity >= 0.7, NaN when no face detected)
    if 'proximity' in segment_frames.columns:
        close_proximity_frames = segment_frames[segment_frames['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD]
        avg_proximity = segment_frames['proximity'].mean()  # Will handle NaN automatically
    else:
        close_proximity_frames = pd.DataFrame()
        avg_proximity = 0.0
    
    return {
        'total_frames': len(segment_frames),
        'proximity_frames': len(close_proximity_frames),
        'proximity_percentage': len(close_proximity_frames) / len(segment_frames) if len(segment_frames) > 0 else 0.0,
        'avg_proximity_score': avg_proximity
    }

def analyze_person_detection(segment_row, frames_df):
    """
    Analyze person detection characteristics within a segment.
    
    Parameters:
    ----------
    segment_row : pd.Series
        Row from segments DataFrame with segment boundaries
    frames_df : pd.DataFrame
        DataFrame with frame-level person detection data
        
    Returns:
    -------
    dict
        Dictionary with person detection metrics
    """
    # Convert segment times to frame numbers (assuming 30 fps)
    start_frame = int(segment_row['segment_start_time'] * 30)
    end_frame = int(segment_row['segment_end_time'] * 30)
    
    # Filter frames to this segment
    segment_frames = frames_df[
        (frames_df['child_id'] == segment_row['child_id']) &
        (frames_df['frame_number'] >= start_frame) &
        (frames_df['frame_number'] <= end_frame)
    ]
    
    if len(segment_frames) == 0:
        return {
            'total_frames': 0,
            'child_detected_frames': 0,
            'other_person_frames': 0,
            'both_present_frames': 0,
            'child_detection_percentage': 0.0,
            'other_detection_percentage': 0.0,
            'co_presence_percentage': 0.0
        }
    
    # Analyze person detection patterns
    child_detected = segment_frames[segment_frames['child_present'] == True] if 'child_present' in segment_frames.columns else pd.DataFrame()
    adult_detected = segment_frames[segment_frames['adult_present'] == True] if 'adult_present' in segment_frames.columns else pd.DataFrame()
    both_present = segment_frames[
        (segment_frames['child_present'] == True) & 
        (segment_frames['adult_present'] == True)
    ] if all(col in segment_frames.columns for col in ['child_present', 'adult_present']) else pd.DataFrame()

    return {
        'total_frames': len(segment_frames),
        'child_detected_frames': len(child_detected),
        'adult_detected_frames': len(adult_detected),
        'both_present_frames': len(both_present),
        'child_detection_percentage': len(child_detected) / len(segment_frames) if len(segment_frames) > 0 else 0.0,
        'adult_detection_percentage': len(adult_detected) / len(segment_frames) if len(segment_frames) > 0 else 0.0,
        'both_presence_percentage': len(both_present) / len(segment_frames) if len(segment_frames) > 0 else 0.0
    }

def load_vocalizations_data():
    """Load all vocalizations from database."""
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    vocalizations_query = """
    SELECT v.vocalization_id, v.video_id, vid.video_name, v.start_time_seconds, v.end_time_seconds, v.speaker
    FROM Vocalizations v
    JOIN Videos vid ON v.video_id = vid.video_id
    """
    vocs_df = pd.read_sql_query(vocalizations_query, conn)
    conn.close()
    
    # Add child_id
    vocs_df['child_id'] = vocs_df['video_name'].apply(extract_child_id)
    return vocs_df

def main():
    """
    Main function to analyze interaction segment composition.
    
    Research Question 6: Which aspects contribute to detected interactions?
    
    This script analyzes:
    1. Audio turn-taking patterns within segments
    2. Proximity characteristics 
    3. Person detection patterns
    4. Correlates these with segment interaction types
    """
    print("🔍 RESEARCH QUESTION 6: INTERACTION COMPOSITION ANALYSIS")
    print("=" * 70)
    print("Analyzing what characteristics contribute to interaction detection...")
    
    # Step 1: Load interaction segments and filter for "Interacting" only
    print("\n📊 Step 1: Loading interaction segments and filtering for interactions...")
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    # rename "category" to "interaction type"
    # Filter for ONLY "Interacting" segments
    interaction_segments = segments_df[segments_df['interaction_type'] == 'Interacting'].copy()
    
    print(f"🎯 Interaction segments to analyze: {len(interaction_segments)}")
    
    # Step 2: Load supporting data
    print("\n📊 Step 2: Loading vocalizations and frame-level data...")
    vocalizations_df = load_vocalizations_data()
    frames_df = pd.read_csv(Inference.FRAME_LEVEL_INTERACTIONS_CSV)
    
    # Add child_id to frames_df (convert video_id to str first)
    frames_df['child_id'] = frames_df['video_id'].astype(str).apply(extract_child_id)
    
    # Step 3: Analyze each interaction segment individually
    print("\n🔍 Step 3: Analyzing individual interaction segment composition...")
    segment_analysis = []

    for _, segment in interaction_segments.iterrows():

        # Analyze audio turn-taking
        audio_metrics = analyze_audio_turn_taking(segment, vocalizations_df)
        
        # Analyze proximity
        proximity_metrics = analyze_proximity_frames(segment, frames_df) if not frames_df.empty else {}
        
        # Analyze person detection (if frame data available)  
        person_metrics = analyze_person_detection(segment, frames_df) if not frames_df.empty else {}
        
        # Combine all metrics for this individual segment
        analysis_row = {
            'child_id': segment['child_id'],
            'age_at_recording': segment.get('age_at_recording', None), 
            'interaction_type': segment['interaction_type'],
            'segment_start_time': segment['segment_start_time'],
            'segment_end_time': segment['segment_end_time'],
            'segment_duration': segment['segment_end_time'] - segment['segment_start_time'],
            **audio_metrics,
            **proximity_metrics,
            **person_metrics
        }
        
        segment_analysis.append(analysis_row)
    
    analysis_df = pd.DataFrame(segment_analysis)
    
    # Sort by child and segment start time for easy review
    analysis_df = analysis_df.sort_values(['child_id', 'segment_start_time']).reset_index(drop=True)

    # Step 4: Save detailed individual segment results
    print(f"\n💾 Step 4: Saving individual segment analysis...")
    analysis_df.to_csv(Inference.INTERACTION_COMPOSITION_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Individual segment analysis saved to: {Inference.INTERACTION_COMPOSITION_CSV}")
    print(f"📊 Each row represents one interaction segment with detailed characteristics")
    print("=" * 70)
    
    return analysis_df

if __name__ == "__main__":
    main()


