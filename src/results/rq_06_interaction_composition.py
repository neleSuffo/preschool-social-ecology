# Research Question: 6. Which aspects contribute to detected interactions?

import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig
from utils import extract_child_id, merge_overlapping_vocalizations

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
    segment_duration = segment_row['end_time_sec'] - segment_row['start_time_sec']
    # Filter vocalizations to this segment - only check if vocalization starts within segment
    segment_vocs = vocalizations_df[
        (vocalizations_df['video_id'] == segment_row['video_id']) &
        (vocalizations_df['start_time_seconds'] >= segment_row['start_time_sec']) &
        (vocalizations_df['start_time_seconds'] <= segment_row['end_time_sec'])
    ].sort_values('start_time_seconds').reset_index(drop=True)

    # Calculate durations, clipping to segment end
    segment_vocs['clipped_end_time'] = segment_vocs['end_time_seconds'].clip(upper=segment_row['end_time_sec'])
    segment_vocs['vocalization_duration'] = segment_vocs['clipped_end_time'] - segment_vocs['start_time_seconds']
    
    # Calculate total durations by speaker
    kchi_vocs = segment_vocs[segment_vocs['speaker'] == 'KCHI']
    other_vocs = segment_vocs[segment_vocs['speaker'] != 'KCHI']
    
    kchi_total_duration = kchi_vocs['vocalization_duration'].sum() if len(kchi_vocs) > 0 else 0.0
    other_total_duration = other_vocs['vocalization_duration'].sum() if len(other_vocs) > 0 else 0.0
    total_vocalization_duration = segment_vocs['vocalization_duration'].sum()
    
    # Add percentage of kchi, other and total vocalizations relative to segment duration
    kchi_percentage = kchi_total_duration / segment_duration if segment_duration > 0 else 0.0
    other_percentage = other_total_duration / segment_duration if segment_duration > 0 else 0.0
    total_percentage = total_vocalization_duration / segment_duration if segment_duration > 0 else 0.0

    if len(segment_vocs) < 2:
        return {
            'has_turn_taking': False,
            'turn_count': 0,
            'kchi_vocalizations': len(kchi_vocs),
            'other_vocalizations': len(other_vocs),
            'total_vocalizations': len(segment_vocs),
            'kchi_voc_duration_seconds': kchi_total_duration,
            'other_voc_duration_seconds': other_total_duration,
            'total_voc_duration_seconds': total_vocalization_duration,
            'kchi_voc_percentage': kchi_percentage,
            'other_voc_percentage': other_percentage,
            'total_voc_percentage': total_percentage
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
            if gap <= InferenceConfig.GAP_MERGE_DURATION_SEC:  # Valid turn
                turn_count += 1
    
    return {
        'has_turn_taking': turn_count > 0,
        'turn_count': turn_count,
        'kchi_vocalizations': len(kchi_vocs),
        'other_vocalizations': len(other_vocs),
        'total_vocalizations': len(segment_vocs),
        'kchi_voc_duration_seconds': kchi_total_duration,
        'other_voc_duration_seconds': other_total_duration,
        'total_voc_duration_seconds': total_vocalization_duration,
        'kchi_voc_percentage': kchi_percentage,
        'other_voc_percentage': other_percentage,
        'total_voc_percentage': total_percentage
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
    start_frame = int(segment_row['start_time_sec'] * DataConfig.FPS)
    end_frame = int(segment_row['end_time_sec'] * DataConfig.FPS)

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
    start_frame = int(segment_row['start_time_sec'] * DataConfig.FPS)
    end_frame = int(segment_row['end_time_sec'] * DataConfig.FPS)

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
    """Load all vocalizations from database and merge overlapping same-speaker vocalizations."""
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
    
    # Merge overlapping vocalizations from same speaker
    vocs_df = merge_overlapping_vocalizations(vocs_df)
    
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
    print("\n📊 Step 1: Loading interaction segments, frame-level data and vocalizations and filtering for interactions...")
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    frames_df = pd.read_csv(Inference.FRAME_LEVEL_INTERACTIONS_CSV)
    vocalizations_df = load_vocalizations_data()

    # Filter for ONLY "Interacting" segments
    interaction_segments = segments_df[segments_df['interaction_type'] == 'Interacting'].copy()

    # Step 2: Analyze each interaction segment individually
    print("\n🔍 Step 2: Analyzing individual interaction segment composition...")
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
            'video_name': segment['video_name'],
            'child_id': segment['child_id'],
            'age_at_recording': segment.get('age_at_recording', None), 
            'interaction_type': segment['interaction_type'],
            'segment_start_time': segment['start_time_sec'],
            'segment_end_time': segment['end_time_sec'],
            'segment_duration': segment['end_time_sec'] - segment['start_time_sec'],
            **audio_metrics,
            **proximity_metrics,
            **person_metrics
        }
        
        segment_analysis.append(analysis_row)
    
    analysis_df = pd.DataFrame(segment_analysis)
    
    # Sort by child and segment start time for easy review
    analysis_df = analysis_df.sort_values(['video_name', 'segment_start_time']).reset_index(drop=True)

    # Step 3: Save detailed individual segment results
    print(f"\n💾 Step 3: Saving individual segment analysis...")
    analysis_df.to_csv(Inference.INTERACTION_COMPOSITION_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Individual segment analysis saved to: {Inference.INTERACTION_COMPOSITION_CSV}")
    print("=" * 70)
    
    return analysis_df

if __name__ == "__main__":
    main()


