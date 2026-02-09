import pandas as pd
from pathlib import Path
from constants import Inference

def enhance_segments_with_presence(segments_df, frame_data_df):
    """
    Enhance interaction segments with presence information using person_or_face_present.
    """
    enhanced_segments = segments_df.copy()
    # Create a single column for general presence
    enhanced_segments['others_present'] = 0
        
    for idx, segment in enhanced_segments.iterrows():
        video_name = segment['video_name']
        segment_start = segment['segment_start']
        segment_end = segment['segment_end']
        
        # Get frame data for this video and segment
        video_frames = frame_data_df[
            (frame_data_df['video_name'] == video_name) &
            (frame_data_df['frame_number'] >= segment_start) &
            (frame_data_df['frame_number'] <= segment_end)
        ]
        
        if len(video_frames) > 0:
            # Check if person_or_face_present is True in any frame of the segment
            if video_frames['person_or_face_present'].any():
                enhanced_segments.loc[idx, 'others_present'] = 1
    
    return enhanced_segments

def main(frame_data_csv: Path = Inference.FRAME_LEVEL_INTERACTIONS_CSV,
        segments_csv: Path = Inference.INTERACTION_SEGMENTS_CSV):
    print("🗣️ RQ 05: PRESENCE ANALYSIS")
    print("=" * 70)
    print("ANALYZING PRESENCE OF ADULTS AND CHILDREN IN INTERACTION SEGMENTS")
    
    # Load frame-level data
    try:
        frame_df = pd.read_csv(frame_data_csv)
    except FileNotFoundError:
        print(f"❌ Error: Frame data file not found at {frame_data_csv}")
        return
    except Exception as e:
        print(f"❌ Error loading frame data: {e}")
        return
    
    # Load existing segments
    try:
        segments_df = pd.read_csv(segments_csv)
    except FileNotFoundError:
        print(f"❌ Error: Segments file not found at {segments_csv}")
        return
    except Exception as e:
        print(f"❌ Error loading segments: {e}")
        return
    
    # Enhance segments with presence information
    enhanced_segments = enhance_segments_with_presence(segments_df, frame_df)
    
    # Keep segments where child is NOT alone
    filtered_segments = enhanced_segments[enhanced_segments['interaction_type'] != 'Alone']
    
    # Save output
    try:
        filtered_segments.to_csv(Inference.PRESENCE_CSV, index=False)
        print(f"\n💾 Output saved to: {Inference.PRESENCE_CSV}") 
    except Exception as e:
        print(f"❌ Error saving enhanced segments: {e}")
    
if __name__ == "__main__":
    main()