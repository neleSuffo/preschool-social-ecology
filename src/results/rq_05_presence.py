import re
import pandas as pd
from pathlib import Path
import sys

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Inference

def enhance_segments_with_presence(segments_df, frame_data_df):
    """
    Enhance interaction segments with adult/child presence information.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        Existing interaction segments data
    frame_data_df : pd.DataFrame
        Frame-level data with presence information
        
    Returns
    -------
    pd.DataFrame
        Enhanced segments with adult_present and child_present flags
    """
    enhanced_segments = segments_df.copy()
    enhanced_segments['adult_present'] = 0
    enhanced_segments['child_present'] = 0
    
    print(f"🔄 Processing {len(enhanced_segments)} segments...")
    
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
            # Check if any frame in this segment has adult or child present
            has_adult = video_frames['adult_present'].sum() > 0
            has_child = video_frames['child_present'].sum() > 0
            
            enhanced_segments.loc[idx, 'adult_present'] = 1 if has_adult else 0
            enhanced_segments.loc[idx, 'child_present'] = 1 if has_child else 0
    
    return enhanced_segments

def main(frame_data_csv: Path = Inference.FRAME_LEVEL_INTERACTIONS_CSV,
        segments_csv: Path = Inference.INTERACTION_SEGMENTS_CSV):
    print("=" * 60)
    print("RQ 05: ENHANCING SEGMENTS WITH PRESENCE INFORMATION")
    print("=" * 60)
    
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
        print(f"� Loaded {len(segments_df):,} existing segments")
    except FileNotFoundError:
        print(f"❌ Error: Segments file not found at {segments_csv}")
        return
    except Exception as e:
        print(f"❌ Error loading segments: {e}")
        return
    
    # Enhance segments with presence information
    enhanced_segments = enhance_segments_with_presence(segments_df, frame_df)
    
    filtered_segments = enhanced_segments[enhanced_segments['interaction_type'] != 'Alone']
    print(f"✅ Enhanced {len(filtered_segments)} segments with presence information")
    
    # Save enhanced segments
    output_file = Inference.BASE_OUTPUT_DIR / 'enhanced_interaction_segments.csv'
    try:
        filtered_segments.to_csv(output_file, index=False)
        print(f"\n💾 Enhanced segments saved to: {output_file}")        
    except Exception as e:
        print(f"❌ Error saving enhanced segments: {e}")
        return
    
if __name__ == "__main__":
    main()