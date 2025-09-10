import pandas as pd
from constants import Inference
from config import DataConfig

segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
frames_df = pd.read_csv(Inference.FRAME_LEVEL_INTERACTIONS_CSV)

# Add is_interaction column based on segment mapping
def add_interaction_column(frames_df, segments_df):
    """
    Add 'is_interaction' column to frames_df based on whether the frame
    falls within an 'Interacting' segment from segments_df.
    """
    # Initialize all frames as non-interaction
    frames_df['is_interaction'] = False
    
    # Filter for only 'Interacting' segments
    interaction_segments = segments_df[segments_df['interaction_type'] == 'Interacting']
    
    # For each interaction segment, mark corresponding frames
    for _, segment in interaction_segments.iterrows():       
        # Mark frames in this video and time range as interaction
        mask = (
            (frames_df['video_id'] == segment['video_id']) &
            (frames_df['frame_number'] >= segment['segment_start']) &
            (frames_df['frame_number'] <= segment['segment_end'])
        )
        frames_df.loc[mask, 'is_interaction'] = True
    
    return frames_df

# Apply the function
frames_df = add_interaction_column(frames_df, segments_df)

print(f"Added 'is_interaction' column to frames_df")
print(f"Total frames: {len(frames_df)}")
print(f"Interaction frames: {frames_df['is_interaction'].sum()}")
print(f"Non-interaction frames: {(~frames_df['is_interaction']).sum()}")

# Option 3: Fill NaN with a special value outside normal range (e.g., -1)
frames_df['proximity_filled'] = frames_df['proximity'].fillna(-1)

# Save the updated frames_df with the new is_interaction column
output_path = Inference.INTERACTION_COMPOSITION_FRAMES_CSV
frames_df.to_csv(output_path, index=False)
print(f"Saved updated frames_df to: {output_path}")