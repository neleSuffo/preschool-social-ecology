import pandas as pd
from constants import Inference
from config import DataConfig

def add_interaction_columns(frames_df, segments_df):
    """
    Add 'is_interaction', 'is_alone', and 'is_available' columns to frames_df
    based on whether the frame falls within the respective segment type.
    """
    # Initialize all frames as False
    frames_df['is_interaction'] = False
    frames_df['is_alone'] = False
    frames_df['is_available'] = False
    
    # Map interaction_type -> column name
    type_to_col = {
        'Interacting': 'is_interaction',
        'Alone': 'is_alone',
        'Available': 'is_available'
    }
    
    # For each segment type, mark corresponding frames
    for _, segment in segments_df.iterrows():
        col = type_to_col.get(segment['interaction_type'])
        if col is None:
            continue  # skip unknown interaction types
        
        mask = (
            (frames_df['video_id'] == segment['video_id']) &
            (frames_df['frame_number'] >= segment['segment_start']) &
            (frames_df['frame_number'] <= segment['segment_end'])
        )
        frames_df.loc[mask, col] = True
    
    return frames_df


def main():
    """
    Main function to add interaction columns to frames and save results.
    
    This script:
    1. Loads interaction segments and frame-level data
    2. Adds boolean interaction columns to frames
    3. Fills missing proximity values
    4. Saves updated frames DataFrame
    """
    print("🔍 INTERACTION FRAMES PROCESSING")
    print("=" * 70)

    # Step 1: Load data
    segments_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    frames_df = pd.read_csv(Inference.FRAME_LEVEL_INTERACTIONS_CSV)

    # Step 2: Add interaction columns
    frames_df = add_interaction_columns(frames_df, segments_df)
    print(f"Interaction frames: {frames_df['is_interaction'].sum()}")
    print(f"Alone frames: {frames_df['is_alone'].sum()}")
    print(f"Available frames: {frames_df['is_available'].sum()}")

    # Step 3: Fill missing proximity values
    frames_df['proximity_filled'] = frames_df['proximity'].fillna(-1)

    # Step 4: Save results
    output_path = Inference.INTERACTION_COMPOSITION_CSV
    frames_df.to_csv(output_path, index=False)

    print(f"\n✅ Processing completed successfully!")
    print(f"📄 Output saved to: {output_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()