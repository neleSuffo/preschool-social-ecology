import sqlite3
import pandas as pd
import re
from pathlib import Path
from constants import DataPaths, Inference
from config import InferenceConfig

def extract_child_id(video_name):
    """
    Extracts the 6-digit child ID from a video name string.
    Example: 'id123456_video.mp4' -> '123456'
    """
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def row_wise_mapping(voc_row, video_segments_df):
    """
    Map a single vocalization to overlapping interaction segment in a video file.
    
    Parameters:
    ----------
    voc_row : pd.Series
        A row from the vocalizations DataFrame with columns:
        ['vocalization_id', 'video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds']
    video_segments_df : pd.DataFrame
        DataFrame containing interaction segments for one video_id with columns:
        ['start_time_sec', 'end_time_sec', 'category']  
    
    Returns:
    -------
    list of dict
        List of dictionaries with mapped vocalizations including:
        ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'interaction_type', 'segment_start_time', 'segment_end_time']
    """
    # Find overlapping segments (already filtered to same video)
    overlaps = video_segments_df[
        (video_segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (video_segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ]
    results = []
    
    # Iterate through each overlapping segment to calculate time-based overlap
    for _, seg in overlaps.iterrows():
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        overlap_end = min(voc_row['end_time_seconds'], seg['end_time_sec'])
        overlap_seconds = max(0, overlap_end - overlap_start)
        
        # Total segment duration should be the segment's own length
        total_segment_duration = seg['end_time_sec'] - seg['start_time_sec']
        
        results.append({
            'video_id': voc_row['video_id'],
            'child_id': voc_row['child_id'],
            'age_at_recording': voc_row['age_at_recording'],
            'start_time_seconds': voc_row['start_time_seconds'],
            'end_time_seconds': voc_row['end_time_seconds'],
            'seconds': overlap_seconds,
            'interaction_type': seg['category'],
            'total_segment_duration': total_segment_duration,
            'segment_start_time': seg['start_time_sec'],
            'segment_end_time': seg['end_time_sec']
        })
    return results

def map_vocalizations_to_segments(db_path: Path = DataPaths.INFERENCE_DB_PATH, segments_csv_path: Path = Inference.INTERACTION_SEGMENTS_CSV):
    """
    Retrieve OTHER (non-KCHI) vocalizations from the SQLite database and map them to interaction segments.
    The mapped_vocalizations DataFrame contains the following columns:
    ['video_id', 'child_id', 'age_at_recording', 'start_time_seconds', 'end_time_seconds', 'seconds', 'interaction_type']

    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'category']
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with mapped vocalizations
    """
    # Load segments DataFrame
    segments_df = pd.read_csv(segments_csv_path)
    
    # Load age data
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)

    # Connect to the SQLite database and query OTHER vocalizations with video names
    conn = sqlite3.connect(db_path)
    # The 'words' column is no longer needed
    vocalizations_query = """
    SELECT v.vocalization_id, v.video_id, vid.video_name, v.start_time_seconds, v.end_time_seconds
    FROM Vocalizations v
    JOIN Videos vid ON v.video_id = vid.video_id
    WHERE v.speaker != 'KCHI'
    """
    other_vocs = pd.read_sql_query(vocalizations_query, conn)
    conn.close()
    
    # Extract child_id from video_name
    other_vocs['child_id'] = other_vocs['video_name'].apply(extract_child_id)
    
    # Merge with age data to get age at recording
    other_vocs = other_vocs.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')
    
    # Check video alignment
    voc_videos = set(other_vocs['video_id'].unique())
    seg_videos = set(segments_df['video_id'].unique())
    common_videos = voc_videos & seg_videos    
    mapped_rows = []
    
    # Process each video separately
    for video_id in common_videos:       
        # Filter data to this specific video
        video_vocalizations = other_vocs[other_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]

        # Map each vocalization in this video to segments in this video
        for _, voc_row in video_vocalizations.iterrows():
            mapped_rows.extend(row_wise_mapping(voc_row, video_segments))
    
    return pd.DataFrame(mapped_rows)

def main():
    """
    Main function to analyze other speech exposure by interaction context.
    
    Research Question 4: How much speech is the key child exposed to (from others)?
    
    This script:
    1. Extracts OTHER (non-KCHI) vocalizations from the database
    2. Maps them to interaction segments (Alone, Co-present, Interacting)  
    3. Outputs mapped vocalization data with overlap durations and speech exposure percentages
    4. Saves results to CSV
    """   
    print("🗣️ RESEARCH QUESTION 4: OTHER SPEECH EXPOSURE ANALYSIS")
    print("=" * 70)
    print("Analyzing other speaker vocalizations (speech exposure) by interaction context...")
    
    # Step 1: Map vocalizations to interaction segments
    print("\n🔄 Step 1: Mapping other vocalizations to interaction segments...")
    mapped_vocalizations = map_vocalizations_to_segments()

    if mapped_vocalizations.empty:
        print("\n⚠️ No vocalizations found. Exiting analysis.")
        return
        
    # Step 2: Create segment-level aggregation
    print("\n📊 Step 2: Aggregating speech time by segment...")
    segment_totals = mapped_vocalizations.groupby([
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time'
    ]).agg(
        total_overlap_seconds=('seconds', 'sum'),
        total_segment_duration=('total_segment_duration', 'max')
    ).reset_index()
    
    # Step 3: Calculate the percentage
    print("\n✅ Step 3: Calculating time-based exposure percentage...")
    segment_totals['other_speech_minutes'] = segment_totals['total_overlap_seconds'] / 60
    segment_totals['segment_duration_minutes'] = segment_totals['total_segment_duration'] / 60
    
    # The exposure percentage is now a true time-based ratio
    segment_totals['speech_exposure_percent'] = (
        segment_totals['total_overlap_seconds'] / segment_totals['total_segment_duration']
    ).fillna(0)
    
    # Sort and save
    segment_totals = segment_totals.sort_values([
        'child_id', 'age_at_recording', 'segment_start_time'
    ]).reset_index(drop=True)

    # Select final columns for output
    final_output = segment_totals[[
        'child_id', 'age_at_recording', 'interaction_type', 
        'segment_start_time', 'segment_end_time', 
        'other_speech_minutes', 'segment_duration_minutes', 'speech_exposure_percent'
    ]].copy()
    
    # Step 4: Save results
    print(f"\n💾 Step 4: Saving segment totals...")
    
    # Save aggregated segment totals
    final_output.to_csv(Inference.CDS_SUMMARY_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Other speech exposure segment totals saved to: {Inference.CDS_SUMMARY_CSV}")
    print(f"📊 Total unique segments: {len(final_output)}")
    print("=" * 70)
    
    return final_output

if __name__ == "__main__":
    main()