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

def count_turns_in_segments(vocalizations_df, segments_df, max_gap=5.0):
    """
    Count conversational turns (KCHI <-> Other) per segment by iterating through
    individual vocalizations. This is the corrected, non-aggregated approach.

    Parameters
    ----------
    vocalizations_df : pd.DataFrame
        Detailed DataFrame with individual vocalizations:
        ['child_id', 'segment_start_time', 'speaker', 'start_time_seconds', 'end_time_seconds']
    segments_df : pd.DataFrame
        DataFrame with interaction segments:
        ['child_id', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes']
    max_gap : float
        Maximum allowed gap (seconds) between turns (default=5.0).

    Returns
    -------
    pd.DataFrame
        Segment-level DataFrame with additional columns:
        ['turn_count', 'turns_per_minute']
    """
    turn_results = []
    
    # Iterate through each unique segment to process its vocalizations
    for _, seg in segments_df.iterrows():
        child_id = seg['child_id']
        seg_start = seg['segment_start_time']
        seg_end = seg['segment_end_time']
        
        # Filter vocalizations to the current segment and sort them chronologically
        segment_vocs = vocalizations_df[
            (vocalizations_df['child_id'] == child_id) &
            (vocalizations_df['segment_start_time'] == seg_start) &
            (vocalizations_df['segment_end_time'] == seg_end)
        ].sort_values('start_time_seconds').reset_index(drop=True)
        
        turn_count = 0
        # Iterate through consecutive utterances within the sorted segment data
        for i in range(len(segment_vocs) - 1):
            curr = segment_vocs.iloc[i]
            nxt = segment_vocs.iloc[i + 1]

            # A turn is an alternation between KCHI and OTHER speakers
            speaker_pair = (curr['speaker'], nxt['speaker'])
            if speaker_pair not in [('KCHI', 'FEM_MAL'), ('FEM_MAL', 'KCHI')]:
                continue
                
            # Check the gap between utterances - handle overlaps by considering them as valid turns
            gap = nxt['start_time_seconds'] - curr['end_time_seconds']
            
            # Count as a turn if:
            # 1. There's no gap or a small gap (0 <= gap <= max_gap), OR
            # 2. There's overlap (gap < 0) - overlapping speech indicates turn-taking
            if gap <= max_gap:
                turn_count += 1
        
        # Store the result for the current segment
        turn_results.append({
            'child_id': child_id,
            'segment_start_time': seg_start,
            'segment_end_time': seg_end,
            'turn_count': turn_count
        })

    turn_df = pd.DataFrame(turn_results)
    
    # Merge turn counts with segment duration to calculate turns per minute
    segments_with_turns = segments_df.merge(
        turn_df,
        on=['child_id', 'segment_start_time', 'segment_end_time'],
        how='left'
    ).fillna({'turn_count': 0})

    segments_with_turns['turns_per_minute'] = (
        segments_with_turns['turn_count'] / segments_with_turns['segment_duration_minutes']
    ).replace([float("inf"), -float("inf")], 0).fillna(0)

    return segments_with_turns

def map_vocalizations_to_segments(db_path: Path, segments_csv_path: Path):
    """
    Retrieves ALL vocalizations from the SQLite database and maps them to interaction segments.
    """
    segments_df = pd.read_csv(segments_csv_path)
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
    
    conn = sqlite3.connect(db_path)
    all_vocalizations_query = """
    SELECT v.vocalization_id, v.video_id, vid.video_name, v.start_time_seconds, v.end_time_seconds, v.words, v.speaker
    FROM Vocalizations v
    JOIN Videos vid ON v.video_id = vid.video_id
    """
    all_vocs = pd.read_sql_query(all_vocalizations_query, conn)
    conn.close()
    
    all_vocs['child_id'] = all_vocs['video_name'].apply(extract_child_id)
    all_vocs = all_vocs.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')
    
    mapped_rows = []
    
    common_videos = set(all_vocs['video_id'].unique()) & set(segments_df['video_id'].unique())
    
    for video_id in common_videos:
        video_vocalizations = all_vocs[all_vocs['video_id'] == video_id]
        video_segments = segments_df[segments_df['video_id'] == video_id]
        
        for _, voc_row in video_vocalizations.iterrows():
            voc_start, voc_end = voc_row['start_time_seconds'], voc_row['end_time_seconds']
            
            overlaps = video_segments[
                (video_segments['start_time_sec'] <= voc_end) &
                (video_segments['end_time_sec'] >= voc_start)
            ].copy()

            if overlaps.empty:
                continue

            total_voc_duration = voc_end - voc_start
            
            for _, seg in overlaps.iterrows():
                overlap_start = max(voc_start, seg['start_time_sec'])
                overlap_end = min(voc_end, seg['end_time_sec'])
                overlap_seconds = overlap_end - overlap_start
                
                if total_voc_duration > 0:
                    proportion = overlap_seconds / total_voc_duration
                    allocated_words = voc_row['words'] * proportion
                else:
                    allocated_words = 0
                
                mapped_rows.append({
                    'video_id': voc_row['video_id'],
                    'child_id': voc_row['child_id'],
                    'age_at_recording': voc_row['age_at_recording'],
                    'speaker': voc_row['speaker'],
                    'start_time_seconds': voc_row['start_time_seconds'],
                    'end_time_seconds': voc_row['end_time_seconds'],
                    'seconds': overlap_seconds,
                    'words': allocated_words,
                    'interaction_type': seg['category'],
                    'segment_start_time': seg['start_time_sec'],
                    'segment_end_time': seg['end_time_sec']
                })

    return pd.DataFrame(mapped_rows)

def main():
    """
    Main function to analyze speech patterns and turn-taking for all speakers by interaction context.
    """   
    print("🗣️ RESEARCH QUESTION 3: SPEECH PATTERNS & TURN-TAKING ANALYSIS")
    print("=" * 70)
    print("Analyzing all vocalizations and turn-taking by interaction context...")
    
    # Step 1: Map vocalizations to interaction segments
    print("\n🔄 Step 1: Mapping all vocalizations to interaction segments...")
    mapped_vocalizations = map_vocalizations_to_segments(
        db_path=DataPaths.INFERENCE_DB_PATH, 
        segments_csv_path=Inference.INTERACTION_SEGMENTS_CSV
    )

    if mapped_vocalizations.empty:
        print("🛑 No vocalizations found. Analysis cannot proceed.")
        return

    # Step 2: Aggregate speech metrics by segment and speaker type
    print("\n📊 Step 2: Aggregating speech by segment and speaker type...")

    mapped_vocalizations['segment_duration_minutes'] = (mapped_vocalizations['segment_end_time'] - mapped_vocalizations['segment_start_time']) / 60
    mapped_vocalizations['speech_minutes'] = mapped_vocalizations['seconds'] / 60
    
    # Group by segment and speaker type to get total speech duration
    segment_speech_summary = mapped_vocalizations.groupby(
        ['child_id', 'age_at_recording', 'interaction_type', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes', 'speaker']
    ).agg(
        total_speech_minutes=('speech_minutes', 'sum'),
        total_words=('words', 'sum')
    ).reset_index()

    # Pivot the table to get separate columns for KCHI and OTHER
    speech_pivot = segment_speech_summary.pivot_table(
        index=['child_id', 'age_at_recording', 'interaction_type', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes'],
        columns='speaker',
        values=['total_speech_minutes', 'total_words'],
        fill_value=0
    ).reset_index()

    # Flatten column names and debug what we actually have
    speech_pivot.columns = [f"{col[1].lower()}_{col[0]}" if col[1] else col[0] for col in speech_pivot.columns]
    
    # Create default columns for missing speaker types
    expected_cols = [
        'kchi_total_speech_minutes', 'fem_mal_total_speech_minutes',
        'kchi_total_words', 'fem_mal_total_words'
    ]
    
    for col in expected_cols:
        if col not in speech_pivot.columns:
            speech_pivot[col] = 0
    
    # Calculate per-minute metrics using the actual column names
    speech_pivot['kchi_speech_per_minute'] = (speech_pivot['kchi_total_speech_minutes'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['other_speech_per_minute'] = (speech_pivot['fem_mal_total_speech_minutes'] / speech_pivot['segment_duration_minutes']).fillna(0)
    
    speech_pivot['kchi_words_per_minute'] = (speech_pivot['kchi_total_words'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['other_words_per_minute'] = (speech_pivot['fem_mal_total_words'] / speech_pivot['segment_duration_minutes']).fillna(0)
    
    # Step 3: Count turns for each segment using the detailed vocalization data
    print("\n🔄 Step 3: Counting conversational turns using sequential vocalization data...")
    
    # Prepare the segments DataFrame with all required columns for turn counting
    segments_for_turns = speech_pivot[['child_id', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes']].drop_duplicates()
    
    # The `count_turns_in_segments` function now correctly takes the detailed, mapped vocalizations
    # and the list of segments to process.
    segments_with_turns = count_turns_in_segments(
        mapped_vocalizations[['child_id', 'segment_start_time', 'segment_end_time', 'speaker', 'start_time_seconds', 'end_time_seconds']].drop_duplicates(), 
        segments_for_turns, 
        max_gap=5.0
    )
    
    # Merge the turn counts back into the main DataFrame
    final_df = speech_pivot.merge(
        segments_with_turns[['child_id', 'segment_start_time', 'segment_end_time', 'turn_count', 'turns_per_minute']],
        on=['child_id', 'segment_start_time', 'segment_end_time'],
        how='left'
    )
    final_df[['turn_count', 'turns_per_minute']] = final_df[['turn_count', 'turns_per_minute']].fillna(0)
    
    # Step 4: Save the final results
    print(f"\n💾 Step 4: Saving comprehensive speech and turn-taking analysis...")

    final_df.to_csv(Inference.TURN_TAKING_CSV, index=False)
    
    print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"📄 Comprehensive speech analysis saved to: {Inference.TURN_TAKING_CSV}")
    print(f"📊 Total unique segments analyzed: {len(final_df)}")
    print("=" * 70)
    
    return final_df

if __name__ == "__main__":
    main()