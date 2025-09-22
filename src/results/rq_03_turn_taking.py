import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths, Inference
from utils import extract_child_id

def count_turns_in_segments(vocalizations_df, segments_df, max_gap=5.0):
    """
    Count conversational turns (KCHI <-> CDS) per segment by iterating through
    individual vocalizations. This focuses only on key child and child-directed speech.

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

            # A turn is an alternation between KCHI and CDS only
            # Skip if either utterance is OHS (overheard speech)
            if curr['speaker'] == 'OHS' or nxt['speaker'] == 'OHS':
                continue
                
            # Only count turns when there's alternation between KCHI and CDS
            if curr['speaker'] != nxt['speaker']:
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
    Retrieves ALL audio classifications from the SQLite database and maps them to interaction segments.
    
    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['video_id', 'start_time_sec', 'end_time_sec', 'interaction_type']
    
    Returns
    -------
    pd.DataFrame
        DataFrame with audio classifications mapped to segments, including:
        ['video_id', 'child_id', 'age_at_recording', 'speaker', 'start_time_seconds', 'end_time_seconds',
         'seconds', 'words', 'interaction_type', 'segment_start_time', 'segment_end_time']
    """
    segments_df = pd.read_csv(segments_csv_path)
    age_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
    
    conn = sqlite3.connect(db_path)
    # Query audio classifications with binary speaker columns
    audio_query = """
    SELECT ac.video_id, vid.video_name, ac.frame_number, ac.has_kchi, ac.has_cds, ac.has_ohs
    FROM AudioClassifications ac
    JOIN Videos vid ON ac.video_id = vid.video_id
    WHERE ac.has_kchi = 1 OR ac.has_cds = 1 OR ac.has_ohs = 1
    ORDER BY ac.video_id, ac.frame_number
    """
    audio_frames = pd.read_sql_query(audio_query, conn)
    conn.close()
    
    if len(audio_frames) == 0:
        print("⚠️ Warning: No speech frames found in audio classifications")
        return pd.DataFrame()
    
    # Convert frame numbers to time intervals
    from config import DataConfig
    audio_frames['start_time_seconds'] = audio_frames['frame_number'] / DataConfig.FPS
    audio_frames['end_time_seconds'] = (audio_frames['frame_number'] + 1) / DataConfig.FPS
    
    # Extract child_id from video_name
    audio_frames['child_id'] = audio_frames['video_name'].apply(extract_child_id)
    
    # Merge with age data
    audio_frames = audio_frames.merge(age_df[['video_name', 'age_at_recording']], on='video_name', how='left')
    
    # Convert binary speaker columns to individual records
    # Create one record per active speaker per frame
    all_vocs = []
    
    for _, frame_row in audio_frames.iterrows():
        base_record = {
            'video_id': frame_row['video_id'],
            'child_id': frame_row['child_id'],
            'age_at_recording': frame_row['age_at_recording'],
            'start_time_seconds': frame_row['start_time_seconds'],
            'end_time_seconds': frame_row['end_time_seconds'],
            'words': 1.0  # Each frame represents a unit of speech activity
        }
        
        # Create records for each active speaker type
        if frame_row['has_kchi'] == 1:
            record = base_record.copy()
            record['speaker'] = 'KCHI'
            all_vocs.append(record)
            
        if frame_row['has_cds'] == 1:
            record = base_record.copy()
            record['speaker'] = 'CDS'
            all_vocs.append(record)
            
        if frame_row['has_ohs'] == 1:
            record = base_record.copy()
            record['speaker'] = 'OHS'
            all_vocs.append(record)
    
    all_vocs = pd.DataFrame(all_vocs)
    
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
                    'interaction_type': seg['interaction_type'],
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
    mapped_vocalizations = map_vocalizations_to_segments(
        db_path=DataPaths.INFERENCE_DB_PATH, 
        segments_csv_path=Inference.INTERACTION_SEGMENTS_CSV
    )

    if mapped_vocalizations.empty:
        print("🛑 No vocalizations found. Analysis cannot proceed.")
        return

    # Step 2: Aggregate speech metrics by segment and speaker type
    mapped_vocalizations['segment_duration_minutes'] = (mapped_vocalizations['segment_end_time'] - mapped_vocalizations['segment_start_time']) / 60
    mapped_vocalizations['speech_minutes'] = mapped_vocalizations['seconds'] / 60
    
    # Group by segment and speaker type to get total speech duration
    segment_speech_summary = mapped_vocalizations.groupby(
        ['child_id', 'age_at_recording', 'interaction_type', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes', 'speaker']
    ).agg(
        total_speech_minutes=('speech_minutes', 'sum'),
        total_words=('words', 'sum')
    ).reset_index()

    # Pivot the table to get separate columns for KCHI, CDS, and OHS
    speech_pivot = segment_speech_summary.pivot_table(
        index=['child_id', 'age_at_recording', 'interaction_type', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes'],
        columns='speaker',
        values=['total_speech_minutes', 'total_words'],
        fill_value=0
    ).reset_index()

    # Flatten column names and handle what we actually have
    speech_pivot.columns = [f"{col[1].lower()}_{col[0]}" if col[1] else col[0] for col in speech_pivot.columns]
    
    # Create default columns for missing speaker types
    expected_cols = [
        'kchi_total_speech_minutes', 'cds_total_speech_minutes', 'ohs_total_speech_minutes',
        'kchi_total_words', 'cds_total_words', 'ohs_total_words'
    ]
    
    for col in expected_cols:
        if col not in speech_pivot.columns:
            speech_pivot[col] = 0
    
    # Calculate per-minute metrics for each speaker type
    speech_pivot['kchi_speech_per_minute'] = (speech_pivot['kchi_total_speech_minutes'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['cds_speech_per_minute'] = (speech_pivot['cds_total_speech_minutes'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['ohs_speech_per_minute'] = (speech_pivot['ohs_total_speech_minutes'] / speech_pivot['segment_duration_minutes']).fillna(0)
    
    speech_pivot['kchi_words_per_minute'] = (speech_pivot['kchi_total_words'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['cds_words_per_minute'] = (speech_pivot['cds_total_words'] / speech_pivot['segment_duration_minutes']).fillna(0)
    speech_pivot['ohs_words_per_minute'] = (speech_pivot['ohs_total_words'] / speech_pivot['segment_duration_minutes']).fillna(0)
    
    # Step 3: Count KCHI-CDS turns for each segment using the detailed vocalization data    
    # Prepare the segments DataFrame with all required columns for turn counting
    segments_for_turns = speech_pivot[['child_id', 'segment_start_time', 'segment_end_time', 'segment_duration_minutes']].drop_duplicates()
    
    # The `count_turns_in_segments` function focuses on KCHI-CDS alternations only
    # and excludes overheard speech (OHS) from turn-taking analysis.
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
    final_df.to_csv(Inference.TURN_TAKING_CSV, index=False)
    
    print(f"\n✅ Analysis completed successfully!")
    print(f"📄 Output saved to: {Inference.TURN_TAKING_CSV}")
    print("=" * 70)
    
    return final_df

if __name__ == "__main__":
    main()