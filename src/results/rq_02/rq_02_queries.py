# Research Question: How much language does the key child produce (utterances and words)?

import sqlite3
import pandas as pd
from constants import DataPaths, ResearchQuestions

def row_wise_mapping(voc_row, segments_df):
    """
    Map a single vocalization to overlapping interaction segments.
    
    Parameters:
    ----------
    voc_row : pd.Series
        A row from the vocalizations DataFrame with columns:
        ['vocalization_id', 'video_id', 'start_time_seconds', 'end_time_seconds', 'words']
    segments_df : pd.DataFrame
        DataFrame containing interaction segments with columns:
        ['start_time_sec', 'end_time_sec', 'category']  
    
    Returns:
    -------
    list of dict
        List of dictionaries with mapped vocalizations including:
        ['vocalization_id', 'video_id', 'seconds', 'words', 'interaction_type']
    """
    # Find overlapping segments
    overlaps = segments_df[
        (segments_df['start_time_sec'] <= voc_row['end_time_seconds']) &
        (segments_df['end_time_sec'] >= voc_row['start_time_seconds'])
    ]
    results = []
    # For each overlapping segment, calculate the overlap duration and proportion of words
    for _, seg in overlaps.iterrows():
        overlap_start = max(voc_row['start_time_seconds'], seg['start_time_sec'])
        overlap_end = min(voc_row['end_time_seconds'], seg['end_time_sec'])
        overlap_seconds = overlap_end - overlap_start + 1
        total_seconds = voc_row['end_time_seconds'] - voc_row['start_time_seconds'] + 1
        prop = overlap_seconds / total_seconds
        results.append({
            'vocalization_id': voc_row['vocalization_id'],
            'video_id': voc_row['video_id'],
            'seconds': overlap_seconds,
            'words': voc_row['words'] * prop,
            'interaction_type': seg['category']
        })
    return results

def map_vocalizations_to_segments(db_path: DataPaths.INFERENCE_DB_PATH, segments_csv_path: ResearchQuestions.INTERACTION_SEGMENTS_CSV):
    """
    Retrieve KCHI vocalizations from the SQLite database and map them to interaction segments.
    
    Parameters:
    ----------
    db_path : str
        Path to the SQLite database.
    segments_csv_path : Path
        Path to the CSV file containing interaction segments with columns:
        ['start_time_sec', 'end_time_sec', 'category']
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with mapped vocalizations including:
        ['vocalization_id', 'video_id', 'seconds', 'words', 'interaction_type']
    """
    # Load segments DataFrame
    segments_df = pd.read_csv(segments_csv_path)

    # Connect to the SQLite database and query KCHI vocalizations
    conn = sqlite3.connect(db_path)
    vocalizations_query = """
    SELECT vocalization_id, video_id, start_time_seconds, end_time_seconds, words
    FROM Vocalizations
    WHERE speaker = 'KCHI'
    """
    kchi_vocs = pd.read_sql_query(vocalizations_query, conn)
    conn.close()
    
    mapped_rows = []
    # Map each vocalization to interaction segments
    for _, voc_row in kchi_vocs.iterrows():
        mapped_rows.extend(row_wise_mapping(voc_row, segments_df))
    
    return pd.DataFrame(mapped_rows)