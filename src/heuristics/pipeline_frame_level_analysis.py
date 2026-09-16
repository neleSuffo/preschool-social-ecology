"""
Frame-Level Social Interaction Analysis Pipeline

This script integrates multimodal data (Face, Person, Audio, Books) from a SQLite 
database and classifies each frame into a social state (Interacting, Available, Alone).
Optimization: Uses vectorized Pandas operations instead of row-wise processing.
"""

import logging
import sqlite3
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Path configuration
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Analysis
from config import AnalysisConfig, DataConfig
from inference.utils import load_processed_videos

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_persistent_tables(conn: sqlite3.Connection):
    """
    Creates persistent indexed tables in SQLite to optimize repeated queries.
    Pre-aggregates detections to avoid re-calculating exclusions and MAX() values.
    
    Parameters
    ----------
    conn: sqlite3.Connection
        Active connection to the SQLite database.
        
    This function creates:
    1. PersistentExclusions: A table of detection IDs that should be excluded (e.g., faces inside books).
    2. CachedFaceAgg: A pre-aggregated table of maximum proximity and face confidence per frame (after exclusions).
    3. CachedPersonAgg: A pre-aggregated table of maximum person confidence per frame (after exclusions).
    Indexes are created on these tables to speed up JOIN operations in the main query.
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE

    # 1. Persistent Exclusion Table (Detections inside Books)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS PersistentExclusions AS
    SELECT fd.detection_id, 'face' as type FROM FaceDetections fd
    JOIN BookDetections bd ON fd.frame_number = bd.frame_number AND fd.video_id = bd.video_id
    WHERE fd.x_min >= bd.x_min AND fd.y_min >= bd.y_min AND fd.x_max <= bd.x_max AND fd.y_max <= bd.y_max
    UNION ALL
    SELECT pc.detection_id, 'person' as type FROM PersonDetections pc
    JOIN BookDetections bd ON pc.frame_number = bd.frame_number AND pc.video_id = bd.video_id
    WHERE pc.x_min >= bd.x_min AND pc.y_min >= bd.y_min AND pc.x_max <= bd.x_max AND pc.y_max <= bd.y_max;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pers_excl ON PersistentExclusions(detection_id);")

    # 2. Aggregated Face Table (Filtered and Sampled)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS CachedFaceAgg AS
    SELECT frame_number, video_id, MAX(proximity) AS proximity, MAX(confidence_score) AS face_conf
    FROM FaceDetections
    WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='face')
      AND frame_number % {sample_rate} = 0
    GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_face_agg ON CachedFaceAgg(video_id, frame_number);")

    # 3. Aggregated Person Table (Filtered and Sampled)
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS CachedPersonAgg AS
    SELECT frame_number, video_id, MAX(confidence_score) AS person_conf
    FROM PersonDetections
    WHERE detection_id NOT IN (SELECT detection_id FROM PersistentExclusions WHERE type='person')
      AND frame_number % {sample_rate} = 0
    GROUP BY video_id, frame_number;
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_person_agg ON CachedPersonAgg(video_id, frame_number);")

def get_all_analysis_data(conn: sqlite3.Connection, 
                          video_list: list) -> pd.DataFrame:
    """
    Fetches and integrates all necessary data for frame-level analysis in a single optimized query.
    
    Parameters
    ----------
    conn: sqlite3.Connection
        Active connection to the SQLite database.
    video_list: list
        List of video names to include in the analysis. If empty, includes all videos.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per sampled frame, containing:
        - frame_number
        - video_id
        - video_name
        - proximity (max proximity from face detections)
        - face_conf (max face confidence)
        - person_conf (max person confidence)
        - has_kchi (binary flag for presence of KCHI in audio)
        - has_ohs (binary flag for presence of OHS in audio)
        - has_cds (binary flag for presence of CDS in audio)
        - instant_presence_conf (max of face_conf and person_conf, used for gating)
    """
    sample_rate = AnalysisConfig.SAMPLE_RATE
    placeholders = ','.join('?' for _ in video_list)
    
    query = f"""
    WITH RECURSIVE FrameGrid AS (
        SELECT video_id, video_name, 0 AS frame_number, max_frame FROM Videos
        WHERE video_name IN ({placeholders})
        UNION ALL
        SELECT video_id, video_name, frame_number + {sample_rate}, max_frame FROM FrameGrid
        WHERE frame_number + {sample_rate} <= max_frame
    )
    SELECT
        fg.frame_number, fg.video_id, fg.video_name,
        COALESCE(fa.proximity, 0) AS proximity,
        COALESCE(fa.face_conf, 0) AS face_conf,
        COALESCE(pa.person_conf, 0) AS person_conf,
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        MAX(COALESCE(pa.person_conf, 0), COALESCE(fa.face_conf, 0)) AS instant_presence_conf
    FROM FrameGrid fg
    LEFT JOIN CachedFaceAgg fa ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN CachedPersonAgg pa ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    LEFT JOIN AudioClassifications af ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    return pd.read_sql(query, conn, params=tuple(video_list))

def calculate_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized calculation of rolling window features (Presence Score, Persistence). 
    In detail, this function computes:
    1. Presence Score: A rolling average of the instant presence confidence to smooth out short-term fluctuations.
    2. Visual Persistence: A binary feature indicating if a person has been seen recently, based on high-confidence anchors and short-term presence.
    3. Sustained Audio Windows: Binary features indicating if there has been sustained CDS or OHS presence over a defined window.   
    
    Parameters    
    ----------
    df: pd.DataFrame
        Input DataFrame containing frame-level data.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional windowed features.
    """
    fps = DataConfig.FPS
    sr = AnalysisConfig.SAMPLE_RATE
    samples_per_sec = fps / sr
    
    # Rolling presence signal for CPD
    df['presence_score'] = df['instant_presence_conf'].rolling(
        window=int(samples_per_sec), min_periods=1, center=True
    ).mean().fillna(0)

    # Visual Persistence
    df['is_high_conf_anchor'] = ((df['proximity'] > AnalysisConfig.HIGH_CONFIDENCE_PROXIMITY_THRESHOLD) | (df['face_conf'] > AnalysisConfig.HIGH_CONFIDENCE_FACE_CONFIDENCE)).astype(int)  
    long_mem = df['is_high_conf_anchor'].rolling(
        window=int(AnalysisConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec), 
        min_periods=1, center=True
    ).max().fillna(0)
    
    short_mem = df['instant_presence_conf'].rolling(
        window=int(AnalysisConfig.SHORT_TERM_VISUAL_MEMORY_SEC * samples_per_sec), min_periods=1, center=True
    ).max().fillna(0) >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD

    df['person_seen_recently'] = (long_mem == 1) | (short_mem)

    # Sustained Audio Windowing
    sustained_win = int(AnalysisConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec)
    df['is_sustained_kcds'] = df['has_cds'].rolling(window=sustained_win).mean() >= AnalysisConfig.SUSTAINED_KCDS_THRESHOLD
    df['is_sustained_ohs'] = df['has_ohs'].rolling(window=sustained_win).mean() >= AnalysisConfig.MIN_PRESENCE_OHS_FRACTION

    return df

def classify_frames(df: pd.DataFrame, 
                    social_state_mode: str = "tertiary") -> pd.DataFrame:
    """
    High-speed social state classifier using Boolean masking.
    Hierarchy: 1 (Interacting) > 2 (Available) > 3 (Alone)
    
    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame with calculated features.
    social_state_mode: str
        "tertiary" for three-class classification, "binary" to merge Available and Alone into one class.
    """
    # 1. Define Boolean Logic Filters
    is_visual_anchor = df['presence_score'] >= AnalysisConfig.AUDIO_VISUAL_GATING_FLOOR
    is_available_visual = df['person_seen_recently'].astype(bool)
    is_visual_confident = df['instant_presence_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD
    is_close = df['proximity'] >= AnalysisConfig.PROXIMITY_THRESHOLD
    
    # Define rules
    rule1_tt = df['is_audio_interaction'].astype(bool) if 'is_audio_interaction' in df else False
    rule2_prox = (is_visual_confident & is_close)
    rule3_kcds = df['is_sustained_kcds'].astype(bool)
    rule4_ohs = df['is_sustained_ohs'].astype(bool)

    df['rule1_turn_taking'] = rule1_tt
    df['rule2_close_proximity'] = rule2_prox
    df['rule3_kcds_speaking'] = (rule3_kcds & is_visual_anchor)
    # ------------------------------------------------------------

    # 2. Initialize Default State: Alone (3)
    df['interaction_type'] = 3
    
    # 3. Apply 'Available' (2)
    available_mask = is_available_visual | (rule4_ohs & is_visual_anchor)
    df.loc[available_mask, 'interaction_type'] = 2
    
    # 4. Apply 'Interacting' (1)
    interacting_mask = (
        rule1_tt | 
        df['rule3_kcds_speaking'] | # Use the gated rule we just saved
        rule2_prox
    )
    df.loc[interacting_mask, 'interaction_type'] = 1

    # 5. Handle Binary Mode
    if social_state_mode == "binary":
    # Everything that is NOT Interacting (1) must become Not Interacting (2)
    # This includes both 'Available' (2) and 'Alone' (3)
        df.loc[df['interaction_type'] != 1, 'interaction_type'] = 2
        
    # We use the INSTANT_CONFIDENCE_THRESHOLD to decide if a detection counts as "present"
    df['has_face'] = (df['face_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df['has_person'] = (df['person_conf'] >= AnalysisConfig.INSTANT_CONFIDENCE_THRESHOLD).astype(int)
    df['person_or_face_present'] = ((df['has_face'] == 1) | (df['has_person'] == 1)).astype(int)
    
    return df

def find_segments(video_df: pd.DataFrame, 
                  column_name: str) -> List[Dict]:
    """
    Identifies continuous segments using vectorized operations.
    (Optimized: Replaced Python loop with NumPy diff)
    """
    segments = []

    if column_name not in video_df.columns:
        return segments

    speech_frames = video_df[video_df[column_name] == 1]
    if speech_frames.empty:
        return segments

    # Determine frame numbers: use index if frame_number isn't explicit
    if 'frame_number' in speech_frames.columns:
        frame_numbers = speech_frames['frame_number'].values.astype(int)
    else:
        frame_numbers = speech_frames.index.values.astype(int)

    if len(frame_numbers) < 2:
        # Handle the single-frame or single-segment case
        return [{
            'start': int(frame_numbers[0]),
            'end': int(frame_numbers[-1]),
            'type': column_name.split('_')[-1]
        }]

    # Vectorized gap calculation (difference between consecutive frames)
    gaps = np.diff(frame_numbers) 

    # Identify indices where the gap exceeds SAMPLE_RATE (segment breaks)
    # np.where returns a tuple, take the first element (the array of indices)
    break_indices = np.where(gaps > AnalysisConfig.SAMPLE_RATE)[0]
    
    # Segment boundaries reconstruction
    current_start = frame_numbers[0]
    
    # Loop over break indices
    for i in break_indices:
        segments.append({
            'start': int(current_start),
            'end': int(frame_numbers[i]),
            'type': column_name.split('_')[-1]
        })
        current_start = frame_numbers[i + 1]

    # Append last segment
    segments.append({
        'start': int(current_start),
        'end': int(frame_numbers[-1]),
        'type': column_name.split('_')[-1]
    })

    return segments

def check_audio_interaction_turn_taking(df: pd.DataFrame, 
                                        fps: int) -> pd.Series:
    """
    Identifies continuous audio interaction bouts where KCHI and CDS segments 
    are linked by a small gap (<= MAX_TURN_TAKING_GAP_SEC) or inter-sperspeaker gap (<= MAX_SAME_SPEAKER_GAP_SEC) if they are the same type.
    Only segments that contain both KCHI and CDS are classified as Interacting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'has_kchi', 'has_cds']
    fps : int
        Frames per second (used for gap calculation)
    
    Returns
    -------
    pd.Series: Boolean Series of 'is_audio_interaction' for all frames.
    """
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [], name='is_audio_interaction')
    
    # Ensure the DataFrame is sorted by video_id and frame_number
    orig_index = df.index
    MAX_GAP_FRAMES = AnalysisConfig.MAX_TURN_TAKING_GAP_SEC * fps
    MAX_SAME_SPEAKER_GAP_FRAMES = AnalysisConfig.MAX_SAME_SPEAKER_GAP_SEC * fps
    all_results = []

    # Loop thorugh videos
    for video_id, video_df in df.groupby('video_id'):
        video_df = video_df.copy()
        video_df['orig_idx'] = video_df.index
        video_df.set_index('frame_number', inplace=True)
        video_df['is_audio_interaction'] = False
        
        # find segments with kchi and cds
        kchi_segments = find_segments(video_df, 'has_kchi')
        kcds_segments = find_segments(video_df, 'has_cds')
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            all_results.append(video_df[['orig_idx', 'is_audio_interaction']])
            continue
            
        interaction_windows = []
        current_window = {
            'start': all_segments[0]['start'],
            'end': all_segments[0]['end'],
            'types': {all_segments[0]['type']}
        }

        # loop through segments and check whether gaps match the criteria
        for seg in all_segments[1:]:
            is_same_type = seg['type'] in current_window['types']
            gap = seg['start'] - current_window['end']
            
            if is_same_type:
                if gap > MAX_SAME_SPEAKER_GAP_FRAMES:
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}
                else:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
            else: 
                if gap <= MAX_GAP_FRAMES:
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
                else:
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    current_window = {'start': seg['start'], 'end': seg['end'], 'types': {seg['type']}}

        if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
            interaction_windows.append(current_window)

        # Mark the entire conversational exchange (including intra-turn gaps) as True
        for window in interaction_windows:
            video_df.loc[window['start'] : window['end'], 'is_audio_interaction'] = True
        
        all_results.append(video_df[['orig_idx', 'is_audio_interaction']])

    # Perfectly re-align back onto original df indices
    result_df = pd.concat(all_results, ignore_index=True).set_index('orig_idx')
    return result_df.reindex(orig_index)['is_audio_interaction'].fillna(False)

def main(db_path: Path, 
         output_dir: Path, 
         social_state_mode: str,
         video_list: list = None,
         hyperparameter_tuning: bool = False):
    """
    Orchestrates the frame-level processing pipeline.
    
    Parameters
    ----------
    db_path: Path
        Path to the SQLite database containing multimodal data.
    output_dir: Path
        Directory where the output CSV will be saved.
        Optional list of video names to process. If None, processes all videos in the database.
    social_state_mode: str
        "binary" - classifies frames into Interacting vs Not-Interacting (Available + Alone)
        "tertiary" - classifies frames into Interacting, Available, Alone (default)
    video_list: list
        Optional list of video names to include in the analysis. If None, includes all videos.
    hyperparameter_tuning: bool
        If True, takes configurations from the parent tuning script and avoids overwriting tuned parameters. Default is False.
    """    
    # 2. Only apply mode if we aren't tuning (avoid overwriting tuned params)
    if not hyperparameter_tuning:
        AnalysisConfig.apply_mode(social_state_mode)
        
    with sqlite3.connect(db_path) as conn:
        prepare_persistent_tables(conn)
        
        all_data = get_all_analysis_data(conn, video_list)
        
        # Audio Interaction Logic (Turn-taking)
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, DataConfig.FPS)
        
        # Calculate Features and Vectorized Classification
        all_data = calculate_window_features(all_data)
        all_data = classify_frames(all_data, social_state_mode=social_state_mode)
        
        output_path = output_dir / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        all_data.to_csv(output_path, index=False)
        logging.info(f"✅ Frame-level analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorized Frame-Level Social Analysis")
    parser.add_argument('--social_state_mode', type=str, choices=['binary', 'tertiary'], default='tertiary')
    parser.add_argument('--video_list', type=str, nargs='+', default=None)
    # New argument to detect if we are part of a tuning run
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--hyperparameter_tuning', action='store_true')

    args = parser.parse_args()
    
    # --- SMART FOLDER LOGIC ---
    if args.output_dir:
        # Tuning/CV mode: Use the folder provided by the parent script
        run_output_dir = Path(args.output_dir)
    else:
        # Inference mode: Create a new timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = Analysis.BASE_OUTPUT_DIR / f"analysis_{timestamp}"
    
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    main(
        db_path=Path(DataPaths.INFERENCE_DB_PATH), 
        output_dir=run_output_dir,
        social_state_mode=args.social_state_mode, 
        video_list=args.video_list,
        hyperparameter_tuning=args.hyperparameter_tuning
    )