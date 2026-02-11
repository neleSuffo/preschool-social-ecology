# Research Question 1. How much time do children spend alone?
#
# This script analyzes frame-level data to determine the amount of time children spend alone.
# It uses a SQLite database to query the relevant data.
# The analysis focuses on identifying frames where children are not in the presence of adults.

import logging
import sqlite3
import argparse
import sys
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from deepface import DeepFace
from typing import List, Dict, Optional, Tuple

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig
from inference.utils import load_processed_videos

# Constants
FPS = DataConfig.FPS # frames per second
SAMPLE_RATE = InferenceConfig.SAMPLE_RATE # every n-th frame is processed (configurable sampling sampling rate)

# Placeholder labels for classification
ROBUST_ALONE_FLAG = 'is_robust_alone'
SUSTAINED_KCDS_FLAG = 'is_sustained_kcds'
ROBUST_PERSON_FLAG = 'is_robust_person'
ROBUST_OHS_FLAG = 'is_sustained_ohs'
RECENT_KCDS_FLAG = 'has_recent_kcds'
MEDIA_INTERACTION_FLAG = 'is_media_interaction'
AUDIOBOOK_FLAG = 'is_audiobook'

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def get_raw_face_bboxes_for_frames(conn, video_id: int, start_frame: int, end_frame: int) -> pd.DataFrame:
    """
    Retrieves face detections (bounding boxes + proximity) for a frame range of a given video.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite connection to the database.
    video_id : int
        ID of the video to query.
    start_frame : int
        Starting frame number (inclusive).
    end_frame : int
        Ending frame number (inclusive).
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing face detections with columns:
        [frame_number, x_min, y_min, x_max, y_max, proximity, has_face, detection_id]
    """
    query = """
        SELECT detection_id, frame_number, x_min, y_min, x_max, y_max  
        FROM FaceDetections
        WHERE video_id = ?
        AND frame_number BETWEEN ? AND ?
        ORDER BY frame_number ASC;
    """

    try:
        df = pd.read_sql_query(
            query,
            conn,
            params=(video_id, start_frame, end_frame)
        )
        return df

    except Exception as e:
        logging.error(f"Database error in get_raw_face_bboxes_for_frames: {e}")
        return pd.DataFrame(columns=[
            "frame_number", "x_min", "y_min", "x_max", "y_max",
            "proximity", "has_face", "detection_id"
        ])

def find_segments(video_df: pd.DataFrame, column_name: str) -> List[Dict]:
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
    break_indices = np.where(gaps > SAMPLE_RATE)[0]
    
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

def get_all_analysis_data(conn, video_list: list):
    """
    Multimodal data integration at configurable frame intervals, 
    EXCLUDING Face/Person Detections fully contained within Book Detections,
    and including the aggregated book presence.
    
    Parameters:
    ----------
    conn : sqlite3.Connection
        SQLite connection to the database.
    video_list : list
        List of video names to filter analysis on.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame containing the integrated analysis data.
    """
    # Ensure video list elements are str
    video_list = [str(v) for v in video_list]
    
    if video_list:
        placeholders = ','.join('?' for _ in video_list)
        video_filter_clause = f"WHERE video_name IN ({placeholders})"
        query_params = tuple(video_list)
        
        logging.info(f"Filtering analysis for {len(video_list)} videos.")
    else:
        video_filter_clause = ""
        query_params = ()
        logging.info("No video filter list provided. Processing all videos.")
        
    # ====================================================================
    # STEP 0: CREATE EXCLUSION TEMPORARY TABLE
    # ====================================================================
    
    # Exclude Face Detections contained in Books
    conn.execute(f"""
    CREATE TEMP TABLE IF NOT EXISTS ExcludedFaceDetections AS
    SELECT 
        fd.detection_id
    FROM FaceDetections fd
    JOIN BookDetections bd 
        ON fd.frame_number = bd.frame_number AND fd.video_id = bd.video_id
    WHERE 
        fd.x_min >= bd.x_min AND 
        fd.y_min >= bd.y_min AND
        fd.x_max <= bd.x_max AND
        fd.y_max <= bd.y_max;
    """)

    # Exclude Person Detections contained in Books
    conn.execute(f"""
    CREATE TEMP TABLE IF NOT EXISTS ExcludedPersonDetections AS
    SELECT 
        pc.detection_id
    FROM PersonDetections pc
    JOIN BookDetections bd 
        ON pc.frame_number = bd.frame_number AND pc.video_id = bd.video_id
    WHERE 
        pc.x_min >= bd.x_min AND 
        pc.y_min >= bd.y_min AND
        pc.x_max <= bd.x_max AND
        pc.y_max <= bd.y_max;
    """)

    # ====================================================================
    # STEP 1A: FACE DETECTION AGGREGATION (FILTERED)
    # ====================================================================
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT
        fd.frame_number, 
        fd.video_id, 
        MAX(fd.proximity) AS proximity,
        1 AS has_face
    FROM FaceDetections fd
    WHERE 
        fd.frame_number % ? = 0 
        AND fd.detection_id NOT IN (SELECT detection_id FROM ExcludedFaceDetections)
    GROUP BY fd.frame_number, fd.video_id;
    """, (SAMPLE_RATE,))
    
    # ====================================================================
    # STEP 1B: PERSON DETECTION AGGREGATION (FILTERED)
    # ====================================================================
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS PersonAgg AS
    SELECT DISTINCT 
        pc.frame_number, 
        pc.video_id, 
        1 AS has_person
    FROM PersonDetections pc
    WHERE 
        pc.frame_number % ? = 0
        AND pc.detection_id NOT IN (SELECT detection_id FROM ExcludedPersonDetections);
    """, (SAMPLE_RATE,))

    # ====================================================================
    # STEP 1C (NEW): BOOK DETECTION AGGREGATION
    # ====================================================================
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS BookAgg AS
    SELECT DISTINCT 
        bd.frame_number, 
        bd.video_id, 
        1 AS has_book
    FROM BookDetections bd
    WHERE 
        bd.frame_number % ? = 0;
    """, (SAMPLE_RATE,))

    # ====================================================================
    # STEP 2: MAIN DATA INTEGRATION QUERY - GENERATE DENSE FRAME GRID
    # ====================================================================
    
    query = f"""
    -- RECURSIVE CTE to generate the full temporal grid for all filtered videos
    WITH RECURSIVE FilteredVideos AS (
        SELECT * FROM Videos
        {video_filter_clause}
    ), 
    FrameGrid AS (
        -- Anchor member: Start at frame 0 for every video in the filtered list
        SELECT
            video_id,
            video_name,
            0 AS frame_number,
            max_frame
        FROM FilteredVideos
        
        UNION ALL
        
        -- Recursive member: Increment frame_number by SAMPLE_RATE
        SELECT
            v.video_id,
            v.video_name,
            fg.frame_number + {SAMPLE_RATE},
            v.max_frame
        FROM FrameGrid fg
        JOIN FilteredVideos v ON fg.video_id = v.video_id
        -- Termination condition: Stop when the next frame exceeds max_frame
        WHERE fg.frame_number + {SAMPLE_RATE} <= v.max_frame
    )
    
    SELECT
        fg.frame_number,
        fg.video_id,
        fg.video_name,
        
        -- PERSON DETECTION MODALITY
        COALESCE(pa.has_person, 0) AS has_person, 
        
        -- AUDIO CLASSIFICATION MODALITY
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        
        -- FACE DETECTION MODALITY
        COALESCE(fa.has_face, 0) AS has_face,
        fa.proximity,
        
        -- MULTIMODAL FUSION FLAGS
        CASE 
            WHEN COALESCE(fa.has_face, 0)=1 OR COALESCE(pa.has_person, 0)=1 THEN 1 
            ELSE 0 
        END AS person_or_face_present,
        
        -- NEW: BOOK DETECTION MODALITY
        COALESCE(ba.has_book, 0) AS has_book

    FROM FrameGrid fg
    LEFT JOIN AudioClassifications af 
        ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    LEFT JOIN FaceAgg fa 
        ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN PersonAgg pa
        ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    -- NEW JOIN: Join to Book Aggregation
    LEFT JOIN BookAgg ba
        ON fg.frame_number = ba.frame_number AND fg.video_id = ba.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    
    df = pd.read_sql(query, conn, params=query_params)
    
    return df

def calculate_window_features(df: pd.DataFrame, fps: int, sample_rate: int) -> pd.DataFrame:
    """
    Calculates all windowed features (Rules 3, 4, Available, Alone) in a single vectorized pass.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'has_cds', 'has_ohs', 'person_or_face_present']
    fps : int
        Frames per second of the video
    sample_rate : int
        Sampling rate used for frame selection
        
    Returns
    -------
    pd.DataFrame
        DataFrame with calculated window features
    """
    # Safeguard: ensure audiobook flag name is always available even if globals were altered
    audiobook_flag = 'is_audiobook'
    # If global constant exists, prefer it
    if 'AUDIOBOOK_FLAG' in globals():
        audiobook_flag = AUDIOBOOK_FLAG
    
    # 1. Setup Window Sizes (in samples/rows)
    samples_per_sec = fps / sample_rate
    
    # Rule 3/Available Window (N)
    N_sustained = int(InferenceConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec)
    N_avail = int(InferenceConfig.PERSON_AVAILABLE_WINDOW_SEC * samples_per_sec)
    N_recent_speech = int(InferenceConfig.PERSON_AUDIO_WINDOW_SEC * samples_per_sec)
    N_alone = int(InferenceConfig.ROBUST_ALONE_WINDOW_SEC * samples_per_sec)
    N_media = int(InferenceConfig.MEDIA_WINDOW_SEC * samples_per_sec)
    
    # --- 1. Sustained KCDS ---
    if N_sustained > 0:
        # 1. Calculate the fraction of the window containing CDS
        df['kcds_frac_lookback'] = df['has_cds'].rolling(window=N_sustained, min_periods=1).mean()
        # 2. Use a threshold (e.g., 0.5 for 50% speech presence)         
        df[SUSTAINED_KCDS_FLAG] = (df['kcds_frac_lookback'] >= InferenceConfig.SUSTAINED_KCDS_THRESHOLD)
        
        # 3. Apply the centering logic to mark the whole window as sustained
        df[SUSTAINED_KCDS_FLAG] = df[SUSTAINED_KCDS_FLAG].rolling(
            window=N_sustained, min_periods=1, center=True
        ).max().fillna(False).astype(bool)
    else:
        df[SUSTAINED_KCDS_FLAG] = False

    # --- 2. Audiobook & Robust OHS Detection ---
    if N_avail > 0:
        df['person_frac_lookback'] = df['person_or_face_present'].rolling(window=N_avail, min_periods=1).mean()
        df['ohs_frac_lookback'] = df['has_ohs'].rolling(window=N_avail, min_periods=1).mean()

        # Identify constant background OHS (Audiobook)
        # Default threshold: OHS present in > 80% of the window
        df[audiobook_flag] = df['ohs_frac_lookback'] >= InferenceConfig.MAX_OHS_FOR_AVAILABLE

        # Robust Person Presence
        df[ROBUST_PERSON_FLAG] = df['person_frac_lookback'].rolling(window=N_avail, min_periods=1, center=True).max().fillna(0) >= InferenceConfig.MIN_PRESENCE_PERSON_FRACTION

        # Robust OHS Presence (Available)
        # Triggered ONLY if OHS is above min threshold AND NOT constant background (audiobook)
        df[ROBUST_OHS_FLAG] = (
            (df['ohs_frac_lookback'].rolling(window=N_avail, min_periods=1, center=True).max().fillna(0) >= InferenceConfig.MIN_PRESENCE_OHS_FRACTION) &
            (~df[audiobook_flag])
        )
    else:
        df['person_frac_lookback'] = 0
        df['ohs_frac_lookback'] = 0
        df[ROBUST_PERSON_FLAG] = df[ROBUST_OHS_FLAG] = df[audiobook_flag] = False

    # --- New: Visual Persistence (Memory of Adult Presence) ---
    # Defines how long we "remember" a person after they disappear from view
    N_persistence = int(InferenceConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec)     
        
    if N_persistence > 0:
        df['person_seen_recently'] = df['person_or_face_present'].rolling(
            window=N_persistence, min_periods=1, center=True
        ).max().fillna(0).astype(bool)
    else:
        df['person_seen_recently'] = df['person_or_face_present'].astype(bool)
        
    # --- 3. Robust Alone Check ---
    if N_alone > 0:
        df['social_signal_total'] = df['person_or_face_present'] + df['has_cds'] + df['has_ohs']
        df['social_signal_frac'] = df['social_signal_total'].rolling(window=N_avail, min_periods=1).mean() / 3
        df[ROBUST_ALONE_FLAG] = df['social_signal_frac'].rolling(window=N_alone, min_periods=1, center=True).min().fillna(1) <= InferenceConfig.MAX_ALONE_FALSE_POSITIVE_FRACTION
    else:
        df[ROBUST_ALONE_FLAG] = True

    # --- 4. Recent KCDS & Media Check ---
    df[RECENT_KCDS_FLAG] = df['has_cds'].rolling(window=max(1, N_recent_speech), min_periods=1).max().shift(1).fillna(0).astype(bool)
    
    if N_media > 0:
        df['book_frac'] = df['has_book'].rolling(window=N_media, min_periods=N_media).mean()
        df['cds_ohs_frac'] = (df['has_cds'] + df['has_ohs']).rolling(window=N_media, min_periods=N_media).mean() / 2
        df['kchi_frac'] = df['has_kchi'].rolling(window=N_media, min_periods=N_media).mean()
        df[MEDIA_INTERACTION_FLAG] = ( (df['book_frac'] >= InferenceConfig.MIN_BOOK_PRESENCE_FRACTION) & (df['cds_ohs_frac'] >= InferenceConfig.MIN_PRESENCE_OHS_KCDS_FRACTION_MEDIA) & (df['kchi_frac'] <= InferenceConfig.MAX_KCHI_FRACTION_FOR_MEDIA) ).rolling(window=N_media, min_periods=1, center=True).max().fillna(False).astype(bool)
    else:
        df[MEDIA_INTERACTION_FLAG] = False
        
    return df[['video_id', 'frame_number', SUSTAINED_KCDS_FLAG, ROBUST_PERSON_FLAG, ROBUST_OHS_FLAG, ROBUST_ALONE_FLAG, RECENT_KCDS_FLAG, MEDIA_INTERACTION_FLAG, audiobook_flag, 
               'person_seen_recently',  # Required for classify_frames
               'ohs_frac_lookback',     # Required for the refined OHS flag
               'person_frac_lookback'   # Required for the refined OHS flag
               ]].copy()

def check_audio_interaction_turn_taking(df, fps):
    """
    Identifies continuous audio interaction bouts where KCHI and CDS segments 
    are linked by a small gap (<= MAX_TURN_TAKING_GAP_SEC).
    
    Crucially, it prevents merging segments of the same type across any gap 
    and filters out non-dual speaker windows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'has_kchi', 'has_cds', 'has_ohs']
    fps : int
        Frames per second (used for gap calculation)
    
    Returns
    -------
    pd.Series: Boolean Series of 'is_audio_interaction' for all frames.
    """
    # Handle empty DataFrame gracefully
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [], name='is_audio_interaction')
    
    MAX_GAP_FRAMES = InferenceConfig.MAX_TURN_TAKING_GAP_SEC * fps
    all_results = []

    # Iterate over the clean, unique initial_df
    for video_id, video_df in df.groupby('video_id'):
        
        # 1. Prepare video segment: Index by frame_number for easy marking
        video_df = video_df.copy() 
        video_df.set_index('frame_number', inplace=True) 
        video_df['is_audio_interaction'] = False
        
        # Identify KCHI and KCDS segments and combine them (only speech, no silence included yet)
        kchi_segments = find_segments(video_df, 'has_kchi')
        kcds_segments = find_segments(video_df, 'has_cds')
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            video_df.reset_index(inplace=True)
            all_results.append(video_df[['frame_number', 'is_audio_interaction']])
            continue
            
        interaction_windows = []
        
        # --- PHASE 2: Merge Segments and Filter for Dual Speakers ---
        current_window = {
            'start': all_segments[0]['start'],
            'end': all_segments[0]['end'],
            'types': {all_segments[0]['type']}
        }

        for seg in all_segments[1:]:
            is_same_type = seg['type'] in current_window['types']
            gap = seg['start'] - current_window['end']
            
            if is_same_type:
                # To prevent merging identical segments across gaps larger than allowed
                if gap > (InferenceConfig.MAX_SAME_SPEAKER_GAP_SEC * fps):
                    # Finalize current window (it's not turn-taking if it's only one speaker type)
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    
                    # Start a new window with the current segment
                    current_window = {
                        'start': seg['start'],
                        'end': seg['end'],
                        'types': {seg['type']}
                    }
                else:
                    # Gap is small enough: extend the current window
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])


            else: # Segments are of DIFFERENT types
                if gap <= MAX_GAP_FRAMES:
                    # Bridge the small gap and extend the window
                    current_window['end'] = seg['end']
                    current_window['types'].add(seg['type'])
                else:
                    # Gap is too long: Finalize the current window and check structural validity
                    if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
                        interaction_windows.append(current_window)
                    
                    # Start a new window with the current segment
                    current_window = {
                        'start': seg['start'],
                        'end': seg['end'],
                        'types': {seg['type']}
                    }

        # Finalize the last window after the loop ends
        if 'kchi' in current_window['types'] and 'cds' in current_window['types']:
            interaction_windows.append(current_window)


        # 4. Final Marking on the DataFrame
        for window in interaction_windows:
            window_start_frame = window['start']
            window_end_frame = window['end']
            
            # Only mark frames within the window that have ANY audio activity ---
            audio_mask = (video_df.loc[window_start_frame : window_end_frame, 'has_kchi'] == 1) | \
                         (video_df.loc[window_start_frame : window_end_frame, 'has_cds'] == 1)
            # Apply the mask to mark only the non-silent frames within the established window
            video_df.loc[
                audio_mask.index[audio_mask], # Index where audio_mask is True
                'is_audio_interaction'
            ] = True
        
        # Reset index, add video_id, and append to results list
        video_df.reset_index(inplace=True)
        video_df['video_id'] = video_id 
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])

    # 5. Combine and Merge Results Back to Original DF
    result_df = pd.concat(all_results, ignore_index=True)

    # save result_df for debugging
    result_df.to_csv("/home/nele_pauline_suffo/outputs/quantex_inference/debug_audio_interaction_segments.csv", index=False)
    
    return result_df['is_audio_interaction']

def classify_frames(row, results_df, included_rules=None):
    """
    Hierarchical social interaction classifier (Optimized: uses pre-calculated flags).
    """
    if included_rules is None:
        included_rules = [1, 2, 3, 4, 5]
    
    current_index = row.name
    
    # --- Retrieve Pre-calculated Flags ---
    
    # Interacting Flags
    # Flags
    rule1_turn_taking = bool(row['is_audio_interaction'])
    rule2_close_proximity = bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD) if pd.notna(row['proximity']) else False
    rule3_kcds_speaking = bool(row[SUSTAINED_KCDS_FLAG])
    person_or_face_present_instant = (row['person_or_face_present'] == 1)
    rule4_person_recent_speech = bool(person_or_face_present_instant and row[RECENT_KCDS_FLAG])
    
    is_sustained_ohs = bool(row[ROBUST_OHS_FLAG])
    is_robust_person_presence = bool(row[ROBUST_PERSON_FLAG])
    is_sustained_absence = bool(row[ROBUST_ALONE_FLAG])
    is_media_interaction = bool(row[MEDIA_INTERACTION_FLAG])
    is_audiobook = bool(row[AUDIOBOOK_FLAG]) # New flag
    
    # Rule 5 Buffered KCHI
    BUFFER_SAMPLES = int(InferenceConfig.KCHI_PERSON_BUFFER_FRAMES / SAMPLE_RATE)
    rule5_buffered_kchi = False
    if 5 in included_rules and row['has_kchi'] == 1:
        start_buffer, end_buffer = max(0, current_index - BUFFER_SAMPLES), min(len(results_df) - 1, current_index + BUFFER_SAMPLES)
        rule5_buffered_kchi = (results_df.loc[start_buffer:end_buffer, 'person_or_face_present'] == 1).any()
            
    # TIER 1: INTERACTING
    active_rules = []
    if 1 in included_rules and rule1_turn_taking: active_rules.append(1)
    if 2 in included_rules and rule2_close_proximity: active_rules.append(2)
    if 3 in included_rules and rule3_kcds_speaking: active_rules.append(3)
    if 4 in included_rules and rule4_person_recent_speech: active_rules.append(4)
    if 5 in included_rules and rule5_buffered_kchi: active_rules.append(5)

    if active_rules:
        interaction_category = "Interacting"
    elif is_sustained_absence: 
        interaction_category = "Alone"
    elif is_audiobook and not is_robust_person_presence:
        # Constant OHS with no visual presence = ALONE
        interaction_category = "Alone"
    elif is_sustained_ohs or is_robust_person_presence:         
        interaction_category = "Alone" if is_media_interaction else "Available"
    else:
        interaction_category = "Alone"
            
    return (interaction_category, rule1_turn_taking, rule2_close_proximity, rule3_kcds_speaking, rule4_person_recent_speech, rule5_buffered_kchi)

def classify_face_category(row):
    """
    Categorize face detection presence using simplified `has_face` indicator.
    """
    try:
        return 'has_face' if int(row.get('has_face', 0)) == 1 else 'no_faces'
    except Exception:
        return 'no_faces'

def classify_person_category(row):
    """
    Categorize raw person detection presence using the `has_person` indicator.
    """
    try:
        # Reverted to use the raw 'has_person' column
        return 'has_person' if int(row.get('has_person', 0)) == 1 else 'no_persons'
    except Exception:
        return 'no_persons'

def classify_fused_category(row):
    """
    Categorize person presence using the fused `person_or_face_present` indicator.
    This tracks the presence of EITHER an adult person OR an adult face.
    """
    try:
        # Uses the fused column
        return 'has_person_or_face' if int(row.get('person_or_face_present', 0)) == 1 else 'no_person_or_face'
    except Exception:
        return 'no_person_or_face'

def get_reference_bb_at_segment_start(
    conn: sqlite3.Connection, 
    video_id: int, 
    current_index: int, 
    video_df: pd.DataFrame
) -> Optional[Tuple]:
    """
    Finds a reference BB near the start of a Media-Alone segment.
    If the exact start frame has no face, searches a +/- 15 frame window.
    """
    anchor_frame = int(video_df.loc[current_index, 'frame_number'])
    window_size = 150

    query = """
    SELECT x_min, y_min, x_max, y_max, frame_number
    FROM FaceDetections 
    WHERE video_id = ? 
      AND frame_number BETWEEN ? AND ?
    ORDER BY ABS(frame_number - ?) ASC
    LIMIT 1
    """
    params = (int(video_id), anchor_frame - window_size, anchor_frame + window_size, anchor_frame)
    
    try:
        cursor = conn.cursor()
        result = cursor.execute(query, params).fetchone()
        if result:
            return (result[0], result[1], result[2], result[3])
        return None
    except Exception as e:
        logging.error(f"Error in robust Face BB search: {e}")
        return None

def cut_and_save_face(video_name: str, frame_num: int, bb: Tuple[int, int, int, int], output_dir: Path, face_id: int, is_gt: bool = False) -> Optional[Path]:
    """Cuts face from the image and saves it to a temporary folder.
    
    Parameters
    ----------
    video_name : str
        Name of the video.
    frame_num : int
        Frame number to cut from.
    bb : Tuple[int, int, int, int]
        Bounding box coordinates (x_min, y_min, x_max, y_max).
    output_dir : Path
        Directory to save the cropped face image.
    face_id : int
        ID of the face in the frame (used for naming).
    is_gt : bool, optional
        Whether this is the ground truth reference face, by default False.
        
    Returns
    -------
    Optional[Path]
        Path to the saved cropped face image, or None if failed.
    """
    # 1. Construct image path
    frame_padded = str(frame_num).zfill(6)
    video_dir = DataPaths.QUANTEX_IMAGES_INPUT_DIR / video_name 
    # check for png or jpg
    img_path_jpg = video_dir / f"{video_name}_{frame_padded}.jpg"
    if img_path_jpg.exists():
        img_path = img_path_jpg
    else:
        img_path = video_dir / f"{video_name}_{frame_padded}.png"
    if not img_path.exists():
        logging.error(f"Image file not found: {img_path}")
        return None

    # 2. Define output filename
    tag = "gt_ref" if is_gt else f"face_{face_id}"
    output_filename = f"{video_name}_{frame_padded}_{tag}.png"
    output_path = output_dir / output_filename

    try:
        # 3. Load image and crop
        img = Image.open(img_path)
        # Ensure BB coordinates are integers (x_min, y_min, x_max, y_max)
        crop_area = tuple(map(int, bb))
        cropped_img = img.crop(crop_area)
        
        # 4. Save
        cropped_img.save(output_path)
        return output_path
    except Exception as e:
        logging.error(f"Error processing image {img_path}: {e}")
        return None
    
def run_deepface_verification(gt_bb_coords: Tuple, gt_frame_num: int, gap_frames_df: pd.DataFrame, video_id: int, video_name: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Runs DeepFace verification against a GT reference face for all frames in the gap.
    
    Parameters:
    ----------
    gt_bb_coords : Tuple[int, int, int, int]
        Bounding box coordinates of the GT reference face.
    gt_frame_num : int
        Frame number of the GT reference face.
    gap_frames_df : pd.DataFrame
        DataFrame containing frames in the gap to verify.
    video_id : int
        ID of the video.
    video_name : str
        Name of the video.
    conn : sqlite3.Connection
        SQLite connection to the database.
        
    Returns:
    -------
    pd.DataFrame
        DataFrame with verification results for each frame in the gap.
    """
    if DeepFace is None:
        logging.error("DeepFace is not available. Verification skipped.")
        return pd.DataFrame()
        
    temp_crop_dir = Inference.TEMP_CUT_FACE_DIR / video_name
    temp_crop_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch all raw BBs for the gap
    gap_start_frame = gap_frames_df['frame_number'].min()
    gap_end_frame = gap_frames_df['frame_number'].max()
    raw_bb_data = get_raw_face_bboxes_for_frames(conn, video_id, gap_start_frame, gap_end_frame)
    
    # 2. Cut and save the GT reference face
    gt_bb_list = [gt_bb_coords]
    gt_face_path = cut_and_save_face(video_name, gt_frame_num, gt_bb_coords, temp_crop_dir, face_id=0, is_gt=True)
    
    if not gt_face_path:
        logging.error("Could not save GT reference face. Aborting verification.")
        return pd.DataFrame()
        
    verification_results = []
    
    # 3. Iterate through gap frames and compare all detected faces against GT
    for frame_num in gap_frames_df['frame_number'].unique():
        frame_bbs = raw_bb_data[raw_bb_data['frame_number'] == frame_num]
        
        if frame_bbs.empty:
            verification_results.append({
                'frame_number': frame_num, 'matched_gt': False, 'bb_matched': False
            })
            continue

        match_found_in_frame = False
        
        for face_idx, bb_row in enumerate(frame_bbs.itertuples()):
            current_bb = (bb_row.x_min, bb_row.y_min, bb_row.x_max, bb_row.y_max)
            
            # Cut and save the detected face in the gap frame
            current_face_path = cut_and_save_face(video_name, frame_num, current_bb, temp_crop_dir, face_id=face_idx, is_gt=False)
            
            if current_face_path:
                try:
                    # Run DeepFace verification
                    result = DeepFace.verify(img1_path=str(gt_face_path), img2_path=str(current_face_path), enforce_detection=False, detector_backend="retinaface")
                    
                    if result[0]['verified']:
                        match_found_in_frame = True
                        break # Found a match in this frame, move to next frame

                except Exception as e:
                    logging.warning(f"DeepFace failed for frame {frame_num}: {e}")
            
        verification_results.append({
            'frame_number': frame_num, 
            'matched_gt': match_found_in_frame, 
            'bb_matched': not frame_bbs.empty # Record if any face was detected at all
        })
        
    return pd.DataFrame(verification_results)
  
def smooth_media_persistence(all_data: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Orchestrates DeepFace verification to persist media exclusion.
    
    Parameters:
    ----------
    all_data : pd.DataFrame
        DataFrame containing frame-level data with MEDIA_INTERACTION_FLAG.
    conn : sqlite3.Connection
        SQLite connection to the database.
        
    Returns
    -------
    pd.DataFrame
        Updated DataFrame with persisted MEDIA_INTERACTION_FLAG.
    """
    updated_data = all_data.copy()
    updated_data[MEDIA_INTERACTION_FLAG] = updated_data[MEDIA_INTERACTION_FLAG].astype(bool)
    MAX_GAP_FRAMES = InferenceConfig.MAX_MEDIA_ALONE_GAP_SEC * FPS

    for video_id, video_df in updated_data.groupby('video_id'):
        video_name = video_df['video_name'].iloc[0]
        indices = video_df.index.tolist()
        i = 0
        
        while i < len(indices):
            row = video_df.loc[indices[i]]
            
            # 1. Look for Anchor A (Start of a Media-Alone segment)
            if not (row['interaction_type'] == 'Alone' and row[MEDIA_INTERACTION_FLAG]):
                i += 1
                continue
                
            start_anchor_idx = indices[i]
            start_frame_num = video_df.loc[start_anchor_idx, 'frame_number']
            
            # 2. Look forward for Anchor B (End of the gap)
            found_anchor_b_idx = -1
            # We look for the NEXT media-alone segment within the MAX_GAP
            for j in range(i + 1, len(indices)):
                future_idx = indices[j]
                future_row = video_df.loc[future_idx]
                
                if future_row['frame_number'] - start_frame_num > MAX_GAP_FRAMES:
                    break
                
                if future_row['interaction_type'] == 'Alone' and future_row[MEDIA_INTERACTION_FLAG]:
                    found_anchor_b_idx = j
                    break
            
            # 3. If we found a potential gap between two Media-Alone anchors
            if found_anchor_b_idx != -1:
                gap_indices = indices[i+1 : found_anchor_b_idx]
                
                if not gap_indices: # No gap to fill
                    i = found_anchor_b_idx
                    continue

                # 4. Get the Reference Face for DeepFace
                gt_bb_coords = get_reference_bb_at_segment_start(conn, video_id, start_anchor_idx, video_df)
                
                if gt_bb_coords:
                    frames_in_gap_df = video_df.loc[gap_indices]
                    verification_df = run_deepface_verification(
                        gt_bb_coords, start_frame_num, frames_in_gap_df, video_id, video_name, conn
                    )

                    if verification_df is not None and not verification_df.empty:
                        # Apply persistence if matches are found
                        match_count = verification_df['matched_gt'].sum()
                        if (match_count / len(verification_df)) >= InferenceConfig.MIN_MEDIA_FACE_MATCH_FRACTION:
                            updated_data.loc[gap_indices, MEDIA_INTERACTION_FLAG] = True
                            logging.info(f"✅ {video_name}: Persisted media over {len(gap_indices)} frames.")

                # Move the pointer to Anchor B to continue the search
                i = found_anchor_b_idx
            else:
                i += 1
                
    return updated_data

def main(db_path: Path, output_dir: Path, hyperparameter_tuning: False, included_rules: list = None):
    """
    Main analysis function that orchestrates multimodal social interaction analysis.
    
    This function performs comprehensive frame-level analysis by calling specialized
    helper functions for each analytical step:
    
    1. Data Integration: Combines person, audio, and face data into a unified dataset.
    2. Audio Turn-Taking Analysis: Detects turn-taking interactions with adaptive windows.
    3. Social Interaction Classification: Classifies frames into interaction categories and tracks individual rule activations in a single pass.
    4. Presence Pattern Categorization: Categorizes face and person detection patterns.
    5. Age Information Merging: Integrates age data from subjects CSV.
    6. Result Saving: Outputs detailed frame-level analysis to CSV with rule info in filename
    
    Parameters
    ----------
    db_path : Path
        Path to the SQLite database containing analysis data.
    output_dir : Path
        Directory where output files will be saved.
    hyperparameter_tuning: Bool
        Whether script runs in hyperparmeter mode or not 
    included_rules : list, optional
        List of rule numbers to include in interaction classification (1, 2, 3, 4).
        If None, uses default rules [2, 3, 4].

    Returns
    -------
    dict: 
        Summary statistics including interaction and presence distributions
    """
    if included_rules is None:
        included_rules = [1, 2, 3, 4]

    # Print which rules are being used
    rule_names = {
        1: "Turn-Taking (KCHI + KCDS)",
        2: "Very Close Proximity",
        3: "KCDS Present", 
        4: "Buffered KCHI + Visual"
    }

    print("🔄 Running comprehensive multimodal social interaction analysis...")
    print(f"📋 Using interaction rules: {[f'{i}: {rule_names[i]}' for i in included_rules]}")

    # ------------------------------------------------------------------
    # 🕒 Create timestamped output folder inside base output directory
    # ------------------------------------------------------------------
    if hyperparameter_tuning:
        run_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"interaction_analysis_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created output folder: {run_dir}")

    # ------------------------------------------------------------------
    # 💾 Save current InferenceConfig snapshot as JSON
    # ------------------------------------------------------------------
    config_snapshot = {attr: getattr(InferenceConfig, attr) 
                       for attr in dir(InferenceConfig) 
                       if not attr.startswith("__") and not callable(getattr(InferenceConfig, attr))}
    config_path = run_dir / "inference_config_snapshot.json"
    with open(config_path, "w") as f:
        json.dump(config_snapshot, f, indent=4)

    # ------------------------------------------------------------------
    # 📜 Copy executed script into the run directory
    # ------------------------------------------------------------------
    try:
        script_path = Path(__file__)
        shutil.copy(script_path, run_dir / script_path.name)
    except NameError:
        print("⚠️ __file__ not defined (likely running in notebook), skipping script copy.")

    # ------------------------------------------------------------------
    # 🔍 Perform analysis
    # ------------------------------------------------------------------
    processed_videos = load_processed_videos(Inference.QUANTEX_VIDEOS_LIST_FILE)

    with sqlite3.connect(db_path) as conn:
        all_data = get_all_analysis_data(conn, processed_videos) 
        
        # --- PHASE 1: AUDIO/TURN-TAKING ANALYSIS (Still sequential, benefits from faster find_segments) ---
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, FPS)
        
        # --- PHASE 2: VECTORIZED WINDOW FEATURE CALCULATION (HUGE SPEEDUP) ---
        window_flags_df = calculate_window_features(all_data, FPS, SAMPLE_RATE)
        # Merge the pre-calculated features back into the main DataFrame
        cols_to_use = window_flags_df.columns.difference(all_data.columns).tolist() + ['video_id', 'frame_number']
        all_data = all_data.merge(window_flags_df[cols_to_use], on=['video_id', 'frame_number'], how='left')
        # --- PHASE 3A: INITIAL HIERARCHICAL CLASSIFICATION (Provides interaction_type and initial MEDIA_INTERACTION flags) ---
        
        print("⏱️ PHASE 3: Running Initial Classification...")
        results = all_data.apply(lambda row: classify_frames(row, all_data, included_rules), axis=1)
        all_data['interaction_type'] = [r[0] for r in results]
        # --- PHASE 3B: BB PERSISTENCE SMOOTHING ---
        # This modifies the 'is_media_interaction' column based on BB tracking.
        print("🧠 PHASE 3B: Running Bounding Box Persistence and Smoothing (DeepFace Check)...")
        all_data = smooth_media_persistence(all_data, conn)

        # --- PHASE 4: FINAL HIERARCHICAL CLASSIFICATION (Applies final interaction_type based on smoothed flag) ---
        print("🔄 PHASE 4: Applying Final Classification based on persistence...")
        final_classification_results = all_data.apply(
            lambda row: classify_frames(row, all_data, included_rules), axis=1
        )

        all_data['interaction_type'] = [result[0] for result in final_classification_results]
        all_data['rule1_turn_taking'] = [result[1] for result in final_classification_results]
        all_data['rule2_close_proximity'] = [result[2] for result in final_classification_results]
        all_data['rule3_kcds_speaking'] = [result[3] for result in final_classification_results]
        all_data['rule4_buffered_kchi'] = [result[4] for result in final_classification_results]
        
        # Categorization
        all_data['face_frame_category'] = all_data.apply(classify_face_category, axis=1)
        all_data['person_frame_category'] = all_data.apply(classify_person_category, axis=1)
        all_data['fused_frame_category'] = all_data.apply(classify_fused_category, axis=1)

        # ------------------------------------------------------------------
        # 💾 Save frame-level CSV to the timestamped output folder
        # ------------------------------------------------------------------
        file_name = Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem + f"_{'_'.join(map(str, included_rules))}.csv"
        csv_path = run_dir / file_name
        all_data.to_csv(csv_path, index=False)
        print(f"✅ Saved detailed frame-level analysis to {csv_path}")

        return all_data

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Frame-level social interaction analysis')
    parser.add_argument('--rules', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=cds-speaking, 4=buffered-kchi-visual). Default: [1, 2, 3, 4]')

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules, hyperparameter_tuning=False)