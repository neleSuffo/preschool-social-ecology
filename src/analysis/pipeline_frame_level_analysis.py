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

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        MAX(fd.confidence_score) AS face_conf, -- Get raw confidence
        1 AS has_face
    FROM FaceDetections fd
    WHERE fd.frame_number % ? = 0 
      AND fd.detection_id NOT IN (SELECT detection_id FROM ExcludedFaceDetections)
    GROUP BY fd.frame_number, fd.video_id;
    """, (SAMPLE_RATE,))    
    
    # ====================================================================
    # STEP 1B: PERSON DETECTION AGGREGATION (FILTERED)
    # ====================================================================
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS PersonAgg AS
    SELECT 
        pc.frame_number, 
        pc.video_id, 
        MAX(pc.confidence_score) AS person_conf, -- Get raw confidence
        1 AS has_person
    FROM PersonDetections pc
    WHERE pc.frame_number % ? = 0
      AND pc.detection_id NOT IN (SELECT detection_id FROM ExcludedPersonDetections)
    GROUP BY pc.frame_number, pc.video_id;
    """, (SAMPLE_RATE,))

    # ====================================================================
    # STEP 1C: BOOK DETECTION AGGREGATION
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
    # STEP 1D: UNIFIED COORDINATE AGGREGATION
    # ====================================================================
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS PresenceCoords AS
    SELECT 
        frame_number, 
        video_id,
        MIN(x_min) AS x_min, 
        MIN(y_min) AS y_min, 
        MAX(x_max) AS x_max, 
        MAX(y_max) AS y_max
    FROM (
        SELECT frame_number, video_id, x_min, y_min, x_max, y_max FROM FaceDetections
        UNION ALL
        SELECT frame_number, video_id, x_min, y_min, x_max, y_max FROM PersonDetections
    )
    GROUP BY frame_number, video_id;
    """)
    
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
        COALESCE(pa.person_conf, 0) AS person_conf,
        COALESCE(fa.face_conf, 0) AS face_conf,
        fa.proximity,
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        COALESCE(ba.has_book, 0) AS has_book,
        MAX(COALESCE(pa.person_conf, 0), COALESCE(fa.face_conf, 0)) AS instant_presence_conf,
        
        COALESCE(pc.x_min, -1) AS last_x_min,
        COALESCE(pc.y_min, -1) AS last_y_min,
        COALESCE(pc.x_max, -1) AS last_x_max,
        COALESCE(pc.y_max, -1) AS last_y_max

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
    LEFT JOIN PresenceCoords pc
        ON fg.frame_number = pc.frame_number AND fg.video_id = pc.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    
    df = pd.read_sql(query, conn, params=query_params)
    
    return df

def calculate_window_features(df: pd.DataFrame, fps: int, sample_rate: int) -> pd.DataFrame:
    """
    Calculates all windowed features in a single vectorized pass.
    
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
    samples_per_sec = fps / sample_rate
    
    # 1. Setup Window Sizes from InferenceConfig
    sustained_window = int(InferenceConfig.SUSTAINED_KCDS_WINDOW_SEC * samples_per_sec)
    avail_window = int(InferenceConfig.PERSON_AVAILABLE_WINDOW_SEC * samples_per_sec)
    audio_window = int(InferenceConfig.PERSON_AUDIO_WINDOW_SEC * samples_per_sec)
    alone_window = int(InferenceConfig.ROBUST_ALONE_WINDOW_SEC * samples_per_sec)
    persistence_window = int(InferenceConfig.VISUAL_PERSISTENCE_SEC * samples_per_sec)
    cooldown_frames = int(InferenceConfig.SOCIAL_COOLDOWN_SEC * samples_per_sec)

    # 2. Presence Score (Confidence Mass)
    df['presence_score'] = df['instant_presence_conf'].rolling(
        window=avail_window, min_periods=1, center=True
    ).mean().fillna(0)

    # 3. INTERACTION HISTORY & DECAY LOGIC
    # Calculate how much interaction happened in the last 60 seconds
    history_window = int(60 * samples_per_sec)
    df['interaction_history_mass'] = df['is_audio_interaction'].rolling(
        window=history_window, min_periods=1
    ).sum() / history_window

    # Calculate frames since the last interaction event
    df['is_not_interacting'] = (~df['is_audio_interaction']).astype(int)
    df['frames_since_interaction'] = df['is_not_interacting'].groupby(
        (df['is_audio_interaction']).cumsum()
    ).cumsum()

    # Calculate History Boost: long interactions (high mass) sustain the state longer
    # Short drive-bys get a 0.2 multiplier; long sessions get 1.0
    df['history_boost'] = (df['interaction_history_mass'] * 5).clip(0.2, 1.0) 
    
    # Linear Decay: 1.0 at start of cooldown, 0.0 at the end
    df['decay_factor'] = (1.0 - (df['frames_since_interaction'] / cooldown_frames)).clip(0, 1)
    df['weighted_cooldown'] = df['decay_factor'] * df['history_boost']

    # 4. EDGE-OF-FRAME TRIPWIRE (Kill Switch)
    edge_margin = getattr(InferenceConfig, 'EDGE_MARGIN', 0.05)
    df['is_at_edge'] = (
        ((df['last_x_min'] >= 0) & (df['last_x_min'] < edge_margin)) | 
        (df['last_x_max'] > (1.0 - edge_margin))
    ).astype(bool)

    # Exit event: person was at edge and instant detection dropped
    df['is_exit_event'] = (df['is_at_edge']) & (df['instant_presence_conf'] == 0)

    # 5. DYNAMIC SOCIAL COOLDOWN
    # Cooldown is active if weighted decay is above threshold and NO exit event
    df['was_interacting_recently'] = (df['weighted_cooldown'] > 0.1) & (~df['is_exit_event'])

    # 6. HYSTERESIS LOGIC
    # Entry threshold is fixed; Exit threshold scales with cooldown strength
    is_high_entry = df['presence_score'] >= InferenceConfig.MIN_PRESENCE_CONFIDENCE_THRESHOLD
    
    standard_exit = InferenceConfig.MIN_PRESENCE_CONFIDENCE_THRESHOLD * InferenceConfig.STANDARD_EXIT_MULTIPLIER
    # Ultra-low exit if there is strong social context (cooldown > 0.5)
    cooldown_exit = InferenceConfig.MIN_PRESENCE_CONFIDENCE_THRESHOLD * InferenceConfig.SOCIAL_COOLDOWN_EXIT_MULTIPLIER
    
    df['effective_exit_threshold'] = np.where(
        df['weighted_cooldown'] > InferenceConfig.SOCIAL_CONTEXT_THRESHOLD, 
        cooldown_exit, 
        standard_exit
    )
    
    is_low_exit = df['presence_score'] >= df['effective_exit_threshold']
    
    # Vectorized Hysteresis
    df['is_robust_person'] = is_high_entry.rolling(
        window=avail_window, min_periods=1, center=True
    ).max().astype(bool) & is_low_exit

    # 7. AUDIOBOOK & VISUAL ANCHOR FOR OHS
    df['ohs_frac_lookback'] = df['has_ohs'].rolling(window=avail_window, min_periods=1).mean()
    df['is_audiobook'] = df['ohs_frac_lookback'] >= InferenceConfig.MAX_OHS_FOR_AVAILABLE

    # Available triggers if OHS is present AND there is a baseline visual floor
    df['is_sustained_ohs'] = (
        (df['ohs_frac_lookback'] >= InferenceConfig.MIN_PRESENCE_OHS_FRACTION) &
        (df['presence_score'] >= InferenceConfig.AUDIO_VISUAL_GATING_FLOOR) & 
        (~df['is_audiobook'])
    )

    # 8. RULE 3: SUSTAINED ADULT SPEECH (KCDS)
    if sustained_window > 0:
        df['kcds_frac_lookback'] = df['has_cds'].rolling(window=sustained_window, min_periods=1).mean()
        df['is_sustained_kcds'] = (df['kcds_frac_lookback'] >= InferenceConfig.SUSTAINED_KCDS_THRESHOLD)
        df['is_sustained_kcds'] = df['is_sustained_kcds'].rolling(
            window=sustained_window, min_periods=1, center=True
        ).max().fillna(False).astype(bool)
    else:
        df['is_sustained_kcds'] = False

    # 9. MEMORY & RULE 4 PREP
    df['is_confident_instant'] = df['instant_presence_conf'] >= InferenceConfig.INSTANT_CONFIDENCE_THRESHOLD
    df['person_seen_recently'] = df['is_confident_instant'].rolling(
        window=max(1, persistence_window), min_periods=1, center=True
    ).max().fillna(0).astype(bool)

    df['recent_kchi_presence'] = df['has_kchi'].rolling(
        window=audio_window, min_periods=1, center=True
    ).max().fillna(0).astype(bool)

    # 10. ROBUST ALONE
    if alone_window > 0:
        df['social_signal_total'] = df['presence_score'] + df['has_cds'] + df['has_ohs']
        df['social_signal_frac'] = df['social_signal_total'].rolling(window=avail_window, min_periods=1).mean() / 3
        df['is_robust_alone'] = df['social_signal_frac'].rolling(
            window=alone_window, min_periods=1, center=True
        ).min().fillna(1) <= InferenceConfig.MAX_ALONE_FALSE_POSITIVE_FRACTION
    else:
        df['is_robust_alone'] = True
    
    return df[['video_id', 'frame_number', 'is_sustained_kcds', 'is_robust_person', 'is_sustained_ohs', 
               'is_robust_alone', 'is_audiobook', 'person_seen_recently', 'presence_score', 
               'instant_presence_conf', 'recent_kchi_presence', 'was_interacting_recently', 'is_exit_event']].copy()

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

def classify_frames(row, included_rules=None):
    """
    Frame-level classification into "Interacting", "Available", or "Alone" based on hierarchical rules.
    
    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing all necessary features for classification.
    included_rules : list, optional
        List of rule numbers to include in the classification. If None, all rules are included.
    """
    if included_rules is None:
        included_rules = [1, 2, 3, 4]
    
    # Permission Gate Multipliers
    interact_thresh = InferenceConfig.MIN_PRESENCE_CONFIDENCE_THRESHOLD * InferenceConfig.INTERACTION_PERMISSION_GATE
    
    is_person_interacting = row['presence_score'] >= interact_thresh
    is_person_available = bool(row['is_robust_person'])
    was_interacting_recently = bool(row['was_interacting_recently'])
    is_visual_confident = row['instant_presence_conf'] >= InferenceConfig.INSTANT_CONFIDENCE_THRESHOLD
    
    rule1_tt = bool(row['is_audio_interaction'])
    rule3_kcds = bool(row['is_sustained_kcds'])
    rule4_kchi_visual = bool(row['recent_kchi_presence']) and is_person_available
    is_book_present = (row['has_book'] == 1)

    # --- HIERARCHY ---
    # TIER 1: INTERACTING
    if 1 in included_rules and rule1_tt:
        interaction_category = 3 if (is_book_present and not is_person_available) else 1 # Alone or Interacting
            
    elif (3 in included_rules and rule3_kcds and is_person_interacting) or \
         (4 in included_rules and rule4_kchi_visual and is_person_interacting) or \
         (2 in included_rules and is_visual_confident and bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD)):
        interaction_category = 1 # Interacting
    elif is_person_available or bool(row['is_sustained_ohs']) or was_interacting_recently:
        interaction_category = 2 # Available
    else:
        interaction_category = 3 # Alone
        
    return (interaction_category, rule1_tt, 
            bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD), 
            rule3_kcds, rule4_kchi_visual)

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
        4: "KCHI + Visual"
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
        
        # --- PHASE 3: HIERARCHICAL CLASSIFICATION ---
        print("⏱️ PHASE 3: Running Initial Classification...")
        results = all_data.apply(lambda row: classify_frames(row, included_rules), axis=1)
        all_data['interaction_type'] = [r[0] for r in results]

        # Update only the classification column
        all_data['interaction_type'] = [result[0] for result in results]
        all_data['rule1_turn_taking'] = [result[1] for result in results]
        all_data['rule2_close_proximity'] = [result[2] for result in results]
        all_data['rule3_kcds_speaking'] = [result[3] for result in results]
        all_data['rule4_kchi_visual'] = [result[4] for result in results]

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
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=kcds-speaking, 4=rule4_kchi_visual). Default: [1, 2, 3, 4]')

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules, hyperparameter_tuning=False)