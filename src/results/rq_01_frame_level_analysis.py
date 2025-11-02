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
from pathlib import Path
from datetime import datetime

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig

# Constants
# Constants
FPS = DataConfig.FPS # frames per second
SAMPLE_RATE = InferenceConfig.SAMPLE_RATE # every n-th frame is processed (configurable sampling rate)

def find_segments(video_df, column_name):
    """
    Identifies continuous audio interaction bouts by chaining segments 
    where the gap between any two adjacent segments (KCHI or CDS) is 
    less than or equal to MAX_TURN_TAKING_GAP_SEC.
    
    The entire chained sequence of KCHI and CDS segments is marked as 
    'is_audio_interaction', provided the intervals contain both KCHI and CDS.

    Parameters:
    ----------
    video_df (pd.DataFrame): 
        DataFrame for a single video.
    column_name (str): 
        The name of the binary column ('has_kchi' or 'has_cds').
        
    Returns:
    ----------
    list of dicts: 
        [{'start': frame_num, 'end': frame_num, 'type': 'kchi'/'kcds'}, ...]
    """
    segments = []

    # If requested binary column doesn't exist, return empty
    if column_name not in video_df.columns:
        return segments

    # Work on a copy to avoid SettingWithCopy warnings and allow index manipulations
    speech_frames = video_df[video_df[column_name] == 1].copy()
    if speech_frames.empty:
        return segments

    # Determine frame numbers: prefer explicit 'frame_number' column, then any column containing 'frame', then the index
    if 'frame_number' in speech_frames.columns:
        frame_numbers = speech_frames['frame_number'].values
    else:
        candidate_cols = [c for c in speech_frames.columns if 'frame' in c.lower()]
        if candidate_cols:
            frame_numbers = speech_frames[candidate_cols[0]].values
        else:
            # If frame_number is the index (common when callers set it), use the index values
            frame_numbers = speech_frames.index.values

    # Ensure frame_numbers is a 1-D numpy array of integers
    try:
        frame_numbers = frame_numbers.astype(int)
    except Exception:
        frame_numbers = pd.Series(frame_numbers).astype(int).values

    # Initialize the start of the first segment
    current_start = int(frame_numbers[0])

    for i in range(1, len(frame_numbers)):
        # Treat gaps larger than SAMPLE_RATE as segment breaks
        if int(frame_numbers[i]) > int(frame_numbers[i-1]) + SAMPLE_RATE:
            segments.append({
                'start': int(current_start),
                'end': int(frame_numbers[i-1]),
                'type': column_name.split('_')[-1]
            })
            current_start = int(frame_numbers[i])

    # Append last segment
    segments.append({
        'start': int(current_start),
        'end': int(frame_numbers[-1]),
        'type': column_name.split('_')[-1]
    })

    return segments

def get_all_analysis_data(conn):
    """
    Multimodal data integration at configurable frame intervals.
    
    ADJUSTED: Uses Videos.max_frame to generate a full temporal grid (FrameGrid) 
    at SAMPLE_RATE intervals. All detection/classification tables are LEFT JOINED 
    onto this dense grid.
    
    Returns:
        pd.DataFrame: Temporally-aligned dataset at SAMPLE_RATE intervals.
    """
    
    # ====================================================================
    # STEP 1A: FACE DETECTION AGGREGATION
    # ====================================================================
    # Aggregate face detections, ensuring they align with the temporal grid.
    
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT 
        frame_number, 
        video_id, 
        proximity,
        1 AS has_face
        
    FROM FaceDetections
    -- Filter FaceDetections to match the SAMPLE_RATE temporal grid
    WHERE 
        frame_number % ? = 0 
        AND confidence_score >= ?        
    GROUP BY frame_number, video_id;
    """, (SAMPLE_RATE, InferenceConfig.FACE_DET_CONFIDENCE_THRESHOLD))
    
    # ====================================================================
    # STEP 1B: PERSON DETECTION AGGREGATION (NEW)
    # ====================================================================
    # Aggregate person detections (single "person" class), aligning with the grid.
    
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS PersonAgg AS
    SELECT 
        frame_number, 
        video_id, 
        1 AS has_person
    FROM PersonClassifications
    WHERE 
        frame_number % ? = 0
        AND confidence_score >= ?
    GROUP BY frame_number, video_id;
    """, (SAMPLE_RATE, InferenceConfig.PERSON_DET_CONFIDENCE_THRESHOLD))
    
    # ====================================================================
    # STEP 2: MAIN DATA INTEGRATION QUERY - GENERATE DENSE FRAME GRID
    # ====================================================================
    
    query = f"""
    -- RECURSIVE CTE to generate the full temporal grid for all videos
    WITH RECURSIVE FrameGrid AS (
        -- Anchor member: Start at frame 0 for every video
        SELECT
            video_id,
            video_name,
            0 AS frame_number,
            max_frame
        FROM Videos
        
        UNION ALL
        
        -- Recursive member: Increment frame_number by SAMPLE_RATE
        SELECT
            v.video_id,
            v.video_name,
            fg.frame_number + {SAMPLE_RATE},
            v.max_frame
        FROM FrameGrid fg
        JOIN Videos v ON fg.video_id = v.video_id
        -- Termination condition: Stop when the next frame exceeds max_frame
        WHERE fg.frame_number + {SAMPLE_RATE} <= v.max_frame
    )
    
    SELECT
        fg.frame_number,
        fg.video_id,
        fg.video_name,
        
        -- PERSON DETECTION MODALITY (LEFT JOIN onto the grid)
        -- Simplified to a single 'has_person' column
        COALESCE(pa.has_person, 0) AS has_person, 
        
        -- AUDIO CLASSIFICATION MODALITY (LEFT JOIN onto the grid)
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        
        -- FACE DETECTION MODALITY (LEFT JOIN onto the grid)
        COALESCE(fa.has_face, 0) AS has_face,
        fa.proximity,
        
        -- MULTIMODAL FUSION FLAGS
        -- person_or_face_present is true if EITHER face OR person detection is present
        CASE 
            WHEN COALESCE(fa.has_face, 0)=1 OR COALESCE(pa.has_person, 0)=1 THEN 1 
            ELSE 0 
        END AS person_or_face_present

    FROM FrameGrid fg
    LEFT JOIN AudioClassifications af 
        ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    LEFT JOIN FaceAgg fa 
        ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
    LEFT JOIN PersonAgg pa
        ON fg.frame_number = pa.frame_number AND fg.video_id = pa.video_id
    ORDER BY fg.video_id, fg.frame_number
    """
    return pd.read_sql(query, conn)


def check_audio_interaction_turn_taking(df, fps):
    """
    Identifies continuous audio interaction bouts based on segment merging.
    
    An interaction bout consists of KCHI and KCDS segments where the gap 
    between any two adjacent segments (regardless of type) is 
    less than or equal to MAX_TURN_TAKING_GAP_SEC.
    
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
        
        # Identify KCHI and KCDS segments and combine them
        kchi_segments = find_segments(video_df, 'has_kchi')
        kcds_segments = find_segments(video_df, 'has_cds')
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            temp = video_df.reset_index()
            all_results.append(temp[['frame_number', 'is_audio_interaction']])
            continue
            
        interaction_windows = []

        # 2. GAP-BASED MERGING (more permissive)
        # Merge adjacent segments if the gap between them is <= MAX_GAP_FRAMES.
        current_segment = all_segments[0]
        current_window_start = current_segment['start']
        current_window_end = current_segment['end']
        segments_in_window = [current_segment]

        for seg in all_segments[1:]:
            gap = seg['start'] - current_window_end
            if gap <= MAX_GAP_FRAMES:
                # extend the current window
                current_window_end = seg['end']
                segments_in_window.append(seg)
            else:
                # finalize current window if it contains both speaker types
                types_in_window = {s['type'] for s in segments_in_window}
                if 'kchi' in types_in_window and 'cds' in types_in_window:
                    interaction_windows.append({
                        'start': current_window_start,
                        'end': current_window_end,
                        'segments': list(segments_in_window)
                    })
                # start a new window
                current_window_start = seg['start']
                current_window_end = seg['end']
                segments_in_window = [seg]

        # finalize last window
        types_in_window = {s['type'] for s in segments_in_window}
        if 'kchi' in types_in_window and 'cds' in types_in_window:
            interaction_windows.append({
                'start': current_window_start,
                'end': current_window_end,
                'segments': list(segments_in_window)
            })

        logging.debug(f"Video {video_id} - all_segments: {all_segments}")
        logging.debug(f"Video {video_id} - interaction_windows: {interaction_windows}")

        # 4. Final Marking on the DataFrame
        for window in interaction_windows:
            video_df.loc[
                window['start'] : window['end'], 
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
    Hierarchical social interaction classifier with dynamic proximity and audio priority.
    Also returns individual rule activations for analysis.
    
    CLASSIFICATION LOGIC:
    1. INTERACTING: Active social engagement (Highest Priority)
    - Turn-taking detected (Rule 1)
    - OR Very close proximity (Rule 2)
    - OR Sustained child-directed speech present (Rule 3)
    - OR Face/Person + recent speech (Rule 4)
    - OR KCHI + Buffered Visual (Rule 5)
    
    2. Available: Passive social presence (Tier 2)
    - OHS is present OR Sustained Person/Face presence is true
    
    3. ALONE: No social presence detected
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row with detection flags and proximity values
    results_df : pd.DataFrame
        The full DataFrame to enable window-based lookups
    included_rules : list, optional
        List of rule numbers to include (1, 2, 3, 4, 5). If None, uses default rules.
        
    Returns
    -------
    tuple
        (interaction_category, rule1_active, rule2_active, rule3_active, rule4_active, rule5_active)
        where interaction_category is str ('Interacting', 'Available', 'Alone')
        and rule flags are boolean
    """
    # Default rules if none specified
    if included_rules is None:
        included_rules = [1, 2, 3, 4, 5]
    
    # Calculate recent speech activity once at the beginning
    current_index = row.name
    
    # --- Lookback Window Setup for Rule 4 (Person + Recent Speech) ---
    window_samples = int(InferenceConfig.PERSON_AUDIO_WINDOW_SEC * FPS / SAMPLE_RATE)
    window_start = max(0, current_index - window_samples)
    
    # Check for recent KCDS in the window based on the original dataframe's index
    recent_speech_exists = (results_df.loc[window_start:current_index, 'has_cds'] == 1).any()
    
    # --- Check Window for Rule 3 (Sustained KCDS, 3 seconds) ---
    window_samples_rule3 = int(InferenceConfig.SUSTAINED_KCDS_SEC * FPS / SAMPLE_RATE)
    window_start_rule3 = max(0, current_index - window_samples_rule3 + 1)
    
    if row['has_cds'] == 1:
        # Check if the sum of 'has_cds' in the window equals the size of the window (all frames must be 1)
        kcds_window_data = results_df.loc[window_start_rule3 : current_index, 'has_cds']
        is_sustained_kcds = (kcds_window_data.sum() == len(kcds_window_data))
    else:
        is_sustained_kcds = False

    # Check for person/face presence and OHS
    person_or_face_present_instant = (row['person_or_face_present'] == 1)
    has_ohs = (row['has_ohs'] == 1)

    # --- Sustained Person/Face Presence for 'Available' (Tier 2 Visual Check) ---
    window_samples_person = int(InferenceConfig.PERSON_AVAILABLE_WINDOW_SEC * FPS / SAMPLE_RATE)
    window_start_person = max(0, current_index - window_samples_person + 1)
    
    window_data_person = results_df.loc[window_start_person : current_index]
    window_size = len(window_data_person)
    
    if window_size > 0:
        person_count_in_window = (window_data_person['person_or_face_present'] == 1).sum()
        presence_fraction = person_count_in_window / window_size
        is_sustained_person_or_face_present = presence_fraction >= InferenceConfig.MIN_PRESENCE_FRACTION
    else:
        is_sustained_person_or_face_present = False

    # --- NEW RULE 5: Buffered KCHI + Visual Presence (30 frame buffer) ---
    BUFFER_SAMPLES = int(InferenceConfig.KCHI_PERSON_BUFFER_FRAMES / SAMPLE_RATE)
    
    is_kchi = (row['has_kchi'] == 1)
    
    rule5_buffered_kchi = False
    if is_kchi:
        # Define window indices relative to the current index
        start_buffer = max(0, current_index - BUFFER_SAMPLES)
        end_buffer = min(len(results_df) - 1, current_index + BUFFER_SAMPLES)
        
        # Check if *any* person/face detection exists in the surrounding buffer
        if end_buffer >= start_buffer:
            visual_in_buffer = (
                results_df.loc[start_buffer:end_buffer, 'person_or_face_present'] == 1
            ).any()
            rule5_buffered_kchi = visual_in_buffer
    
    # Evaluate all rules and track their activation
    rule1_turn_taking = bool(row['is_audio_interaction'])
    rule2_close_proximity = bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD) if pd.notna(row['proximity']) else False
    
    # Activated only for sustained KCDS
    rule3_kcds_speaking = is_sustained_kcds
    
    # Rule 4: Person (Face) present + recent speech (uses the instantaneous fused flag)
    rule4_person_recent_speech = bool(person_or_face_present_instant and recent_speech_exists)
    
    # Tier 1: INTERACTING (Active engagement) - check only included rules
    active_rules = []
    
    if 1 in included_rules and rule1_turn_taking:
        active_rules.append(1)
    
    if 2 in included_rules and rule2_close_proximity:
        active_rules.append(2)

    if 3 in included_rules and rule3_kcds_speaking:
        active_rules.append(3)
    
    if 4 in included_rules and rule4_person_recent_speech:
        active_rules.append(4)

    if 5 in included_rules and rule5_buffered_kchi:
        active_rules.append(5)

    # Determine interaction category
    if active_rules:
        interaction_category = "Interacting"
    # Trigger 'Available' if (OHS is heard) OR (Sustained Person/Face Presence is true)
    elif has_ohs or is_sustained_person_or_face_present: 
        interaction_category = "Available"
    else:
        interaction_category = "Alone"
    
    return (
        interaction_category, 
        rule1_turn_taking, 
        rule2_close_proximity, 
        rule3_kcds_speaking, 
        rule4_person_recent_speech, 
        rule5_buffered_kchi
    )

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

def merge_age_information(df):
    """
    Merge age information from subjects CSV into the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Frame-level data with video_name column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with age information merged
    """    
    try:
        subjects_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
        
        # Merge age information based on video_name
        df_with_age = df.merge(
            subjects_df[['video_name', 'age_at_recording', 'child_id']], 
            on='video_name', 
            how='left'
        )
        
        # Check merge success
        missing_age_count = df_with_age['age_at_recording'].isna().sum()
        if missing_age_count > 0:
            print(f"⚠️ Warning: {missing_age_count} frames missing age data ({missing_age_count/len(df_with_age)*100:.1f}%)")
            
            # Show some examples of unmatched video names
            unmatched_videos = df_with_age[df_with_age['age_at_recording'].isna()]['video_name'].unique()[:5]
            print(f"Examples of unmatched video names: {list(unmatched_videos)}")

        # Reorder columns to put age near the beginning
        cols = df_with_age.columns.tolist()
        new_order = ['frame_number', 'video_id', 'video_name', 'child_id', 'age_at_recording'] + [col for col in cols if col not in ['frame_number', 'video_id', 'video_name', 'age_at_recording', 'child_id']]
        df_with_age = df_with_age[new_order]
        
        return df_with_age
        
    except FileNotFoundError:
        print(f"⚠️ Warning: Subjects CSV not found at {DataPaths.SUBJECTS_CSV_PATH}")
        print("Proceeding without age information")
        return df
    except Exception as e:
        print(f"⚠️ Warning: Error loading subjects data: {e}")
        print("Proceeding without age information")
        return df

def main(db_path: Path, output_dir: Path, included_rules: list = None):
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
    included_rules : list, optional
        List of rule numbers to include in interaction classification (1, 2, 3, 4).
        If None, uses default rules [2, 3, 4].

    Returns
    -------
    dict: 
        Summary statistics including interaction and presence distributions
    """
    if included_rules is None:
        included_rules = [1, 2, 3, 4, 5]

    # Print which rules are being used
    rule_names = {
        1: "Turn-Taking (KCHI + KCDS)",
        2: "Very Close Proximity",
        3: "KCDS Present", 
        4: "Face/Person + Recent KCDS",
        5: "Buffered KCHI + Visual"
    }

    print("🔄 Running comprehensive multimodal social interaction analysis...")
    print(f"📋 Using interaction rules: {[f'{i}: {rule_names[i]}' for i in included_rules]}")

    # ------------------------------------------------------------------
    # 🕒 Create timestamped output folder inside base output directory
    # ------------------------------------------------------------------
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
    with sqlite3.connect(db_path) as conn:
        all_data = get_all_analysis_data(conn)
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, FPS)
        
        classification_results = all_data.apply(
            lambda row: classify_frames(row, all_data, included_rules), axis=1
        )

        all_data['interaction_type'] = [result[0] for result in classification_results]
        all_data['rule1_turn_taking'] = [result[1] for result in classification_results]
        all_data['rule2_close_proximity'] = [result[2] for result in classification_results]
        all_data['rule3_kcds_speaking'] = [result[3] for result in classification_results]
        all_data['rule4_person_recent_speech'] = [result[4] for result in classification_results]
        all_data['rule5_buffered_kchi'] = [result[5] for result in classification_results] # ADDED 5

        # Categorization
        all_data['face_frame_category'] = all_data.apply(classify_face_category, axis=1)
        all_data['person_frame_category'] = all_data.apply(classify_person_category, axis=1)
        all_data['fused_frame_category'] = all_data.apply(classify_fused_category, axis=1)

        # Merge age info
        all_data = merge_age_information(all_data)

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
    parser.add_argument('--rules', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=cds-speaking, 4=adult-face-recent-speech, 5=buffered-kchi-visual). Default: [1, 2, 3, 4, 5]') # UPDATED HELP TEXT

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4, 5]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules)