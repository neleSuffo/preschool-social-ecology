# Research Question 1. How much time do children spend alone?
#
# This script analyzes frame-level data to determine the amount of time children spend alone.
# It uses a SQLite database to query the relevant data.
# The analysis focuses on identifying frames where children are not in the presence of adults.

import sqlite3
import argparse
import sys
import pandas as pd
from pathlib import Path

# Get the src directory (2 levels up from current notebook location)
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference
from config import DataConfig, InferenceConfig

# Constants
FPS = DataConfig.FPS # frames per second

def get_all_analysis_data(conn):
    """
    Comprehensive multimodal data integration query.
    
    This function creates a unified dataset by combining three detection modalities:
    
    1. FACE AGGREGATION (FaceAgg temp table):
    - Aggregates multiple face detections per frame into binary presence flags
    - Computes proximity values (0=far, 1=close) for social distance analysis
    - Groups by frame to handle multiple faces of same age class
    
    2. VOCALIZATION PROCESSING (VocalizationFrames temp table):
    - Converts time-based vocalizations to frame-based mapping
    - Implements KCHI priority: child speech overrides adult speech in same frame
    - Handles overlapping speech segments by prioritizing child vocalizations
    - Time conversion: seconds → frames with 30fps and 10-frame rounding
    
    3. MAIN DATA INTEGRATION:
    - Joins person classifications with face detections and vocalizations
    - Creates combined presence flags (person OR face = presence)
    - Preserves frame-level granularity for temporal analysis
    
    Returns:
        pd.DataFrame: Frame-level dataset with all modalities integrated
                    Columns: frame_number, video_id, person flags, face flags, 
                            proximity, combined presence, speaker
    """
    # ====================================================================
    # STEP 1: FACE DETECTION AGGREGATION
    # ====================================================================
    # Purpose: Convert multiple face detections per frame into binary flags
    # 
    # Input: FaceDetections table with individual face detections
    #        - Each row = one face detection with age_class (0=child, 1=adult)
    #        - Multiple faces possible per frame
    #        - Proximity values indicate social distance (0=far, 1=close)
    #
    # Output: FaceAgg temp table with frame-level aggregation
    #         - One row per frame
    #         - Binary flags: has_child_face, has_adult_face
    #         - Single proximity value per frame (from any detected face)
    #
    # Logic: MAX() function converts any face detection to 1, else 0
    #        Groups by frame to collapse multiple detections
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT 
        frame_number, 
        video_id, 
        proximity,
        MAX(CASE WHEN age_class = 0 THEN 1 ELSE 0 END) AS has_child_face,
        MAX(CASE WHEN age_class = 1 THEN 1 ELSE 0 END) AS has_adult_face
    FROM FaceDetections
    GROUP BY frame_number, video_id;
    """)
    
    # ====================================================================
    # STEP 2: VOCALIZATION PROCESSING WITH PRIORITY HANDLING
    # ====================================================================
    # Purpose: Map time-based vocalizations to frame-based data with conflict resolution
    #
    # Input: Vocalizations table with time intervals
    #        - start_time_seconds, end_time_seconds: continuous time intervals
    #        - speaker: 'KCHI' (child) or other speakers (adult/unknown)
    #        - Overlapping intervals possible (child and adult speaking simultaneously)
    #
    # Processing Steps:
    #   1. Convert time intervals to frame ranges using 30fps
    #   2. Round to 10-frame intervals (ROUND_TO=10) for data consistency
    #   3. Expand intervals into individual frame assignments
    #   4. Resolve conflicts using KCHI priority rule
    #
    # Conflict Resolution Logic:
    #   - If KCHI and other speakers detected in same frame → assign KCHI
    #   - If only other speakers detected → assign that speaker
    #   - If no speakers detected → NULL
    #
    # Time-to-Frame Conversion Formula:
    #   frame = ROUND(time_seconds * 30 / 10) * 10
    #   Example: 5.3 seconds → frame 160 (5.3 * 30 = 159 → round to 160)
    #
    # Output: VocalizationFrames temp table
    #         - One row per frame with speech
    #         - Single speaker per frame (conflicts resolved)
    #         - Consistent with PersonClassifications frame numbering
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS VocalizationFrames AS
    SELECT 
        video_id,
        frame_number,
        REPLACE(GROUP_CONCAT(DISTINCT speaker), ',', ';') AS speaker
    FROM (
        SELECT 
            v.video_id,
            v.speaker,
            pf.frame_number
        FROM Vocalizations v
        JOIN PersonClassifications pf ON v.video_id = pf.video_id
        WHERE pf.frame_number BETWEEN 
            CAST(ROUND(v.start_time_seconds * 30 / 10) * 10 AS INTEGER) AND 
            CAST(ROUND(v.end_time_seconds * 30 / 10) * 10 AS INTEGER)
    )
    GROUP BY video_id, frame_number;
    """)
    
    # ====================================================================
    # STEP 3: MAIN DATA INTEGRATION QUERY
    # ====================================================================
    # Purpose: Combine all modalities into a unified frame-level dataset
    #
    # Data Sources:
    #   1. PersonClassifications (pd) - PRIMARY: person body detections per frame
    #   2. FaceAgg (fa) - SECONDARY: aggregated face detections with proximity
    #   3. VocalizationFrames (vf) - TERTIARY: speech activity per frame
    #   4. Videos (v) - METADATA: video information and naming
    #
    # Join Strategy:
    #   - LEFT JOIN ensures all PersonClassifications frames are preserved
    #   - Missing face/speech data filled with NULL/0 values using COALESCE
    #   - Frame and video IDs used as joint keys for temporal alignment
    #   - Videos table joined to provide video metadata (name, etc.)
    #
    # Combined Presence Logic:
    #   - child_present = 1 IF (child face detected OR child person detected)
    #   - adult_present = 1 IF (adult face detected OR adult person detected)
    #   - OR logic maximizes detection sensitivity across modalities
    #
    # Output Columns:
    #   - Identifiers: frame_number, video_id, video_name
    #   - Person flags: has_child_person, has_adult_person (original)
    #   - Face flags: has_child_face, has_adult_face (aggregated)
    #   - Social distance: proximity (0=far, 1=close, NULL=no faces)
    #   - Combined flags: child_present, adult_present (multimodal fusion)
    #   - Speech activity: speaker ('KCHI', other, or NULL)
    #
    # Temporal Granularity:
    #   - Frame-level analysis enables precise temporal segmentation
    #   - 10-frame intervals (1/3 second) balance precision with computational efficiency
    #   - Consistent timestamps across all modalities for synchronized analysis
    query = """
    SELECT
        pd.frame_number,
        pd.video_id,
        v.video_name,                -- Video metadata: human-readable video name
        
        -- PERSON DETECTION MODALITY
        -- Raw person body detections from YOLO model
        pd.has_child_person,        -- Binary: child body detected in frame
        pd.has_adult_person,        -- Binary: adult body detected in frame
        
        -- FACE DETECTION MODALITY  
        -- Aggregated face detections with social distance measures
        COALESCE(fa.has_child_face, 0) AS has_child_face,    -- Binary: any child face
        COALESCE(fa.has_adult_face, 0) AS has_adult_face,    -- Binary: any adult face
        fa.proximity,                                        -- Float: social distance (0=far, 1=close)
        
        -- MULTIMODAL FUSION FLAGS
        -- Combined presence indicators using OR logic across modalities
        CASE WHEN COALESCE(fa.has_child_face,0)=1 OR pd.has_child_person=1 THEN 1 ELSE 0 END AS child_present,
        CASE WHEN COALESCE(fa.has_adult_face,0)=1 OR pd.has_adult_person=1 THEN 1 ELSE 0 END AS adult_present,
        
        -- VOCALIZATION MODALITY
        -- Speech activity with conflict resolution (KCHI priority)
        vf.speaker                   -- String: 'KCHI', other speaker, or NULL
    FROM PersonClassifications pd
    LEFT JOIN FaceAgg fa ON pd.frame_number = fa.frame_number AND pd.video_id = fa.video_id
    LEFT JOIN VocalizationFrames vf ON pd.frame_number = vf.frame_number AND pd.video_id = vf.video_id
    LEFT JOIN Videos v ON pd.video_id = v.video_id
    """
    return pd.read_sql(query, conn)

def check_audio_interaction_turn_taking(df, fps, base_window_sec, extended_window_sec):
    """
    Adaptive turn-taking detection with temporal gap constraint:
    - Uses smaller window (base_window_sec) when child is alone
    - Expands window (extended_window_sec) if a person or face is visible
    - Only considers it turn-taking if gap between KCHI and other speech is ≤ 5 seconds

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'speaker', 'child_present', 'adult_present']
    fps : int
        Frames per second
    base_window_sec : int
        Base window size in seconds (when child is alone)
    extended_window_sec : int
        Extended window size in seconds (when someone else is present)
    """
    base_window_frames = base_window_sec * fps
    extended_window_frames = extended_window_sec * fps

    df_copy = df.copy()   
    df_copy['has_kchi'] = df_copy['speaker'].str.contains('KCHI', na=False).astype(int)
    df_copy['has_other'] = df_copy['speaker'].str.contains('FEM_MAL', na=False).astype(int)

    all_results = []

    for video_id, video_df in df_copy.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').reset_index(drop=True)
        interaction_flags = []

        for _, row in video_df.iterrows():
            current_frame = row['frame_number']

            # Check initial small window for person/face
            half_base_window = base_window_frames // 2
            base_start = current_frame - half_base_window
            base_end = current_frame + half_base_window

            base_window_data = video_df[
                (video_df['frame_number'] >= base_start) &
                (video_df['frame_number'] <= base_end)
            ]

            # Check for ANY person (child or adult) present
            person_present_in_base = (base_window_data['child_present'].sum() > 0) or (base_window_data['adult_present'].sum() > 0)
            # Check for ANY face (child or adult) present
            face_present_in_base = (base_window_data['has_child_face'].sum() > 0) or (base_window_data['has_adult_face'].sum() > 0)

            # Decide window size based on context
            if person_present_in_base or face_present_in_base:
                # Use extended window
                half_window = extended_window_frames // 2
            else:
                # Use base window
                half_window = half_base_window

            window_start = current_frame - half_window
            window_end = current_frame + half_window

            # Get actual window data
            window_data = video_df[
                (video_df['frame_number'] >= window_start) &
                (video_df['frame_number'] <= window_end)
            ]

            # Check for interaction with gap constraint
            has_kchi_in_window = window_data['has_kchi'].sum() > 0
            has_other_in_window = window_data['has_other'].sum() > 0
            
            # Only consider it turn-taking if both speakers present AND gap <= 5 seconds
            is_interaction = False
            if has_kchi_in_window and has_other_in_window:
                # Find frames with KCHI and other speech
                kchi_frames = window_data[window_data['has_kchi'] == 1]['frame_number'].values
                other_frames = window_data[window_data['has_other'] == 1]['frame_number'].values
                
                if len(kchi_frames) > 0 and len(other_frames) > 0:
                    # Calculate minimum gap between any KCHI and other speech frames
                    min_gap_frames = float('inf')
                    for kchi_frame in kchi_frames:
                        for other_frame in other_frames:
                            gap_frames = abs(kchi_frame - other_frame)
                            min_gap_frames = min(min_gap_frames, gap_frames)
                    
                    # Convert gap from frames to seconds (5 seconds = 5 * fps frames)
                    max_gap_frames = InferenceConfig.MAX_TURN_TAKING_GAP_SEC * fps
                    is_interaction = min_gap_frames <= max_gap_frames
            
            interaction_flags.append(is_interaction)

        video_df['is_audio_interaction'] = interaction_flags
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])

    # Combine and merge back
    result_df = pd.concat(all_results, ignore_index=True)
    df_with_interaction = df.merge(result_df, on=['frame_number', 'video_id'], how='left')
    df_with_interaction['is_audio_interaction'] = df_with_interaction['is_audio_interaction'].fillna(False)

    return df_with_interaction['is_audio_interaction']

def classify_interaction_with_audio(row, results_df, included_rules=None):
    """
    Hierarchical social interaction classifier with dynamic proximity and audio priority.
    
    CLASSIFICATION LOGIC:
    1. INTERACTING: Active social engagement
    - Turn-taking audio interaction detected (highest priority)
    - OR Very close proximity (>= PROXIMITY_THRESHOLD)
    - OR other person speaking
    - OR an adult face (proximity doesn't matter) + recent speech
    
    2. CO-PRESENT SILENT: Passive social presence
    - Person detected but no active interaction indicators
    
    3. ALONE: No social presence detected
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row with detection flags and proximity values
    results_df : pd.DataFrame
        The full DataFrame to enable window-based lookups
    included_rules : list, optional
        List of rule numbers to include (1, 2, 3, 4). If None, uses default rules.
        
    Returns
    -------
    str
        Interaction category ('Interacting', 'Co-present Silent', 'Alone')
    """
    # Default rules if none specified (currently excluding rule 1 - turn taking)
    if included_rules is None:
        included_rules = [1, 2, 3, 4]  # Default: turn taking, proximity, other speaking, adult face + recent speech
    
    # Calculate recent proximity once at the beginning
    current_index = row.name
    window_start = max(0, current_index - InferenceConfig.PERSON_AUDIO_WINDOW_SEC)
    recent_speech_exists = (results_df.loc[window_start:current_index, 'fem_mal_speech_present'] == 1).any()

    # Check if a person is present at all, using the combined flags
    person_is_present = (row['child_present'] == 1) or (row['adult_present'] == 1)
    
    # Tier 1: INTERACTING (Active engagement) - evaluate only included rules
    active_rules = []
    
    # Rule 1: Turn-taking audio interaction (highest priority)
    if 1 in included_rules and row['is_audio_interaction']:
        active_rules.append(1)
    
    # Rule 2: Very close proximity (>= PROXIMITY_THRESHOLD)
    if 2 in included_rules and (row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD):
        active_rules.append(2)
    
    # Rule 3: Other person speaking
    if 3 in included_rules and row['fem_mal_speech_present']:
        active_rules.append(3)
    
    # Rule 4: Adult face + recent speech
    if 4 in included_rules and (row['has_adult_face'] == 1 and recent_speech_exists):
        active_rules.append(4)

    if active_rules:
        return "Interacting"

    # Tier 2: CO-PRESENT SILENT (Passive presence)
    if person_is_present:
        return "Co-present Silent"

    # Tier 3: ALONE (No presence)
    return "Alone"

def classify_interaction_rules(row, results_df):
    """
    Classify which specific interaction rules are active for each frame.
    
    This function evaluates each of the four interaction rules separately
    to enable analysis of which rules are most commonly triggering interactions.
    
    Rules:
    1. Turn-taking audio interaction (highest priority)
    2. Very close proximity (>= PROXIMITY_THRESHOLD)
    3. Other person speaking
    4. Adult face + recent speech
    
    Parameters
    ----------
    row : pd.Series
        DataFrame row with detection flags and proximity values
    results_df : pd.DataFrame
        The full DataFrame to enable window-based lookups for recent speech
        
    Returns
    -------
    tuple
        (rule1_active, rule2_active, rule3_active, rule4_active) - all boolean
    """
    # Calculate recent speech for rule 4
    current_index = row.name
    window_start = max(0, current_index - InferenceConfig.PERSON_AUDIO_WINDOW_SEC)
    recent_speech_exists = (results_df.loc[window_start:current_index, 'fem_mal_speech_present'] == 1).any()
    
    # Rule 1: Turn-taking audio interaction detected (highest priority)
    rule1_turn_taking = bool(row['is_audio_interaction'])
    
    # Rule 2: Very close proximity (>= PROXIMITY_THRESHOLD)
    rule2_close_proximity = bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD) if pd.notna(row['proximity']) else False
    
    # Rule 3: Other person speaking
    rule3_other_speaking = bool(row['fem_mal_speech_present'])
    
    # Rule 4: Adult face + recent speech
    rule4_adult_face_recent_speech = bool(row['has_adult_face'] == 1 and recent_speech_exists)
    
    return rule1_turn_taking, rule2_close_proximity, rule3_other_speaking, rule4_adult_face_recent_speech

def classify_face_category(row):
    """Categorize face detection patterns for attention analysis."""
    if row['has_child_face'] and not row['has_adult_face']:
        return 'only_child'
    elif row['has_adult_face'] and not row['has_child_face']:
        return 'only_adult'
    elif row['has_child_face'] and row['has_adult_face']:
        return 'both_faces'
    else:
        return 'no_faces'

def classify_person_category(row):
    """Categorize person detection patterns for presence analysis."""
    if row['has_child_person'] and not row['has_adult_person']:
        return 'only_child'
    elif row['has_adult_person'] and not row['has_child_person']:
        return 'only_adult'
    elif row['has_child_person'] and row['has_adult_person']:
        return 'both_persons'
    else:
        return 'no_persons'

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
    
    1. Data integration (multimodal query)
    2. Audio turn-taking analysis
    3. Speech feature extraction
    4. Social interaction classification
    5. Presence pattern categorization
    6. Age information merging
    7. Results export
    
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
        included_rules = [1, 2, 3, 4]  # Default: include all rules

    # Print which rules are being used
    rule_names = {
        1: "Turn-taking audio interaction",
        2: "Very close proximity",
        3: "Other person speaking", 
        4: "Adult face + recent speech"
    }
    
    print("🔄 Running comprehensive multimodal social interaction analysis...")
    print(f"📋 Using interaction rules: {[f'{i}: {rule_names[i]}' for i in included_rules]}")
    
    with sqlite3.connect(db_path) as conn:
        # Step 1: Data integration
        all_data = get_all_analysis_data(conn)

        # Step 2: Audio turn-taking analysis
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(
            all_data, FPS, InferenceConfig.TURN_TAKING_BASE_WINDOW_SEC, InferenceConfig.TURN_TAKING_EXT_WINDOW_SEC
        )

        # Step 3: Speech feature extraction        
        for cls in InferenceConfig.SPEECH_CLASSES:
            col_name = f"{cls.lower()}_speech_present"
            all_data[col_name] = all_data['speaker'].str.contains(cls, na=False).astype(int)
        
        # Step 4: Social interaction classification with specified rules
        all_data['interaction_type'] = all_data.apply(
            lambda row: classify_interaction_with_audio(row, all_data, included_rules), axis=1
        )
        
        # Step 4b: Interaction rule analysis - track which specific rules are active
        print("🔍 Analyzing interaction rule patterns...")
        rule_results = all_data.apply(
            lambda row: classify_interaction_rules(row, all_data), axis=1
        )
        
        # Unpack the rule results into separate columns
        all_data['rule1_turn_taking'] = [result[0] for result in rule_results]
        all_data['rule2_close_proximity'] = [result[1] for result in rule_results]
        all_data['rule3_other_speaking'] = [result[2] for result in rule_results]
        all_data['rule4_adult_face_recent_speech'] = [result[3] for result in rule_results]

        # Step 5: Presence pattern categorization       
        all_data['face_frame_category'] = all_data.apply(classify_face_category, axis=1)
        all_data['person_frame_category'] = all_data.apply(classify_person_category, axis=1)

        # Step 7: Merge age information
        all_data = merge_age_information(all_data)
                
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 8: Save results with information about which rules were included e.g. 01_frame_level_social_interactions_1_2_3_4.csv
        file_name = Inference.FRAME_LEVEL_INTERACTIONS_CSV.name + f"_{'_'.join(map(str, included_rules))}.csv"  
        all_data.to_csv(output_dir / file_name, index=False)

        print(f"✅ Saved detailed frame-level analysis to {output_dir / file_name}")
        
        return all_data

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Frame-level social interaction analysis')
    parser.add_argument('--rules', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=other-speaking, 4=adult-face-recent-speech). Default: [1, 2, 3, 4]')

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules)