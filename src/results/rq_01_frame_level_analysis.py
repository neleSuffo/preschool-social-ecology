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
SAMPLE_RATE = InferenceConfig.SAMPLE_RATE # every n-th frame is processed (configurable sampling rate)

def get_all_analysis_data(conn):
    """
    Multimodal data integration at configurable frame intervals.
    
    This function creates a unified dataset by combining three detection modalities:
    
    1. FACE AGGREGATION (FaceAgg temp table):
    - Aggregates multiple face detections per frame into binary presence flags
    - Computes proximity values (0=far, 1=close) for social distance analysis
    
    2. PERSON CLASSIFICATION (PRIMARY temporal grid):
    - Uses PersonClassifications as PRIMARY table (every sample rate-th frame)
    
    3. AUDIO SAMPLING (filtered to match person frames):
    - Samples AudioClassifications to every sample rate-th frame only
    
    4. MAIN DATA INTEGRATION:
    - PersonClassifications defines temporal grid (every sample rate-th frame)
    - LEFT JOINs audio and face data to matching frames
    - Creates combined presence flags (person OR face = presence)
    - Optimizes processing efficiency with aligned temporal sampling
    
    Returns:
        pd.DataFrame: Temporally-aligned dataset at sample rate-frame intervals
                    Columns: frame_number, video_id, person flags, audio flags, 
                            face flags, proximity, combined presence
                    
    Temporal Resolution:
        - Every {SAMPLE_RATE}-th frame = every {SAMPLE_RATE/FPS:.2f} second at FPS
        - Efficient for social interaction analysis
        - Aligned with person classification temporal sampling
    """
    # ====================================================================
    # STEP 1: FACE DETECTION AGGREGATION
    # ====================================================================
    # Purpose: Convert multiple face detections per frame into binary flags

    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT 
        frame_number, 
        video_id, 
        MAX(CASE WHEN age_class IN (0,1) THEN 1 ELSE 0 END) AS has_face
    FROM FaceDetections
    GROUP BY frame_number, video_id;
    """)
    
    # ====================================================================
    # STEP 2: DIRECT AUDIO INTEGRATION APPROACH
    # ====================================================================
    # Purpose: Use AudioClassifications table directly as primary data source
    
    # ====================================================================
    # STEP 3: MAIN DATA INTEGRATION QUERY - OPTIMIZED FOR CONFIGURABLE FRAME ALIGNMENT
    # ====================================================================
    # Purpose: Combine all modalities into a unified frame-level dataset
    # 
    # Data Sources:
    #   1. PersonClassifications (pd) - PRIMARY: every {SAMPLE_RATE}-th frame, defines temporal grid
    #   2. AudioClassifications (af) - SECONDARY: filter to matching frames only  
    #   3. FaceAgg (fa) - TERTIARY: aggregated face detections with proximity
    #   4. Videos (v) - METADATA: video information and naming
    #
    # Join Strategy:
    #   - PersonClassifications as PRIMARY defines the {SAMPLE_RATE}-frame temporal grid
    #   - LEFT JOIN AudioClassifications WHERE frame matches (efficient filtering)
    #   - LEFT JOIN FaceAgg on same temporal grid    
    query = f"""
    SELECT
        pd.frame_number,
        pd.video_id,
        v.video_name,
        
        -- PERSON DETECTION MODALITY
        -- Raw person body detections from YOLO + BiLSTM model for every SAMPLE_RATE-th frame
        pd.has_child_person,         -- Binary: child body detected
        pd.has_adult_person,         -- Binary: adult body detected
        
        -- AUDIO CLASSIFICATION MODALITY (sampled to match person frames)
        -- Frame-level audio classifications filtered to every SAMPLE_RATE-th frame
        COALESCE(af.has_kchi, 0) AS has_kchi,        -- Binary: child speech detected
        COALESCE(af.has_ohs, 0) AS has_ohs,          -- Binary: other human speech detected  
        COALESCE(af.has_cds, 0) AS has_kcds          -- Binary: key child-directed speech detected

        -- FACE DETECTION MODALITY
        -- Aggregated face detections with social distance measures
        ,COALESCE(fa.has_face, 0) AS has_face    -- Binary: face present

        -- MULTIMODAL FUSION FLAGS
        -- Combined presence indicators using OR logic across modalities
        ,CASE WHEN COALESCE(fa.has_face,0)=1 OR pd.has_child_person=1 OR pd.has_adult_person=1 THEN 1 ELSE 0 END AS person_present

    FROM PersonClassifications pd
    LEFT JOIN AudioClassifications af ON pd.frame_number = af.frame_number AND pd.video_id = af.video_id
    LEFT JOIN FaceAgg fa ON pd.frame_number = fa.frame_number AND pd.video_id = fa.video_id
    LEFT JOIN Videos v ON pd.video_id = v.video_id
    ORDER BY pd.video_id, pd.frame_number
    """
    return pd.read_sql(query, conn)

def check_audio_interaction_turn_taking(df, fps, base_window_sec, extended_window_sec):
    """
    Adaptive turn-taking detection with temporal gap constraint:
    - Uses smaller window (base_window_sec) when child is alone
    - Expands window (extended_window_sec) if a person or face is visible
    - Only considers it turn-taking if gap between KCHI and KCDS is ≤ 5 seconds

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ['video_id', 'frame_number', 'has_kchi', 'has_kcds', 'child_present', 'adult_present']
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
    # Use the new audio classification columns
    df_copy['has_kchi'] = df_copy['has_kchi'].astype(int)  # Child speech
    df_copy['has_kcds'] = df_copy['has_kcds'].astype(int)    # Key Child-directed speech

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

            # Check for ANY person presence
            person_present_in_base = (base_window_data['person_present'].sum() > 0) or (base_window_data['has_face'].sum() > 0)

            # Decide window size based on context
            if person_present_in_base:
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
            has_kcds_in_window = window_data['has_kcds'].sum() > 0
            
            # Only consider it turn-taking if both KCHI and KCDS present AND gap <= 5 seconds
            is_interaction = False
            if has_kchi_in_window and has_kcds_in_window:
                # Find frames with KCHI and KCDS
                kchi_frames = window_data[window_data['has_kchi'] == 1]['frame_number'].values
                kcds_frames = window_data[window_data['has_kcds'] == 1]['frame_number'].values

                if len(kchi_frames) > 0 and len(kcds_frames) > 0:
                    # Calculate minimum gap between any KCHI and KCDS frames
                    min_gap_frames = float('inf')
                    for kchi_frame in kchi_frames:
                        for cds_frame in kcds_frames:
                            gap_frames = abs(kchi_frame - cds_frame)
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

def classify_frames(row, results_df, included_rules=None):
    """
    Hierarchical social interaction classifier with dynamic proximity and audio priority.
    Also returns individual rule activations for analysis.
    
    CLASSIFICATION LOGIC:
    1. INTERACTING: Active social engagement
    - Turn-taking detected (highest priority)
    - OR Very close proximity (>= PROXIMITY_THRESHOLD)
    - OR Key child-directed speech present
    - OR a Person or Face (proximity doesn't matter) + recent speech
    
    2. Available: Passive social presence
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
    tuple
        (interaction_category, rule1_active, rule2_active, rule3_active, rule4_active)
        where interaction_category is str ('Interacting', 'Available', 'Alone')
        and rule flags are boolean
    """
    # Default rules if none specified
    if included_rules is None:
        included_rules = [1, 2, 3, 4]  # Default: all rules
    
    # Calculate recent speech activity once at the beginning
    current_index = row.name
    window_start = max(0, current_index - InferenceConfig.PERSON_AUDIO_WINDOW_SEC)
    recent_speech_exists = (results_df.loc[window_start:current_index, 'has_kcds'] == 1).any()
    person_present = (row['person_present'] == 1)
    
    # Evaluate all rules and track their activation
    rule1_turn_taking = bool(row['is_audio_interaction'])
    rule2_close_proximity = bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD) if pd.notna(row['proximity']) else False
    rule3_kcds_speaking = bool(row['has_kcds'])
    rule4_person_recent_speech = bool(person_present and recent_speech_exists)
    
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

    # Determine interaction category
    if active_rules:
        interaction_category = "Interacting"
    elif person_present:
        interaction_category = "Available"
    else:
        interaction_category = "Alone"
    
    return interaction_category, rule1_turn_taking, rule2_close_proximity, rule3_kcds_speaking, rule4_person_recent_speech

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
        included_rules = [1, 2, 3, 4]  # Default: include all rules

    # Print which rules are being used
    rule_names = {
        1: "Turn-Taking (KCHI + KCDS)",
        2: "Very Close Proximity",
        3: "KCDS Present", 
        4: "Face/Person + Recent KCDS"
    }
    
    print("🔄 Running comprehensive multimodal social interaction analysis...")
    print(f"📋 Using interaction rules: {[f'{i}: {rule_names[i]}' for i in included_rules]}")
    
    with sqlite3.connect(db_path) as conn:
        # Data integration
        all_data = get_all_analysis_data(conn)

        # Audio turn-taking analysis
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(
            all_data, FPS, InferenceConfig.TURN_TAKING_BASE_WINDOW_SEC, InferenceConfig.TURN_TAKING_EXT_WINDOW_SEC
        )
        
        # Social interaction classification with specified rules
        classification_results = all_data.apply(
            lambda row: classify_frames(row, all_data, included_rules), axis=1
        )
        
        # Unpack the classification results into separate columns
        all_data['interaction_type'] = [result[0] for result in classification_results]
        all_data['rule1_turn_taking'] = [result[1] for result in classification_results]
        all_data['rule2_close_proximity'] = [result[2] for result in classification_results]
        all_data['rule3_kcds_speaking'] = [result[3] for result in classification_results]
        all_data['rule4_person_recent_speech'] = [result[4] for result in classification_results]

        # Step 5: Presence pattern categorization       
        all_data['face_frame_category'] = all_data.apply(classify_face_category, axis=1)
        all_data['person_frame_category'] = all_data.apply(classify_person_category, axis=1)

        # Step 7: Merge age information
        all_data = merge_age_information(all_data)
                
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 8: Save results with information about which rules were included e.g. 01_frame_level_social_interactions_1_2_3_4.csv
        file_name = Inference.FRAME_LEVEL_INTERACTIONS_CSV.stem + f"_{'_'.join(map(str, included_rules))}.csv"  
        all_data.to_csv(output_dir / file_name, index=False)

        print(f"✅ Saved detailed frame-level analysis to {output_dir / file_name}")
        
        return all_data

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Frame-level social interaction analysis')
    parser.add_argument('--rules', type=int, nargs='+', default=[1, 2, 3, 4],
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=KCDS-speech, 4=adult-face-recent-speech). Default: [1, 2, 3, 4]')

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules)