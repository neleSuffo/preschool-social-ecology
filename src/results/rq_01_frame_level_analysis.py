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
    
    # Ensure the column is integer type for accurate diff calculation
    # Use .copy() to avoid SettingWithCopyWarning if speech_frames is modified later
    speech_frames = video_df[video_df[column_name] == 1].copy()
    
    if speech_frames.empty:
        return segments

    frame_numbers = speech_frames['frame_number'].values
    
    # Initialize the start of the first segment
    current_start = frame_numbers[0]
    
    for i in range(1, len(frame_numbers)):
        # If the gap between this frame and the previous one is > 1 frame,
        # it means the previous segment ended and a new one starts here (due to SAMPLE_RATE=1 assumption for continuity).
        if frame_numbers[i] > frame_numbers[i-1] + SAMPLE_RATE: 
            # Finalize the previous segment
            segments.append({
                'start': current_start,
                'end': frame_numbers[i-1],
                'type': column_name.split('_')[-1] # 'kchi' or 'kcds'
            })
            # Start a new segment
            current_start = frame_numbers[i]

    # Append the last segment
    segments.append({
        'start': current_start,
        'end': frame_numbers[-1],
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
    # STEP 1: FACE DETECTION AGGREGATION
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
    WHERE frame_number % ? = 0 
    GROUP BY frame_number, video_id;
    """, (SAMPLE_RATE,)) 
    
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
        
        -- PERSON DETECTION MODALITY (Set to 0 as PersonClassifications is excluded)
        0 AS has_child_person,
        0 AS has_adult_person,
        
        -- AUDIO CLASSIFICATION MODALITY (LEFT JOIN onto the grid)
        COALESCE(af.has_kchi, 0) AS has_kchi,
        COALESCE(af.has_ohs, 0) AS has_ohs,
        COALESCE(af.has_cds, 0) AS has_cds,
        
        -- FACE DETECTION MODALITY (LEFT JOIN onto the grid)
        COALESCE(fa.has_face, 0) AS has_face,
        fa.proximity,
        
        -- MULTIMODAL FUSION FLAGS
        CASE 
            WHEN COALESCE(fa.has_face, 0)=1 THEN 1 
            ELSE 0 
        END AS person_present -- Simplified: person_present = has_face in this mode

    FROM FrameGrid fg
    LEFT JOIN AudioClassifications af 
        ON fg.frame_number = af.frame_number AND fg.video_id = af.video_id
    LEFT JOIN FaceAgg fa 
        ON fg.frame_number = fa.frame_number AND fg.video_id = fa.video_id
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
    
    # Handle empty DataFrame
    if df.empty:
        return pd.Series([], dtype=bool, name='is_audio_interaction')

    # 5 seconds in frames
    MAX_GAP_FRAMES = InferenceConfig.MAX_TURN_TAKING_GAP_SEC * fps
    
    all_results = []

    for video_id, video_df in df.groupby('video_id'):
        
        # 1. Identify all KCHI and KCDS segments
        video_df = video_df.copy() 
        video_df['is_audio_interaction'] = False
        
        kchi_segments = find_segments(video_df, 'has_kchi')
        kcds_segments = find_segments(video_df, 'has_cds')
        
        # 2. Combine and sort all segments by start time
        all_segments = sorted(kchi_segments + kcds_segments, key=lambda x: x['start'])
        
        if not all_segments:
            all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])
            continue
            
        interaction_windows = []
        
        # 3. Merge segments into continuous interaction intervals
        current_window_start = all_segments[0]['start']
        current_window_end = all_segments[0]['end']
        
        # Track all segment types within the current window to enforce KCHI+CDS requirement
        segments_in_window = [all_segments[0]]
        
        for i in range(1, len(all_segments)):
            prev_segment_end = current_window_end
            current_segment_start = all_segments[i]['start']
            current_segment_end = all_segments[i]['end']

            # Gap is the frame difference between the end of the previous segment/window and the start of the current segment
            gap = current_segment_start - prev_segment_end
            
            if gap <= MAX_GAP_FRAMES:
                # Merge: Extend the current window
                current_window_end = current_segment_end
                segments_in_window.append(all_segments[i])
            else:
                # Break: The current interaction window is finalized.
                interaction_windows.append({
                    'start': current_window_start, 
                    'end': current_window_end,
                    'segments': segments_in_window
                })
                # Start a new window
                current_window_start = current_segment_start
                current_window_end = current_segment_end
                segments_in_window = [all_segments[i]]
                
        # Append the last window
        interaction_windows.append({
            'start': current_window_start, 
            'end': current_window_end,
            'segments': segments_in_window
        })
        
        # The key filter: An interaction bout must contain *both* KCHI and CDS
        for window in interaction_windows:
            # Check segment types in the window (more efficient than summing frames)
            has_kchi = any(s['type'] == 'kchi' for s in window['segments'])
            has_cds = any(s['type'] == 'cds' for s in window['segments'])
            
            if has_kchi and has_cds:
                video_df.loc[
                    (video_df['frame_number'] >= window['start']) & 
                    (video_df['frame_number'] <= window['end']), 
                    'is_audio_interaction'
                ] = True
        
        # FIX: Include 'video_id'
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])
    # Combine all results 
    result_df = pd.concat(all_results, ignore_index=True)
    
    # Create unique keys for reliable merge
    result_df['temp_merge_key'] = result_df['frame_number'].astype(str) + '_' + result_df['video_id'].astype(str)
    df['temp_merge_key'] = df['frame_number'].astype(str) + '_' + df['video_id'].astype(str)
    
    # Merge the calculated interaction flags back to the original full dataframe (df)
    df_merged = df.merge(
        result_df[['temp_merge_key', 'is_audio_interaction']], 
        on='temp_merge_key',
        how='left'
    )

    # Clean up and return
    df.drop(columns=['temp_merge_key'], inplace=True, errors='ignore') 
        
    # Ensure the returned Series has the correct name for assignment in main()
    return df_merged['is_audio_interaction'].fillna(False).rename('is_audio_interaction')

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
    
    # Calculate the window start index based on the frame rate and sample rate
    # Window size in samples = (Window size in seconds * FPS) / SAMPLE_RATE
    window_samples = int(InferenceConfig.PERSON_AUDIO_WINDOW_SEC * FPS / SAMPLE_RATE)
    window_start = max(0, current_index - window_samples)
    
    # Check for recent KCDS in the window based on the original dataframe's index
    recent_speech_exists = (results_df.loc[window_start:current_index, 'has_cds'] == 1).any()
    
    # In this simplified mode, person_present = has_face
    person_present = (row['person_present'] == 1) 
    
    # Evaluate all rules and track their activation
    rule1_turn_taking = bool(row['is_audio_interaction'])
    rule2_close_proximity = bool(row['proximity'] >= InferenceConfig.PROXIMITY_THRESHOLD) if pd.notna(row['proximity']) else False
    rule3_kcds_speaking = bool(row['has_cds'])
    
    # Rule 4: Person (Face) present + recent speech
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
    """
    Categorize face detection presence using simplified `has_face` indicator.
    (Note: In the current simplified mode, this captures any face detection).
    """
    try:
        return 'has_face' if int(row.get('has_face', 0)) == 1 else 'no_faces'
    except Exception:
        return 'no_faces'

def classify_person_category(row):
    """
    Categorize person presence using simplified `person_present` indicator.
    (Note: In the current simplified mode, this is equivalent to has_face).
    """
    try:
        return 'person_present' if int(row.get('person_present', 0)) == 1 else 'no_persons'
    except Exception:
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
        # Data integration (uses adjusted query with Recursive CTE)
        all_data = get_all_analysis_data(conn)
        
        # Audio turn-taking analysis (uses segment merging logic)
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(all_data, FPS)
        
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
                    help='List of interaction rules to include (1=turn-taking, 2=proximity, 3=cds-speaking, 4=adult-face-recent-speech). Default: [1, 2, 3, 4]')

    args = parser.parse_args()
    
    # Validate rule numbers
    valid_rules = [1, 2, 3, 4]
    if not all(rule in valid_rules for rule in args.rules):
        print(f"❌ Error: Invalid rule numbers. Valid options are: {valid_rules}")
        sys.exit(1)

    main(db_path=Path(DataPaths.INFERENCE_DB_PATH), output_dir=Inference.BASE_OUTPUT_DIR, included_rules=args.rules)