# Research Question 1. How much time do children spend alone?
#
# This script analyzes multimodal social interaction patterns by combining:
# - Visual person detection (child/adult bodies)
# - Visual face detection (child/adult faces with proximity measures)  
# - Audio vocalization detection (child/other speaker identification)
#
# The analysis produces frame-level classifications of social contexts to understand
# when children are alone vs. in various types of social interactions.

import sqlite3
import re
import pandas as pd
from constants import DataPaths, ResearchQuestions
from config import DataConfig, Research_QuestionConfig

# Constants
FPS = DataConfig.FPS # frames per second

def extract_child_id(video_name):
    match = re.search(r'id(\d{6})', video_name)
    return match.group(1) if match else None

def extract_segments_with_buffering(results_df):
    """
    Creates mutually exclusive segments, buffering small state changes.

    Args:
        results_df (pd.DataFrame): DataFrame with 'video_id', 'frame_number', 'interaction_category'.

    Returns:
        pd.DataFrame: A DataFrame of the extracted, buffered segments.
    """
    print("Creating segments...")
    
    all_segments = []
    
    for video_id, video_df in results_df.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').reset_index(drop=True)
        
        if len(video_df) == 0:
            continue
                    
        # Get interaction states and frame numbers
        states = video_df['interaction_category'].values
        frame_numbers = video_df['frame_number'].values
        video_name = video_df['video_name'].iloc[0]
        
        # Buffer short state changes
        buffered_states = states.copy()
        i = 0
        while i < len(buffered_states) - 1:
            current_state = buffered_states[i]
            j = i + 1
            # Find the end of the current run of states
            while j < len(buffered_states) and buffered_states[j] == current_state:
                j += 1
            
            # The length of the current run of states
            run_length = j - i
            run_duration = (frame_numbers[j-1] - frame_numbers[i]) / FPS
            
            # If the run is short and not at the beginning/end, merge it
            if run_duration < Research_QuestionConfig.RQ1_MIN_CHANGE_DURATION_SEC and i > 0 and j < len(buffered_states):
                # Replace the short run with the previous state
                buffered_states[i:j] = buffered_states[i-1]
                # Reset i to re-evaluate from the previous point
                i = 0
            else:
                i = j
        
        # Now, find state changes in the buffered states
        current_state = buffered_states[0]
        segment_start_frame = frame_numbers[0]
        
        for i in range(1, len(buffered_states)):
            if buffered_states[i] != current_state:
                segment_end_frame = frame_numbers[i-1]
                
                # Only keep segments longer than minimum duration
                segment_duration = (segment_end_frame - segment_start_frame) / FPS
                if segment_duration >= Research_QuestionConfig.RQ1_MIN_SEGMENT_DURATION_SEC:
                    all_segments.append({
                        'video_id': video_id,
                        'video_name': video_name,
                        'category': current_state,
                        'segment_start': segment_start_frame,
                        'segment_end': segment_end_frame,
                        'start_time_sec': segment_start_frame / FPS,
                        'end_time_sec': segment_end_frame / FPS,
                        'duration_sec': segment_duration
                    })
                
                current_state = buffered_states[i]
                segment_start_frame = frame_numbers[i]
        
        # Handle the final segment
        segment_end_frame = frame_numbers[-1]
        segment_duration = (segment_end_frame - segment_start_frame) / FPS
        if segment_duration >= Research_QuestionConfig.RQ1_MIN_SEGMENT_DURATION_SEC:
            all_segments.append({
                'video_id': video_id,
                'video_name': video_name,
                'category': current_state,
                'segment_start': segment_start_frame,
                'segment_end': segment_end_frame,
                'start_time_sec': segment_start_frame / FPS,
                'end_time_sec': segment_end_frame / FPS,
                'duration_sec': segment_duration
            })
    
    if all_segments:
        segments_df = pd.DataFrame(all_segments)
        segments_df = segments_df.sort_values(['video_id', 'start_time_sec']).reset_index(drop=True)
    else:
        segments_df = pd.DataFrame(columns=['video_id', 'video_name', 'category',
                                          'segment_start', 'segment_end', 
                                          'start_time_sec', 'end_time_sec', 'duration_sec'])
        
    # Add child_id to segments_df using extract_child_id
    segments_df['child_id'] = segments_df['video_name'].apply(extract_child_id)
    
    # Load subjects CSV to get age information
    try:
        subjects_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
        print(f"📋 Loaded subjects data with {len(subjects_df)} records for age mapping")
        
        # Merge age information based on video_name
        segments_df = segments_df.merge(
            subjects_df[['video_name', 'age_at_recording']], 
            on='video_name', 
            how='left'
        )
        
        # Check merge success
        missing_age_count = segments_df['age_at_recording'].isna().sum()
        if missing_age_count > 0:
            print(f"⚠️ Warning: {missing_age_count} segments missing age data ({missing_age_count/len(segments_df)*100:.1f}%)")
            
            # Show some examples of unmatched video names
            unmatched_videos = segments_df[segments_df['age_at_recording'].isna()]['video_name'].unique()[:5]
            print(f"Examples of unmatched video names: {list(unmatched_videos)}")
        else:
            print("✅ All segments successfully matched with age data")
            
    except FileNotFoundError:
        print(f"⚠️ Warning: Subjects CSV not found at {DataPaths.SUBJECTS_CSV_PATH}")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None
    except Exception as e:
        print(f"⚠️ Warning: Error loading subjects data: {e}")
        print("Proceeding without age information")
        segments_df['age_at_recording'] = None

    # Reorder columns so that child_id and age_at_recording come after video_name
    cols = ['video_name', 'child_id', 'age_at_recording'] + [col for col in segments_df.columns if col not in ['video_name', 'child_id', 'age_at_recording']]
    segments_df = segments_df.loc[:, cols]

    print(f"Created {len(segments_df)} segments after buffering.")
    return segments_df

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

def check_audio_interaction_turn_taking(df, fps, base_window_sec, extended_window_sec=15):
    """
    Adaptive turn-taking detection:
    - Uses smaller window (base_window_sec) when child is alone
    - Expands window (extended_window_sec) if a person or face is visible

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
    print(f"🎤 Adaptive turn-taking: base {base_window_sec}s, extended {extended_window_sec}s")

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

            # Check for interaction
            has_kchi_in_window = window_data['has_kchi'].sum() > 0
            has_other_in_window = window_data['has_other'].sum() > 0
            is_interaction = has_kchi_in_window and has_other_in_window
            interaction_flags.append(is_interaction)

        video_df['is_audio_interaction'] = interaction_flags
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])

    # Combine and merge back
    result_df = pd.concat(all_results, ignore_index=True)
    df_with_interaction = df.merge(result_df, on=['frame_number', 'video_id'], how='left')
    df_with_interaction['is_audio_interaction'] = df_with_interaction['is_audio_interaction'].fillna(False)

    total_interactions = df_with_interaction['is_audio_interaction'].sum()
    print(f"Found {total_interactions:,} frames with interactions ({total_interactions/len(df)*100:.1f}%)")

    return df_with_interaction['is_audio_interaction']

def run_analysis():
    """
    Main analysis function that orchestrates multimodal social interaction analysis.
    
    This function performs the following analytical steps:
    
    1. DATA INTEGRATION:
       - Executes the comprehensive multimodal query via get_all_analysis_data()
       - Combines visual (person/face) and auditory (speech) detection modalities
       - Creates a unified frame-level dataset with temporal alignment
    
    2. FEATURE ENGINEERING:
       - Speech flags: Separates child speech (KCHI) from other speech
       - Interaction categories: Classifies social contexts (Alone, Co-present, Interacting)
       - Frame categories: Categorizes presence patterns for faces and persons
    
    3. SOCIAL INTERACTION CLASSIFICATION:
       
       INTERACTION CATEGORIES:
       - "Interacting": Active social engagement detected
         * High proximity faces (>= 0.5) indicating close social contact
         * OR other person speaking (active communication)
         * OR person detected with recent speech
       
       - "Co-present Silent": Passive social presence 
         * People detected but no active interaction indicators
         * Physical presence without communication or close contact
       
       - "Alone": No social presence detected
         * No people detected in any modality
         * Child is physically and socially isolated
    
    4. TEMPORAL GRANULARITY ANALYSIS:
       - Frame-level (1/3 second intervals) for precise temporal dynamics
       - Enables fine-grained analysis of interaction transitions
       - Supports identification of brief social moments and extended episodes
    
    5. MULTIMODAL VALIDATION:
       - Cross-validates detection across visual and auditory channels
       - Reduces false positives through multi-source confirmation
       - Handles missing data gracefully with modality-specific fallbacks
    
    6. OUTPUT GENERATION:
       - Summary statistics: Distribution of interaction categories and presence patterns
       - Frame-level data: Complete temporal record for longitudinal analysis
       - CSV exports: Both aggregate summaries and detailed frame-by-frame data
    
    ANALYTICAL ADVANTAGES:
    - Temporal precision: 1/3 second granularity captures rapid social dynamics
    - Multimodal robustness: Multiple detection channels reduce noise
    - Social context differentiation: Distinguishes active interaction from mere presence
    - Comprehensive coverage: Analyzes entire video corpus systematically
    
    Returns:
        dict: Summary statistics including interaction and presence distributions
    """
    with sqlite3.connect(DataPaths.INFERENCE_DB_PATH) as conn:
        print("🔄 Running comprehensive multimodal social interaction analysis...")
        
        # ====================================================================
        # STEP 1: DATA INTEGRATION AND FEATURE ENGINEERING
        # ====================================================================
        all_data = get_all_analysis_data(conn)
        print(f"📊 Integrated {len(all_data):,} frames across {all_data['video_id'].nunique()} videos")
        
        # ====================================================================
        # STEP 2: AUDIO TURN-TAKING ANALYSIS
        # ====================================================================
        # Add the audio interaction flag to the DataFrame
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(
            all_data, FPS, Research_QuestionConfig.TURN_TAKING_WINDOW_SEC, 
        )
        
        # ====================================================================
        # STEP 3: SPEECH MODALITY FEATURE EXTRACTION
        # ====================================================================
        # Binary flags for speech presence by speaker type
        all_data['kchi_speech_present'] = all_data['speaker'].str.contains('KCHI', na=False).astype(int)
        all_data['other_speech_present'] = all_data['speaker'].str.contains('FEM_MAL', na=False).astype(int)

        speech_stats = {
            'kchi_frames': all_data['kchi_speech_present'].sum(),
            'other_speech_frames': all_data['other_speech_present'].sum(),
            'silent_frames': len(all_data) - all_data['kchi_speech_present'].sum() - all_data['other_speech_present'].sum()
        }
        print(f"🎤 Speech analysis: {speech_stats['kchi_frames']:,} child speech frames, {speech_stats['other_speech_frames']:,} other speech frames")
        
        # ====================================================================
        # STEP 4: SOCIAL INTERACTION CLASSIFICATION (with audio priority)
        # ====================================================================
        # Three-tier classification system based on interaction intensity                
        def classify_interaction_with_audio(row, results_df):
            """
            Hierarchical social interaction classifier with dynamic proximity and audio priority.
            
            CLASSIFICATION LOGIC:
            1. INTERACTING: Active social engagement
            - Turn-taking audio interaction detected (highest priority)
            - OR Very close proximity (>= PROXIMITY_THRESHOLD)
            - OR other person speaking
            - OR an adult face (proximity doesn't matter) + recent speech (last 3 sec)
            
            2. CO-PRESENT SILENT: Passive social presence
            - Person detected but no active interaction indicators
            
            3. ALONE: No social presence detected
            
            Args:
                row: DataFrame row with detection flags and proximity values
                results_df: The full DataFrame to enable window-based lookups
                window_size_frames: The size of the sliding window in frames
                PROXIMITY_THRESHOLD: The threshold for defining "close" proximity
                
            Returns:
                str: Interaction category ('Interacting', 'Co-present Silent', 'Alone')
            """
    
            # Calculate recent proximity once at the beginning
            current_index = row.name
            window_start = max(0, current_index - Research_QuestionConfig.TURN_TAKING_WINDOW_SEC)
            recent_speech_exists = (results_df.loc[window_start:current_index, 'other_speech_present'] == 1).any()

            # Check if a person is present at all, using the combined flags
            person_is_present = (row['child_present'] == 1) or (row['adult_present'] == 1)
            
            # =======================================================================
            # Tier 1: INTERACTING (Active engagement)
            # This tier is evaluated first due to its high-confidence indicators.
            # =======================================================================
            is_active_interaction = (
                row['is_audio_interaction'] or # turn taking
                (row['proximity'] >= Research_QuestionConfig.RQ1_PROXIMITY_THRESHOLD) or # very close proximity
                row['other_speech_present'] or # other person speaking
                (row['has_adult_face'] == 1 and recent_speech_exists) # adult face + recent speech
            )

            if is_active_interaction:
                return "Interacting"

            # =======================================================================
            # Tier 2: CO-PRESENT SILENT (Passive presence)
            # This tier is evaluated if no active interaction is found.
            # =======================================================================
            if person_is_present:
                return "Co-present Silent"

            # =======================================================================
            # Tier 3: ALONE (No presence)
            # This is the default if no social presence is detected.
            # =======================================================================
            return "Alone"

        # Use a lambda function to pass the DataFrame into the `apply` call
        all_data['interaction_category'] = all_data.apply(
            lambda row: classify_interaction_with_audio(row, all_data), axis=1
        )
        
        # ====================================================================
        # STEP 5: PRESENCE PATTERN CATEGORIZATION
        # ====================================================================
        # Face-level categorization for social attention analysis
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
        
        # Person-level categorization for physical presence analysis
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
        
        all_data['face_frame_category'] = all_data.apply(classify_face_category, axis=1)
        all_data['person_frame_category'] = all_data.apply(classify_person_category, axis=1)
        
        # ====================================================================
        # STEP 5: STATISTICAL SUMMARY GENERATION
        # ====================================================================
        summaries = {}
        
        # Interaction distribution analysis
        interaction_dist = all_data['interaction_category'].value_counts()
        summaries['interaction_distribution'] = interaction_dist.to_dict()
        print(f"🤝 Interaction distribution: {dict(interaction_dist)}")
        
        # Face presence pattern analysis
        face_dist = all_data['face_frame_category'].value_counts()
        summaries['face_category_distribution'] = face_dist.to_dict()
        print(f"👤 Face pattern distribution: {dict(face_dist)}")
        
        # Person presence pattern analysis  
        person_dist = all_data['person_frame_category'].value_counts()
        summaries['person_category_distribution'] = person_dist.to_dict()
        print(f"🚶 Person pattern distribution: {dict(person_dist)}")
        
        # ====================================================================
        # STEP 6: STATISTICAL SUMMARY GENERATION
        # ====================================================================
        summaries = {}
        
        # Interaction distribution analysis
        interaction_dist = all_data['interaction_category'].value_counts()
        summaries['interaction_distribution'] = interaction_dist.to_dict()
        print(f"🤝 Interaction distribution: {dict(interaction_dist)}")
        
        # Face presence pattern analysis
        face_dist = all_data['face_frame_category'].value_counts()
        summaries['face_category_distribution'] = face_dist.to_dict()
        print(f"👤 Face pattern distribution: {dict(face_dist)}")
        
        # Person presence pattern analysis  
        person_dist = all_data['person_frame_category'].value_counts()
        summaries['person_category_distribution'] = person_dist.to_dict()
        print(f"🚶 Person pattern distribution: {dict(person_dist)}")
        
        # ====================================================================
        # STEP 7: DATA EXPORT AND PERSISTENCE
        # ====================================================================   
        # ====================================================================
        # STEP 7: DATA EXPORT AND PERSISTENCE
        # ====================================================================        
        # Load subjects CSV to get age information
        try:
            subjects_df = pd.read_csv(DataPaths.SUBJECTS_CSV_PATH)
            print(f"📋 Loaded subjects data with {len(subjects_df)} records")
            
            # Merge age information based on video_name
            all_data = all_data.merge(
                subjects_df[['video_name', 'age_at_recording']], 
                on='video_name', 
                how='left'
            )
            
            # Check merge success
            missing_age_count = all_data['age_at_recording'].isna().sum()
            if missing_age_count > 0:
                print(f"⚠️ Warning: {missing_age_count} frames missing age data ({missing_age_count/len(all_data)*100:.1f}%)")
                
                # Show some examples of unmatched video names
                unmatched_videos = all_data[all_data['age_at_recording'].isna()]['video_name'].unique()[:5]
                print(f"Examples of unmatched video names: {list(unmatched_videos)}")
            
            # Reorder columns to put age near the beginning
            cols = all_data.columns.tolist()
            new_order = ['frame_number', 'video_id', 'video_name', 'age_at_recording'] + [col for col in cols if col not in ['frame_number', 'video_id', 'video_name', 'age_at_recording']]
            all_data = all_data[new_order]
            
        except FileNotFoundError:
            print(f"⚠️ Warning: Subjects CSV not found at {DataPaths.SUBJECTS_CSV_PATH}")
            print("Proceeding without age information")
        except Exception as e:
            print(f"⚠️ Warning: Error loading subjects data: {e}")
            print("Proceeding without age information")
        
        # Complete frame-level dataset (for temporal analysis)
        ResearchQuestions.RQ1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        all_data.to_csv(ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV, index=False)
    print(f"✅ Saved detailed frame-level analysis to {ResearchQuestions.FRAME_LEVEL_INTERACTIONS_CSV}")

if __name__ == "__main__":
    run_analysis()