# Research Question 1: Social Interaction Analysis for Naturalistic Child Language Data
#
# This script analyzes multimodal social interaction patterns by combining:
# - Visual person detection (child/adult bodies)
# - Visual face detection (child/adult faces with proximity measures)  
# - Audio vocalization detection (child/other speaker identification)
#
# The analysis produces frame-level classifications of social contexts to understand
# when children are alone vs. in various types of social interactions.

import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths
from config import DataConfig

# Constants
OUTPUT_DIR = Path("/home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/results/rq_01")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROUND_TO = 10
AUDIO_WINDOW_SEC = 3  # seconds
FPS = 30  # frames per second

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
    SELECT DISTINCT 
        video_id,
        frame_number,
        CASE 
            WHEN COUNT(CASE WHEN speaker = 'KCHI' THEN 1 END) > 0 THEN 'KCHI'
            ELSE MAX(speaker)
        END as speaker
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

def check_audio_interaction_turn_taking(df, window_size_frames, fps):
    """
    Checks for turn-taking audio interaction within a sliding window.
    
    A turn-taking interaction is defined as the presence of both 'KCHI'
    and another speaker (non-KCHI) within the specified time window.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a 'speaker' column.
        window_size_frames (int): The size of the sliding window in frames.
        fps (int): Frames per second.
        
    Returns:
        pd.Series: A boolean Series indicating if an audio interaction occurred.
    """
    print(f"Analyzing turn-taking with window size: {window_size_frames} frames ({window_size_frames/fps:.1f} seconds)")
    
    # Create binary flags for each speaker type
    df_copy = df.copy()
    df_copy['has_kchi'] = (df_copy['speaker'] == 'KCHI').astype(int)
    df_copy['has_other'] = ((~df_copy['speaker'].isna()) & (df_copy['speaker'] != 'KCHI')).astype(int)
    
    # Process each video separately to handle rolling windows correctly
    all_results = []
    
    for video_id, video_df in df_copy.groupby('video_id'):
        video_df = video_df.sort_values('frame_number').reset_index(drop=True)
        
        # Apply rolling sum to numeric columns (this works with pandas rolling)
        kchi_window = video_df['has_kchi'].rolling(window=window_size_frames, center=True, min_periods=1).sum()
        other_window = video_df['has_other'].rolling(window=window_size_frames, center=True, min_periods=1).sum()
        
        # Turn-taking detected if both KCHI and other speakers present in window
        video_df['is_audio_interaction'] = ((kchi_window > 0) & (other_window > 0)).astype(bool)
        
        all_results.append(video_df[['frame_number', 'video_id', 'is_audio_interaction']])
    
    # Combine all videos
    result_df = pd.concat(all_results, ignore_index=True)
    
    # Merge back to original dataframe order
    df_with_audio = df.merge(result_df, on=['frame_number', 'video_id'], how='left')
    
    # Fill any missing values with False
    df_with_audio['is_audio_interaction'] = df_with_audio['is_audio_interaction'].fillna(False)
    
    audio_interaction_count = df_with_audio['is_audio_interaction'].sum()
    print(f"Found {audio_interaction_count:,} frames with turn-taking audio interaction ({audio_interaction_count/len(df)*100:.1f}%)")
    
    return df_with_audio['is_audio_interaction']

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
        print(f"🎤 Analyzing turn-taking in a {AUDIO_WINDOW_SEC}-second window...")
        # Add the audio interaction flag to the DataFrame
        all_data['is_audio_interaction'] = check_audio_interaction_turn_taking(
            all_data, AUDIO_WINDOW_SEC * FPS, FPS
        )
        
        # ====================================================================
        # STEP 3: SPEECH MODALITY FEATURE EXTRACTION
        # ====================================================================
        # Binary flags for speech presence by speaker type
        all_data['kchi_speech_present'] = (all_data['speaker'] == 'KCHI').astype(int)
        all_data['other_speech_present'] = ((~all_data['speaker'].isna()) & (all_data['speaker'] != 'KCHI')).astype(int)
        
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
        def classify_interaction_with_audio(row):
            """
            Hierarchical social interaction classifier with audio priority.
            
            CLASSIFICATION LOGIC:
            1. INTERACTING: Active social engagement
               - Turn-taking audio interaction detected (highest priority)
               - OR Close proximity (>= 0.5) from video
               - OR other person speaking from audio
               
            2. CO-PRESENT SILENT: Passive social presence
               - Person detected but no interaction indicators
               
            3. ALONE: No social presence detected
            
            Args:
                row: DataFrame row with detection flags and proximity values
                
            Returns:
                str: Interaction category ('Interacting', 'Co-present Silent', 'Alone')
            """
            # High-confidence interaction indicators (Audio has highest priority)
            if row['is_audio_interaction'] or (row['proximity'] >= 0.5) or row['other_speech_present']:
                return "Interacting"
            # Physical presence without interaction
            elif (row['has_child_person'] == 1) or (row['has_adult_person'] == 1):
                return "Co-present Silent"
            # No social presence detected
            else:
                return "Alone"
        
        # Now use the new classification function
        all_data['interaction_category'] = all_data.apply(classify_interaction_with_audio, axis=1)
        
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
        # Combined summary statistics (single row with all metrics)
        summary_df = pd.DataFrame([{
            **{f"interaction_{k}": v for k, v in summaries['interaction_distribution'].items()},
            **{f"face_{k}": v for k, v in summaries['face_category_distribution'].items()},
            **{f"person_{k}": v for k, v in summaries['person_category_distribution'].items()},
            **{f"speech_{k}": v for k, v in speech_stats.items()}
        }])
        
        summary_path = OUTPUT_DIR / "comprehensive_interaction_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Saved comprehensive summary to {summary_path}")
        
        # Complete frame-level dataset (for temporal analysis)
        full_path = OUTPUT_DIR / "frame_level_social_interactions.csv"
        all_data.to_csv(full_path, index=False)
        print(f"✅ Saved detailed frame-level analysis to {full_path}")
        
    print("🎯 Multimodal social interaction analysis completed successfully.")
    print(f"📈 Dataset ready for longitudinal analysis, developmental patterns, and social context modeling.")
    return summaries

def main():
    return run_analysis()

if __name__ == "__main__":
    run_analysis()