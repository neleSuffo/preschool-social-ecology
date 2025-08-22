# Optimized query analysis for naturalistic social data
import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths

# Constants
OUTPUT_DIR = DataPaths.INFERENCE_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FPS = 30
ROUND_TO = 10

def get_all_analysis_data(conn):
    """
    Single optimized query that gets all required data in one go.
    """
    # Create temp table once
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT 
        frame_number, video_id,
        MAX(CASE WHEN age_class = 0 THEN 1 ELSE 0 END) AS has_child_face,
        MAX(CASE WHEN age_class = 1 THEN 1 ELSE 0 END) AS has_adult_face
    FROM FaceDetections
    GROUP BY frame_number, video_id;
    """)
    
    # Create simplified vocalization mapping with KCHI priority
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
    
    # Main query without ambiguous joins
    query = """
    SELECT
        pd.frame_number,
        pd.video_id,
        -- Person presence
        pd.has_child_person,
        pd.has_adult_person,
        -- Face presence  
        COALESCE(fa.has_child_face, 0) AS has_child_face,
        COALESCE(fa.has_adult_face, 0) AS has_adult_face,
        -- Combined presence
        CASE WHEN COALESCE(fa.has_child_face,0)=1 OR pd.has_child_person=1 THEN 1 ELSE 0 END AS child_present,
        CASE WHEN COALESCE(fa.has_adult_face,0)=1 OR pd.has_adult_person=1 THEN 1 ELSE 0 END AS adult_present,
        -- Vocalizations
        vf.speaker
    FROM PersonClassifications pd
    LEFT JOIN FaceAgg fa ON pd.frame_number = fa.frame_number AND pd.video_id = fa.video_id
    LEFT JOIN VocalizationFrames vf ON pd.frame_number = vf.frame_number AND pd.video_id = vf.video_id
    """
    return pd.read_sql(query, conn)

def run_analysis():
    """
    Simplified main analysis function using a single comprehensive query.
    """
    with sqlite3.connect(DataPaths.INFERENCE_DB_PATH) as conn:
        # Get comprehensive data in one query
        print("🔄 Running comprehensive analysis...")
        all_data = get_all_analysis_data(conn)
        
        # Calculate summaries
        summaries = {}
        
        # Face categories summary
        face_counts = all_data.groupby([
            (all_data['has_child_face'] == 1) & (all_data['has_adult_face'] == 0),  # only_child
            (all_data['has_child_face'] == 0) & (all_data['has_adult_face'] == 1),  # only_adult  
            (all_data['has_child_face'] == 1) & (all_data['has_adult_face'] == 1)   # both_faces
        ]).size()
        
        summaries['face_categories'] = {
            'only_child_face': face_counts.get((True, False, False), 0),
            'only_adult_face': face_counts.get((False, True, False), 0), 
            'both_faces': face_counts.get((False, False, True), 0),
            'no_faces': len(all_data) - face_counts.sum()
        }
        
        # Person categories summary  
        person_counts = all_data.groupby([
            (all_data['has_child_person'] == 1) & (all_data['has_adult_person'] == 0),
            (all_data['has_child_person'] == 0) & (all_data['has_adult_person'] == 1),
            (all_data['has_child_person'] == 1) & (all_data['has_adult_person'] == 1)
        ]).size()
        
        summaries['person_categories'] = {
            'only_child_person': person_counts.get((True, False, False), 0),
            'only_adult_person': person_counts.get((False, True, False), 0),
            'both_persons': person_counts.get((False, False, True), 0), 
            'no_persons': len(all_data) - person_counts.sum()
        }
        
        # Combined presence summary
        combined_counts = all_data.groupby([
            (all_data['child_present'] == 1) & (all_data['adult_present'] == 0),
            (all_data['child_present'] == 0) & (all_data['adult_present'] == 1),
            (all_data['child_present'] == 1) & (all_data['adult_present'] == 1)
        ]).size()
        
        summaries['combined_presence'] = {
            'only_child_present': combined_counts.get((True, False, False), 0),
            'only_adult_present': combined_counts.get((False, True, False), 0),
            'both_present': combined_counts.get((False, False, True), 0),
            'no_one_present': len(all_data) - combined_counts.sum(),
            'total_frames_analyzed': len(all_data)
        }
        
        # Save individual summaries as CSV
        for name, data in summaries.items():
            df = pd.DataFrame([data])
            output_path = OUTPUT_DIR / f"{name}.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {name} to {output_path}")
        
        # Add speech flags and save comprehensive dataset
        all_data['kchi_speech_present'] = (all_data['speaker'] == 'KCHI').astype(int)
        all_data['other_speech_present'] = (all_data['speaker'] == 'OTH').astype(int)
        all_data['not_alone_detection'] = (
            (all_data['adult_present'] == 1) | 
            (all_data['child_present'] == 1) | 
            (all_data['speaker'] == 'OTH')
        ).astype(int)
        
        # Save comprehensive dataset
        output_path = OUTPUT_DIR / "alone_vs_not_alone.csv" 
        all_data.to_csv(output_path, index=False)
        print(f"✅ Saved comprehensive analysis to {output_path}")
        
    print("🎯 Analysis completed successfully.")
    return summaries

def main():
    """Legacy function for backward compatibility."""
    return run_analysis()

if __name__ == "__main__":
    run_analysis()