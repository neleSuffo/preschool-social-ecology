# analysis/query_frame_categories.py
import sqlite3
import pandas as pd
from pathlib import Path
from constants import DataPaths

# SQL queries
face_frame_categories_query = """
WITH UniqueFrames AS (
    SELECT 
        frame_number,
        video_id,
        MAX(CASE WHEN age_class = 0 THEN 1 ELSE 0 END) as has_child_face,
        MAX(CASE WHEN age_class = 1 THEN 1 ELSE 0 END) as has_adult_face
    FROM FaceDetections 
    GROUP BY frame_number, video_id
),
FrameCategories AS (
    SELECT 
        frame_number,
        video_id,
        CASE 
            WHEN has_child_face = 1 AND has_adult_face = 1 THEN 'both_faces'
            WHEN has_child_face = 1 AND has_adult_face = 0 THEN 'only_child'
            WHEN has_child_face = 0 AND has_adult_face = 1 THEN 'only_adult'
            ELSE 'no_faces'
        END as frame_category
    FROM UniqueFrames
)
SELECT 
    frame_number,
    video_id,
    frame_category
FROM FrameCategories
"""

person_frame_categories_query = """
WITH FrameCategories AS (
    SELECT 
        frame_number,
        video_id,
        CASE 
            WHEN has_child_person = 1 AND has_adult_person = 1 THEN 'both_persons'
            WHEN has_child_person = 1 AND has_adult_person = 0 THEN 'only_child'
            WHEN has_child_person = 0 AND has_adult_person = 1 THEN 'only_adult'
            WHEN has_child_person = 0 AND has_adult_person = 0 THEN 'no_persons'
        END as frame_category
    FROM PersonClassifications
)
SELECT 
    COUNT(CASE WHEN frame_category = 'only_child' THEN 1 END) AS only_child_person,
    COUNT(CASE WHEN frame_category = 'only_adult' THEN 1 END) AS only_adult_person,
    COUNT(CASE WHEN frame_category = 'both_persons' THEN 1 END) AS both_persons,
    COUNT(CASE WHEN frame_category = 'no_persons' THEN 1 END) AS no_persons
FROM FrameCategories
"""

combined_face_person_query = """
SELECT
    SUM(CASE WHEN child_present = 1 AND adult_present = 0 THEN 1 ELSE 0 END) AS only_child_present,
    SUM(CASE WHEN child_present = 0 AND adult_present = 1 THEN 1 ELSE 0 END) AS only_adult_present,
    SUM(CASE WHEN child_present = 1 AND adult_present = 1 THEN 1 ELSE 0 END) AS both_present,
    SUM(CASE WHEN child_present = 0 AND adult_present = 0 THEN 1 ELSE 0 END) AS no_one_present,
    COUNT(*) AS total_frames_analyzed
FROM (
    SELECT
        pd.frame_number,
        pd.video_id,
        COALESCE(fa.has_child_face, 0) AS has_child_face,
        COALESCE(fa.has_adult_face, 0) AS has_adult_face,
        pd.has_child_person,
        pd.has_adult_person,
        CASE WHEN COALESCE(fa.has_child_face,0)=1 OR pd.has_child_person=1 THEN 1 ELSE 0 END AS child_present,
        CASE WHEN COALESCE(fa.has_adult_face,0)=1 OR pd.has_adult_person=1 THEN 1 ELSE 0 END AS adult_present
    FROM PersonClassifications pd
    LEFT JOIN FaceAgg fa
    ON pd.frame_number = fa.frame_number AND pd.video_id = fa.video_id
)
"""

def run_combined_query(conn, output_path):
    """
    Creates a temporary FaceAgg table and runs the combined face-person query.
    """
    # Create temporary table
    conn.execute("""
    CREATE TEMP TABLE IF NOT EXISTS FaceAgg AS
    SELECT 
        frame_number,
        video_id,
        MAX(CASE WHEN age_class = 0 THEN 1 ELSE 0 END) AS has_child_face,
        MAX(CASE WHEN age_class = 1 THEN 1 ELSE 0 END) AS has_adult_face
    FROM FaceDetections
    GROUP BY frame_number, video_id;
    """)
    
    # Run combined query
    df = pd.read_sql(combined_face_person_query, conn)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved combined face-person analysis to {output_path}")
    return df

def run_query_to_csv(conn, query, output_path, description):
    """
    Executes an SQL query, saves the result to CSV, and logs the output.
    """
    df = pd.read_sql(query, conn)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {description} to {output_path}")
    return df

def round_to_nearest(x, base=ROUND_TO):
    return int(base * round(float(x) / base))

def load_person_face_data(conn):
    query = """
        SELECT
            p.video_id,
            p.frame_number,
            CASE WHEN p.has_child_person=1 OR f_c.age_class=0 THEN 1 ELSE 0 END AS child_present,
            CASE WHEN p.has_adult_person=1 OR f_a.age_class=1 THEN 1 ELSE 0 END AS adult_present
        FROM PersonClassifications p
        LEFT JOIN (SELECT video_id, frame_number, age_class FROM FaceDetections WHERE age_class = 0) f_c
            ON p.video_id = f_c.video_id AND p.frame_number = f_c.frame_number
        LEFT JOIN (SELECT video_id, frame_number, age_class FROM FaceDetections WHERE age_class = 1) f_a
            ON p.video_id = f_a.video_id AND p.frame_number = f_a.frame_number
    """
    return pd.read_sql(query, conn)

def load_vocalizations(conn):
    vocal_df = pd.read_sql("SELECT video_id, start_time_seconds, end_time_seconds, speaker FROM Vocalizations;", conn)
    vocal_df['start_frame'] = (vocal_df['start_time_seconds'] * FPS).apply(round_to_nearest)
    vocal_df['end_frame'] = (vocal_df['end_time_seconds'] * FPS).apply(round_to_nearest)
    return vocal_df

def expand_speech(vocal_df):
    vocal_df['frame_range'] = vocal_df.apply(
        lambda row: list(range(row['start_frame'], row['end_frame'] + ROUND_TO, ROUND_TO)), axis=1
    )
    return vocal_df.explode('frame_range').rename(columns={'frame_range': 'frame_number'})[['video_id', 'frame_number', 'speaker']]

def main():
    OUTPUT_DIR = DataPaths.INFERENCE_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FPS = 30
    ROUND_TO = 10

    # Connect to database
    with sqlite3.connect(DataPaths.INFERENCE_DB_PATH) as conn:
        # Face categories
        face_df = run_query_to_csv(
            conn,
            face_frame_categories_query,
            OUTPUT_DIR / "face_frame_categories.csv",
            "face frame categories"
        )

        # Person categories
        person_df = run_query_to_csv(
            conn,
            person_frame_categories_query,
            OUTPUT_DIR / "person_frame_categories.csv",
            "person frame categories"
        )

        # Combined face and person analysis
        combined_df = run_combined_query(
            conn,
            OUTPUT_DIR / "combined_face_person_categories.csv"
        )
        
        person_alone_df = load_person_face_data(conn)
        vocal_df = load_vocalizations(conn)
        speech_df = expand_speech(vocal_df)
        alone_not_alone_df = person_alone_df.merge(speech_df, on=['video_id', 'frame_number'], how='left')

        # Add flags
        alone_not_alone_df['kchi_speech_present'] = (alone_not_alone_df['speaker'] == 'KCHI').astype(int)
        alone_not_alone_df['other_speech_present'] = (alone_not_alone_df['speaker'] == 'OTH').astype(int)
        alone_not_alone_df['not_alone_detection'] = (alone_not_alone_df['adult_present'] == 1) | (alone_not_alone_df['child_present'] == 1) | (alone_not_alone_df['speaker'] == 'OTH')
        alone_not_alone_df = alone_not_alone_df[['only_child_present', 'only_adult_present', 'both_present', 'no_one_present', 'total_frames_analyzed']]
        alone_not_alone_df.to_csv(OUTPUT_DIR / "alone_not_alone_analysis.csv", index=False)

    print("🎯 All queries executed successfully.")

if __name__ == "__main__":
    main()