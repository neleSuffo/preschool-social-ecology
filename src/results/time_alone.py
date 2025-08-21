import sqlite3
import pandas as pd
import numpy as np

# Connect to your database
db_path = '/home/nele_pauline_suffo/outputs/quantex_inference/interaction_inference.db'
conn = sqlite3.connect(db_path)

# Load all data into a single DataFrame
df = pd.read_sql("""
    SELECT
        p.video_id,
        p.frame_number,
        CASE WHEN p.has_child_person=1 OR f_c.has_child_face=1 THEN 1 ELSE 0 END AS child_present,
        CASE WHEN p.has_adult_person=1 OR f_a.has_adult_face=1 THEN 1 ELSE 0 END AS adult_present,
        CASE WHEN a.speech_id IS NOT NULL THEN 1 ELSE 0 END AS speech_present,
        a.speaker_type
    FROM PersonClassifications p
    LEFT JOIN (SELECT video_id, frame_number, has_child_face FROM FaceDetections WHERE age_class = 0) f_c
        ON p.video_id = f_c.video_id AND p.frame_number = f_c.frame_number
    LEFT JOIN (SELECT video_id, frame_number, has_adult_face FROM FaceDetections WHERE age_class = 1) f_a
        ON p.video_id = f_a.video_id AND p.frame_number = f_a.frame_number
    LEFT JOIN AudioDiarization a
        ON p.video_id = a.video_id AND p.frame_number >= a.start_frame AND p.frame_number <= a.end_frame
    ORDER BY p.video_id, p.frame_number;
""", conn)
conn.close()

# Ensure integer types for video_id and frame_number
df['video_id'] = df['video_id'].astype(int)
df['frame_number'] = df['frame_number'].astype(int)

# Create a boolean column for "not alone"
df['not_alone_detection'] = (df['adult_present'] == 1) | \
                           (df['speaker_type'].isin(['adult_male', 'adult_female', 'other_child', 'key_child_parent']))

# The `speaker_type` field might require careful mapping,
# but the general logic is to mark any presence other than the key child as "not alone."
# You've mentioned `another_child` and `another_adult`,
# so adjust the `isin` list to match your exact data.