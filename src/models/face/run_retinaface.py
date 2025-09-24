import os
import cv2
import sqlite3
from retinaface import RetinaFace
from tqdm import tqdm

# =========================
# CONFIG
# =========================
FRAMES_ROOT = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed"
OUTPUT_DB = "/home/nele_pauline_suffo/outputs/quantex_inference/inference_short.db"

VIDEOS_TO_PROCESS = ["quantex_at_home_id255237_2022_05_08_03",
                     "quantex_at_home_id254922_2022_05_21_01",
                     "quantex_at_home_id254922_2022_04_12_01",
                     "quantex_at_home_id255237_2022_05_26_01",
                     "quantex_at_home_id255695_2022_02_12_01",
                     "quantex_at_home_id255695_2022_02_21_01",
                     "quantex_at_home_id255695_2022_02_21_02",
                     "quantex_at_home_id255706_2022_04_16_01",
                     "quantex_at_home_id255944_2022_03_25_01",
                     "quantex_at_home_id255944_2022_03_25_02",
                     "quantex_at_home_id256354_2021_08_14_02",
                     "quantex_at_home_id255695_2022_02_12_02"]

# =========================
# DATABASE SETUP
# =========================
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RetinaFace (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT,
            frame_path TEXT,
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            landmark_eye_left_x REAL, landmark_eye_left_y REAL,
            landmark_eye_right_x REAL, landmark_eye_right_y REAL,
            landmark_nose_x REAL, landmark_nose_y REAL,
            landmark_mouth_left_x REAL, landmark_mouth_left_y REAL,
            landmark_mouth_right_x REAL, landmark_mouth_right_y REAL
        )
    """)
    conn.commit()
    return conn

# =========================
# PROCESSING
# =========================
def process_frames(root_dir, conn):
    cursor = conn.cursor()

    # Process only the specified videos in VIDEOS_TO_PROCESS
    for video_id in VIDEOS_TO_PROCESS:
        subdir = os.path.join(root_dir, video_id)
        
        # Check if the video folder exists
        if not os.path.exists(subdir):
            print(f"⚠️ Warning: Video folder not found: {subdir}")
            continue
            
        # Get all files in this specific video folder
        try:
            files = os.listdir(subdir)
        except OSError as e:
            print(f"❌ Error accessing folder {subdir}: {e}")
            continue

        # Process only if there are image files (either .jpg or .PNG)
        files = [f for f in files if f.endswith(".jpg")] + [f for f in files if f.endswith(".PNG")]
        if not files:
            print(f"⚠️ No image files found in: {video_id}")
            continue

        for file in tqdm(files, desc=f"Processing {video_id}"):
            img_path = os.path.join(subdir, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            detections = RetinaFace.detect_faces(img)
            if not isinstance(detections, dict):
                continue  # No detections

            for face in detections.values():
                facial_area = face['facial_area']
                # Ensure we have the correct format and convert to integers
                if isinstance(facial_area, (list, tuple)) and len(facial_area) == 4:
                    x1, y1, x2, y2 = map(int, facial_area)
                else:
                    print(f"Warning: Invalid facial_area format: {facial_area}")
                    continue
                    
                landmarks = face['landmarks']

                # Insert into DB
                cursor.execute("""
                    INSERT INTO RetinaFace (
                        video_id, frame_path, 
                        x1, y1, x2, y2,
                        landmark_eye_left_x, landmark_eye_left_y,
                        landmark_eye_right_x, landmark_eye_right_y,
                        landmark_nose_x, landmark_nose_y,
                        landmark_mouth_left_x, landmark_mouth_left_y,
                        landmark_mouth_right_x, landmark_mouth_right_y
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video_id, img_path,
                    x1, y1, x2, y2,
                    float(landmarks['left_eye'][0]), float(landmarks['left_eye'][1]),
                    float(landmarks['right_eye'][0]), float(landmarks['right_eye'][1]),
                    float(landmarks['nose'][0]), float(landmarks['nose'][1]),
                    float(landmarks['mouth_left'][0]), float(landmarks['mouth_left'][1]),
                    float(landmarks['mouth_right'][0]), float(landmarks['mouth_right'][1])
                ))

        conn.commit()

# =========================
# MAIN
# =========================
def main():
    print("Initializing database...")
    conn = init_db(OUTPUT_DB)

    print("Processing frames...")
    process_frames(FRAMES_ROOT, conn)

    conn.close()
    print(f"✅ Finished. Results stored in: {OUTPUT_DB}")


if __name__ == "__main__":
    main()