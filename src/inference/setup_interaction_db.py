import sqlite3
import logging
import pandas as pd
import cv2
from pathlib import Path
from constants import DataPaths

logging.basicConfig(level=logging.INFO)

def get_max_frame_count(video_path: Path) -> int:
    """
    Returns the maximum number of frames in a video using OpenCV.
    If the video cannot be opened, returns None.
    """
    if not video_path.exists():
        logging.error(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def store_video_data(age_group_df: pd.DataFrame, conn: sqlite3.Connection):
    """
    Stores video information from age_group.csv directly in the Videos table.
    Also computes and stores max_frame for each video.
    """
    cursor = conn.cursor()
    video_data = []

    for _, row in age_group_df.iterrows():
        try:
            video_name = row['video_name']
            child_id = row['child_id']
            recording_date_str = row['recording_date']
            age_at_recording = row['age_at_recording']
            age_group = row['age_group']

            # Parse recording date (format: dd.mm.yyyy)
            recording_date = pd.to_datetime(recording_date_str, format="%d.%m.%Y")

            # Determine max frame
            # check for .MP4 and .mp4
            
            video_base = Path(DataPaths.QUANTEX_VIDEOS_INPUT_DIR) / video_name

            # Find the actual file independent of extension
            matched_files = list(video_base.parent.glob(f"{video_base.stem}.*"))

            if not matched_files:
                raise FileNotFoundError(f"No file found for base name: {video_name}")

            # Normalize extension to handle .mp4 / .MP4 / .mP4 / .Mp4 etc.
            video_path = matched_files[0]
            ext = video_path.suffix.lower()

            if ext == ".mp4":
                # use video_path from here
                max_frame = get_max_frame_count(video_path)
            else:
                raise ValueError(f"Unsupported extension: {ext}")

            # Prepare table output log
            video_data.append({
                'video_name': video_name,
                'child_id': child_id,
                'recording_date': recording_date.strftime('%Y-%m-%d'),
                'age_at_recording': f"{age_at_recording:.2f}",
                'age_group': age_group,
                'max_frame': max_frame
            })

            # Insert or update video data including max_frame
            cursor.execute('''
                INSERT INTO Videos (video_name, child_id, recording_date, age_at_recording, age_group, max_frame)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_name) DO UPDATE SET
                    child_id=excluded.child_id,
                    recording_date=excluded.recording_date,
                    age_at_recording=excluded.age_at_recording,
                    age_group=excluded.age_group,
                    max_frame=excluded.max_frame
            ''', (
                video_name,
                child_id,
                recording_date.strftime('%Y-%m-%d'),
                age_at_recording,
                age_group,
                max_frame
            ))

        except Exception as e:
            logging.error(f"Error processing video {row.get('video_name', 'unknown')}: {str(e)}")

    conn.commit()

def setup_interaction_db(db_path: Path):
    """
    Sets up the SQLite database for storing detection results.
    Only creates database and tables if they don't exist.
    """
    if db_path.exists():
        logging.info(f"Database already exists at {db_path}. Skipping creation.")
        return
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Videos (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT UNIQUE,
            child_id INTEGER,
            recording_date DATE,
            age_at_recording FLOAT,
            age_group INTEGER,
            max_frame INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            description TEXT,
            output_variables TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS FaceDetections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            confidence_score REAL,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            age_class INTEGER CHECK(age_class IN (0, 1)),
            proximity REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PersonDetections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            confidence_score REAL,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            age_class INTEGER CHECK(age_class IN (0, 1)),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS BookDetections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            confidence_score REAL,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS AudioClassifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            has_kchi INTEGER CHECK(has_kchi IN (0, 1)),
            kchi_confidence_score REAL,
            has_cds INTEGER CHECK(has_cds IN (0, 1)),
            cds_confidence_score REAL,
            has_ohs INTEGER CHECK(has_ohs IN (0, 1)),
            ohs_confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    # Insert default models
    cursor.execute('''
        INSERT OR IGNORE INTO Models (model_name, description, output_variables) VALUES 
        ('yolo_face_detection', 'YOLO model for face detection with age classification', 
        '{"age_class": {"0": "child_face", "1": "adult_face"}, "proximity": "continuous_value"}'),
        ('yolo_person_detection', 'CNN+RNN model for person presence classification',
        '{"has_adult_person": {"0": "no", "1": "yes"}, "has_child_person": {"0": "no", "1": "yes"}}'),
        ('audio_voice_classification', 'Audio model for voice type classification',
        '{"has_kchi": {"0": "no", "1": "yes"}, "has_cds": {"0": "no", "1": "yes"}, "has_ohs": {"0": "no", "1": "yes"}}'),
        ('kchi_vocalization', 'ALICE for KCHI vocalization analysis',
        '{"phonemes": "float", "syllables": "float", "words": "float"}'),
        ('yolo_book_detection', 'YOLO model for book detection',
        '')
    ''')
    
    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")


def main(db_path: Path = DataPaths.INFERENCE_DB_PATH):
    """
    Main function to set up the database and store video data.
    
    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database file
    """
    # Step 1: Setup database
    setup_interaction_db(db_path)

    # Step 2: Load CSV
    try:
        age_group_df = pd.read_csv(
            DataPaths.SUBJECTS_CSV_PATH,
            header=0, sep=';', encoding='utf-8'
        )
        age_group_df['age_at_recording'] = (
            age_group_df['age_at_recording']
            .astype(str)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
    except Exception as e:
        logging.error(f"Failed to load CSV at {DataPaths.SUBJECTS_CSV_PATH}: {str(e)}")
        return

    # Step 3: Store video data
    try:
        conn = sqlite3.connect(db_path)
        store_video_data(age_group_df, conn)
    except Exception as e:
        logging.error(f"Database operation failed: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()