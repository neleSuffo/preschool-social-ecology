import sqlite3
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path
from constants import DataPaths

logging.basicConfig(level=logging.INFO)
    
def store_video_data(age_group_df: pd.DataFrame, conn):
    """
    Stores video information from age_group.csv directly in the Videos table.
    Logs video information in a tabular format.
    
    Parameters:
    ----------
    age_group_df : pd.DataFrame
        DataFrame containing video information from age_group.csv
    conn : sqlite3.Connection
        SQLite connection object
    """
    cursor = conn.cursor()
    video_data = []

    for _, row in age_group_df.iterrows():
        try:
            # Get data from the CSV
            video_name = row['video_name']
            child_id = row['child_id']
            recording_date_str = row['recording_date']
            age_at_recording = row['age_at_recording']
            age_group = row['age_group']
            
            # Parse recording date (format: dd.mm.yyyy)
            recording_date = pd.to_datetime(recording_date_str, format="%d.%m.%Y")
            
            # Construct video path
            video_path = str(DataPaths.VIDEOS_INPUT_DIR / f"{video_name}.MP4")

            # Add data to video_data list for logging
            video_data.append({
                'video_name': video_name,
                'child_id': child_id,
                'recording_date': recording_date.strftime('%Y-%m-%d'),
                'age_at_recording': f"{age_at_recording:.2f}",
                'age_group': age_group
            })

            # Insert or update video data
            cursor.execute('''
                INSERT INTO Videos (video_name, video_path, child_id, recording_date, age_at_recording, age_group)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_name) DO UPDATE SET
                    video_path=excluded.video_path,
                    child_id=excluded.child_id,
                    recording_date=excluded.recording_date,
                    age_at_recording=excluded.age_at_recording,
                    age_group=excluded.age_group
                ''', (video_name, video_path, child_id, recording_date.strftime('%Y-%m-%d'), 
                    age_at_recording, age_group))

        except Exception as e:
            logging.error(f"Error processing video {row.get('video_name', 'unknown')}: {str(e)}")
            continue
    
    conn.commit()
    
    # Create DataFrame and log it
    if video_data:
        df = pd.DataFrame(video_data)
        logging.info("\nVideo Information:\n" + df.to_string())
    
    logging.info(f"Stored {len(video_data)} videos in the database.")
    
def setup_detection_database(db_path: Path = DataPaths.INFERENCE_DB_PATH):
    """
    This function sets up the SQLite database for storing detection results.
    If the database already exists, it skips creation.
    
    Parameters:
    ----------
    db_path : Path
        Path to the SQLite database file, defaults to DetectionPaths.detection_db_path
    """
    # Check if database already exists
    if db_path.exists():
        logging.info(f"Database already exists at {db_path}. Skipping creation.")
        return
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database (create it since we know it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Videos (
            video_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_name TEXT UNIQUE,
            video_path TEXT,
            child_id INTEGER,
            recording_date DATE,
            age_at_recording FLOAT,
            age_group INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE Models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE,
            description TEXT,
            output_variables TEXT  -- JSON string describing the output variables
        )
    ''')
    
    # Face detections with bounding boxes and proximity
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
            age_class INTEGER CHECK(age_class IN (0, 1)),  -- 0: child, 1: adult
            proximity REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    # Person classification (frame-level, no bounding boxes)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS PersonClassifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            has_adult_person INTEGER CHECK(has_adult_person IN (0, 1)),  -- 0: no, 1: yes
            adult_confidence_score REAL,
            has_child_person INTEGER CHECK(has_child_person IN (0, 1)),  -- 0: no, 1: yes
            child_confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    # Voice/Audio classifications
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS AudioClassifications (
            classification_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            model_id INTEGER,
            has_kchi INTEGER CHECK(has_kchi IN (0, 1)),  -- 0: no, 1: yes
            kchi_confidence_score REAL,
            has_cds INTEGER CHECK(has_cds IN (0, 1)),  -- 0: no, 1: yes
            cds_confidence_score REAL,
            has_ohs INTEGER CHECK(has_ohs IN (0, 1)),  -- 0: no, 1: yes
            ohs_confidence_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')

    # Insert default models with their output variable descriptions
    cursor.execute('''
        INSERT OR IGNORE INTO Models (model_name, description, output_variables) VALUES 
        ('yolo_face_detection', 'YOLO model for face detection with age classification', 
         '{"age_class": {"0": "child_face", "1": "adult_face"}, "proximity": "continuous_value"}'),
        ('cnn_rnn_person_classification', 'CNN+RNN model for person presence classification',
         '{"has_adult_person": {"0": "no", "1": "yes"}, "has_child_person": {"0": "no", "1": "yes"}}'),
        ('audio_voice_classification', 'Audio model for voice type classification',
         '{"has_kchi": {"0": "no", "1": "yes"}, "has_cds": {"0": "no", "1": "yes"}, "has_ohs": {"0": "no", "1": "yes"}}')
    ''')
    
    # Load age group data from CSV
    age_group_df = pd.read_csv(
        DataPaths.SUBJECTS_CSV_PATH,
        header=0, sep=',', encoding='utf-8'
    )
    
    # Store video data directly in Videos table
    store_video_data(age_group_df, conn)
    
    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")