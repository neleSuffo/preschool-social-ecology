import sqlite3
import logging
import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path
from constants import DetectionPaths, DetectionPipeline, BasePaths

logging.basicConfig(level=logging.INFO)

def get_age_group(age: float) -> int:
    """
    Determines age group based on age.
    
    Parameters:
    ----------
    age : float
        Age in years
        
    Returns:
    -------
    int
        Age group (3, 4, or 5)
    """
    if age < 4:
        return 3
    elif age < 5:
        return 4
    else:
        return 5
    
def store_subject_data(subjects_df: pd.DataFrame, video_paths: list, conn):
    """
    Stores subject information (ID, birthday, gender, age at recording, age group) in the Subjects table.
    Logs video information in a tabular format.
    """
    cursor = conn.cursor()
    video_data = []

    for video_path in video_paths:
        video_name = Path(video_path).stem

        import re
        id_match = re.search(r'(?<=id)\d{6}', video_name)
        date_match = re.search(r'(\d{4})_(\d{2})_(\d{2})', video_name)

        if id_match and date_match:
            try:
                child_id = int(id_match.group(0))
                year, month, day = map(int, date_match.groups())
                recording_date = pd.to_datetime(f"{day}.{month}.{year}", format="%d.%m.%Y")

                if child_id not in subjects_df['id'].values:
                    logging.warning(f"Child ID {child_id} not found in subjects data for video {video_name}")
                    continue
                
                child_birthday = subjects_df.loc[subjects_df['id'] == child_id, 'birthday'].iloc[0]
                gender = subjects_df.loc[subjects_df['id'] == child_id, 'gender'].iloc[0]
                
                # Calculate age at recording
                delta = relativedelta(recording_date, child_birthday)
                age_at_recording = round(delta.years + (delta.months / 12) + (delta.days / 365.25),2)

                age_group = get_age_group(age_at_recording)

                # Add data to video_data list
                video_data.append({
                    'video_name': video_name,
                    'child_id': child_id,
                    'birthday': child_birthday.strftime('%Y-%m-%d'),
                    'recording_date': recording_date.strftime('%Y-%m-%d'),
                    'gender': gender,
                    'age_at_recording': f"{age_at_recording:.2f}",
                    'age_group': age_group
                })

                # Insert or update subject data
                cursor.execute('''
                    INSERT INTO Subjects (video_name, child_id, birthday, recording_date, gender, age_at_recording, age_group)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(video_name) DO UPDATE SET
                        child_id=excluded.child_id,
                        birthday=excluded.birthday,
                        recording_date=excluded.recording_date,
                        gender=excluded.gender,
                        age_at_recording=excluded.age_at_recording,
                        age_group=excluded.age_group
                    ''', (video_name, child_id, child_birthday.strftime('%Y-%m-%d'), 
                        recording_date.strftime('%Y-%m-%d'), gender, age_at_recording, age_group))

            except Exception as e:
                logging.error(f"Error processing video {video_name}: {str(e)}")
                continue
    
    conn.commit()
    
    # Create DataFrame and log it
    if video_data:
        df = pd.DataFrame(video_data)
        logging.info("\nVideo and Subject Information:\n" + df.to_string())
    
    logging.info("Stored subject data in the database.")
    
def setup_detection_database(db_path: Path = DetectionPaths.detection_db_path):
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
            video_path TEXT UNIQUE,
            child_id INTEGER,
            recording_date DATE,
            age_at_recording FLOAT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Frames (
            frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            UNIQUE(video_id, frame_number),
            FOREIGN KEY (video_id) REFERENCES Videos(video_id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE Models (
            model_id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT UNIQUE
        )
    ''')

    cursor.execute('''
        CREATE TABLE Classes (
            class_id INTEGER,
            model_id INTEGER,
            class_name TEXT,
            PRIMARY KEY (model_id, class_name),
            FOREIGN KEY (model_id) REFERENCES Models(model_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER,
            frame_number INTEGER,
            object_class TEXT,
            confidence_score REAL,
            x_min REAL,
            y_min REAL,
            x_max REAL,
            y_max REAL,
            age_class INTEGER,
            age_confidence REAL,
            face_adjusted BOOLEAN,
            gaze_direction INTEGER,
            gaze_confidence REAL,
            proximity REAL,
            FOREIGN KEY (video_id, frame_number) REFERENCES Frames(video_id, frame_number)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VideoStatistics (
            video_id INTEGER PRIMARY KEY,
            total_frames INTEGER,
            processed_frames INTEGER,
            child_count INTEGER,
            adult_count INTEGER,
            child_face_count INTEGER,
            adult_face_count INTEGER,
            book_count INTEGER,
            toy_count INTEGER,
            kitchenware_count INTEGER,
            screen_count INTEGER,
            food_count INTEGER,
            other_object_count INTEGER,
            FOREIGN KEY (video_id) REFERENCES Videos(video_id)
        )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Subjects (
        video_name TEXT PRIMARY KEY,
        child_id INTEGER,  
        birthday DATE,
        recording_date DATE,
        gender TEXT,
        age_at_recording FLOAT,
        age_group INTEGER
        )
    ''')

    # Get list of video paths
    video_paths = list(Path(DetectionPaths.quantex_videos_input_dir).rglob("*.MP4"))
    logging.info(f"Found {len(video_paths)} video files.")
       
    # Load subjects data from CSV
    quantex_subjects_df = pd.read_csv(
        DetectionPipeline.quantex_subjects,
        header=0, sep=';', encoding='utf-8', parse_dates=['birthday'], dayfirst=True
    )
    
    # Store subject data in the database
    store_subject_data(quantex_subjects_df, video_paths, conn)
    
    conn.commit()
    conn.close()
    logging.info(f"Detection database created at {db_path}")