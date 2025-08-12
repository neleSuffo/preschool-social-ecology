import sqlite3
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from constants import DataPaths, PersonClassification
from config import PersonConfig, DataConfig

logging.basicConfig(level=logging.INFO)

# --- GLOBAL CONSTANTS ---
DB_PATH = DataPaths.ANNO_DB_PATH
FRAME_BASE_PATH = DataPaths.IMAGES_INPUT_DIR
DATABASE_CATEGORY_IDS = PersonConfig.DATABASE_CATEGORY_IDS # (1: 'person', 2: 'reflection')
AGE_GROUP_TO_CLASS_ID = PersonConfig.AGE_GROUP_TO_CLASS_ID # (inf:0, child: 0, teen: 1, adult: 1)
MODEL_CLASS_ID_TO_LABEL = PersonConfig.MODEL_CLASS_ID_TO_LABEL # (0: 'child', 1: 'adult')
TARGET_LABELS = PersonConfig.TARGET_LABELS # ['child', 'adult']
TRAIN_SPLIT_RATIO = DataConfig.TRAIN_SPLIT_RATIO
OUTPUT_DIR = PersonClassification.PERSON_DATA_INPUT_DIR

def fetch_presence_per_frame(category_ids: List[int]) -> List[Tuple]:
    """
    Fetches all annotations for specified categories, including age group, from the SQLite database.

    Parameters
    ----------
    category_ids (List[int])
        A list of integer IDs (e.g., [1, 2]) for the categories to fetch.

    Returns
    -------
    List[Tuple]
        A list of tuples, where each tuple contains (video_id, frame_id, file_name, category_id, age_group)
        for all detected instances of the specified categories.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    placeholders = ", ".join("?" for _ in category_ids)
    query = f"""
    SELECT
        i.video_id,
        i.frame_id,
        i.file_name,
        a.category_id,
        a.person_age AS age_group
    FROM annotations a
    JOIN images i
        ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v
        ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders})
      AND a.outside = 0
    ORDER BY i.video_id, i.frame_id
    """
    cursor.execute(query, category_ids)
    rows = cursor.fetchall()
    conn.close()

    logging.info(f"Fetched {len(rows)} annotation rows")
    return rows

def build_frame_level_labels(rows: List[Tuple], age_group_mapping: Dict[str, int], model_class_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Aggregates all detections in the same frame into binary presence labels for 'adult' and 'child'.

    This function takes a list of annotation rows (which now includes age_group) and converts them
    into a DataFrame where each row represents a unique video frame. The columns 'adult' and 'child'
    are set to 1 if a person of that age group is present.

    Parameters
    ----------
    rows (List[Tuple])
        A list of tuples, where each tuple contains (video_id, frame_id, file_name, category_id, age_group).
    age_group_mapping (Dict[str, int])
        A dictionary mapping age group strings (e.g., 'child', 'adult') to integer class IDs (e.g., 0, 1).
    model_class_mapping (Dict[int, str])
        A dictionary mapping integer class IDs (e.g., 0, 1) to their corresponding string labels (e.g., 'child', 'adult').

    Returns
    -------
        pd.DataFrame
            A DataFrame with columns ['video_id', 'frame_id', 'file_path', 'adult', 'child']
            containing binary presence labels.
    """
    frame_dict = {}

    for video_id, frame_id, file_name, cat_id, age_group in rows:
        # We only care about category_id 1 (person), not 2 (reflection)
        if cat_id != 1:
            continue

        key = (video_id, frame_id, file_name)
        if key not in frame_dict:
            # Initialize with 0 for both child and adult
            frame_dict[key] = {label: 0 for label in TARGET_LABELS}

        # Map the age_group to the target label using both mappings
        if age_group in age_group_mapping:
            label_id = age_group_mapping[age_group]
            label_name = model_class_mapping.get(label_id)

            if label_name:
                frame_dict[key][label_name] = 1

    # Convert the dictionary of frames to a DataFrame
    data = []
    for (video_id, frame_id, file_name), labels in frame_dict.items():
        row = {
            "video_id": video_id,
            "frame_id": frame_id,
            "file_path": str(FRAME_BASE_PATH / file_name),
            **labels
        }
        data.append(row)
    df = pd.DataFrame(data)
    return df

def split_by_child_id(df: pd.DataFrame, train_ratio: float = TRAIN_SPLIT_RATIO) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets, stratified by child_id balance.
    (This function remains unchanged from the previous version)
    """
    val_ratio = (1 - train_ratio) / 2
    
    conn = sqlite3.connect(DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    def extract_child_id(file_name: str) -> int:
        try:
            return int(file_name.split('id')[1].split('_')[0])
        except (IndexError, ValueError) as e:
            logging.error(f"Failed to extract child_id from file_name: {file_name} - {e}")
            return None
    
    video_map['child_id'] = video_map['file_name'].apply(extract_child_id)
    df = df.merge(video_map.drop(columns='file_name'), on="video_id", how="left")

    df.dropna(subset=['child_id'], inplace=True)
    df['child_id'] = df['child_id'].astype(int)

    child_group_counts = df.groupby('child_id')[['adult', 'child']].sum()
    child_group_counts['adult_ratio'] = child_group_counts['adult'] / (child_group_counts['adult'] + child_group_counts['child'])
    child_group_counts['adult_ratio'].fillna(0, inplace=True)

    sorted_child_ids = child_group_counts['adult_ratio'].sort_values().index.tolist()
    
    n_total = len(sorted_child_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    train_ids, val_ids, test_ids = [], [], []
    
    split_indices = np.arange(n_total)
    random.shuffle(split_indices)
    
    for i in range(n_total):
        child_id = sorted_child_ids[i]
        
        if len(train_ids) < n_train:
            train_ids.append(child_id)
        elif len(val_ids) < n_val:
            val_ids.append(child_id)
        else:
            test_ids.append(child_id)

    unassigned_ids = set(sorted_child_ids) - set(train_ids) - set(val_ids) - set(test_ids)
    test_ids.extend(list(unassigned_ids))
    
    logging.info(f"Splitting {n_total} unique child IDs: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    df_train = df[df["child_id"].isin(train_ids)].drop(columns=["child_id"])
    df_val = df[df["child_id"].isin(val_ids)].drop(columns=["child_id"])
    df_test = df[df["child_id"].isin(test_ids)].drop(columns=["child_id"])

    logging.info(f"Train set class counts:\n{df_train[['adult', 'child']].sum()}")
    logging.info(f"Validation set class counts:\n{df_val[['adult', 'child']].sum()}")
    logging.info(f"Test set class counts:\n{df_test[['adult', 'child']].sum()}")

    return df_train, df_val, df_test

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = fetch_presence_per_frame(DATABASE_CATEGORY_IDS)
    
    if not rows:
        logging.warning("No annotations found for the specified categories. DataFrame will be empty. Exiting script.")
        exit()

    df = build_frame_level_labels(rows, AGE_GROUP_TO_CLASS_ID, MODEL_CLASS_ID_TO_LABEL)

    expected_columns = ['child', 'adult']
    if not all(col in df.columns for col in expected_columns):
        logging.error(f"DataFrame is missing expected columns: {expected_columns}. Available columns: {df.columns.tolist()}")
        exit()

    df_train, df_val, df_test = split_by_child_id(df)

    df_train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    df_val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    df_test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    logging.info(f"Generated CSV files at {OUTPUT_DIR}")
    logging.info(f"Train samples: {len(df_train)}")
    logging.info(f"Validation samples: {len(df_val)}")
    logging.info(f"Test samples: {len(df_test)}")