import sqlite3
import random
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from constants import DataPaths, PersonClassification, BasePaths
from config import PersonConfig, DataConfig

logging.basicConfig(level=logging.INFO)

def fetch_all_annotations(category_ids: List[int]) -> List[Tuple]:
    """Fetch annotations for given category IDs from the SQLite database.
    
    Parameters
    ----------
    category_ids : List[int]
        List of category IDs to filter annotations.
        
    Returns
    -------
    List[Tuple]       
        List of tuples containing annotation data.
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    placeholders = ", ".join("?" * len(category_ids))
    excluded_videos = [
        'quantex_at_home_id260275_2022_05_27_01.mp4',
        'quantex_at_home_id260275_2022_04_16_01.mp4', 
        'quantex_at_home_id260275_2022_04_12_01.mp4',
        'quantex_at_home_id258704_2022_05_07_03.mp4',
        'quantex_at_home_id258704_2022_05_07_04.mp4',
        'quantex_at_home_id258704_2022_05_10_02.mp4',
        'quantex_at_home_id258704_2022_05_15_01.mp4',
        'quantex_at_home_id262565_2022_05_26_03.mp4',
    ]

    # Create a SQL string like: 'video1', 'video2', ...
    excluded_videos_sql = ", ".join(f"'{v}'" for v in DataConfig.EXCLUDED_VIDEOS)

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
      AND v.file_name NOT IN ({excluded_videos_sql})
    ORDER BY i.video_id, i.frame_id
    """
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        results = cursor.fetchall()
        
    logging.info(f"Excluded {len(excluded_videos)} videos from query")
    logging.info(f"Excluded videos: {excluded_videos}")
    
    return results

def build_frame_level_labels(rows: List[Tuple], age_group_mapping: Dict[str, int], model_class_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Aggregates all detections in the same frame into binary presence labels for 'adult' and 'child'.

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
        if cat_id != 1:
            continue

        key = (video_id, frame_id, file_name)
        if key not in frame_dict:
            frame_dict[key] = {label: 0 for label in PersonConfig.TARGET_LABELS}

        if age_group in age_group_mapping:
            label_id = age_group_mapping[age_group]
            label_name = model_class_mapping.get(label_id)

            if label_name:
                frame_dict[key][label_name] = 1

    data = []
    for (video_id, frame_id, file_name), labels in frame_dict.items():
        # Extract folder name (everything before the last underscore + frame number)
        frame_folder = "_".join(file_name.split("_")[:-1])
        
        # Try to find image with any valid extension
        image_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        # Skip if no valid image file found
        if image_path is None:
            logging.warning(f"Image not found with any valid extension: {file_name}")
            continue

        row = {
            "video_id": video_id,
            "frame_id": frame_id,
            "file_path": image_path,
            **labels
        }
        data.append(row)
    df = pd.DataFrame(data)
    return df

def split_by_child_id(df: pd.DataFrame, train_ratio: float = PersonConfig.TRAIN_SPLIT_RATIO):
    """
    Splits the DataFrame into training, validation, and test sets using child IDs as the unit,
    while keeping each split's adult/child ratio close to the global dataset ratio.
    """

    # --- Map video_id -> child_id ---
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    def extract_child_id(file_name: str) -> str:
        try:
            return 'id' + file_name.split('id')[1].split('_')[0]
        except (IndexError, ValueError):
            return None

    video_map['child_id'] = video_map['file_name'].apply(extract_child_id)
    df = df.merge(video_map.drop(columns='file_name'), on="video_id", how="left")
    df.dropna(subset=['child_id'], inplace=True)

    # --- Counts per child ---
    child_group_counts = df.groupby('child_id')[PersonConfig.TARGET_LABELS].sum()

    # Global target ratio (adult proportion)
    global_counts = child_group_counts[PersonConfig.TARGET_LABELS].sum()
    target_adult_ratio = global_counts[PersonConfig.TARGET_LABELS[1]] / global_counts.sum()

    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    if n_total < 6:
        logging.error(f"Not enough unique children to create balanced splits. Found {n_total}, need at least 6.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []

    n_train = max(int(n_total * train_ratio), 2)
    remaining_ids = n_total - n_train
    n_val = max(int(remaining_ids / 2), 2)
    n_test = n_total - n_train - n_val

    if n_test < 2:
        n_val -= (2 - n_test)
        if n_val < 2:
            logging.error("Could not meet minimum ID requirements for all splits.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []
        n_test = 2

    # --- Initialize splits ---
    split_info = {
        'train': {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
        'val':   {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
        'test':  {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
    }

    # Randomize assignment order to reduce bias
    random.shuffle(sorted_child_ids)

    def deviation_from_target(current_counts, child_counts):
        """Absolute difference from target adult ratio after adding this child."""
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0:
            return 0
        new_ratio = new_counts[PersonConfig.TARGET_LABELS[1]] / new_counts.sum()
        return abs(new_ratio - target_adult_ratio)

    for child_id in sorted_child_ids:
        child_counts = child_group_counts.loc[child_id, PersonConfig.TARGET_LABELS]

        # Find eligible splits (still have room for more IDs)
        eligible_splits = []
        for split_name, target_size in zip(['train', 'val', 'test'], [n_train, n_val, n_test]):
            if len(split_info[split_name]['ids']) < target_size:
                eligible_splits.append(split_name)

        if not eligible_splits:
            logging.warning(f"No split has space left for child {child_id}. Skipping.")
            continue

        # Pick split that gets closest to global ratio
        best_split = min(
            eligible_splits,
            key=lambda s: deviation_from_target(split_info[s]['current_counts'], child_counts)
        )

        # Assign child
        split_info[best_split]['ids'].append(child_id)
        split_info[best_split]['current_counts'] += child_counts

    # --- Build final DataFrames ---
    train_ids = split_info['train']['ids']
    val_ids = split_info['val']['ids']
    test_ids = split_info['test']['ids']

    df_train = df[df["child_id"].isin(train_ids)].drop(columns=["child_id"])
    df_val   = df[df["child_id"].isin(val_ids)].drop(columns=["child_id"])
    df_test  = df[df["child_id"].isin(test_ids)].drop(columns=["child_id"])

    # Log final split information
    for split_name, split_df in zip(["Train", "Val", "Test"], [df_train, df_val, df_test]):
        if len(split_df) > 0:
            ratio = split_df[PersonConfig.TARGET_LABELS[1]].sum() / len(split_df)

    return df_train, df_val, df_test, train_ids, val_ids, test_ids

def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List):
    """
    Generates a statistics file with dataset split information, including percentages.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_person_cls_{timestamp}.txt"
    
    total_images = len(df)
    
    with open(file_path, "w") as f:
        f.write(f"Dataset Split Information - {timestamp}\n\n")
        
        total_0 = df['child'].sum()
        total_1 = df['adult'].sum()

        f.write(f"Initial Distribution:\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Class 0 {PersonConfig.TARGET_LABELS[0]}: {total_0} images ({total_0 / total_images:.2%})\n")
        f.write(f"Class 1 {PersonConfig.TARGET_LABELS[1]}: {total_1} images ({total_1 / total_images:.2%})\n\n")

        f.write("Split Distribution:\n")
        f.write("--------------------------------------------------\n\n")

        def write_split_info(split_name, split_df):
            total_split = len(split_df)
            total_split_percent = total_split / total_images if total_images > 0 else 0

            count_0 = split_df[PersonConfig.TARGET_LABELS[0]].sum()
            count_1 = split_df[PersonConfig.TARGET_LABELS[1]].sum()
            
            ratio_0 = count_0 / total_split if total_split > 0 else 0
            ratio_1 = count_1 / total_split if total_split > 0 else 0
            
            f.write(f"{split_name} Set: ({total_split_percent:.2%})\n")
            f.write(f"Total Images: {total_split}\n")
            f.write(f"{PersonConfig.TARGET_LABELS[0]}: {count_0} ({ratio_0:.2%})\n")
            f.write(f"{PersonConfig.TARGET_LABELS[1]}: {count_1} ({ratio_1:.2%})\n\n")

        write_split_info("Validation", df_val)
        write_split_info("Test", df_test)
        write_split_info("Train", df_train)

        f.write("ID Distribution:\n")
        f.write(f"Training IDs: {len(train_ids)}, {train_ids}\n")
        f.write(f"Validation IDs: {len(val_ids)}, {val_ids}\n")
        f.write(f"Test IDs: {len(test_ids)}, {test_ids}\n\n")

        f.write("ID Overlap Check:\n")
        train_val_overlap = set(train_ids).intersection(val_ids)
        train_test_overlap = set(train_ids).intersection(test_ids)
        val_test_overlap = set(val_ids).intersection(test_ids)
        if train_val_overlap or train_test_overlap or val_test_overlap:
            f.write("Overlap found: Yes\n")
        else:
            f.write("Overlap found: No\n")
    
    logging.info(f"Statistics file generated at {file_path}")

if __name__ == "__main__":  
    # Fetch all annotations for the specified categories
    rows = fetch_all_annotations(PersonConfig.DATABASE_CATEGORY_IDS)
    
    if not rows:
        logging.warning("No annotations found for the specified categories. DataFrame will be empty. Exiting script.")
        exit()

    df = build_frame_level_labels(rows, PersonConfig.AGE_GROUP_TO_CLASS_ID, PersonConfig.MODEL_CLASS_ID_TO_LABEL)

    if not all(col in df.columns for col in PersonConfig.TARGET_LABELS):
        logging.error(f"DataFrame is missing expected columns: {PersonConfig.TARGET_LABELS}. Available columns: {df.columns.tolist()}")
        exit()

    df_train, df_val, df_test, train_ids, val_ids, test_ids = split_by_child_id(df)

    # Save CSVs
    PersonClassification.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(PersonClassification.INPUT_DIR / "train.csv", index=False)
    df_val.to_csv(PersonClassification.INPUT_DIR / "val.csv", index=False)
    df_test.to_csv(PersonClassification.INPUT_DIR / "test.csv", index=False)

    # Generate and save statistics file
    generate_statistics_file(df, df_train, df_val, df_test, train_ids, val_ids, test_ids)

    logging.info(f"Generated CSV files at {PersonClassification.INPUT_DIR}")