import sqlite3
import random
from datetime import datetime
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# --- Conceptual Imports (Ensure these are imported from your actual files) ---
# Assuming PersonConfig, DataConfig, DataPaths, and PersonClassification are available
# from config import PersonConfig, DataConfig
# from constants import DataPaths, PersonClassification
# from utils import get_child_id_from_filename # Assuming a utility function for this exists

# --- Mock Classes for Execution Context (REPLACE WITH REAL IMPORTS) ---
class PersonConfig:
    # User-defined age groups and class IDs
    AGE_GROUP_TO_CLASS_ID = {'Inf': 0, 'Child': 0, 'Teen': 1, 'Adult': 1}
    MODEL_CLASS_ID_TO_LABEL = {0: "child", 1: "adult"}
    DATABASE_CATEGORY_IDS = [1, 2] # Person (1) or other relevant category (2)
    
    # Target classes for the two modes
    TARGET_LABELS_AGE_BINARY = ['child', 'adult'] 
    TARGET_LABELS_PERSON_ONLY = ['person']
    
    TRAIN_SPLIT_RATIO = 0.8
    # NOTE: These values must match your actual config
    
class DataConfig:
    EXCLUDED_VIDEOS = []
    VALID_EXTENSIONS = ['.jpg', '.png']
class DataPaths:
    ANNO_DB_PATH = '/path/to/your/anno.db'
class BasePaths:
    LOGGING_DIR = Path('./logs')
class PersonClassification:
    INPUT_DIR = Path('./data/person_cls_output')
    TRAIN_CSV_PATH = INPUT_DIR / 'train.csv'
    VAL_CSV_PATH = INPUT_DIR / 'val.csv'
    TEST_CSV_PATH = INPUT_DIR / 'test.csv'
    IMAGES_INPUT_DIR = Path('./images_input_dir')

def get_child_id_from_filename(file_name: str) -> str:
    """Utility to extract child ID from video filename (e.g., 'quantex_at_home_id255237_...' -> 'id255237')"""
    try:
        return 'id' + file_name.split('id')[1].split('_')[0]
    except (IndexError, ValueError):
        return None
# --------------------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)

# ==============================
# Database Query Functions
# ==============================

def fetch_all_annotations(category_ids: List[int]) -> List[Tuple]:
    """Fetch all relevant person annotations from the SQLite database.
    
    Parameters:
    ----------
    category_ids : List[int]
        List of category IDs to filter annotations (e.g., [1, 2] for person-related).
        
    Returns:
    -------
    List[Tuple]
        List of tuples containing (video_id, frame_id, file_name, category_id, age_group).
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    placeholders = ", ".join("?" * len(category_ids))

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
      AND a.outside = 0  -- Only include people inside the frame
    ORDER BY i.video_id, i.frame_id
    """
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        results = cursor.fetchall()
        
    logging.info(f"Found {len(results)} annotations.")
    return results

def get_all_video_frames(video_ids: List[int]) -> pd.DataFrame:
    """Get ALL frames from specified videos for sequential modeling."""
    if not video_ids:
        return pd.DataFrame()
        
    placeholders = ", ".join("?" * len(video_ids))
    excluded_videos_sql = ", ".join(f"'{v}'" for v in DataConfig.EXCLUDED_VIDEOS)
    
    query = f"""
    SELECT DISTINCT
        i.video_id,
        i.frame_id,
        i.file_name
    FROM images i
    JOIN videos v ON i.video_id = v.id
    WHERE i.video_id IN ({placeholders})
      AND v.file_name NOT IN ({excluded_videos_sql})
    ORDER BY i.video_id, i.frame_id
    """
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, video_ids)
        results = cursor.fetchall()
    
    df = pd.DataFrame(results, columns=['video_id', 'frame_id', 'file_name'])
    return df

# ==============================
# Data Processing Functions
# ==============================

def build_frame_level_labels(rows: List[Tuple], classification_mode: str) -> pd.DataFrame:
    """
    Convert individual person annotations into frame-level binary labels based on mode.
    
    Parameters:
    ----------
    rows : List[Tuple]
        List of tuples from the database containing (video_id, frame_id, file_name, category_id, age_group).
    classification_mode : str
        'age-binary' for child/adult classification, 'person-only' for person presence
    """
    if classification_mode == 'age-binary':
        target_labels = PersonConfig.TARGET_LABELS_AGE_BINARY
        age_mapping = PersonConfig.AGE_GROUP_TO_CLASS_ID
        class_mapping = PersonConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
    elif classification_mode == 'person-only':
        target_labels = PersonConfig.TARGET_LABELS_PERSON_ONLY
        age_mapping = {k: 0 for k in PersonConfig.AGE_GROUP_TO_CLASS_ID}
        class_mapping = PersonConfig.MODEL_CLASS_ID_TO_LABEL_PERSON_ONLY
    else:
        raise ValueError(f"Invalid classification_mode: {classification_mode}")

    frame_dict = {}

    for video_id, frame_id, file_name, cat_id, age_group in rows:
        # Check if category is relevant to person detection (Cat ID 1 or 2 as defined)
        if cat_id not in PersonConfig.DATABASE_CATEGORY_IDS: continue

        key = (video_id, frame_id, file_name)
        if key not in frame_dict:
            frame_dict[key] = {label: 0 for label in target_labels}

        if classification_mode == 'age-binary':
            # Map age group to a target label ('child' or 'adult')
            if age_group in age_mapping:
                label_id = age_mapping[age_group]
                label_name = class_mapping.get(label_id)
                if label_name and label_name in target_labels:
                    frame_dict[key][label_name] = 1
        
        elif classification_mode == 'person-only':
            # If any person/face annotation exists, mark 'person' as 1
            frame_dict[key]['person'] = 1

    # Convert to DataFrame and verify image files exist (only for annotated frames initially)
    data = []
    for (video_id, frame_id, file_name), labels in frame_dict.items():
        frame_folder = "_".join(file_name.split("_")[:-1])
        image_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
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
    
    return pd.DataFrame(data), target_labels


def build_sequential_dataset(annotated_frames_df: pd.DataFrame, train_ids: List[str], val_ids: List[str], test_ids: List[str], target_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build complete sequential datasets (including ALL frames) for LSTM training.
    """    
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()
    
    video_map['child_id'] = video_map['file_name'].apply(get_child_id_from_filename)
    video_map.dropna(subset=['child_id'], inplace=True)
    
    # Create annotation lookup for quick access
    annotation_lookup = {}
    for _, row in annotated_frames_df.iterrows():
        key = (row['video_id'], row['frame_id'])
        annotation_lookup[key] = {label: row[label] for label in target_labels}
    
    def build_split_data(child_ids: List[str], split_name: str) -> pd.DataFrame:
        """Build complete frame sequence for a split."""
        if not child_ids:
            return pd.DataFrame()
            
        split_videos = video_map[video_map['child_id'].isin(child_ids)]['video_id'].tolist()
        
        if not split_videos:
            logging.warning(f"No videos found for {split_name} children: {child_ids}")
            return pd.DataFrame()
        
        # Get ALL frames from these videos
        all_frames_df = get_all_video_frames(split_videos)
        
        data = []
        
        for _, frame_row in all_frames_df.iterrows():
            video_id = frame_row['video_id']
            frame_id = frame_row['frame_id']
            file_name = frame_row['file_name']
            
            # Check if image file exists
            frame_folder = "_".join(file_name.split("_")[:-1])
            image_path = None
            for ext in DataConfig.VALID_EXTENSIONS:
                potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
                if potential_path.exists():
                    image_path = str(potential_path)
                    break
            
            if image_path is None: continue
            
            # Get labels from annotation lookup or default to 0
            key = (video_id, frame_id)
            if key in annotation_lookup:
                labels = annotation_lookup[key]
            else:
                # Frame without people annotations
                labels = {label: 0 for label in target_labels}
            
            row = {
                "video_id": video_id,
                "frame_id": frame_id,
                "file_path": image_path,
                **labels
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    df_train_full = build_split_data(train_ids, "Train")
    df_val_full = build_split_data(val_ids, "Validation") 
    df_test_full = build_split_data(test_ids, "Test")
    
    return df_train_full, df_val_full, df_test_full


def split_by_child_id(df: pd.DataFrame, target_labels: List[str], train_ratio: float = PersonConfig.TRAIN_SPLIT_RATIO):
    """
    Split dataset by child IDs to prevent data leakage, balancing splits based on target label counts.
    """
    # 1. Merge child_id into main DataFrame
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    video_map['child_id'] = video_map['file_name'].apply(get_child_id_from_filename)
    df = df.merge(video_map.drop(columns='file_name'), on="video_id", how="left")
    df.dropna(subset=['child_id'], inplace=True)
    
    # 2. Calculate class distribution per child
    child_group_counts = df.groupby('child_id')[target_labels].sum()
    
    # Total counts across all targeted labels
    global_counts = child_group_counts[target_labels].sum()

    # Use the ratio of the first target class (e.g., 'child' or 'person') as the balance metric
    target_ratio = global_counts[target_labels[0]] / global_counts.sum() if global_counts.sum() > 0 else 0

    # 3. Determine split sizes
    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    if n_total < 6:
        logging.error(f"Not enough unique children to create balanced splits. Found {n_total}, need at least 6.")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    n_train = max(int(n_total * train_ratio), 2)
    remaining_ids = n_total - n_train
    n_val = max(int(remaining_ids / 2), 2)
    n_test = n_total - n_train - n_val

    if n_test < 2:
        n_val -= (2 - n_test)
        if n_val < 2:
            logging.error("Could not meet minimum ID requirements for all splits.")
            return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        n_test = 2

    # 4. Initialize splits
    split_info = {
        'train': {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
        'val':   {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
        'test':  {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
    }

    random.shuffle(sorted_child_ids)

    def deviation_from_target(current_counts, child_counts):
        """Calculate how much adding this child would deviate from the target ratio."""
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0:
            return 0
        new_ratio = new_counts[target_labels[0]] / new_counts.sum()
        return abs(new_ratio - target_ratio)

    # 5. Assign children to splits
    for child_id in sorted_child_ids:
        child_counts = child_group_counts.loc[child_id, target_labels]

        eligible_splits = []
        for split_name, target_size in zip(['train', 'val', 'test'], [n_train, n_val, n_test]):
            if len(split_info[split_name]['ids']) < target_size:
                eligible_splits.append(split_name)

        if not eligible_splits:
            logging.warning(f"No split has space left for child {child_id}. Skipping.")
            continue

        best_split = min(
            eligible_splits,
            key=lambda s: deviation_from_target(split_info[s]['current_counts'], child_counts)
        )

        split_info[best_split]['ids'].append(child_id)
        split_info[best_split]['current_counts'] += child_counts

    # 6. Return assigned IDs (the dataframes are built later by build_sequential_dataset)
    train_ids = split_info['train']['ids']
    val_ids = split_info['val']['ids']
    test_ids = split_info['test']['ids']

    return train_ids, val_ids, test_ids


# ==============================
# Statistics and Logging
# ==============================

def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List, classification_mode: str, target_labels: List[str]):
    """Generate detailed statistics file about the dataset splits."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_person_cls_{classification_mode}_{timestamp}.txt"
    
    with open(file_path, "w") as f:
        f.write(f"Person Classification Dataset Split Report - {timestamp}\n")
        f.write(f"*** CLASSIFICATION MODE: {classification_mode.upper()} ***\n\n")
        
        full_df = pd.concat([df_train, df_val, df_test])
        total_images = len(full_df)

        f.write(f"=== OVERALL DATASET STATISTICS ===\n")
        f.write(f"Total Images: {total_images}\n")
        
        # Calculate overall person presence (if any of the target labels are 1)
        if target_labels:
            person_present_series = full_df[target_labels].any(axis=1).sum()
        else:
            person_present_series = 0
            
        f.write(f"Frames with ANY person: {person_present_series} ({person_present_series / total_images:.2%})\n")
        f.write(f"Frames with NO person: {total_images - person_present_series} ({1 - person_present_series / total_images:.2%})\n")

        for label in target_labels:
            count = full_df[label].sum()
            f.write(f"  Total '{label}' frames: {count} ({count / total_images:.2%})\n")
        f.write("\n")

        f.write("=== SPLIT DISTRIBUTION ===\n")
        f.write("-" * 50 + "\n\n")

        def write_split_info(split_name, split_df):
            total_split = len(split_df)
            split_percentage = total_split / total_images if total_images > 0 else 0

            f.write(f"{split_name} Set ({split_percentage:.1%} of total):\n")
            f.write(f"  Total Images: {total_split}\n")
            
            for label in target_labels:
                count = split_df[label].sum()
                ratio = count / total_split if total_split > 0 else 0
                f.write(f"  {label.capitalize()}: {count} ({ratio:.2%})\n")
            f.write("\n")

        write_split_info("Training", df_train)
        write_split_info("Validation", df_val)
        write_split_info("Test", df_test)

        f.write("=== CHILD ID ASSIGNMENTS ===\n")
        f.write(f"Training Children ({len(train_ids)}): {sorted(train_ids)}\n")
        f.write(f"Validation Children ({len(val_ids)}): {sorted(val_ids)}\n")
        f.write(f"Test Children ({len(test_ids)}): {sorted(test_ids)}\n\n")

        f.write("=== DATA LEAKAGE CHECK ===\n")
        train_val_overlap = set(train_ids).intersection(set(val_ids))
        train_test_overlap = set(train_ids).intersection(set(test_ids))
        val_test_overlap = set(val_ids).intersection(set(test_ids))
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            f.write("❌ OVERLAP DETECTED: Yes\n")
        else:
            f.write("✅ OVERLAP DETECTED: No\n")
    
    logging.info(f"Statistics report saved to: {file_path}")

# ==============================
# Main Execution
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Create input CSV files for person classification (sequential data)')
    parser.add_argument('--classification-mode', choices=["person-only", "age-binary"], default="age-binary",
                       help='Select the classification mode: person-only (single binary class) or age-binary (child/adult).')
    
    args = parser.parse_args()
    mode = args.classification_mode
    
    logging.info(f"Starting person classification dataset pipeline in '{mode}' mode...")

    # Step 1: Fetch annotations from database
    rows = fetch_all_annotations(PersonConfig.DATABASE_CATEGORY_IDS)
    
    if not rows:
        logging.error("No annotations found for the specified categories. Exiting.")
        return

    # Step 2: Build frame-level labels based on mode
    df_annotated, target_labels = build_frame_level_labels(rows, mode)

    if df_annotated.empty:
        logging.error("No valid frame-level annotations created. Exiting.")
        return

    # Step 3: Split data by child IDs (using annotated frames for assignment)
    logging.info(f"Splitting children by IDs based on '{target_labels[0]}' label...")
    train_ids, val_ids, test_ids = split_by_child_id(df_annotated, target_labels)

    if not train_ids or not val_ids or not test_ids:
        logging.error("Failed to create valid child ID splits.")
        return

    # Step 4: Build complete sequential datasets with ALL frames
    df_train_full, df_val_full, df_test_full = build_sequential_dataset(df_annotated, train_ids, val_ids, test_ids, target_labels)

    # Validate full datasets were created successfully
    if df_train_full.empty or df_val_full.empty or df_test_full.empty:
        logging.error("Failed to create complete sequential datasets.")
        return

    # Step 5: Save CSV files
    logging.info("Saving complete train/val/test CSV files...")
    PersonClassification.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    BasePaths.LOGGING_DIR.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
    
    df_train_full.to_csv(PersonClassification.TRAIN_CSV_PATH, index=False)
    df_val_full.to_csv(PersonClassification.VAL_CSV_PATH, index=False)
    df_test_full.to_csv(PersonClassification.TEST_CSV_PATH, index=False)

    # Step 6: Generate statistics report
    generate_statistics_file(df_annotated, df_train_full, df_val_full, df_test_full, train_ids, val_ids, test_ids, mode, target_labels)

    # Final summary
    logging.info(f"✅ Sequential dataset pipeline completed successfully for mode: {mode}!")
    logging.info(f"📁 CSV files saved to: {PersonClassification.INPUT_DIR}")
        
if __name__ == "__main__":
    main()