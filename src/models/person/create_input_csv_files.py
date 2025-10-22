import sqlite3
import random
import logging
import pandas as pd
import argparse
import cv2
import json
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from constants import BasePaths, DataPaths, PersonClassification
from config import DataConfig, PersonConfig

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
    
    Parameters
    ----------
    category_ids : List[int]
        List of category IDs to fetch (e.g., [1, 2] for person
        
    Returns
    -------
    List[Tuple]
        List of tuples containing (video_id, frame_id, file_name, category_id, age_group)
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    placeholders = ", ".join("?" * len(category_ids))

    query = f"""
    SELECT DISTINCT
        a.category_id, a.bbox, i.file_name, a.person_age
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders})
      AND a.outside = 0
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
    if not video_ids: return pd.DataFrame()
        
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
# YOLO Generation and Labeling Functions
# ==============================

def convert_to_yolo_format(width: int, height: int, bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert [xtl, ytl, xbr, ybr] to YOLO (x_center, y_center, width, height).
    
    Parameters
    ----------
    width : int
        Image width in pixels
    height : int 
        Image height in pixels
    bbox : List[float]
        Bounding box in format [xtl, ytl, xbr, ybr] (absolute coordinates)
        
    Returns
    -------
    Tuple[float, float, float, float]
        YOLO format: (x_center_norm, y_center_norm, width_norm, height_norm)
        All values normalized to [0, 1]
    """
    xtl, ytl, xbr, ybr = bbox
    
    # Calculate center coordinates (absolute)
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    
    # Calculate width and height (absolute)
    bbox_width = xbr - xtl
    bbox_height = ybr - ytl
    
    # Normalize to [0, 1] by dividing by image dimensions
    x_center_norm = x_center / width
    y_center_norm = y_center / height
    width_norm = bbox_width / width
    height_norm = bbox_height / height
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)

def write_annotations(file_path: Path, lines: List[str]) -> None:
    file_path.write_text("".join(lines))

def generate_yolo_labels_and_frame_data(annotations: List[Tuple], classification_mode: str, output_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    1. Generates YOLO labels (.txt files).
    2. Returns a DataFrame of ALL frames (positives + negatives) with their binary classification labels.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Determine Labeling Scheme
    if classification_mode == 'age-binary':
        yolo_class_mapping = PersonConfig.AGE_GROUP_TO_CLASS_ID_AGE_BINARY
        target_labels = PersonConfig.TARGET_LABELS_AGE_BINARY
    elif classification_mode == 'person-only':
        yolo_class_mapping = PersonConfig.AGE_GROUP_TO_CLASS_ID_PERSON_ONLY
        target_labels = PersonConfig.TARGET_LABELS_PERSON_ONLY
    else:
        raise ValueError(f"Invalid classification_mode: {classification_mode}")
    
    # Map from class ID (0/1) back to the label string ('child'/'adult'/'person') for the CSV
    label_id_to_string = {v: k for k, v in yolo_class_mapping.items()}
    
    files = defaultdict(list)
    positive_frames_info = defaultdict(lambda: {label: 0 for label in target_labels})
    
    logging.info("Starting YOLO label generation and positive frame mapping...")
    processed, skipped = 0, 0

    # Iterate through all annotations to generate YOLO lines and mark frames as positive
    for _, bbox_json, file_name, age_group in annotations:
        
        frame_key = file_name
        
        frame_folder = "_".join(file_name.split("_")[:-1])
        img_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None: continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                skipped += 1
                continue
            
            height, width = img.shape[:2]
            
            bbox = json.loads(bbox_json)
            yolo_bbox = convert_to_yolo_format(width, height, bbox)
            yolo_class_id = yolo_class_mapping.get(age_group, 99)

            if yolo_class_id == 99:
                logging.warning(f"Unknown age group '{age_group}' for {file_name}")
                skipped += 1
                continue

            yolo_line = f"{yolo_class_id} " + " ".join(map(str, yolo_bbox)) + "\n"
            files[file_name].append(yolo_line)
            processed += 1

            # Mark frame as positive for the corresponding target label
            target_label = label_id_to_string.get(yolo_class_id)
            if target_label and target_label in target_labels:
                positive_frames_info[frame_key][target_label] = 1

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    # Save YOLO annotation files
    logging.info(f"Writing {len(files)} YOLO label files to {output_dir}...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        for file_name, lines in files.items():
            output_file = output_dir / f"{file_name}.txt"
            executor.submit(write_annotations, output_file, lines)
    
    logging.info(f"Processed {processed}, skipped {skipped}")

    # 2. Create the master DataFrame (`df_full_frames_with_labels`)
    
    # A. Get all video/frame IDs from the database
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    all_db_frames = pd.read_sql_query("SELECT file_name FROM images", conn)
    conn.close()

    all_db_frames['file_path'] = None # Placeholder for file path
    
    # B. Merge with annotations/paths/labels
    final_data_list = []
    
    for _, row in all_db_frames.iterrows():
        file_name = row['file_name']

        key = file_name

        # Determine file path (Necessary because 'images' table only holds file_name, not full path)
        frame_folder = "_".join(file_name.split("_")[:-1])
        image_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        if image_path is None: continue

        entry = {"file_name": file_name, "file_path": image_path}
        
        # Add classification labels (0 for negatives, 1 for positives derived from YOLO labels)
        if key in positive_frames_info:
            entry.update(positive_frames_info[key])
        else:
            entry.update({label: 0 for label in target_labels})
            
        final_data_list.append(entry)

    df_full_frames_with_labels = pd.DataFrame(final_data_list)
    
    return df_full_frames_with_labels, target_labels


# ==============================
# Sequential Data Pipeline Functions
# ==============================

def build_sequential_dataset(df_combined: pd.DataFrame, train_ids: List[str], val_ids: List[str], test_ids: List[str], target_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build complete sequential datasets by filtering the master df by child ID.

    Parameters
    ----------
    df_combined : pd.DataFrame
        Master DataFrame with all frames and labels.
    train_ids : List[str]
        List of child IDs for training set.
    val_ids : List[str]
        List of child IDs for validation set.
    test_ids : List[str]
        List of child IDs for test set.
    target_labels : List[str]
        List of target label column names.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames for the training, validation, and test sets.
    """    
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()
    
    video_map['child_id'] = video_map['file_name'].apply(get_child_id_from_filename)
    
    # Merge child_id into the master dataframe
    df_with_child_id = df_combined.merge(
        video_map[['file_name', 'child_id']], 
        on='file_name', 
        how='left'
    )
    df_with_child_id.dropna(subset=['child_id'], inplace=True)
    
    def filter_split_data(child_ids: List[str]) -> pd.DataFrame:
        df_split = df_with_child_id[df_with_child_id['child_id'].isin(child_ids)].copy()
        
        # Sort by video and frame number to maintain sequence order
        df_split.sort_values(by=['file_name'], inplace=True)
        
        # Final set of required columns (excluding the temporary 'child_id')
        required_cols = ['file_name', 'file_path'] + target_labels
        return df_split[required_cols]

    df_train_full = filter_split_data(train_ids)
    df_val_full = filter_split_data(val_ids) 
    df_test_full = filter_split_data(test_ids)
    
    return df_train_full, df_val_full, df_test_full


def split_by_child_id(df_combined: pd.DataFrame, target_labels: List[str], train_ratio: float = PersonConfig.TRAIN_SPLIT_RATIO):
    """
    Split dataset by child IDs, balancing based on positive frames only.
    """
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    video_map['child_id'] = video_map['file_name'].apply(get_child_id_from_filename)
    
    # Merge child_id into the combined dataframe
    df = df_combined.merge(video_map[['video_id', 'child_id']], on='video_id', how='left')
    df.dropna(subset=['child_id'], inplace=True)

    # KEY: Filter to use ONLY positive frames for balancing calculation
    df_positive = df[df[target_labels].any(axis=1)].copy() 
    
    if df_positive.empty:
        logging.warning("No positively labeled frames found. Cannot perform balanced split.")
        child_group_counts = df.groupby('child_id')[target_labels].sum()
    else:
        child_group_counts = df_positive.groupby('child_id')[target_labels].sum()
    
    # Calculate target ratio for the first label (e.g., 'child' or 'person')
    global_counts = child_group_counts[target_labels].sum()
    target_ratio = global_counts[target_labels[0]] / global_counts.sum() if global_counts.sum() > 0 else 0

    # Determine split sizes
    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    # Basic split size calculation (retains previous logic)
    if n_total < 3 * PersonConfig.MIN_CHILDREN_PER_SPLIT: return [], [], []
    n_train = max(int(n_total * train_ratio), PersonConfig.MIN_CHILDREN_PER_SPLIT)
    remaining_ids = n_total - n_train
    n_val = max(int(remaining_ids / 2), PersonConfig.MIN_CHILDREN_PER_SPLIT)
    n_test = n_total - n_train - n_val
    if n_test < PersonConfig.MIN_CHILDREN_PER_SPLIT: n_val -= (PersonConfig.MIN_CHILDREN_PER_SPLIT - n_test); n_test = PersonConfig.MIN_CHILDREN_PER_SPLIT
    
    # Initialize splits
    split_info = {
        'train': {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
        'val':   {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
        'test':  {'ids': [], 'current_counts': pd.Series([0] * len(target_labels), index=target_labels)},
    }
    random.shuffle(sorted_child_ids)

    def deviation_from_target(current_counts, child_counts):
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0: return 0
        new_ratio = new_counts[target_labels[0]] / new_counts.sum()
        return abs(new_ratio - target_ratio)

    # Assign children to splits
    for child_id in sorted_child_ids:
        child_counts = child_group_counts.loc[child_id, target_labels]
        eligible_splits = []
        for split_name, target_size in zip(['train', 'val', 'test'], [n_train, n_val, n_test]):
            if len(split_info[split_name]['ids']) < target_size:
                eligible_splits.append(split_name)

        if not eligible_splits: continue

        best_split = min(
            eligible_splits,
            key=lambda s: deviation_from_target(split_info[s]['current_counts'], child_counts)
        )

        split_info[best_split]['ids'].append(child_id)
        split_info[best_split]['current_counts'] += child_counts

    return split_info['train']['ids'], split_info['val']['ids'], split_info['test']['ids']


# ==============================
# Statistics and Logging
# (generate_statistics_file remains the same as previously defined)
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
    parser.add_argument('--classification-mode', choices=["person-only", "age-binary"], default="person-only",
                       help='Select the classification mode: person-only (single binary class) or age-binary (child/adult).')
    
    args = parser.parse_args()
    mode = args.classification_mode
    
    logging.info(f"Starting person classification dataset pipeline in '{mode}' mode...")

    # Step 1: Fetch annotations (including BBox)
    anns = fetch_all_annotations(PersonConfig.DATABASE_CATEGORY_IDS)
    
    if not anns:
        logging.error("No annotations found for the specified categories. Exiting.")
        return

    # Step 2: Generate YOLO labels (.txt) and build the master DF (df_combined)
    df_combined, target_labels = generate_yolo_labels_and_frame_data(anns, mode, PersonClassification.LABELS_INPUT_DIR)

    if df_combined.empty:
        logging.error("No valid frames (positive or negative) created. Exiting.")
        return

    # Step 3: Split data by child IDs (Balancing is based on positive frames only, extracted from df_combined)
    logging.info(f"Splitting children by IDs based on positive frame distribution...")
    train_ids, val_ids, test_ids = split_by_child_id(df_combined, target_labels)

    if not train_ids or not val_ids or not test_ids:
        logging.error("Failed to create valid child ID splits.")
        return

    # Step 4: Build complete sequential datasets (selects ALL frames corresponding to the split IDs from df_combined)
    df_train_full, df_val_full, df_test_full = build_sequential_dataset(df_combined, train_ids, val_ids, test_ids, target_labels)

    if df_train_full.empty or df_val_full.empty or df_test_full.empty:
        logging.error("Failed to create complete sequential datasets.")
        return

    # Step 5: Save CSV files
    logging.info("Saving complete train/val/test CSV files...")
    PersonClassification.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    BasePaths.LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    
    df_train_full.to_csv(PersonClassification.TRAIN_CSV_PATH, index=False)
    df_val_full.to_csv(PersonClassification.VAL_CSV_PATH, index=False)
    df_test_full.to_csv(PersonClassification.TEST_CSV_PATH, index=False)

    # Step 6: Generate statistics report
    generate_statistics_file(df_combined, df_train_full, df_val_full, df_test_full, train_ids, val_ids, test_ids, mode, target_labels)

    logging.info(f"✅ Sequential dataset pipeline completed successfully for mode: {mode}!")
    logging.info(f"📁 CSV files saved to: {PersonClassification.INPUT_DIR}")
    logging.info(f"📦 YOLO label files saved to: {PersonClassification.LABELS_INPUT_DIR}")
        
if __name__ == "__main__":
    main()