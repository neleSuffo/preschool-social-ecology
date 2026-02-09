import sqlite3
import random
import logging
import pandas as pd
import argparse
import cv2
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from torch import mode
from tqdm import tqdm
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
    """Fetch all relevant person annotations, including bounding boxes.
    Parameters:
    -----------
    category_ids: List[int]
        List of category IDs to fetch from the database.
    
    Returns:
    --------
    List[Tuple]
        A list of tuples containing the fetched annotations.
    """
    logging.info(f"Fetching annotations for category IDs: {category_ids}")
    placeholders = ", ".join("?" * len(category_ids))

    query = f"""
    SELECT 
        i.video_id,
        i.frame_id,
        i.file_name,
        a.category_id,
        a.person_age AS age_group,
        a.bbox
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON i.video_id = v.id
    WHERE a.category_id IN ({placeholders})
      AND a.outside = 0
    ORDER BY i.video_id, i.frame_id
    """
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        # Returns: (video_id, frame_id, file_name, category_id, age_group, bbox)
        results = cursor.fetchall()
        
    logging.info(f"Found {len(results)} annotations.")
    return results

def fetch_noisy_frames() -> List[str]:
    """Fetch file_names of frames that contain *only* the category_id = -1 (noisy frames)."""
    query = """
    SELECT i.file_name
    FROM images i JOIN annotations a ON i.frame_id = a.image_id AND i.video_id = a.video_id
    WHERE a.category_id = -1
    GROUP BY i.file_name
    HAVING COUNT(CASE WHEN a.category_id != -1 THEN 1 END) = 0
    """
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
    
# ==============================
# YOLO Generation and Master DF Creation (Combines labeling and file system scan)
# ==============================

def convert_to_yolo_format(width: int, height: int, bbox: List[float]) -> Tuple[float, float, float, float]:
    """Convert [xtl, ytl, xbr, ybr] to YOLO."""
    xtl, ytl, xbr, ybr = bbox
    x_center = (xtl + xbr) / 2.0
    y_center = (ytl + ybr) / 2.0
    bbox_width = xbr - xtl
    bbox_height = ybr - ytl
    
    x_center_norm = x_center / width
    y_center_norm = y_center / height
    width_norm = bbox_width / width
    height_norm = bbox_height / height
    
    return (x_center_norm, y_center_norm, width_norm, height_norm)

def write_annotations(file_path: Path, lines: List[str]) -> None:
    file_path.write_text("".join(lines))

def save_annotations(annotations: List[Tuple], output_dir: Path, mode: str, positive_frames_info_path: Path) -> Dict[Tuple, Dict[str, int]]:
    """Convert annotations to YOLO format and save them in parallel.
    
    Parameters
    ----------
    annotations : List[Tuple]
        List of tuples containing (video_id, frame_id, file_name, category_id, age_group, bbox_json)
    output_dir : Path, optional
        Directory to save annotation files
    mode : str
        Detection mode to use for saving annotations (default: "face-only")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if mode == 'age-binary':
        yolo_class_mapping = PersonConfig.AGE_GROUP_TO_CLASS_ID_AGE_BINARY
        target_labels = PersonConfig.TARGET_LABELS_AGE_BINARY
        label_map_csv = PersonConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
    elif mode == 'person-only':
        yolo_class_mapping = PersonConfig.AGE_GROUP_TO_CLASS_ID_PERSON_ONLY
        target_labels = PersonConfig.TARGET_LABELS_PERSON_ONLY
        label_map_csv = PersonConfig.MODEL_CLASS_ID_TO_LABEL_PERSON_ONLY
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # label_map_csv maps model class id -> label string (e.g., {0: 'person'})
    label_id_to_string = {int(k): v for k, v in label_map_csv.items()}

    files = defaultdict(list)
    positive_frames_info = {}
    
    for video_id, frame_id, file_name, _, age_group, bbox_json in annotations:
        
        frame_key_tuple = (video_id, frame_id, file_name)
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
            if img is None: continue
            height, width = img.shape[:2]
            
            bbox = json.loads(bbox_json)
            yolo_bbox = convert_to_yolo_format(width, height, bbox)
            yolo_class_id = yolo_class_mapping.get(age_group, 99)

            if yolo_class_id == 99: 
                logging.warning(f"Unknown age group '{age_group}' for {file_name}")
                continue

            yolo_line = f"{yolo_class_id} " + " ".join(map(str, yolo_bbox)) + "\n"
            files[file_name].append(yolo_line)

            target_label = label_id_to_string.get(int(yolo_class_id))
            if target_label and target_label in target_labels:
                if frame_key_tuple not in positive_frames_info:
                    positive_frames_info[frame_key_tuple] = {label: 0 for label in target_labels}
                positive_frames_info[frame_key_tuple][target_label] = 1

        except Exception as e:
            logging.error(f"Error processing {file_name}: {e}")

    # Save YOLO annotation files (write in parallel and show progress)
    total_files = len(files)
    futures_map = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for file_name, lines in files.items():
            output_file = output_dir / f"{file_name}.txt"
            fut = executor.submit(write_annotations, output_file, lines)
            futures_map[fut] = output_file

        # iterate as futures complete and update progress bar
        for _ in tqdm(as_completed(futures_map), total=total_files, desc="Writing YOLO files", unit="file"):
            try:
                # retrieving result will re-raise exceptions from the worker
                _.result()
            except Exception as e:
                logging.error(f"Error writing YOLO file: {e}")

    # Save positive frames info to JSON
    string_keyed_info = {
        f"{vid}_{fid}_{fname}": labels for (vid, fid, fname), labels in positive_frames_info.items()
    }

    # Ensure parent directory exists and write safely
    try:
        positive_frames_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(positive_frames_info_path, "w") as f:
            json.dump(string_keyed_info, f, indent=2)
            f.flush()
        logging.info(f"Saved positive frames info to {positive_frames_info_path} ({positive_frames_info_path.stat().st_size} bytes)")
    except Exception as e:
        logging.error(f"Failed to write positive frames info JSON: {e}")

def is_valid_negative_candidate(file_name_stem: str, frame_id: int, video_name: str, noisy_frames_set: set) -> bool:
    """
    Checks if a frame is a valid candidate for a negative sample based on the user's rules:
    1. Must not be flagged as noisy.
    2. Must be a sampled frame (multiple of FPS, potentially shifted).    
    
    Parameters:
    -----------
    file_name_stem : str
        The stem of the file name (without extension).
    frame_id : int
        The ID of the frame.
    video_name : str
        The name of the video file.
    noisy_frames_set : set
        A set of noisy frame identifiers.

    Returns:
    --------
    bool
        True if the frame is a valid negative candidate, False otherwise.
    """
    DEFAULT_OFFSET = 0
    
    # 1. Check exclusion list (noisy frames)
    if file_name_stem in noisy_frames_set:
        return False
    
    # 2. Check FPS/Sampling Rule (e.g., multiples of 30)
    if frame_id <= 0:
        return False
        
    is_sampled_frame = False
    
    exception_map = DataConfig.SHIFTED_VIDEOS_OFFSETS
    
    if video_name in exception_map:
        # Logic for exception videos with frame shift (Rule 2)
        start_frame, shift = exception_map[video_name]
        
        if frame_id < start_frame:
            # Rule 1: Before the exception start, use standard sampling
            if frame_id % DataConfig.FPS == DEFAULT_OFFSET:
                is_sampled_frame = True
        else:
            # Rule 2: From the exception start onward, use the shifted modulo rule
            if (frame_id - start_frame) % DataConfig.FPS == DEFAULT_OFFSET:
                is_sampled_frame = True
            
    else:
        # Default rule for all other videos: multiples of DataConfig.FPS
        if frame_id % DataConfig.FPS == DEFAULT_OFFSET:
            is_sampled_frame = True
            
    return is_sampled_frame

def build_master_frame_df(pos_frames_file_path: Path, mode: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a master DataFrame of all frames (positive and negative) with labels 
    by scanning the file system.
    
    Parameters:
    -----------
    pos_frames_file_path : Path
        Path to the JSON file containing positive frames info.
    mode : str  
        Classification mode ('age-binary' or 'person-only').
        
    Returns:
    --------
    Tuple[pd.DataFrame, List[str]]
        A tuple containing the master DataFrame and the list of target labels.
    """
    if mode == 'age-binary':
        target_labels = PersonConfig.TARGET_LABELS_AGE_BINARY
    elif mode == 'person-only':
        target_labels = PersonConfig.TARGET_LABELS_PERSON_ONLY
    else:
        raise ValueError(f"Invalid classification_mode: {mode}")

    # --- Load and Deserialize Positive Frames Info ---
    try:
        with open(pos_frames_file_path, "r") as f:
            string_keyed_info = json.load(f)
    except FileNotFoundError:
        logging.warning("Positive frames info JSON not found. Assuming zero positive frames.")
        string_keyed_info = {}

    positive_frames_info = {}
    for key_str, labels in string_keyed_info.items():
        parts = key_str.split('_')
        if len(parts) < 3: continue

        file_name = "_".join(parts[2:])
        try:
             video_id = int(parts[0])
             frame_id = int(parts[1])
             positive_frames_info[(video_id, frame_id, file_name)] = labels
        except ValueError:
             continue
    
    annotated_frames_set = set(positive_frames_info.keys())

    # --- Fetch noisy frames to exclude from negative pool ---
    noisy_frames_set = set(fetch_noisy_frames())

    # --- Scan File System for All Frames --- 
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    final_data_list = []

    # --- Iterate through all videos and frames ---
    for _, row in video_map.iterrows():
        video_id = row['video_id']
        video_file_name = row['file_name']
        
        video_folder_name = video_file_name.replace('.avi', '').replace('.mp4', '') 
        folder_path = PersonClassification.IMAGES_INPUT_DIR / video_folder_name
        
        if not folder_path.is_dir(): continue

        for image_file in folder_path.iterdir():
            file_name_stem = image_file.stem

            if image_file.suffix.lower() not in DataConfig.VALID_EXTENSIONS:
                continue

            try:
                frame_num_str = file_name_stem.split('_')[-1]
                frame_id = int(frame_num_str)
            except (ValueError, IndexError):
                continue

            frame_key = (video_id, frame_id, file_name_stem)
            image_path = str(image_file)

            entry = {"video_id": video_id, "frame_id": frame_id, "file_name": file_name_stem, "file_path": image_path}
            
            is_positive = frame_key in annotated_frames_set

            if is_positive:
                entry.update(positive_frames_info[frame_key])
                final_data_list.append(entry)
            else:
                # Negative frame logic: ONLY include if it meets sampling criteria
                if is_valid_negative_candidate(file_name_stem, frame_id, video_file_name, noisy_frames_set):
                    entry.update({label: 0 for label in target_labels})
                    final_data_list.append(entry)

    df_full_frames_with_labels = pd.DataFrame(final_data_list)
    
    return df_full_frames_with_labels, target_labels

# ==============================
# Sequential Data Pipeline Functions
# ==============================

def build_sequential_dataset(df: pd.DataFrame, train_ids: List[str], val_ids: List[str], test_ids: List[str], target_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build complete sequential datasets for train, val, and test splits. The function filters
    the master DataFrame based on child IDs and sorts frames by video_id and frame_id.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Master DataFrame containing all frames with labels.
    train_ids : List[str]
        List of child IDs for the training set.
    val_ids : List[str]
        List of child IDs for the validation set.
    test_ids : List[str]
        List of child IDs for the test set.
    target_labels : List[str]
        List of target label column names.
    """        
    df['child_id'] = df['file_name'].apply(get_child_id_from_filename)
        
    def filter_split_data(child_ids: List[str]) -> pd.DataFrame:
        df_split = df[df['child_id'].isin(child_ids)].copy()
        
        df_split.sort_values(by=['video_id', 'frame_id'], inplace=True)
        
        required_cols = ['video_id', 'frame_id', 'file_name', 'file_path'] + target_labels
        return df_split[[col for col in required_cols if col in df_split.columns]]

    df_train_full = filter_split_data(train_ids)
    df_val_full = filter_split_data(val_ids) 
    df_test_full = filter_split_data(test_ids)
    
    return df_train_full, df_val_full, df_test_full


def split_by_child_id(df: pd.DataFrame, target_labels: List[str], train_ratio: float = PersonConfig.TRAIN_SPLIT_RATIO):
    """Split dataset by child IDs, balancing based on positive frames only."""

    df['child_id'] = df['file_name'].apply(get_child_id_from_filename)
    
    # Explicitly cast target label columns to integer/boolean before filtering
    df[target_labels] = df[target_labels].astype(int)
    
    print(f"df: {df.head(10)}")
    df_positive = df[df[target_labels].any(axis=1)].copy()
    
    if df_positive.empty:
        logging.warning("No positively labeled frames found. Cannot perform balanced split.")
        child_group_counts = df.groupby('child_id')[target_labels].sum()
    else:
        child_group_counts = df_positive.groupby('child_id')[target_labels].sum()
    
    global_counts = child_group_counts[target_labels].sum()
    target_ratio = global_counts[target_labels[0]] / global_counts.sum() if global_counts.sum() > 0 else 0

    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    if n_total < 3 * PersonConfig.MIN_IDS_PER_SPLIT: return [], [], []
    n_train = max(int(n_total * train_ratio), PersonConfig.MIN_IDS_PER_SPLIT)
    remaining_ids = n_total - n_train
    n_val = max(int(remaining_ids / 2), PersonConfig.MIN_IDS_PER_SPLIT)
    n_test = n_total - n_train - n_val
    if n_test < PersonConfig.MIN_IDS_PER_SPLIT: n_val -= (PersonConfig.MIN_IDS_PER_SPLIT - n_test); n_test = PersonConfig.MIN_IDS_PER_SPLIT
    
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
# ==============================
def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List, mode: str, target_labels: List[str]):
    """Generate detailed statistics file about the dataset splits."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_person_cls_{mode}_{timestamp}.txt"
    
    with open(file_path, "w") as f:
        f.write(f"Person Classification Dataset Split Report - {timestamp}\n")
        f.write(f"*** CLASSIFICATION MODE: {mode.upper()} ***\n\n")
        
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
    parser.add_argument('--mode', choices=["person-only", "age-binary"], default="person-only",
                       help='Select the classification mode: person-only (single binary class) or age-binary (child/adult).')
    parser.add_argument('--fetch_annotations', action='store_true',
                       help='Fetch annotations from DB and generate YOLO .txt files (default: False).')
    
    args = parser.parse_args()
    mode = args.mode
    
    logging.info(f"Starting person classification dataset pipeline in '{mode}' mode...")
    pos_frames_file_path = PersonClassification.LABELS_INPUT_DIR / "positive_frames_info.json"
    
    # 1. Fetch annotations from the DB and save YOLO labels
    if args.fetch_annotations:
        anns = fetch_all_annotations(PersonConfig.DATABASE_CATEGORY_IDS)
        save_annotations(anns, PersonClassification.LABELS_INPUT_DIR, mode, pos_frames_file_path)
        
    # 2. Build master frame list by scanning file system and annotating frames
    df_combined, target_labels = build_master_frame_df(pos_frames_file_path, mode)
        
    if df_combined.empty:
        logging.error("No valid frames (positive or negative) created. Exiting.")
        return

    # Step 3: Split data by child IDs (Balancing is based on positive frames only)
    logging.info(f"Splitting children by IDs based on positive frame distribution...")
    train_ids, val_ids, test_ids = split_by_child_id(df_combined, target_labels)

    if not train_ids or not val_ids or not test_ids:
        logging.error("Failed to create valid child ID splits.")
        return

    # Step 4: Build complete sequential datasets (filter df_combined into splits)
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
    if args.fetch_annotations:
        logging.info(f"📦 YOLO label files saved to: {PersonClassification.LABELS_INPUT_DIR}")
        
if __name__ == "__main__":
    main()