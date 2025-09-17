import json
import logging
import sqlite3
import cv2
import re
import shutil
import random
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from constants import DataPaths, BasePaths, FaceDetection
from config import FaceConfig, DataConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==============================
# Database and Query
# ==============================
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

    # Create a SQL string like: 'video1', 'video2', ...
    excluded_videos_sql = ", ".join(f"'{v}'" for v in DataConfig.EXCLUDED_VIDEOS)

    query = f"""
    SELECT DISTINCT 
        a.category_id, a.bbox, i.file_name, a.person_age
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders}) 
        AND a.outside = 0 
        AND v.file_name NOT IN ({excluded_videos_sql})
    ORDER BY a.video_id, a.image_id
    """

    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        results = cursor.fetchall()
        
    logging.info(f"Excluded {len(DataConfig.EXCLUDED_VIDEOS)} videos from query")
    
    return results

# ==============================
# Image and Bounding Box Utilities
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

# ==============================
# Annotation Writing
# ==============================
def write_annotations(file_path: Path, lines: List[str]) -> None:
    file_path.write_text("".join(lines))

def save_annotations(annotations: List[Tuple]) -> None:
    """Convert annotations to YOLO format and save them in parallel.
    
    Parameters
    ----------
    annotations : List[Tuple]
        List of tuples containing annotation data (category_id, bbox, file_name, person_age).
        
    """
    output_dir = FaceDetection.LABELS_INPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    files = defaultdict(list)
    processed, skipped = 0, 0

    for cat_id, bbox_json, img_name, age in annotations:
        # img_name comes from database and should be the image filename without extension
        # Example: quantex_at_home_id255237_2022_05_08_04_000240 -> quantex_at_home_id255237_2022_05_08_04
        img_name_parts = img_name.split("_")
        if len(img_name_parts) < 9:
            logging.warning(f"Invalid image name format: {img_name} (expected at least 9 parts)")
            skipped += 1
            continue

        # Video folder name is the image name without the last part (frame number)
        video_folder_name = "_".join(img_name.split("_")[:-1])
        
        # Try to find image with any valid extension
        img_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = FaceDetection.IMAGES_INPUT_DIR / video_folder_name / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            logging.warning(f"Image not found: {img_name} in folder {video_folder_name}")
            logging.debug(f"Searched paths: {[FaceDetection.IMAGES_INPUT_DIR / video_folder_name / f'{img_name}{ext}' for ext in DataConfig.VALID_EXTENSIONS]}")
            skipped += 1
            continue

        try:
            # Load image to get actual dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                skipped += 1
                continue
            
            height, width = img.shape[:2]  # cv2 returns (height, width, channels)
            
            bbox = json.loads(bbox_json)
            yolo_bbox = convert_to_yolo_format(width, height, bbox)
            class_id = FaceConfig.AGE_GROUP_TO_CLASS_ID.get(age, 99)
            
            # Skip invalid class IDs
            if class_id == 99:
                logging.warning(f"Unknown age group '{age}' for {img_name}")
                skipped += 1
                continue

            # Use img_name (without extension) as the key for the annotation file
            files[img_name].append(f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n")
            processed += 1
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            skipped += 1

    # Save annotation files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name, lines in files.items():
            # Create .txt file with same name as image (without extension)
            output_file = output_dir / f"{img_name}.txt"
            executor.submit(write_annotations, output_file, lines)

    logging.info(f"Processed {processed}, skipped {skipped}")

def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = FaceDetection.IMAGES_INPUT_DIR) -> list:
    """
    This function returns frames that have any face annotations plus an equal number of random frames without faces.
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
        
    Returns
    -------
    list
        the annotated frames plus negative samples (formatted as a list of tuples)
        e.g. [(image_path, image_id), ...]
    """
    video_names = set()
    annotated_images = []
    negative_images = []
    annotated_frames = set()
    
    # Step 1: Get all frames that have ANY face annotations (adult, child, or both)
    for annotation_file in label_path.glob('*.txt'):
        # Only include files that have actual content (any face annotations)
        if annotation_file.stat().st_size > 0:
            try:
                with open(annotation_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # File has actual annotations
                        parts = annotation_file.stem.split('_')
                        # Example: quantex_at_home_id255237_2022_05_08_04_000240 -> quantex_at_home_id255237_2022_05_08_04
                        video_name = "_".join(parts[:8]) 
                        video_names.add(video_name)
                        annotated_frames.add(annotation_file.stem)
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")
        
    logging.info(f"Found {len(video_names)} unique video names")
    logging.info(f"Found {len(annotated_frames)} frames with any face annotations")

    # Step 2: Get annotated frames and collect potential negative frames
    video_negative_candidates = {}  # Initialize the dictionary here
    
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            video_negative_candidates[video_name] = []
            
            # Iterate through all frames in the video folder
            for frame in video_path.iterdir():
                if frame.is_file():
                    image_name = frame.name
                    parts = image_name.split("_")
                    
                    # Extract image ID
                    if len(parts) > 3 and parts[3].startswith('id'):
                        image_id = parts[3].replace("id", "")
                    else:
                        image_id = frame.stem
                    
                    if frame.stem in annotated_frames:
                        # This is an annotated frame (has any faces)
                        annotated_images.append((str(frame.resolve()), image_id))
                    else:
                        # This is a potential negative frame (no faces)
                        video_negative_candidates[video_name].append((str(frame.resolve()), image_id))
    
    # Step 3: Randomly sample negative frames to match annotated frames exactly
    random.seed(42)  # For reproducible results
    num_annotated = len(annotated_images)
    
    # Collect negative samples from each video proportionally
    total_candidates = sum(len(candidates) for candidates in video_negative_candidates.values())
    
    if total_candidates == 0:
        logging.warning("No negative frames found. Using only annotated frames.")
        total_images = annotated_images
    else:
        # Sample exactly the same number of negative frames as frames with any faces
        if total_candidates >= num_annotated:
            # We have enough candidates, sample 75% of the amount of num_annotated
            all_candidates = []
            for candidates in video_negative_candidates.values():
                all_candidates.extend(candidates)

            negative_images = random.sample(all_candidates, int(num_annotated * 0.75))
        else:
            # Use all available candidates if we don't have enough
            negative_images = []
            for candidates in video_negative_candidates.values():
                negative_images.extend(candidates)
            logging.warning(f"Only {len(negative_images)} negative frames available, less than {num_annotated} frames with faces")
        
        total_images = annotated_images + negative_images
    
    logging.info(f"Total frames with any face annotations: {len(annotated_images)}")
    logging.info(f"Total negative frames (without faces): {len(negative_images)}")
    logging.info(f"Total frames for training: {len(total_images)}")
    logging.info(f"Face coverage ratio: {len(annotated_images) / len(total_images):.2%}")
    
    return total_images

def get_class_distribution(total_images: list, annotation_folder: Path) -> pd.DataFrame:
    """
    Reads label files and groups images based on their class distribution.

    Parameters:
    ----------
    total_images: list
        List of tuples containing image paths and IDs.
    annotation_folder: Path
        Path to the directory containing label files.

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames, IDs, and their corresponding one-hot encoded class labels.
    """   
    if not total_images:
        logging.error("No images provided to get_class_distribution")
        return pd.DataFrame()
    
    image_class_mapping = []

    # Step 1: Read each image and its corresponding annotation file
    for i, (image_path, image_id) in enumerate(total_images):
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        # Get labels from annotation file
        labels = []
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            try:
                with open(annotation_file, 'r') as f:
                    content = f.read().strip()
                    if content:  # Only process non-empty files
                        lines = content.split('\n')
                        class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                        labels = [FaceConfig.MODEL_CLASS_ID_TO_LABEL[cid] for cid in class_ids if cid in FaceConfig.MODEL_CLASS_ID_TO_LABEL]
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")

        # Create one-hot encoded dictionary for the image
        try:
            # Get all possible class names from the configuration
            all_class_names = list(FaceConfig.MODEL_CLASS_ID_TO_LABEL.values())
            
            mapping_entry = {
                "filename": image_file.stem,
                "id": image_id,
                "has_annotation": bool(labels),  # True if labels are found, False otherwise
            }
            
            # Add one-hot encoding for each class
            for class_name in all_class_names:
                mapping_entry[class_name] = 1 if class_name in labels else 0
                
            image_class_mapping.append(mapping_entry)
            
        except Exception as e:
            logging.error(f"Error creating mapping entry for {image_file.name}: {e}")
            continue
        
    df = pd.DataFrame(image_class_mapping)
    
    return df

def get_first_n_minutes_frames(child_ids: List[str], image_folder: Path, minutes: int = 5) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Get frames from the first N minutes of recordings for specified child IDs.
    Splits them between training (first 80%) and validation (remaining 20%).

    Parameters
    ----------
    child_ids : List[str]
        List of child IDs to get frames for
    image_folder : Path
        Path to the image folder
    minutes : int
        Number of minutes from the beginning to extract (default: 5)
        
    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        Two lists of tuples: (train_frames, val_frames) containing (image_path, image_id)
    """
    train_frames = []
    val_frames = []
    
    total_frames_for_minutes = minutes * 60 * DataConfig.FPS

    # Split: first 80% to training, remaining 20% to validation
    train_frame_limit = int(total_frames_for_minutes * 0.8)  # First ~4 minutes
    val_frame_limit = total_frames_for_minutes - train_frame_limit  # Next ~1 minute

    logging.info(f"Extracting first {minutes} minutes ({total_frames_for_minutes} frames at {DataConfig.FPS} FPS) for test children")
    logging.info(f"Split: {train_frame_limit} frames to training, {val_frame_limit} frames to validation")
    
    for child_id in child_ids:
        child_train_frames = []
        child_val_frames = []
        
        # Find all video folders for this child
        for video_folder in sorted(image_folder.iterdir()):
            if video_folder.is_dir() and child_id in video_folder.name:
                # Get all frames from this video folder and sort them by frame number
                video_frames = []
                for frame in video_folder.iterdir():
                    if frame.is_file() and frame.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Extract frame number from filename
                        # Example: quantex_at_home_id255237_2022_05_08_04_000240.jpg -> frame 240
                        parts = frame.stem.split("_")
                        if len(parts) >= 9:
                            try:
                                frame_number = int(parts[-1])  # Last part should be frame number
                                video_frames.append((frame, frame_number))
                            except ValueError:
                                logging.warning(f"Could not extract frame number from {frame.name}")
                                continue
                
                # Sort by frame number to ensure we get the FIRST frames chronologically
                video_frames.sort(key=lambda x: x[1])
                
                current_train_count = len(child_train_frames)
                current_val_count = len(child_val_frames)
                
                for frame_path, frame_number in video_frames:
                    # Stop if we've collected enough frames for both sets
                    if current_train_count >= train_frame_limit and current_val_count >= val_frame_limit:
                        break
                    
                    # Extract image ID
                    parts = frame_path.name.split("_")
                    if len(parts) > 3 and parts[3].startswith('id'):
                        image_id = parts[3].replace("id", "")
                    else:
                        image_id = frame_path.stem
                    
                    # First assign to training, then to validation
                    if current_train_count < train_frame_limit:
                        child_train_frames.append((str(frame_path.resolve()), image_id))
                        current_train_count += 1
                    elif current_val_count < val_frame_limit:
                        child_val_frames.append((str(frame_path.resolve()), image_id))
                        current_val_count += 1
        
        train_frames.extend(child_train_frames)
        val_frames.extend(child_val_frames)
        
        logging.info(f"Child {child_id}: {len(child_train_frames)} frames to training, {len(child_val_frames)} frames to validation")
    
    logging.info(f"Total first {minutes} minutes: {len(train_frames)} frames to training, {len(val_frames)} frames to validation")
    return train_frames, val_frames

def get_all_frames_for_children(child_ids: List[str], image_folder: Path) -> List[Tuple[str, str]]:
    """
    Get ALL frames (not just annotated ones) for specified child IDs from all their videos.
    
    Parameters
    ----------
    child_ids : List[str]
        List of child IDs to get frames for
    image_folder : Path
        Path to the image folder
        
    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing (image_path, image_id)
    """
    all_frames = []
    
    for child_id in child_ids:
        # Find all video folders for this child
        for video_folder in image_folder.iterdir():
            if video_folder.is_dir() and child_id in video_folder.name:
                # Get all frames from this video folder
                for frame in video_folder.iterdir():
                    if frame.is_file() and frame.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Extract image ID
                        parts = frame.name.split("_")
                        if len(parts) > 3 and parts[3].startswith('id'):
                            image_id = parts[3].replace("id", "")
                        else:
                            image_id = frame.stem
                        
                        all_frames.append((str(frame.resolve()), image_id))
    
    logging.info(f"Found {len(all_frames)} total frames for test children: {child_ids}")
    return all_frames

def split_by_child_id(df: pd.DataFrame, train_ratio: float = FaceConfig.TRAIN_SPLIT_RATIO, add_first_minutes: bool = False, minutes: int = 5):
    """
    Splits the DataFrame into training, validation, and test sets using child IDs as the unit,
    while keeping each split's class ratio close to the global dataset ratio.
    
    For test set: Keeps ALL frames from test children (not just annotated ones) to reflect real-world scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image data and annotations
    train_ratio : float
        Ratio for training split
    add_first_minutes : bool
        If True, adds first N minutes of test children to training/validation
    minutes : int
        Number of minutes to add from beginning of test children recordings
    """
   
    # Check if 'filename' column exists, if not, return empty splits
    if 'filename' not in df.columns:
        logging.error(f"'filename' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Add child_id to our dataframe by mapping through image names
    def get_child_id_from_filename(filename: str) -> str:
        match = re.search(r'id(\d+)_', filename)
        if match:
            return 'id' + match.group(1)
        return None
    
    df['child_id'] = df['filename'].apply(get_child_id_from_filename)
    
    # Debug: Check child_id extraction
    logging.info(f"Child IDs found: {df['child_id'].unique()}")
    
    df.dropna(subset=['child_id'], inplace=True)

    # --- Counts per child (including face coverage) ---
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    
    if not class_columns:
        logging.error("No class columns found in DataFrame")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    child_group_counts = df.groupby('child_id')[class_columns].sum()
    
    # Also calculate face coverage per child
    child_face_coverage = df.groupby('child_id').agg({
        'has_annotation': ['sum', 'count']
    }).round(2)
    child_face_coverage.columns = ['faces_count', 'total_count']
    child_face_coverage['face_ratio'] = child_face_coverage['faces_count'] / child_face_coverage['total_count']

    # Global target ratio (first class proportion)
    global_counts = child_group_counts[class_columns].sum()
    target_class_0_ratio = global_counts[class_columns[0]] / global_counts.sum()

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

    # --- Initialize splits ---
    split_info = {
        'train': {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns), 'face_count': 0, 'total_count': 0},
        'val':   {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns), 'face_count': 0, 'total_count': 0},
        'test':  {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns), 'face_count': 0, 'total_count': 0},
    }

    # Sort children by their face count (descending) to ensure balanced distribution
    child_face_counts = [(child_id, child_face_coverage.loc[child_id, 'faces_count']) for child_id in sorted_child_ids]
    child_face_counts.sort(key=lambda x: x[1], reverse=True)
    
    logging.info(f"Child face counts: {child_face_counts}")

    def deviation_from_target(current_counts, child_counts):
        """Absolute difference from target class ratio after adding this child."""
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0:
            return 0
        new_ratio = new_counts[class_columns[0]] / new_counts.sum()
        return abs(new_ratio - target_class_0_ratio)
    
    def face_coverage_deviation(current_face_count, current_total_count, child_face_count, child_total_count):
        """Calculate how much the face coverage would deviate from 50% after adding this child."""
        new_face_count = current_face_count + child_face_count
        new_total_count = current_total_count + child_total_count
        if new_total_count == 0:
            return 0
        new_face_ratio = new_face_count / new_total_count
        return abs(new_face_ratio - 0.5)  # Target 50% face coverage

    # Assign children using a balanced approach that considers both class ratio and face coverage
    for child_id, child_face_count in child_face_counts:
        child_counts = child_group_counts.loc[child_id, class_columns]
        child_total_count = child_face_coverage.loc[child_id, 'total_count']

        # Find eligible splits (still have room for more IDs)
        eligible_splits = []
        for split_name, target_size in zip(['train', 'val', 'test'], [n_train, n_val, n_test]):
            if len(split_info[split_name]['ids']) < target_size:
                eligible_splits.append(split_name)

        if not eligible_splits:
            logging.warning(f"No split has space left for child {child_id}. Skipping.")
            continue

        # Pick split that minimizes both class ratio deviation AND face coverage deviation
        best_split = min(
            eligible_splits,
            key=lambda s: (
                deviation_from_target(split_info[s]['current_counts'], child_counts) * 0.3 +  # 30% weight on class balance
                face_coverage_deviation(split_info[s]['face_count'], split_info[s]['total_count'], 
                                       child_face_count, child_total_count) * 0.7  # 70% weight on face coverage balance
            )
        )

        # Assign child
        split_info[best_split]['ids'].append(child_id)
        split_info[best_split]['current_counts'] += child_counts
        split_info[best_split]['face_count'] += child_face_count
        split_info[best_split]['total_count'] += child_total_count

    # --- Build final DataFrames ---
    train_ids = split_info['train']['ids']
    val_ids = split_info['val']['ids']
    test_ids = split_info['test']['ids']

    train_df = df[df["child_id"].isin(train_ids)].copy()
    val_df = df[df["child_id"].isin(val_ids)].copy()
    test_df = df[df["child_id"].isin(test_ids)].copy()
    
    # Log face coverage for train/val splits before test enhancement
    train_face_coverage = len(train_df[train_df['has_annotation'] == True]) / len(train_df) if len(train_df) > 0 else 0
    val_face_coverage = len(val_df[val_df['has_annotation'] == True]) / len(val_df) if len(val_df) > 0 else 0
    logging.info(f"Pre-test enhancement - Train face coverage: {train_face_coverage:.2%}, Val face coverage: {val_face_coverage:.2%}")
    
    # Handle first N minutes addition if requested
    first_minutes_train_frames = []
    first_minutes_val_frames = []
    
    if add_first_minutes:
        logging.info(f"Adding first {minutes} minutes from test children to training/validation sets")
        first_minutes_train_frames, first_minutes_val_frames = get_first_n_minutes_frames(
            test_ids, FaceDetection.IMAGES_INPUT_DIR, minutes
        )
        
        # Add first minutes frames to training set
        if first_minutes_train_frames:
            additional_train_entries = []
            for image_path, image_id in first_minutes_train_frames:
                image_file = Path(image_path)
                if image_file.stem not in train_df['filename'].values:
                    # Check if this frame has annotations
                    annotation_file = FaceDetection.LABELS_INPUT_DIR / f"{image_file.stem}.txt"
                    has_annotation = annotation_file.exists() and annotation_file.stat().st_size > 0
                    
                    entry = {
                        "filename": image_file.stem,
                        "id": image_id,
                        "has_annotation": has_annotation,
                        "child_id": get_child_id_from_filename(image_file.stem)
                    }
                    
                    # Get annotation labels if they exist
                    labels = []
                    if has_annotation:
                        try:
                            with open(annotation_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    lines = content.split('\n')
                                    class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                                    labels = [FaceConfig.MODEL_CLASS_ID_TO_LABEL[cid] for cid in class_ids if cid in FaceConfig.MODEL_CLASS_ID_TO_LABEL]
                        except Exception as e:
                            logging.warning(f"Error reading annotation file {annotation_file}: {e}")
                    
                    # Add class labels
                    for class_name in class_columns:
                        entry[class_name] = 1 if class_name in labels else 0
                    
                    additional_train_entries.append(entry)
            
            if additional_train_entries:
                additional_train_df = pd.DataFrame(additional_train_entries)
                train_df = pd.concat([train_df, additional_train_df], ignore_index=True)
                logging.info(f"Added {len(additional_train_entries)} first-minutes frames to training set")
        
        # Add first minutes frames to validation set
        if first_minutes_val_frames:
            additional_val_entries = []
            for image_path, image_id in first_minutes_val_frames:
                image_file = Path(image_path)
                if image_file.stem not in val_df['filename'].values:
                    # Check if this frame has annotations
                    annotation_file = FaceDetection.LABELS_INPUT_DIR / f"{image_file.stem}.txt"
                    has_annotation = annotation_file.exists() and annotation_file.stat().st_size > 0
                    
                    entry = {
                        "filename": image_file.stem,
                        "id": image_id,
                        "has_annotation": has_annotation,
                        "child_id": get_child_id_from_filename(image_file.stem)
                    }
                    
                    # Get annotation labels if they exist
                    labels = []
                    if has_annotation:
                        try:
                            with open(annotation_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    lines = content.split('\n')
                                    class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                                    labels = [FaceConfig.MODEL_CLASS_ID_TO_LABEL[cid] for cid in class_ids if cid in FaceConfig.MODEL_CLASS_ID_TO_LABEL]
                        except Exception as e:
                            logging.warning(f"Error reading annotation file {annotation_file}: {e}")
                    
                    # Add class labels
                    for class_name in class_columns:
                        entry[class_name] = 1 if class_name in labels else 0
                    
                    additional_val_entries.append(entry)
            
            if additional_val_entries:
                additional_val_df = pd.DataFrame(additional_val_entries)
                val_df = pd.concat([val_df, additional_val_df], ignore_index=True)
                logging.info(f"Added {len(additional_val_entries)} first-minutes frames to validation set")

    # For test set: Add ALL frames from test child videos (not just annotated ones)
    # But EXCLUDE the first N minutes if they were added to train/val
    logging.info(f"Adding all frames from test children: {test_ids}")
    additional_test_frames = get_all_frames_for_children(test_ids, FaceDetection.IMAGES_INPUT_DIR)
    
    # Create set of first-minutes frame names to exclude from test set
    first_minutes_frame_names = set()
    if add_first_minutes:
        for image_path, _ in first_minutes_train_frames + first_minutes_val_frames:
            first_minutes_frame_names.add(Path(image_path).stem)
    
    # Add these additional frames to test_df (excluding first minutes if applicable)
    if additional_test_frames:
        additional_entries = []
        for image_path, image_id in additional_test_frames:
            image_file = Path(image_path)
            # Skip if this frame is already in our DataFrame or is part of first minutes
            if (image_file.stem not in test_df['filename'].values and 
                image_file.stem not in first_minutes_frame_names):
                # Create entry for this additional frame
                entry = {
                    "filename": image_file.stem,
                    "id": image_id,
                    "has_annotation": False,  # These are additional frames without annotations
                    "child_id": get_child_id_from_filename(image_file.stem)
                }
                # Add zero values for all class columns (no annotations)
                for class_name in class_columns:
                    entry[class_name] = 0
                additional_entries.append(entry)
        
        if additional_entries:
            additional_df = pd.DataFrame(additional_entries)
            test_df = pd.concat([test_df, additional_df], ignore_index=True)
            logging.info(f"Added {len(additional_entries)} additional frames to test set")
            if add_first_minutes:
                logging.info(f"Excluded {len(first_minutes_frame_names)} first-minutes frames from test set")

    # Log final split information
    for split_name, split_df in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        if len(split_df) > 0:
            ratio_0 = split_df[class_columns[0]].sum() / len(split_df)
            ratio_1 = split_df[class_columns[1]].sum() / len(split_df)

    return (train_df['filename'].tolist(), val_df['filename'].tolist(), test_df['filename'].tolist(),
            train_df, val_df, test_df)

def move_images(image_names: list, 
                split_type: str, 
                label_path: Path,
                n_workers: int = 4) -> Tuple[int, int]:
    """
    Move images and their corresponding labels to the specified split directory for face detection.
    Uses multithreading for faster processing.
    
    Parameters
    ----------
    image_names: list
        List of image names to process
    split_type: str
        Split type (train, val, or test)
    label_path: Path
        Path to label directory
    n_workers: int
        Number of worker threads for parallel processing
        
    Returns
    -------
    Tuple[int, int]
        Number of successful and failed moves
    """
    if not image_names:
        logging.info(f"No images to move for face detection {split_type}")
        return (0, 0)

    image_dst_dir = FaceDetection.INPUT_DIR / "images" / split_type
    label_dst_dir = FaceDetection.INPUT_DIR / "labels" / split_type
    
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    def process_single_image(image_name: str) -> bool:
        """Process a single image and its label."""
        try:
            
            # Handle face detection cases - get video folder from image name
            image_parts = image_name.split("_")
            if len(image_parts) < 9:
                logging.debug(f"Invalid image name format: {image_name}")
                return False
            
            # from quantex_at_home_id255237_2022_05_08_04_000240.jpg to quantex_at_home_id255237_2022_05_08_04
            image_folder = "_".join(image_parts[:8])
            
            # Try to find image with any valid extension
            image_src = None
            for ext in DataConfig.VALID_EXTENSIONS:
                potential_path = FaceDetection.IMAGES_INPUT_DIR / image_folder / f"{image_name}{ext}"
                if potential_path.exists():
                    image_src = potential_path
                    break
            
            if image_src is None:
                logging.debug(f"Image not found with any valid extension: {image_name}")
                return False
            
            label_src = label_path / f"{image_name}.txt"
            # Keep original extension in destination
            image_dst = image_dst_dir / f"{image_name}{image_src.suffix}"
            label_dst = label_dst_dir / f"{image_name}.txt"

            # Handle label file
            if not label_src.exists():
                label_dst.touch()
            else:
                shutil.copy2(label_src, label_dst)

            # Handle image file
            shutil.copy2(image_src, image_dst)
            return True

        except Exception as e:
            logging.error(f"Error processing {image_name}: {str(e)}")
            return False

    # Process images in parallel with progress bar
    successful = failed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_image, img) for img in image_names]
        
        from concurrent.futures import as_completed
        from tqdm import tqdm
        with tqdm(total=len(image_names), desc=f"Moving {split_type} images") as pbar:
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)

    # Log results
    logging.info(f"\nCompleted moving {split_type} images:")    
    return successful, failed
    
def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List, add_first_minutes: bool = False, minutes: int = 5):
    """
    Generates a statistics file with dataset split information, including percentages.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_face_det_{timestamp}.txt"
    
    # Calculate totals based on original balanced dataset (before test enhancement)
    original_total = len(df_train) + len(df_val) + len(df_test[df_test['has_annotation'] == True])
    
    with open(file_path, "w") as f:
        f.write(f"Dataset Split Information - {timestamp}\n")
        if add_first_minutes:
            f.write(f"*** FIRST {minutes} MINUTES MODE ENABLED ***\n")
            f.write(f"First {minutes} minutes of test children added to train/val\n")
        f.write("\n")
        
        class_0, class_1 = FaceConfig.TARGET_LABELS
        
        # Original distribution (before test enhancement)
        original_0 = df_train[class_0].sum() + df_val[class_0].sum() + df_test[df_test['has_annotation'] == True][class_0].sum()
        original_1 = df_train[class_1].sum() + df_val[class_1].sum() + df_test[df_test['has_annotation'] == True][class_1].sum()
        
        f.write(f"Original Balanced Distribution (before test enhancement):\n")
        f.write(f"Total Images: {original_total}\n")
        f.write(f"Class 0 {FaceConfig.TARGET_LABELS[0]}: {original_0} images ({original_0 / original_total:.2%})\n")
        f.write(f"Class 1 {FaceConfig.TARGET_LABELS[1]}: {original_1} images ({original_1 / original_total:.2%})\n\n")

        f.write("Split Distribution (within each split):\n")
        f.write("--------------------------------------------------\n\n")

        def write_split_info(split_name, split_df):
            total_split = len(split_df)
            
            count_0 = split_df[FaceConfig.TARGET_LABELS[0]].sum()
            count_1 = split_df[FaceConfig.TARGET_LABELS[1]].sum()
            
            # Calculate percentages WITHIN the split
            ratio_0 = count_0 / total_split if total_split > 0 else 0
            ratio_1 = count_1 / total_split if total_split > 0 else 0
            
            # Calculate face coverage (frames with any faces)
            frames_with_faces = split_df[split_df['has_annotation'] == True]
            face_coverage = len(frames_with_faces) / total_split if total_split > 0 else 0
            
            f.write(f"{split_name} Set:\n")
            f.write(f"Total Images: {total_split}\n")
            f.write(f"Face Coverage: {len(frames_with_faces)} ({face_coverage:.2%}) - frames with any faces\n")
            f.write(f"No Face: {total_split - len(frames_with_faces)} ({1-face_coverage:.2%}) - frames without faces\n")
            f.write(f"{FaceConfig.TARGET_LABELS[0]}: {count_0} ({ratio_0:.2%}) - within split\n")
            f.write(f"{FaceConfig.TARGET_LABELS[1]}: {count_1} ({ratio_1:.2%}) - within split\n\n")

        write_split_info("Train", df_train)
        write_split_info("Validation", df_val)
        write_split_info("Test", df_test)

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
    
    # Also log the key statistics to console
    train_face_coverage = len(df_train[df_train['has_annotation'] == True]) / len(df_train) if len(df_train) > 0 else 0
    val_face_coverage = len(df_val[df_val['has_annotation'] == True]) / len(df_val) if len(df_val) > 0 else 0
    test_face_coverage = len(df_test[df_test['has_annotation'] == True]) / len(df_test) if len(df_test) > 0 else 0
    
    logging.info(f"Face Coverage - Train: {train_face_coverage:.2%}, Val: {val_face_coverage:.2%}, Test: {test_face_coverage:.2%}")
    logging.info(f"Train images: {len(df_train)}, Val images: {len(df_val)}, Test images: {len(df_test)}")

def split_yolo_data(annotation_folder: Path, add_first_minutes: bool = False, minutes: int = 5):
    """
    This function prepares the dataset for face detection YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    add_first_minutes : bool
        If True, adds first N minutes of test children to training/validation
    minutes : int
        Number of minutes to add from beginning of test children recordings
    """
    logging.info("Starting dataset preparation for face detection")

    try:
        # Get annotated frames
        total_images = get_total_number_of_annotated_frames(annotation_folder)
        
        if not total_images:
            logging.error("No annotated images found. Check your annotation folder and image paths.")
            return
        
        # Multi-class detection case for face detection
        df = get_class_distribution(total_images, annotation_folder)
        
        if df.empty:
            logging.error("DataFrame is empty. Check class distribution function.")
            return
                
        # Split data grouped by child id 
        train, val, test, df_train, df_val, df_test = split_by_child_id(df, add_first_minutes=add_first_minutes, minutes=minutes)

        # Get the IDs for logging
        train_ids = df_train['child_id'].unique().tolist() if 'child_id' in df_train.columns else []
        val_ids = df_val['child_id'].unique().tolist() if 'child_id' in df_val.columns else []
        test_ids = df_test['child_id'].unique().tolist() if 'child_id' in df_test.columns else []
        
        # Use the original balanced DataFrame (df) for statistics, not the enhanced test set
        generate_statistics_file(df, df_train, df_val, df_test, train_ids, val_ids, test_ids, 
                                add_first_minutes=add_first_minutes, minutes=minutes)
        
        # Move images for each split
        for split_name, split_set in [("train", train), 
                                      ("val", val), 
                                      ("test", test)]:
            if split_set:
                successful, failed = move_images(
                    image_names=split_set,
                    split_type=split_name,
                    label_path=annotation_folder,
                    n_workers=4
                )
                logging.info(f"{split_name}: Moved {successful}, Failed {failed}")
            else:
                logging.warning(f"No images for {split_name} split")
    
    except Exception as e:
        logging.error(f"Error processing face detection: {str(e)}")
        raise
    
    logging.info("Completed dataset preparation for face detection")

# ==============================
# Main
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Create input files for face detection YOLO training')
    parser.add_argument('--add-first-minutes', action='store_true', 
                       help='Add first N minutes of test children recordings to training/validation sets')
    parser.add_argument('--minutes', type=int, default=5,
                       help='Number of minutes from the beginning to add to train/val (default: 5)')
    parser.add_argument('--fetch-annotations', action='store_true',
                       help='Fetch and save annotations from database (default: False)')
    
    args = parser.parse_args()
    
    if args.add_first_minutes:
        logging.info(f"First minutes mode enabled: adding first {args.minutes} minutes of test children to train/val")
    
    try:
        if args.fetch_annotations:
            anns = fetch_all_annotations(FaceConfig.DATABASE_CATEGORY_IDS)
            logging.info(f"Fetched {len(anns)} annotations.")
            save_annotations(anns)
        
        split_yolo_data(FaceDetection.LABELS_INPUT_DIR, 
                       add_first_minutes=args.add_first_minutes, 
                       minutes=args.minutes)
        logging.info("Conversion completed successfully.")
    except Exception as e:
        logging.error(f"Failed: {e}")
        raise

if __name__ == "__main__":
    main()