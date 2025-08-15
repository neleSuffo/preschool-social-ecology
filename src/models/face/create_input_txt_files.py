import json
import logging
import sqlite3
import cv2
import os
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    query = f"""
    SELECT DISTINCT 
        a.category_id, a.bbox, i.file_name, a.person_age
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders}) 
      AND a.outside = 0 
      AND v.file_name NOT LIKE '%id255237_2022_05_08_04%'
    ORDER BY a.video_id, a.image_id
    """

    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        return cursor.fetchall()

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
        parent_folder = "_".join(img_name.split("_")[:-1])
        
        # Try to find image with any valid extension
        img_path = None
        for ext in FaceConfig.VALID_EXTENSIONS:
            potential_path = FaceDetection.IMAGES_INPUT_DIR / parent_folder / f"{img_name}{ext}"
            if potential_path.exists():
                img_path = potential_path
                break
        
        if img_path is None:
            logging.warning(f"Image not found with any valid extension: {img_name}")
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
                
            files[img_name].append(f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n")
            processed += 1
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            skipped += 1

    # Save annotation files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name, lines in files.items():
            output_file = output_dir / f"{Path(img_name).stem}.txt"
            executor.submit(write_annotations, output_file, lines)

    logging.info(f"Processed {processed}, skipped {skipped}")

def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = FaceDetection.IMAGES_INPUT_DIR) -> list:
    """
    This function returns only the frames that have annotations (every 30th frame).
    
    Parameters
    ----------
    label_path : Path
        the path to the label files
    image_folder : Path
        the path to the image folder
        
    Returns
    -------
    list
        the annotated frames (formatted as a list of tuples)
        e.g. [(image_path, image_id), ...]
    """
    video_names = set()
    total_images = []
    
    # Step 1: Get unique video names and annotated frames
    annotated_frames = set()
    for annotation_file in label_path.glob('*.txt'):
        parts = annotation_file.stem.split('_')
        video_name = "_".join(parts[:8])
        video_names.add(video_name)
        annotated_frames.add(annotation_file.stem)
    
    logging.info(f"Found {len(video_names)} unique video names")

    # Step 2: Only get the corresponding annotated frames from image folders
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            for video_file in video_path.iterdir():
                if video_file.is_file():
                    # Only include if the frame is in our annotated frames set
                    if video_file.stem in annotated_frames:
                        image_name = video_file.name
                        parts = image_name.split("_")
                        image_id = parts[3].replace("id", "")
                        total_images.append((str(video_file.resolve()), image_id))
                        
    logging.info(f"Total annotated frames found: {len(total_images)}")
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
    image_class_mapping = []

    # Step 1: Read each image and its corresponding annotation file
    for image_path, image_id in total_images:
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        # Get labels from annotation file
        labels = []
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}
                labels = [FaceConfig.MODEL_CLASS_ID_TO_LABEL[cid] for cid in class_ids if cid in FaceConfig.MODEL_CLASS_ID_TO_LABEL]

        # Create one-hot encoded dictionary for the image
        image_class_mapping.append({
            "filename": image_file.stem,
            "id": image_id,
            "has_annotation": bool(labels),  # True if labels are found, False otherwise
            **{class_name: (1 if class_name in labels else 0) 
               for class_name in FaceConfig.MODEL_CLASS_ID_TO_LABEL.values()}
        })

    return pd.DataFrame(image_class_mapping)

def multilabel_stratified_split(df: pd.DataFrame,
                              train_ratio: float = FaceConfig.TRAIN_SPLIT_RATIO,
                              random_seed: int = DataConfig.RANDOM_SEED,
                              min_ids_per_split: int = 2):
    """
    Performs a group-based stratified split balancing three factors:
    1. Group Integrity: Keeps all frames from an ID in one split.
    2. Class Representation: Prioritizes ensuring all classes are present in val/test.
    3. True Frame Distribution: Bases the split ratio on the number of frames.
    4. Minimum ID Requirements: Ensures at least min_ids_per_split IDs in val/test sets.
    
    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame containing image filenames, IDs, and one-hot encoded class labels.
    train_ratio: float
        Target ratio for the training set based on frame count.
    random_seed: int
        Random seed for reproducibility.
    min_ids_per_split: int
        Minimum number of IDs required in validation and test sets.
        
    Returns:
    -------
    Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Lists of filenames and DataFrames for train, val, and test splits.
    """
    # Set random seed
    random.seed(random_seed)
    
    val_ratio = (1.0 - train_ratio) / 2  # Split remaining equally between val and test
    test_ratio = val_ratio

    # --- 1. Setup and Pre-analysis ---
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = BasePaths.LOGGING_DIR / f"split_distribution_face_det_{timestamp}.txt"

    logging.info(f"Starting split for {df.shape[0]} frames and {df['id'].nunique()} unique IDs.")
    if df.empty:
        raise ValueError("Input DataFrame is empty!")

    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation']]
    id_class_map = df.groupby('id')[class_columns].sum().apply(lambda s: s[s > 0].index.to_list(), axis=1).to_dict()

    # Pre-calculate the 'weight' (frame count) of each ID
    id_frame_count_map = df['id'].value_counts().to_dict()
    total_frame_count = df.shape[0]
    
    # Check if we have enough IDs for minimum requirements
    total_ids = len(id_class_map)
    if total_ids < (min_ids_per_split * 2 + 1):  # Need at least 2 for val + 2 for test + 1 for train
        logging.warning(f"Not enough IDs ({total_ids}) to guarantee {min_ids_per_split} IDs in each of val and test sets")
        # Adjust minimum requirement if necessary
        min_ids_per_split = max(1, (total_ids - 1) // 2)
        logging.warning(f"Adjusted min_ids_per_split to {min_ids_per_split}")

    # --- 2. Main Hybrid Scoring Algorithm ---
    available_ids = sorted(list(id_class_map.keys()))
    random.shuffle(available_ids) # Shuffle to prevent any bias from sorting order

    train_ids, val_ids, test_ids = set(), set(), set()
    
    # Track split sizes by cumulative frame count
    train_frames, val_frames, test_frames = 0, 0, 0
    
    # Track classes still needed in val and test
    uncovered_val_classes = set(class_columns)
    uncovered_test_classes = set(class_columns)

    # Define weights for the scoring model. Coverage bonus must be very high.
    COVERAGE_BONUS_WEIGHT = 1000.0
    RATIO_PENALTY_WEIGHT = 1.0 / total_frame_count # Normalize penalty by total frames
    ID_REQUIREMENT_WEIGHT = 2000.0  # Very high weight for meeting minimum ID requirements

    for id_to_assign in available_ids:
        id_classes = set(id_class_map[id_to_assign])
        id_frame_count = id_frame_count_map[id_to_assign]
        
        scores = {'train': 0.0, 'val': 0.0, 'test': 0.0}

        # === Score for assigning to VALIDATION set ===
        # 1. Coverage Score: Huge bonus for covering a needed class
        val_coverage_gain = len(id_classes.intersection(uncovered_val_classes))
        scores['val'] += val_coverage_gain * COVERAGE_BONUS_WEIGHT
        
        # 2. ID Requirement Score: High bonus if we still need more IDs
        if len(val_ids) < min_ids_per_split:
            scores['val'] += ID_REQUIREMENT_WEIGHT
        
        # 3. Ratio Score: Penalty for making the split larger than its target frame count
        if val_frames >= total_frame_count * val_ratio and len(val_ids) >= min_ids_per_split:
            potential_new_val_frames = val_frames + id_frame_count
            overage_penalty = potential_new_val_frames - (total_frame_count * val_ratio)
            scores['val'] -= overage_penalty * RATIO_PENALTY_WEIGHT

        # === Score for assigning to TEST set ===
        # 1. Coverage Score
        test_coverage_gain = len(id_classes.intersection(uncovered_test_classes))
        scores['test'] += test_coverage_gain * COVERAGE_BONUS_WEIGHT
        
        # 2. ID Requirement Score: High bonus if we still need more IDs
        if len(test_ids) < min_ids_per_split:
            scores['test'] += ID_REQUIREMENT_WEIGHT

        # 3. Ratio Score
        test_ratio_calc = 1.0 - train_ratio - val_ratio
        if test_frames >= total_frame_count * test_ratio_calc and len(test_ids) >= min_ids_per_split:
            potential_new_test_frames = test_frames + id_frame_count
            overage_penalty = potential_new_test_frames - (total_frame_count * test_ratio_calc)
            scores['test'] -= overage_penalty * RATIO_PENALTY_WEIGHT
            
        # === Assign ID to the split with the highest score ===
        best_split = max(scores, key=scores.get)
        
        if best_split == 'val':
            val_ids.add(id_to_assign)
            val_frames += id_frame_count
            uncovered_val_classes.difference_update(id_classes)
        elif best_split == 'test':
            test_ids.add(id_to_assign)
            test_frames += id_frame_count
            uncovered_test_classes.difference_update(id_classes)
        else: # best_split == 'train'
            train_ids.add(id_to_assign)
            train_frames += id_frame_count

    # --- 3. Finalization and Enhanced Logging ---
    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()

    # Verify minimum ID requirements
    if len(val_ids) < min_ids_per_split:
        logging.warning(f"⚠️  Validation set has only {len(val_ids)} ID(s), target was {min_ids_per_split}")
    if len(test_ids) < min_ids_per_split:
        logging.warning(f"⚠️  Test set has only {len(test_ids)} ID(s), target was {min_ids_per_split}")
    
    if len(val_ids) >= min_ids_per_split and len(test_ids) >= min_ids_per_split:
        logging.info(f"✓ Both validation and test sets have at least {min_ids_per_split} IDs")

    # Final check to warn about impossible splits
    for class_col in class_columns:
        if val_df[class_col].sum() == 0:
            logging.error(f"WARNING: Class '{class_col}' has 0 instances in the Validation set.")
        if test_df[class_col].sum() == 0:
            logging.error(f"WARNING: Class '{class_col}' has 0 instances in the Test set.")

    train_class_counts = train_df[class_columns].sum()
    val_class_counts = val_df[class_columns].sum()
    test_class_counts = test_df[class_columns].sum()
    class_counts = train_class_counts + val_class_counts + test_class_counts

    # Prepare detailed log report in desired style
    split_info = [f"Dataset Split Information - {timestamp}\n"]

    # Initial Distribution
    split_info.append("Initial Distribution:")
    split_info.append(f"Total Frames: {total_frame_count}")
    split_info.append(f"child_face: {class_counts['child_face']} images ({class_counts['child_face']/total_frame_count:.2%})")
    split_info.append(f"adult_face: {class_counts['adult_face']} images ({class_counts['adult_face']/total_frame_count:.2%})\n")

    # Split Distribution
    split_info.append("Split Distribution:")
    split_info.append("-" * 50)

    def add_split_summary(name, frame_count, child_count, adult_count):
        split_info.append(f"{name} Set: ({frame_count/total_frame_count:.2%})")
        split_info.append(f"Total Frames: {frame_count}")
        split_info.append(f"child_face: {child_count} ({child_count/frame_count:.2%})")
        split_info.append(f"adult_face: {adult_count} ({adult_count/frame_count:.2%})\n")

    add_split_summary("Validation", val_frames, val_class_counts['child_face'], val_class_counts['adult_face'])
    add_split_summary("Test", test_frames, test_class_counts['child_face'], test_class_counts['adult_face'])
    add_split_summary("Train", train_frames, train_class_counts['child_face'], train_class_counts['adult_face'])

    # ID Distribution
    split_info.append("ID Distribution:")
    split_info.append(f"Training IDs: {len(train_ids)}, {sorted(list(train_ids))}")
    split_info.append(f"Validation IDs: {len(val_ids)}, {sorted(list(val_ids))}")
    split_info.append(f"Test IDs: {len(test_ids)}, {sorted(list(test_ids))}\n")

    # ID Overlap Check
    overlap = set(train_ids) & set(val_ids) | set(train_ids) & set(test_ids) | set(val_ids) & set(test_ids)
    split_info.append("ID Overlap Check:")
    split_info.append(f"Overlap found: {'Yes' if overlap else 'No'}")

    # Join and print
    log_report = "\n".join(split_info)
    
    # Write report to file and log to console
    report_string = '\n'.join(split_info)
    with open(output_file, 'w') as f:
        f.write(report_string)
    
    logging.info(f"\nSplit distribution report saved to: {output_file}\n")
    
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

    image_dst_dir = FaceDetection.DATA_INPUT_DIR / "images" / split_type
    label_dst_dir = FaceDetection.DATA_INPUT_DIR / "labels" / split_type
    
    image_dst_dir.mkdir(parents=True, exist_ok=True)
    label_dst_dir.mkdir(parents=True, exist_ok=True)

    def process_single_image(image_name: str) -> bool:
        """Process a single image and its label."""
        try:
            # Handle face detection cases
            image_parts = image_name.split("_")[:8]
            image_folder = "_".join(image_parts)
            
            # Try to find image with any valid extension
            image_src = None
            for ext in FaceConfig.VALID_EXTENSIONS:
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
                import shutil
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
    
def split_yolo_data(annotation_folder: Path):
    """
    This function prepares the dataset for face detection YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    """
    logging.info("Starting dataset preparation for face detection")

    try: 
        # Get annotated frames
        total_images = get_total_number_of_annotated_frames(annotation_folder)
        
        # Multi-class detection case for face detection
        df = get_class_distribution(total_images, annotation_folder)
        
        # Split data grouped by id with minimum ID requirements
        train, val, test, *_ = multilabel_stratified_split(df, min_ids_per_split=2)

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
    logging.info(f"Starting YOLO annotation conversion")
    try:
        anns = fetch_all_annotations(FaceConfig.DATABASE_CATEGORY_IDS)
        logging.info(f"Fetched {len(anns)} annotations.")
        save_annotations(anns)
        split_yolo_data(FaceDetection.LABELS_INPUT_DIR)
        logging.info("Conversion completed successfully.")
    except Exception as e:
        logging.error(f"Failed: {e}")

if __name__ == "__main__":
    main()