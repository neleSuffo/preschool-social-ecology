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
    object_ids = [3, 4, 5, 6, 7, 8, 12]
    object_placeholders = ", ".join(map(str, object_ids))

    query = f"""
    SELECT DISTINCT 
        a.category_id, a.bbox, a.object_interaction, i.file_name,
        a.gaze_directed_at_child, a.person_age
    FROM annotations a
    JOIN images i ON a.image_id = i.frame_id AND a.video_id = i.video_id
    JOIN videos v ON a.video_id = v.id
    WHERE a.category_id IN ({placeholders}) 
      AND a.outside = 0 
      AND v.file_name NOT LIKE '%id255237_2022_05_08_04%'
      AND (
        (a.category_id IN ({object_placeholders}) AND a.object_interaction = 'Yes')
        OR (a.category_id NOT IN ({object_placeholders}))
      )
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
    """Convert annotations to YOLO format and save them in parallel."""
    output_dir = FaceDetection.LABELS_INPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    files = defaultdict(list)
    processed, skipped = 0, 0

    for cat_id, bbox_json, obj_inter, img_name, gaze, age in annotations:
        parent_folder = "_".join(img_name.split("_")[:-1])
        img_path = FaceDetection.IMAGES_INPUT_DIR / parent_folder / img_name

        if not img_path.exists():
            logging.warning(f"{img_path} does not exist")
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
    # Define class mappings for face detection
    id_to_name = {0: "child_face", 1: "adult_face"}

    image_class_mapping = []

    for image_path, image_id in total_images:
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        # Get labels from annotation file
        labels = []
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                class_ids = {int(line.split()[0]) for line in f if line.split()}
                labels = [id_to_name[cid] for cid in class_ids if cid in id_to_name]

        # Create one-hot encoded dictionary for the image
        image_class_mapping.append({
            "filename": image_file.stem,
            "id": image_id,
            "has_annotation": bool(labels),  # True if labels are found, False otherwise
            **{class_name: (1 if class_name in labels else 0) 
               for class_name in id_to_name.values()}
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
    import random
    import pandas as pd
    
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
            for ext in FaceDetection.VALID_EXTENSION:
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_id(filename):
    """Extract the ID from a filename (e.g., 'id255237' from 'quantex_at_home_id255237_...')."""
    parts = filename.split('_')
    for part in parts:
        if part.startswith('id'):
            return part
    return None

def get_original_image_and_index(input_image: str) -> Tuple[str, int]:
    """Extract the original image name and detection index from an input image filename.
    
    Works for both "_face_" and "_person_" formats by always extracting the last `_`-separated number.

    Parameters
    ----------
    input_image: str
        The name of the input image file.
    
    Returns
    -------
    Tuple[str, int]
        The original image name and the detection index.
    """
    parts = input_image.rsplit('_', 2)  # Split at most twice from the right
    base_name = parts[0]  # Everything before the last two `_` parts
    index = int(parts[-1].split('.')[0])  # Extract last numeric part (index)
    
    original_image = base_name + '.jpg'
    return original_image, index

def get_class(input_image: str, annotation_folder: Path) -> Optional[int]:
    """Retrieve the class ID (0 or 1) for an input image from its annotation file.
    
    Parameters
    ----------
    input_image: str
        The name of the input image file.
    annotation_folder: Path
        Path to the directory containing annotation files.
        
    Returns
    -------
    Optional[int]
        The class ID (0 or 1) if found, otherwise None.
    """
    
    original_image, index = get_original_image_and_index(input_image)
    annotation_file = annotation_folder / Path(original_image).with_suffix('.txt').name
    if annotation_file.exists():
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            if index < len(lines):
                class_id = lines[index].strip().split()[0]
                return int(class_id)
    return None

def compute_id_counts(input_images: list,
                      annotation_folder: Path) -> Tuple[defaultdict, int]:
    """Compute the number of images per ID and their class distribution.
    
    Parameters
    ----------
    input_images: list
        List of image filenames.
    annotation_folder: Path
        Path to the directory containing annotation files.
    
    Returns
    -------
    Tuple[defaultdict, int]
        A dictionary with IDs as keys and a list of counts [n0, n1] as values,
        where n0 is the count of class 0 and n1 is the count of class 1.
        Also returns the number of images without annotations.
    """
    id_counts = defaultdict(lambda: [0, 0])  # [n0, n1] for each ID
    missing_annotations = 0
    missing_images = []
    for input_image in input_images:
        id_ = extract_id(input_image)
        class_id = get_class(input_image, annotation_folder)
        if class_id == 0:
            id_counts[id_][0] += 1
        elif class_id == 1:
            id_counts[id_][1] += 1
        else:
            missing_annotations += 1
            missing_images.append(input_image)
    logging.info(f"Images without annotations: {missing_annotations}")
    return id_counts, missing_annotations

def find_best_split(all_ids, id_counts, total_samples, num_trials=100, min_ratio=0.05, min_ids_per_split=2):
    """Find the best split of IDs that maintains class distribution in val and test sets.
    
    Parameters
    ----------
    all_ids : list
        List of all available IDs
    id_counts : dict
        Dictionary mapping IDs to [class_0_count, class_1_count]
    total_samples : int
        Total number of samples
    num_trials : int
        Number of random trials to find best split
    min_ratio : float
        Minimum ratio of each class needed in val/test sets
    min_ids_per_split : int
        Minimum number of IDs required in val and test sets
    """
    best_score = float('inf')
    best_split = None
    total_class_0 = sum(counts[0] for counts in id_counts.values())
    total_class_1 = sum(counts[1] for counts in id_counts.values())
    min_class_0_needed = total_class_0 * min_ratio
    min_class_1_needed = total_class_1 * min_ratio
    
    # Check if we have enough IDs for the minimum requirements
    if len(all_ids) < (min_ids_per_split * 2 + 1):  # Need at least 2 for val + 2 for test + 1 for train
        logging.warning(f"Not enough IDs ({len(all_ids)}) to guarantee {min_ids_per_split} IDs in each of val and test sets")
        # Adjust minimum requirement if necessary
        min_ids_per_split = max(1, (len(all_ids) - 1) // 2)
        logging.warning(f"Adjusted min_ids_per_split to {min_ids_per_split}")

    for trial in range(num_trials):
        # Sort IDs by ratio of classes to try balancing first
        id_ratios = [(id_, counts[0]/(counts[0] + counts[1]) if (counts[0] + counts[1]) > 0 else 0) 
                     for id_, counts in id_counts.items()]
        sorted_ids = sorted(id_ratios, key=lambda x: abs(x[1] - 0.5))  # Sort by how balanced they are
        trial_ids = [id_ for id_, _ in sorted_ids]
        random.shuffle(trial_ids)  # Add some randomness while keeping generally balanced IDs first

        # Initialize trial splits
        val_ids = []
        test_ids = []
        val_class_0 = val_class_1 = test_class_0 = test_class_1 = 0

        # First ensure minimum IDs in validation set
        ids_added_to_val = 0
        for id_ in trial_ids[:]:
            if ids_added_to_val >= min_ids_per_split and (val_class_0 >= min_class_0_needed and val_class_1 >= min_class_1_needed):
                break
                
            val_ids.append(id_)
            trial_ids.remove(id_)
            val_class_0 += id_counts[id_][0]
            val_class_1 += id_counts[id_][1]
            ids_added_to_val += 1

        # Then ensure minimum IDs in test set
        ids_added_to_test = 0
        for id_ in trial_ids[:]:
            if ids_added_to_test >= min_ids_per_split and (test_class_0 >= min_class_0_needed and test_class_1 >= min_class_1_needed):
                break
                
            test_ids.append(id_)
            trial_ids.remove(id_)
            test_class_0 += id_counts[id_][0]
            test_class_1 += id_counts[id_][1]
            ids_added_to_test += 1

        # Remaining IDs go to train
        train_ids = trial_ids

        # Skip this trial if we don't meet minimum ID requirements
        if len(val_ids) < min_ids_per_split or len(test_ids) < min_ids_per_split:
            continue

        # Calculate how well balanced the splits are
        val_total = val_class_0 + val_class_1
        test_total = test_class_0 + test_class_1
        
        if val_total > 0 and test_total > 0:
            val_ratio = val_class_0 / val_total
            test_ratio = test_class_0 / test_total
            overall_ratio = total_class_0 / total_samples
            
            # Score based on:
            # 1. How close ratios are to overall ratio
            # 2. Whether minimum requirements are met
            # 3. How close to target split sizes (10% each)
            # 4. Whether minimum ID requirements are met
            ratio_score = abs(val_ratio - overall_ratio) + abs(test_ratio - overall_ratio)
            
            min_req_score = 0
            if val_class_0 < min_class_0_needed or val_class_1 < min_class_1_needed:
                min_req_score += 100
            if test_class_0 < min_class_0_needed or test_class_1 < min_class_1_needed:
                min_req_score += 100
            
            # Penalty for not meeting minimum ID requirements
            id_req_score = 0
            if len(val_ids) < min_ids_per_split:
                id_req_score += 1000
            if len(test_ids) < min_ids_per_split:
                id_req_score += 1000
                
            size_score = abs(val_total - 0.1 * total_samples) + abs(test_total - 0.1 * total_samples)
            
            total_score = ratio_score + min_req_score + id_req_score + size_score * 0.1
            
            if total_score < best_score:
                best_score = total_score
                best_split = (val_ids, test_ids, train_ids)

    # Final check and logging
    if best_split is None:
        logging.error("Could not find a valid split meeting all requirements!")
        # Fallback: simple split ensuring minimum IDs
        random.shuffle(all_ids)
        val_ids = all_ids[:min_ids_per_split]
        test_ids = all_ids[min_ids_per_split:min_ids_per_split*2]
        train_ids = all_ids[min_ids_per_split*2:]
        best_split = (val_ids, test_ids, train_ids)
        logging.warning(f"Using fallback split: val={len(val_ids)} IDs, test={len(test_ids)} IDs, train={len(train_ids)} IDs")
    else:
        val_ids, test_ids, train_ids = best_split
        logging.info(f"Best split found: val={len(val_ids)} IDs, test={len(test_ids)} IDs, train={len(train_ids)} IDs")

    return best_split

def balance_train_set(train_input_images: list, 
                      annotation_folder: Path,
                      min_ratio: float = 0.45) -> list:
    """
    Balance the training set only if class ratio exceeds specified threshold.
    
    Parameters
    ----------
    train_input_images: list
        List of training image filenames.
    annotation_folder: Path
        Path to the directory containing annotation files.
    min_ratio: float
        Minimum ratio of minority class to total images to trigger balancing.
        
    Returns
    -------
    list
        A balanced list of training image filenames.
    """
    # Separate images by class
    train_class_0 = []
    train_class_1 = []
    
    for img in train_input_images:
        class_id = get_class(img, annotation_folder)
        if class_id == 0:
            train_class_0.append(img)
        elif class_id == 1:
            train_class_1.append(img)

    n0 = len(train_class_0)
    n1 = len(train_class_1)
    total = n0 + n1
    
    # Calculate class ratios
    ratio_0 = n0 / total if total > 0 else 0
    ratio_1 = n1 / total if total > 0 else 0
    
        
    # Only balance if ratio exceeds threshold
    if min(ratio_0, ratio_1) < min_ratio:
        logging.info("Class imbalance detected, performing balancing...")
        if len(train_class_0) > len(train_class_1):
            target_size = len(train_class_1)
            train_class_0 = random.sample(train_class_0, target_size)
        else:
            target_size = len(train_class_0)
            train_class_1 = random.sample(train_class_1, target_size)
    
        balanced_ratio = len(train_class_0) / (len(train_class_0) + len(train_class_1))
        logging.info(f"After balancing - Class ratio: {balanced_ratio:.3f}")
        return train_class_0 + train_class_1
    else:
        logging.info("Class distribution within acceptable range, no balancing needed")
        return train_input_images
    
def split_dataset(input_folder: str, 
                  annotation_folder: str,
                  target: str,
                  class_mapping: dict = None) -> Tuple[list, list, list]:
    """
    Split the dataset into train, val, and test sets while keeping test set unbalanced for representative evaluation.
    
    Parameters
    ----------
    input_folder: str
        Path to the folder containing input images.
    annotation_folder: str
        Path to the folder containing annotation files.
    target: str
        The target type for YOLO detection or classification.
    class_mapping: dict
        Optional mapping of class IDs to names.
    """
    # Get all input images
    input_folder = Path(input_folder)
    annotation_folder = Path(annotation_folder)
    all_input_images = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    
    # Create output directory for split information
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = LOGGING_DIR / f"split_distribution_{target}_{timestamp}.txt"

    split_info = []
    split_info.append(f"Dataset Split Information - {timestamp}\n")
    split_info.append(f"Found {len(all_input_images)} {target} images in {input_folder}\n")
    
    # Compute class counts per ID
    id_counts, missing_annotations = compute_id_counts(all_input_images, annotation_folder)
    all_ids = list(id_counts.keys())
    
    # Get total distribution
    N0 = sum(counts[0] for counts in id_counts.values())
    N1 = sum(counts[1] for counts in id_counts.values())
    total_samples = N0 + N1
    
    # Log initial distribution
    split_info.append("Initial Distribution:")
    split_info.append(f"Class 0 {class_mapping[0][0]}: {N0} images")
    split_info.append(f"Class 1 {class_mapping[1][0]}: {N1} images")
    split_info.append(f"Total: {total_samples} images")
    split_info.append(f"Missing Annotations: {missing_annotations} images\n")
    split_info.append(f"Overall {class_mapping[0][0]}-to-Total Ratio: {N0 / total_samples:.3f}\n")

    # Find the best split with minimum ID requirements
    val_ids, test_ids, train_ids = find_best_split(
        all_ids, id_counts, total_samples, 
        num_trials=100, min_ratio=0.05, min_ids_per_split=2
    )
    
    # Assign input images to splits
    val_input_images = [f for f in all_input_images if extract_id(f) in val_ids]
    test_input_images = [f for f in all_input_images if extract_id(f) in test_ids]
    train_input_images = [f for f in all_input_images if extract_id(f) in train_ids]
    
    # Balance only training and validation sets - keep test set unbalanced for representative evaluation
    train_balanced = balance_train_set(train_input_images, annotation_folder)
    val_balanced = balance_train_set(val_input_images, annotation_folder)
    
    # Log detailed split information
    split_info.append("Split Distribution:")
    split_info.append("-" * 50)
    
    for split_name, split_images in [
        ("Train (Original)", train_input_images),
        ("Train (Balanced)", train_balanced),
        ("Validation (Original)", val_input_images), 
        ("Validation (Balanced)", val_balanced),
        ("Test (Unbalanced - Representative)", test_input_images)
    ]:
        n0 = sum(1 for f in split_images if get_class(f, annotation_folder) == 0)
        n1 = len(split_images) - n0
        n_split = len(split_images)
        ratio = n0 / n_split if n_split > 0 else 0
        
        split_details = [
            f"\n{split_name} Set:",
            f"Total Images: {n_split}",
            f"{class_mapping[0][0]} (Class 0): {n0}",
            f"{class_mapping[1][0]} (Class 1): {n1}",
            f"{class_mapping[0][0]}-to-Total Ratio: {ratio:.3f}"
        ]
        split_info.extend(split_details)
        
        # Also log to console
        logging.info("\n".join(split_details))
    
    # Add ID distribution information
    split_info.extend([
        f"\nID Distribution:",
        f"Training IDs: {len(train_ids)}, {train_ids}",
        f"Validation IDs: {len(val_ids)}, {val_ids}",
        f"Test IDs: {len(test_ids)}, {test_ids}",
    ])
    
    # Log ID counts to console as well
    logging.info(f"\nID Distribution Summary:")
    logging.info(f"Training set: {len(train_ids)} IDs")
    logging.info(f"Validation set: {len(val_ids)} IDs")
    logging.info(f"Test set: {len(test_ids)} IDs")
    
    # Verify minimum requirements
    if len(val_ids) < 2:
        logging.warning(f"⚠️  Validation set has only {len(val_ids)} ID(s), recommended minimum is 2")
    if len(test_ids) < 2:
        logging.warning(f"⚠️  Test set has only {len(test_ids)} ID(s), recommended minimum is 2")
    
    if len(val_ids) >= 2 and len(test_ids) >= 2:
        logging.info("✓ Both validation and test sets have at least 2 IDs")
    
    # Check for ID overlap
    train_id_set = set(train_ids)
    val_id_set = set(val_ids)
    test_id_set = set(test_ids)
    
    overlap = train_id_set & val_id_set | train_id_set & test_id_set | val_id_set & test_id_set
    split_info.append(f"\nID Overlap Check:")
    split_info.append(f"Overlap found: {'Yes' if overlap else 'No'}")
    if overlap:
        split_info.append(f"Overlapping IDs: {overlap}")
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(split_info))
    
    logging.info(f"\nSplit distribution saved to: {output_file}")
    
    return train_balanced, val_balanced, test_input_images

def calculate_original_class_counts(image_list: list, annotation_folder: Path) -> dict:
    """
    Calculate the original class counts for a list of images before balancing.
    
    Parameters
    ----------
    image_list: list
        List of image filenames.
    annotation_folder: Path
        Path to the directory containing annotation files.
        
    Returns
    -------
    dict
        Dictionary containing class counts for no_gaze and gaze.
    """
    counts = {"no_gaze": 0, "gaze": 0}
    
    for image_filename in image_list:
        annotation_file = annotation_folder / Path(image_filename).with_suffix('.txt').name
        
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            # Count all detections in this image
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id == 0:
                        counts["no_gaze"] += 1
                    elif class_id == 1:
                        counts["gaze"] += 1
    
    return counts

def create_json_from_image_list(image_list: list, annotation_folder: Path, total_images: list) -> list:
    """
    Create JSON annotations from a list of image filenames.
    
    Parameters
    ----------
    image_list: list
        List of image filenames for this split.
    annotation_folder: Path
        Path to the directory containing annotation files.
    total_images: list
        List of tuples containing all image paths and IDs.
        
    Returns
    -------
    list
        List of annotation dictionaries in ViT format.
    """
    # Create a mapping from filename to full path
    filename_to_path = {}
    for image_path, image_id in total_images:
        filename = Path(image_path).name
        filename_to_path[filename] = image_path
    
    annotations = []
    
    for image_filename in tqdm(image_list, desc="Creating JSON annotations"):
        # Get full path from filename
        image_path = filename_to_path.get(image_filename)
        if not image_path:
            logging.warning(f"Could not find full path for {image_filename}")
            continue
            
        annotation_file = annotation_folder / Path(image_filename).with_suffix('.txt').name
        
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            # Get actual image dimensions
            img_width, img_height = get_image_dimensions(image_path)
            
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            # Process each gaze detection in the image
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    
                    # Only process gaze detections (0: no_gaze, 1: gaze)
                    if class_id in [0, 1]:
                        # Convert YOLO format to absolute coordinates
                        x_center, y_center, width, height = map(float, parts[1:5])
                        
                        # Convert from YOLO format (normalized) to absolute coordinates
                        x_center_abs = x_center * img_width
                        y_center_abs = y_center * img_height
                        width_abs = width * img_width
                        height_abs = height * img_height
                        
                        # Convert to [x1, y1, x2, y2] format
                        x1 = max(0, x_center_abs - width_abs / 2)
                        y1 = max(0, y_center_abs - height_abs / 2)
                        x2 = min(img_width, x_center_abs + width_abs / 2)
                        y2 = min(img_height, y_center_abs + height_abs / 2)
                        
                        # Ensure bbox is valid
                        if x2 > x1 and y2 > y1:
                            annotation = {
                                "image_path": str(image_path),
                                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                "label": class_id  # 0: no_gaze, 1: gaze
                            }
                            annotations.append(annotation)
    
    return annotations

def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get actual image dimensions. This is a helper function that should be 
    called to get real image dimensions instead of using placeholders.
    
    Parameters
    ----------
    image_path: str
        Path to the image file
        
    Returns
    -------
    Tuple[int, int]
        Width and height of the image
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        logging.warning(f"Could not get dimensions for {image_path}: {e}")
        return 1920, 1080  # Default fallback
       
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