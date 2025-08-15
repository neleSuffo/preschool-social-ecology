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
        for ext in DataConfig.VALID_EXTENSIONS:
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

def split_by_child_id(df: pd.DataFrame, train_ratio: float = FaceConfig.TRAIN_SPLIT_RATIO):
    """
    Splits the DataFrame into training, validation, and test sets using child IDs as the unit,
    while keeping each split's class ratio close to the global dataset ratio.
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
    
    # Add child_id to our dataframe by mapping through image names
    def get_child_id_from_filename(filename: str) -> str:
        try:
            return 'id' + filename.split('id')[1].split('_')[0]
        except (IndexError, ValueError):
            return None
    
    df['child_id'] = df['filename'].apply(get_child_id_from_filename)
    df.dropna(subset=['child_id'], inplace=True)

    # --- Counts per child ---
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    child_group_counts = df.groupby('child_id')[class_columns].sum()

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
        'train': {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns)},
        'val':   {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns)},
        'test':  {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns)},
    }

    # Randomize assignment order to reduce bias
    random.shuffle(sorted_child_ids)

    def deviation_from_target(current_counts, child_counts):
        """Absolute difference from target class ratio after adding this child."""
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0:
            return 0
        new_ratio = new_counts[class_columns[0]] / new_counts.sum()
        return abs(new_ratio - target_class_0_ratio)

    for child_id in sorted_child_ids:
        child_counts = child_group_counts.loc[child_id, class_columns]

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

    train_df = df[df["child_id"].isin(train_ids)].drop(columns=["child_id"])
    val_df = df[df["child_id"].isin(val_ids)].drop(columns=["child_id"])
    test_df = df[df["child_id"].isin(test_ids)].drop(columns=["child_id"])

    # Log final split information
    for split_name, split_df in zip(["Train", "Val", "Test"], [train_df, val_df, test_df]):
        if len(split_df) > 0:
            ratio = split_df[class_columns[0]].sum() / len(split_df)
            logging.info(f"{split_name} set: {len(split_df)} images, {class_columns[0]} ratio: {ratio:.3f}")

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
        
        # Split data grouped by child id 
        train, val, test, *_ = split_by_child_id(df)

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