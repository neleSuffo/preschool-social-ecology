import shutil
import random
import logging
import argparse
import os
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from constants import FaceDetection, BasePaths, DataPaths, VALID_TARGETS
from config import DataConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit as MSSS
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: Test sets are kept unbalanced for representative evaluation
# Only training and validation sets undergo balancing to prevent data leakage
# and ensure test results reflect real-world performance




       

         


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
           
def split_yolo_data(annotation_folder: Path, target: str):
    """
    This function prepares the dataset for YOLO training by splitting the images into train, val, and test sets.
    
    Parameters:
    ----------
    annotation_folder: Path
        Path to the directory containing label files.
    target: str
        The target object for YOLO detection or classification.
    """
    logging.info(f"Starting dataset preparation for {target}")

    try:
        # Define mappings for different target types
        target_mappings = {
            "face_cls": {
                1: ("adult_face", FaceDetection.FACE_IMAGES_INPUT_DIR),
                0: ("child_face", FaceDetection.FACE_IMAGES_INPUT_DIR)
            },
            "person_cls": {
                1: ("adult_person", PersonClassification.PERSON_IMAGES_INPUT_DIR),
                0: ("child_person", PersonClassification.PERSON_IMAGES_INPUT_DIR)
            },
        }       
        # Get annotated frames
        total_images = get_total_number_of_annotated_frames(annotation_folder)
        
        if target in target_mappings:
            # Get source directories based on target type
            original_target = target  # Preserve original target name
            input_folder = target_mappings[original_target][0][1]  # Use first mapping's input dir
            class_mapping = target_mappings[original_target]
            # Get custom splits
            train_images, val_images, test_images = split_dataset(
                input_folder, annotation_folder, original_target, class_mapping)
                
            # Process each split
            for split_name, split_images in [("train", train_images), 
                                           ("val", val_images), 
                                           ("test", test_images)]:
                # Separate images into classes
                for class_id in [0, 1]:
                    class_images = [img for img in split_images 
                                  if get_class(img, annotation_folder) == class_id]
                    if class_images:
                        target_name = target_mappings[original_target][class_id][0]
                        successful, failed = move_images(
                            target=target_name,
                            image_names=class_images,
                            split_type=split_name,
                            label_path=annotation_folder,
                            n_workers=4
                        )   
                        logging.info(f"{split_name} {target_name}: Moved {successful}, Failed {failed}")
                    else:
                        target_name = target_mappings[original_target][class_id][0]
                        logging.warning(f"No {target_name} images for {split_name}")
        else:         
            # Multi-class detection case
            df = get_class_distribution(total_images, annotation_folder, target)
            # split data grouped by id with minimum ID requirements
            train, val, test, *_ = multilabel_stratified_split(
                df, target=target, min_ids_per_split=2
            )

            # Move images for each split
            for split_name, split_set in [("train", train), 
                                        ("val", val), 
                                        ("test", test)]:
                if split_set:
                    successful, failed = move_images(
                        target=target,
                        image_names=split_set,
                        split_type=split_name,
                        label_path=annotation_folder,
                        n_workers=4
                    )
                    logging.info(f"{split_name}: Moved {successful}, Failed {failed}")
                else:
                    logging.warning(f"No images for {split_name} split")
    
    except Exception as e:
        logging.error(f"Error processing target {target}: {str(e)}")
        raise
    
    logging.info(f"\nCompleted dataset preparation for {target}")

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