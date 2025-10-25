import json
import logging
import sqlite3
import cv2
import re
import sys
import shutil
import random
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from pandera import parser

from constants import DataPaths, BasePaths, FaceDetection
from config import FaceConfig, DataConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def parse_retrain_file(retrain_file: Path) -> Dict[str, List[str]]:
    """
    Parses the ID Distribution section from the statistics file.
    
    Parameters
    ----------
    retrain_file : Path
        Path to the "Dataset Split Information" text file.
        
    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing 'train_ids', 'val_ids', 'test_ids'.
    """
    logging.info(f"Parsing existing split IDs from {retrain_file.name}")
    data = {}
    try:
        content = retrain_file.read_text()
        
        # Regex to find ID Distribution block
        train_match = re.search(r"Training IDs: \d+, \[(.*?)\]", content)
        val_match = re.search(r"Validation IDs: \d+, \[(.*?)\]", content)
        test_match = re.search(r"Test IDs: \d+, \[(.*?)\]", content)
        
        if not train_match or not val_match or not test_match:
            raise ValueError("Could not find all required ID distribution blocks.")
            
        def parse_ids(match):
            # Strip quotes and split by comma, then strip whitespace
            ids_str = match.group(1).replace("'", "").replace('"', "")
            return [i.strip() for i in ids_str.split(',') if i.strip()]

        data['train_ids'] = parse_ids(train_match)
        data['val_ids'] = parse_ids(val_match)
        data['test_ids'] = parse_ids(test_match)

        logging.info(f"Loaded {len(data['train_ids'])} Train IDs, {len(data['val_ids'])} Val IDs, {len(data['test_ids'])} Test IDs.")
        return data

    except FileNotFoundError:
        logging.error(f"Retrain file not found at {retrain_file}")
        raise
    except Exception as e:
        logging.error(f"Error parsing retrain file: {e}")
        raise
    
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
    ORDER BY a.video_id, a.image_id
    """

    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        results = cursor.fetchall()
     
    logging.info(f"Found {len(results)} annotations.")   
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

def save_annotations(annotations: List[Tuple], output_dir: Path = None, mode: str = "face-only") -> None:
    """Convert annotations to YOLO format and save them in parallel.
    
    Parameters
    ----------
    annotations : List[Tuple]
        List of tuples containing (category_id, bbox_json, img_name, age)
    output_dir : Path, optional
        Directory to save annotation files
    mode : str
        Detection mode to use for saving annotations (default: "face-only")
    """
    if output_dir is None:
        output_dir = FaceDetection.LABELS_INPUT_DIR
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = defaultdict(list)
    processed, skipped = 0, 0

    for _, bbox_json, img_name, age in annotations:
        # img_name comes from database and should be the image filename without extension
        # Example: quantex_at_home_id255237_2022_05_08_04_000240 -> quantex_at_home_id255237_2022_05_08_04
        img_name_parts = img_name.split("_")
        if len(img_name_parts) < 9:
            logging.warning(f"Invalid image name format: {img_name} (expected at least 9 parts)")
            skipped += 1
            continue

        video_folder_name = "_".join(img_name.split("_")[:-1])
        
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
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Failed to load image: {img_path}")
                skipped += 1
                continue
            
            height, width = img.shape[:2]
            
            bbox = json.loads(bbox_json)
            yolo_bbox = convert_to_yolo_format(width, height, bbox)
            
            if mode == "face-only":
                class_id = FaceConfig.AGE_GROUP_TO_CLASS_ID_FACE_ONLY.get(age, 99)
            elif mode == "age-binary":
                class_id = FaceConfig.AGE_GROUP_TO_CLASS_ID_AGE_BINARY.get(age, 99)
            else:
                logging.error(f"Unknown detection mode: {mode}")
                skipped += 1
                continue
            # Skip invalid class IDs
            if class_id == 99:
                logging.warning(f"Unknown age group '{age}' for {img_name}")
                skipped += 1
                continue

            # Write the dynamically determined class_id (0 for all faces in 'face-only')
            files[img_name].append(f"{class_id} " + " ".join(map(str, yolo_bbox)) + "\n")
            processed += 1
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            skipped += 1

    # Save annotation files in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        for img_name, lines in files.items():
            output_file = output_dir / f"{img_name}.txt"
            executor.submit(write_annotations, output_file, lines)

    logging.info(f"Processed {processed}, skipped {skipped}")

def fetch_noisy_frames() -> List[str]:
    """
    Fetch file_names of frames that contain *only* the category_id = -1
    (i.e., frames that were flagged as 'bad' or 'no face' but have no other faces).
    
    Returns
    -------
    List[str]
        List of file_names (stems) to be excluded from the negative sample pool.
    """
    query = """
    SELECT
        i.file_name
    FROM images i
    JOIN annotations a ON i.frame_id = a.image_id AND i.video_id = a.video_id
    WHERE a.category_id = -1
    GROUP BY i.file_name
    HAVING COUNT(CASE WHEN a.category_id != -1 THEN 1 END) = 0
    """

    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        # Fetch results and flatten the list of tuples (file_name,) into a list of strings
        results = [row[0] for row in cursor.fetchall()]

    return results

def get_total_number_of_annotated_frames(label_path: Path, image_folder: Path = FaceDetection.IMAGES_INPUT_DIR) -> Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]:
    """
    Returns ALL annotated (positive) frames and ALL potential clean negative frame candidates (sampled by frame offset).
    
    Parameters
    ----------
    label_path : Path
        Path to the directory containing label files.
    image_folder : Path
        Path to the image folder containing video subfolders.
    
    Returns
    -------
    Tuple[List[Tuple[str, str]], Dict[str, List[Tuple[str, str]]]]
        A tuple containing:
        - List of tuples (image_path, image_id) for all positive annotated frames.
        - Dictionary mapping video names to lists of tuples (image_path, image_id) for potential negative frame candidates.
    """
    DEFAULT_OFFSET = 0

    # 1. Fetch frames to be explicitly excluded from negative samples (only -1 annotation)
    frames_to_exclude = fetch_noisy_frames()
    excluded_frames_set = set(frames_to_exclude)
    
    video_names = set()
    positive_images = []
    positive_frame_stems = set()
    
    # 2. Find all videos that have any valid annotations
    for annotation_file in label_path.glob("*.txt"):
        if annotation_file.stat().st_size > 0:
            try:
                # Extract video name from annotation file stem
                # match example: quantex_at_home_id255237_2022_05_08_04_000240.txt -> positive_frame_stems: quantex_at_home_id255237_2022_05_08_04_000240
                # video_name: quantex_at_home_id255237_2022_05_08_04_000240
                stem = annotation_file.stem
                match = re.match(r"(.+)_\d{6}$", stem)
                if match:
                    video_name = match.group(1)
                    video_names.add(video_name)
                    positive_frame_stems.add(stem)
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")

    logging.info(f"Found {len(video_names)} unique video names with annotations.")

    # 3. Get annotated frames and collect ALL valid potential negative frames (sampled frames only)
    video_negative_candidates = defaultdict(list) # Use defaultdict for cleaner grouping
    
    for video_name in video_names:
        video_path = image_folder / video_name
        if video_path.exists() and video_path.is_dir():
            
            exception_map = DataConfig.SHIFTED_VIDEOS_OFFSETS
            
            # Iterate through all frames in the video folder
            for frame in video_path.iterdir():
                if frame.is_file():
                    stem = frame.stem
                    parts = stem.split("_")
                    
                    # --- Frame Number Check ---
                    frame_number = -1
                    if len(parts) >= 9:
                        try:
                            frame_number = int(parts[-1])
                        except ValueError:
                            continue # Skip if frame number is not an integer
                    
                    # Determine if the frame is a sampled frame
                    is_sampled_frame = False
                    
                    if video_name in exception_map:
                        # Logic for exception videos with frame shift
                        start_frame, shift = exception_map[video_name]
                        
                        if frame_number < start_frame:
                            # Rule 1: Before the exception start, use standard sampling
                            if frame_number % DataConfig.FPS == DEFAULT_OFFSET:
                                is_sampled_frame = True
                        else:
                            # Rule 2: From the exception start onward, use the shifted modulo rule
                            # Sampling is shifted to start at start_frame and then every DataConfig.FPS frames
                            if (frame_number - start_frame) % DataConfig.FPS == DEFAULT_OFFSET:
                                is_sampled_frame = True
                            
                    else:
                        # Default rule for all other videos: multiples of DataConfig.FPS
                        if frame_number % DataConfig.FPS == DEFAULT_OFFSET:
                            is_sampled_frame = True
                    
                    # A. If this frame has a valid annotation, include it as a positive
                    if stem in positive_frame_stems:
                        positive_images.append((str(frame.resolve()), stem))
                        continue

                    # B. If it's a sampled frame AND a clean negative, add to candidates
                    if is_sampled_frame and stem not in excluded_frames_set:
                        video_negative_candidates[video_name].append((str(frame.resolve()), stem))
    
    logging.info(f"Total positive (annotated) frames found: {len(positive_images)}")
    logging.info(f"Total potential negative candidates found: {sum(len(c) for c in video_negative_candidates.values())}")
    
    # Return positive images list and the dict of negative candidates
    return positive_images, dict(video_negative_candidates)

def get_class_distribution(total_images: list, annotation_folder: Path, mode: str) -> pd.DataFrame:
    """
    Reads label files and groups images based on their class distribution for the given detection mode.

    Parameters:
    ----------
    total_images: list
        List of tuples containing image paths and IDs.
    annotation_folder: Path
        Path to the directory containing label files.
    mode: str
        The mode for class mapping ('face-only' or 'age-binary').

    Returns:
    -------
    pd.DataFrame
        DataFrame containing image filenames, IDs, and their corresponding one-hot encoded class labels.
    """   
    if not total_images:
        logging.error("No images provided to get_class_distribution")
        return pd.DataFrame()
        
    image_class_mapping = []

    if mode == "face-only":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_FACE_ONLY
        target_class_names = FaceConfig.TARGET_LABELS_FACE_ONLY
    elif mode == "age-binary":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
        target_class_names = FaceConfig.TARGET_LABELS_AGE_BINARY
    else:
        logging.error(f"Unknown detection mode: {mode}")
        return pd.DataFrame()

    # Step 1: Read each image and its corresponding annotation file
    for i, (image_path, image_id) in enumerate(total_images):
        image_file = Path(image_path)
        annotation_file = annotation_folder / image_file.with_suffix('.txt').name

        mode_labels = set()
        if annotation_file.exists() and annotation_file.stat().st_size > 0:
            try:
                with open(annotation_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\n')
                        class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                        
                        original_labels = [id_to_label_map[cid] for cid in class_ids if cid in id_to_label_map]
                        mode_labels.update(original_labels)
                            
            except Exception as e:
                logging.warning(f"Error reading annotation file {annotation_file}: {e}")

        # Create one-hot encoded dictionary for the image
        try:
            mapping_entry = {
                "filename": image_file.stem,
                "id": image_id,
                "has_annotation": bool(mode_labels),
            }

            for class_name in target_class_names:
                mapping_entry[class_name] = 1 if class_name in mode_labels else 0
                
            image_class_mapping.append(mapping_entry)
            
        except Exception as e:
            logging.error(f"Error creating mapping entry for {image_file.name}: {e}")
            continue
        
    df = pd.DataFrame(image_class_mapping)
    
    return df

def get_first_n_minutes_frames(child_ids: List[str], image_folder: Path, minutes: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Get frames from the first N minutes of recordings for specified child IDs.
    Filters based on actual frame numbers, not just file count.
    Splits them between training (first 80%) and validation (remaining 20%).

    Parameters
    ----------
    child_ids : List[str]
        List of child IDs to get frames for
    image_folder : Path
        Path to the image folder
    minutes : int
        Number of minutes from the beginning to extract
        
    Returns
    -------
    Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]
        Two lists of tuples: (train_frames, val_frames) containing (image_path, image_id)
    """
    train_frames = []
    val_frames = []
    
    # Calculate the maximum frame number for N minutes
    max_frame_number = minutes * 60 * DataConfig.FPS  # e.g., 5 * 60 * 30 = 9000
    
    # Calculate frame number splits (80% for training, 20% for validation)
    train_frame_max = int(max_frame_number * 0.8)  # e.g., frame 7200 for 4 minutes
    val_frame_max = max_frame_number  # e.g., frame 9000 for 5 minutes total
    
    for child_id in child_ids:
        child_train_frames = []
        child_val_frames = []
        
        # Find all video folders for this child
        for video_folder in sorted(image_folder.iterdir()):
            if video_folder.is_dir() and child_id in video_folder.name:
                # Get all frames from this video folder and filter by frame number
                video_frames = []
                for frame in video_folder.iterdir():
                    if frame.is_file() and frame.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Extract frame number from filename
                        # Example: quantex_at_home_id255237_2022_05_08_04_000240.jpg -> frame 240
                        parts = frame.stem.split("_")
                        if len(parts) >= 9:
                            try:
                                frame_number = int(parts[-1])  # Last part should be frame number
                                
                                # Only include frames within the first N minutes
                                if frame_number <= max_frame_number:
                                    video_frames.append((frame, frame_number))
                            except ValueError:
                                logging.warning(f"Could not extract frame number from {frame.name}")
                                continue
                
                # Sort by frame number to ensure chronological order
                video_frames.sort(key=lambda x: x[1])
                
                # Assign frames based on frame number thresholds
                for frame_path, frame_number in video_frames:
                    # Extract image ID
                    parts = frame_path.name.split("_")
                    if len(parts) > 3 and parts[3].startswith('id'):
                        image_id = parts[3].replace("id", "")
                    else:
                        image_id = frame_path.stem
                    
                    # Assign to training (first 80% of time) or validation (remaining 20%)
                    if frame_number <= train_frame_max:
                        child_train_frames.append((str(frame_path.resolve()), image_id))
                    elif frame_number <= val_frame_max:
                        child_val_frames.append((str(frame_path.resolve()), image_id))
        
        train_frames.extend(child_train_frames)
        val_frames.extend(child_val_frames)
        
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
    
    return all_frames

# =======================================================
# Retrain Split Function (Uses Fixed IDs and HNM List)
# =======================================================

def retrain_split_by_child_id(df: pd.DataFrame, negative_candidates: Dict[str, List[Tuple[str, str]]], train_ids: List[str], val_ids: List[str], test_ids: List[str], hard_neg_file: Path, labels_input_dir: Path = None, mode: str = "face-only") -> Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data using FIXED child IDs loaded from a file, incorporates Hard Negatives 
    into the training set, and samples the remaining negative frames.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image data and annotations
    negative_candidates : Dict[str, List[Tuple[str, str]]]
        Dictionary of potential negative frame candidates per video
    train_ids : List[str]   
        List of child IDs for training set
    val_ids : List[str]
        List of child IDs for validation set
    test_ids : List[str]
        List of child IDs for test set
    hard_neg_file : Path
        Path to the hard negative file
    labels_input_dir : Path
        Path to the labels input directory
    mode : str
        Mode for the retraining process ('face-only' or 'age-binary')
        
    Returns
    -------
    Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - List of child IDs for training set
        - List of child IDs for validation set
        - List of child IDs for test set
        - DataFrame for training set
        - DataFrame for validation set
        - DataFrame for test set
    """
    logging.info("Running RETRAIN mode with fixed ID distribution.")

    if labels_input_dir is None:
        labels_input_dir = FaceDetection.LABELS_INPUT_DIR

    # --- Infer Detection Mode and Get Correct Mapping ---
    if mode == "face-only":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_FACE_ONLY
    elif mode == "age-binary":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
    else:
        id_to_label_map = {} 
    
    def get_child_id_from_filename(filename: str) -> str:
        match = re.search(r'id(\d+)_', filename)
        return 'id' + match.group(1) if match else None

    df['child_id'] = df['filename'].apply(get_child_id_from_filename)
    df.dropna(subset=['child_id'], inplace=True)

    # --- Prepare Split DataFrames (Positive Frames Only) ---
    train_df_pos = df[df["child_id"].isin(train_ids) & (df["has_annotation"] == True)].copy()
    val_df_pos = df[df["child_id"].isin(val_ids) & (df["has_annotation"] == True)].copy()
    
    # Test set is initialized with ALL frames from test children
    test_df = df[df["child_id"].isin(test_ids)].copy()
    
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    
    # --- 2. Compile Negative Pools and Load Hard Negatives ---
    
    train_negative_candidates = []
    val_negative_candidates = []
    hard_negative_stems = set()

    # Load Hard Negatives from the provided file
    try:
        hard_neg_content = hard_neg_file.read_text().splitlines()
        hard_negative_stems = {line.strip() for line in hard_neg_content if line.strip()}
        logging.info(f"Loaded {len(hard_negative_stems)} Hard Negative image stems.")
    except Exception as e:
        logging.warning(f"Could not load Hard Negative file {hard_neg_file}: {e}")

    def get_child_id_from_video_name(video_name):
        match = re.search(r'id(\d+)', video_name)
        return 'id' + match.group(1) if match else None
        
    hard_negatives = []
    soft_negatives = []

    # Filter negative candidates into Hard (in file) and Soft (not in file)
    for video_name, candidates in negative_candidates.items():
        child_id = get_child_id_from_video_name(video_name)
        
        for image_path, image_id in candidates:
            filename = Path(image_path).stem
            
            if child_id in train_ids:
                if filename in hard_negative_stems:
                    hard_negatives.append((image_path, image_id))
                else:
                    soft_negatives.append((image_path, image_id))
            
            elif child_id in val_ids:
                val_negative_candidates.append((image_path, image_id))

    logging.info(f"Train negative pool: {len(hard_negatives)} Hard Negatives, {len(soft_negatives)} Soft Negatives.")
    
    # --- 3. Apply Negative Sampling (HNM Priority for Train) ---
    random.seed(DataConfig.RANDOM_SEED)
    
    # TRAIN NEGATIVE SAMPLING (Prioritize HNM)
    num_train_pos = len(train_df_pos)
    target_train_neg = int(num_train_pos * FaceConfig.NEGATIVE_SAMPLING_RATIO)
    
    # 3.1. Start with ALL Hard Negatives
    sampled_train_neg = hard_negatives[:]
    
    # 3.2. Fill remaining quota with Soft Negatives
    remaining_quota = target_train_neg - len(sampled_train_neg)
    
    if remaining_quota > 0:
        if len(soft_negatives) >= remaining_quota:
            sampled_train_neg.extend(random.sample(soft_negatives, remaining_quota))
        else:
            sampled_train_neg.extend(soft_negatives)
            logging.warning(f"Train: Only {len(sampled_train_neg)} negative frames available after hard negs, target was {target_train_neg}.")
    
    # VAL NEGATIVE SAMPLING (Random sampling retained)
    num_val_pos = len(val_df_pos)
    target_val_neg = int(num_val_pos * FaceConfig.NEGATIVE_SAMPLING_RATIO)
    
    if len(val_negative_candidates) >= target_val_neg:
        sampled_val_neg = random.sample(val_negative_candidates, target_val_neg)
    else:
        sampled_val_neg = val_negative_candidates
        logging.warning(f"Val: Only {len(sampled_val_neg)} negative frames available, target was {target_val_neg}.")

    # --- 4. Finalize DataFrames ---
    
    def create_neg_df(sampled_neg_list, child_id_getter, class_cols):
        entries = []
        for image_path, image_id in sampled_neg_list:
            filename = Path(image_path).stem
            entries.append({
                "filename": filename,
                "id": image_id,
                "has_annotation": False,
                "child_id": child_id_getter(filename),
                **{col: 0 for col in class_cols}
            })
        return pd.DataFrame(entries)

    train_neg_df = create_neg_df(sampled_train_neg, get_child_id_from_filename, class_columns)
    val_neg_df = create_neg_df(sampled_val_neg, get_child_id_from_filename, class_columns)
    
    train_df = pd.concat([train_df_pos, train_neg_df], ignore_index=True)
    val_df = pd.concat([val_df_pos, val_neg_df], ignore_index=True)
    
    return (train_df['filename'].tolist(), val_df['filename'].tolist(), test_df['filename'].tolist(),
            train_df, val_df, test_df)
    
def split_by_child_id(df: pd.DataFrame, negative_candidates: Dict[str, List[Tuple[str, str]]], train_ratio: float = FaceConfig.TRAIN_SPLIT_RATIO, add_first_minutes: bool = False, minutes: int = 5, labels_input_dir: Path = None, mode: str = "face-only") -> Tuple[List[str], List[str], List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the DataFrame into training, validation, and test sets using child IDs as the unit, while balancing class distributions.

    For test set: Keeps ALL frames from test children (not just annotated ones) to reflect real-world scenarios.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with image data and annotations
    negative_candidates : Dict[str, List[Tuple[str, str]]]
        Dictionary of potential negative frame candidates per video
    train_ratio : float
        Ratio for training split
    add_first_minutes : bool
        If True, adds first N minutes of test children to training/validation
    minutes : int
        Number of minutes to add from beginning of test children recordings
    labels_input_dir : Path
        Path to labels directory (if None, uses default FaceDetection.LABELS_INPUT_DIR)
    mode : str
        Detection mode to use for splitting ('face-only' or 'age-binary')
    """
    # Define minimum number of face images required in the test set as fixed percentage of positive images
    if labels_input_dir is None:
        labels_input_dir = FaceDetection.LABELS_INPUT_DIR
        
    if 'filename' not in df.columns:
        logging.error(f"'filename' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # --- Infer Detection Mode and Get Correct Mapping ---
    if mode == "face-only":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_FACE_ONLY
    elif mode == "age-binary":
        id_to_label_map = FaceConfig.MODEL_CLASS_ID_TO_LABEL_AGE_BINARY
    else:
        id_to_label_map = {} 
        logging.error("Unknown detection mode specified.")

    def get_child_id_from_filename(filename: str) -> str:
        match = re.search(r'id(\d+)_', filename)
        if match:
            return 'id' + match.group(1)
        return None
    
    df['child_id'] = df['filename'].apply(get_child_id_from_filename)
    df.dropna(subset=['child_id'], inplace=True)

    # --- Calculate Counts per Child ---
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    if not class_columns:
        logging.error("No class columns found in DataFrame")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    child_group_counts = df.groupby('child_id')[class_columns].sum()
    child_face_coverage = df.groupby('child_id').agg({'has_annotation': ['sum', 'count']})
    child_face_coverage.columns = ['faces_count', 'total_count']
    
    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    if n_total < 3 * FaceConfig.MIN_IDS_PER_SPLIT:
        logging.error(f"Not enough unique children to create splits. Found {n_total}, need at least {3 * FaceConfig.MIN_IDS_PER_SPLIT}.")
        return [], [], [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # --- Prepare Child Face Counts Sorted ---
    child_face_counts_sorted = [(child_id, child_face_coverage.loc[child_id, 'faces_count']) 
                                for child_id in sorted_child_ids]
    child_face_counts_sorted.sort(key=lambda x: x[1], reverse=True) 
    
    global_counts = child_group_counts[class_columns].sum()
    target_class_0_ratio = global_counts[class_columns[0]] / global_counts.sum() if global_counts.sum() > 0 else 0

    # --- Utility Functions for Balancing ---
    def deviation_from_target(current_counts, child_counts):
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0: return 0
        new_ratio = new_counts[class_columns[0]] / new_counts.sum()
        return abs(new_ratio - target_class_0_ratio)
    
    def face_coverage_deviation(current_face_count, current_total_count, child_face_count, child_total_count):
        new_face_count = current_face_count + child_face_count
        new_total_count = current_total_count + child_total_count
        if new_total_count == 0: return 0
        new_face_ratio = new_face_count / new_total_count
        return abs(new_face_ratio - 0.5) 

    # --- 1. ID SELECTION (Weighted Balancing for ALL three splits) ---
    train_ids = []
    val_ids = []
    test_ids = []
    
    # Calculate target split sizes based on the total children
    n_train_target = max(int(n_total * train_ratio), FaceConfig.MIN_IDS_PER_SPLIT)
    remaining_ids = n_total - n_train_target
    n_val_target = max(int(remaining_ids / 2), FaceConfig.MIN_IDS_PER_SPLIT)
    n_test_target = max(n_total - n_train_target - n_val_target, FaceConfig.MIN_IDS_PER_SPLIT)
    
    # Adjust targets if necessary to ensure sum is n_total
    n_sum = n_train_target + n_val_target + n_test_target
    if n_sum > n_total:
        n_test_target = max(n_total - n_train_target - n_val_target, FaceConfig.MIN_IDS_PER_SPLIT)
        n_train_target = max(n_total - n_val_target - n_test_target, FaceConfig.MIN_IDS_PER_SPLIT)
        n_val_target = max(n_total - n_train_target - n_test_target, FaceConfig.MIN_IDS_PER_SPLIT)

    split_targets = {'train': n_train_target, 'val': n_val_target, 'test': n_test_target}
    
    split_info_all = {
        s: {'ids': [], 'current_counts': pd.Series([0] * len(class_columns), index=class_columns), 'face_count': 0, 'total_count': 0, 'target': split_targets[s]}
        for s in ['train', 'val', 'test']
    }
    
    # Iterate over all children, assigning the highest face-count children first to the split 
    for child_id, child_face_count in child_face_counts_sorted:
        child_counts = child_group_counts.loc[child_id, class_columns]
        child_total_count = child_face_coverage.loc[child_id, 'total_count']

        eligible_splits = []
        for s in ['train', 'val', 'test']:
            if len(split_info_all[s]['ids']) < split_info_all[s]['target']:
                eligible_splits.append(s)

        if not eligible_splits:
            logging.warning(f"No space left for child {child_id}. Skipping.")
            continue

        # Pick split that minimizes the weighted deviation (0.3 class + 0.7 coverage)
        best_split = min(
            eligible_splits,
            key=lambda s: (
                deviation_from_target(split_info_all[s]['current_counts'], child_counts) * 0.3 +
                face_coverage_deviation(split_info_all[s]['face_count'], split_info_all[s]['total_count'], 
                                       child_face_count, child_total_count) * 0.7
            )
        )

        # Assign child
        split_info_all[best_split]['ids'].append(child_id)
        split_info_all[best_split]['current_counts'] += child_counts
        split_info_all[best_split]['face_count'] += child_face_count
        split_info_all[best_split]['total_count'] += child_total_count

    train_ids = split_info_all['train']['ids']
    val_ids = split_info_all['val']['ids']
    test_ids = split_info_all['test']['ids']

    current_test_faces = split_info_all['test']['face_count']

    logging.info(f"Test split face coverage: {current_test_faces} faces from {len(test_ids)} children.")
    logging.info(f"Train/Val split: {len(train_ids)} train IDs, {len(val_ids)} val IDs.")

    # --- 2. Build Final DataFrames and Apply Negative Sampling ---

    # 2a. Initial split of POSITIVE/ANNOTATED images based on IDs
    train_df_pos = df[df["child_id"].isin(train_ids) & (df["has_annotation"] == True)].copy()
    val_df_pos = df[df["child_id"].isin(val_ids) & (df["has_annotation"] == True)].copy()
    
    # Test set still gets ALL frames from assigned children for true imbalance representation
    test_df = df[df["child_id"].isin(test_ids)].copy() 

    # 2b. Compile negative pools for Train and Val (Uses negative_candidates passed to the function)
    train_negative_candidates = []
    val_negative_candidates = []
    
    def get_child_id_from_video_name(video_name):
        match = re.search(r'id(\d+)', video_name)
        return 'id' + match.group(1) if match else None

    for video_name, candidates in negative_candidates.items():
        child_id = get_child_id_from_video_name(video_name)
        if child_id in train_ids:
            train_negative_candidates.extend(candidates)
        elif child_id in val_ids:
            val_negative_candidates.extend(candidates)
                
    # 2c. Apply 75% negative sampling to Train and Val sets
    random.seed(DataConfig.RANDOM_SEED)
    
    # TRAIN NEGATIVE SAMPLING
    num_train_pos = len(train_df_pos)
    target_train_neg = int(num_train_pos * FaceConfig.NEGATIVE_SAMPLING_RATIO)
    
    if len(train_negative_candidates) >= target_train_neg:
        sampled_train_neg = random.sample(train_negative_candidates, target_train_neg)
    else:
        sampled_train_neg = train_negative_candidates
        logging.warning(f"Train: Only {len(sampled_train_neg)} negative frames available, target was {target_train_neg}.")

    # VAL NEGATIVE SAMPLING
    num_val_pos = len(val_df_pos)
    target_val_neg = int(num_val_pos * FaceConfig.NEGATIVE_SAMPLING_RATIO)
    
    if len(val_negative_candidates) >= target_val_neg:
        sampled_val_neg = random.sample(val_negative_candidates, target_val_neg)
    else:
        sampled_val_neg = val_negative_candidates
        logging.warning(f"Val: Only {len(sampled_val_neg)} negative frames available, target was {target_val_neg}.")

    # 2d. Convert sampled negative list to DataFrame rows
    def create_neg_df(sampled_neg_list, child_id_getter, class_cols):
        entries = []
        for image_path, image_id in sampled_neg_list:
            filename = Path(image_path).stem
            entries.append({
                "filename": filename,
                "id": image_id,
                "has_annotation": False,
                "child_id": child_id_getter(filename),
                **{col: 0 for col in class_cols}
            })
        return pd.DataFrame(entries)

    train_neg_df = create_neg_df(sampled_train_neg, get_child_id_from_filename, class_columns)
    val_neg_df = create_neg_df(sampled_val_neg, get_child_id_from_filename, class_columns)
    
    # 2e. Finalize Train and Val DataFrames
    train_df = pd.concat([train_df_pos, train_neg_df], ignore_index=True)
    val_df = pd.concat([val_df_pos, val_neg_df], ignore_index=True)
    
    # --- 3. Handle First N Minutes and Final Test Set Enhancement (retained) ---
    
    first_minutes_train_frames = []
    first_minutes_val_frames = []
    
    if add_first_minutes:
        
        first_minutes_train_frames, first_minutes_val_frames = get_first_n_minutes_frames(
            test_ids, FaceDetection.IMAGES_INPUT_DIR, minutes
        )
        
        # Add first minutes frames to training set
        if first_minutes_train_frames:
            additional_train_entries = []
            for image_path, image_id in first_minutes_train_frames:
                image_file = Path(image_path)
                if image_file.stem not in train_df['filename'].values:
                    annotation_file = labels_input_dir / f"{image_file.stem}.txt"
                    has_annotation = annotation_file.exists() and annotation_file.stat().st_size > 0
                    
                    entry = {
                        "filename": image_file.stem,
                        "id": image_id,
                        "has_annotation": has_annotation,
                        "child_id": get_child_id_from_filename(image_file.stem)
                    }
                    
                    labels = []
                    if has_annotation:
                        try:
                            with open(annotation_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    lines = content.split('\n')
                                    class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                                    labels = [id_to_label_map[cid] for cid in class_ids if cid in id_to_label_map]
                        except Exception as e:
                            logging.warning(f"Error reading annotation file {annotation_file}: {e}")
                    
                    for class_name in class_columns:
                        entry[class_name] = 1 if class_name in labels else 0
                    
                    additional_train_entries.append(entry)
            
            if additional_train_entries:
                additional_train_df = pd.DataFrame(additional_train_entries)
                train_df = pd.concat([train_df, additional_train_df], ignore_index=True)
        
        # Add first minutes frames to validation set
        if first_minutes_val_frames:
            additional_val_entries = []
            for image_path, image_id in first_minutes_val_frames:
                image_file = Path(image_path)
                if image_file.stem not in val_df['filename'].values:
                    annotation_file = labels_input_dir / f"{image_file.stem}.txt"
                    has_annotation = annotation_file.exists() and annotation_file.stat().st_size > 0
                    
                    entry = {
                        "filename": image_file.stem,
                        "id": image_id,
                        "has_annotation": has_annotation,
                        "child_id": get_child_id_from_filename(image_file.stem)
                    }
                    
                    labels = []
                    if has_annotation:
                        try:
                            with open(annotation_file, 'r') as f:
                                content = f.read().strip()
                                if content:
                                    lines = content.split('\n')
                                    class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                                    labels = [id_to_label_map[cid] for cid in class_ids if cid in id_to_label_map]
                        except Exception as e:
                            logging.warning(f"Error reading annotation file {annotation_file}: {e}")
                    
                    for class_name in class_columns:
                        entry[class_name] = 1 if class_name in labels else 0
                    
                    additional_val_entries.append(entry)
            
            if additional_val_entries:
                additional_val_df = pd.DataFrame(additional_val_entries)
                val_df = pd.concat([val_df, additional_val_df], ignore_index=True)

    # For test set: Add ALL frames from videos that have annotations for test children    
    test_videos_with_annotations = set()
    for filename in df[df['child_id'].isin(test_ids) & (df['has_annotation'] == True)]['filename']:
        video_name = "_".join(filename.split("_")[:-1])
        test_videos_with_annotations.add(video_name)
    
    first_minutes_frame_names = set()
    if add_first_minutes:
        for image_path, _ in first_minutes_train_frames + first_minutes_val_frames:
            first_minutes_frame_names.add(Path(image_path).stem)
    
    # Add ALL frames from videos that have annotations (not just annotated frames)
    additional_entries = []
    for video_name in test_videos_with_annotations:
        video_path = FaceDetection.IMAGES_INPUT_DIR / video_name
        if video_path.exists() and video_path.is_dir():
            for frame in video_path.iterdir():
                if frame.is_file() and frame.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    parts = frame.name.split("_")
                    if len(parts) > 3 and parts[3].startswith('id'):
                        image_id = parts[3].replace("id", "")
                    else:
                        image_id = frame.stem
                    
                    if (frame.stem not in test_df['filename'].values and 
                        frame.stem not in first_minutes_frame_names):
                        
                        annotation_file = labels_input_dir / f"{frame.stem}.txt"
                        has_annotation = annotation_file.exists() and annotation_file.stat().st_size > 0
                        
                        entry = {
                            "filename": frame.stem,
                            "id": image_id,
                            "has_annotation": has_annotation,
                            "child_id": get_child_id_from_filename(frame.stem)
                        }
                        
                        labels = []
                        if has_annotation:
                            try:
                                with open(annotation_file, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        lines = content.split('\n')
                                        class_ids = {int(line.split()[0]) for line in lines if line.strip()}
                                        labels = [id_to_label_map[cid] for cid in class_ids if cid in id_to_label_map]
                            except Exception as e:
                                logging.warning(f"Error reading annotation file {annotation_file}: {e}")
                        
                        for class_name in class_columns:
                            entry[class_name] = 1 if class_name in labels else 0
                        
                        additional_entries.append(entry)
    
    if additional_entries:
        additional_df = pd.DataFrame(additional_entries)
        test_df = pd.concat([test_df, additional_df], ignore_index=True)

    return (train_df['filename'].tolist(), val_df['filename'].tolist(), test_df['filename'].tolist(),
            train_df, val_df, test_df)

def move_images(image_names: list, 
                split_type: str, 
                label_path: Path,
                input_dir: Path = None,
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
    input_dir: Path
        Custom input directory (if None, uses default FaceDetection.INPUT_DIR)
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

    # Use custom input_dir if provided, otherwise use default
    if input_dir is None:
        input_dir = FaceDetection.INPUT_DIR

    image_dst_dir = input_dir / "images" / split_type
    label_dst_dir = input_dir / "labels" / split_type
    
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
    
def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List, add_first_minutes: bool = False, minutes: int = 5, mode: str = "face-only"):
    """
    Generates a statistics file with dataset split information, including percentages.
    
    Parameters:
    ----------
    df : pd.DataFrame
        Original DataFrame with all images and annotations.
    df_train : pd.DataFrame
        Training set DataFrame.
    df_val : pd.DataFrame
        Validation set DataFrame.
    df_test : pd.DataFrame
        Test set DataFrame.
    train_ids : List
        List of training child IDs.
    val_ids : List
        List of validation child IDs.
    test_ids : List
        List of test child IDs.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_face_det_{timestamp}.txt"
    
    class_columns = [col for col in df.columns if col not in ['filename', 'id', 'has_annotation', 'child_id']]
    
    # Calculate totals based on original balanced dataset (before test enhancement)
    original_total = len(df_train) + len(df_val) + len(df_test[df_test['has_annotation'] == True])
    
    with open(file_path, "w") as f:
        f.write(f"Dataset Split Information - {timestamp}\n")
        f.write(f"*** DETECTION MODE: {mode.upper()} ***\n")
        if add_first_minutes:
            f.write(f"First {minutes} minutes of test children added to train/val\n")
        f.write("\n")
        
        # Original distribution (before test enhancement)
        f.write(f"Original Balanced Distribution (before test enhancement):\n")
        f.write(f"Total Images: {original_total}\n")
        
        if class_columns:
            # If age-binary, include class id numbers using FaceConfig mapping if available
            try:
                label_to_id = {v: k for k, v in FaceConfig.MODEL_CLASS_ID_TO_LABEL.items()}
            except Exception:
                label_to_id = {}

            for col in class_columns:
                original_count = df_train[col].sum() + df_val[col].sum() + df_test[df_test['has_annotation'] == True][col].sum()
                if original_total > 0:
                    pct = original_count / original_total
                else:
                    pct = 0

                if col in label_to_id:
                    f.write(f"Class {label_to_id[col]} {col}: {original_count} images ({pct:.2%})\n")
                else:
                    f.write(f"Class {col}: {original_count} images ({pct:.2%})\n")
        f.write("\n")

        f.write("Split Distribution (within each split):\n")
        f.write("--------------------------------------------------\n\n")

        def write_split_info(split_name, split_df):
            total_split = len(split_df)
            
            # Calculate face coverage (frames with any faces)
            frames_with_faces = split_df[split_df['has_annotation'] == True]
            face_coverage = len(frames_with_faces) / total_split if total_split > 0 else 0
            
            f.write(f"{split_name} Set:\n")
            f.write(f"Total Images: {total_split}\n")
            f.write(f"Face Coverage: {len(frames_with_faces)} ({face_coverage:.2%}) - frames with any faces\n")
            f.write(f"No Face: {total_split - len(frames_with_faces)} ({1-face_coverage:.2%}) - frames without faces\n")
            
            for col in class_columns:
                count = split_df[col].sum()
                ratio = count / total_split if total_split > 0 else 0
                # Print label names in lowercase (e.g., 'child', 'adult') for readability
                f.write(f"{col}: {count} ({ratio:.2%}) - within split\n")
            f.write("\n")

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

def split_data(annotation_folder: Path, add_first_minutes: bool = False, minutes: int = 5, mode: str = "face-only", data_distribution_file: Path = None, hard_neg_file: Path = None):
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
    mode: str
        The mode for class mapping ('face-only' or 'age-binary').
    data_distribution_file: Path
        Path to data distribution file (if any).
    hard_neg_file: Path
        Path to hard negative samples file (if any).
    """
    logging.info(f"Starting dataset preparation for face detection in mode: {mode}")

    # --- 1. Determine Output Directory ---
    if data_distribution_file:
        input_dir = FaceDetection.INPUT_DIR.parent / (FaceDetection.INPUT_DIR.name + "_retrain")
    else:
        input_dir = FaceDetection.INPUT_DIR
    
    try:
        # --- 2. Get All Positive Images and Negative Candidates ---
        positive_images, negative_candidates = get_total_number_of_annotated_frames(annotation_folder)
        
        if not positive_images:
            logging.error("No annotated images found.")
            return
        
        # --- 3. Get initial DataFrame from POSITIVE images ---
        df = get_class_distribution(positive_images, annotation_folder, mode)
        
        if df.empty:
            logging.error("DataFrame is empty. Check class distribution function.")
            return
        
        # --- 4. Select Splitting Strategy ---
        if data_distribution_file:
            # RETRAIN MODE: Load fixed IDs and use HNM sampling
            id_data = parse_retrain_file(data_distribution_file)
            train, val, test, df_train, df_val, df_test = retrain_split_by_child_id(
                df, negative_candidates, id_data['train_ids'], id_data['val_ids'], id_data['test_ids'],
                hard_neg_file, annotation_folder, mode
            )
        elsed:
            train, val, test, df_train, df_val, df_test = split_by_child_id(
                df, negative_candidates, len(positive_images), add_first_minutes, minutes, annotation_folder, mode
            )

        # Get the IDs for logging
        train_ids = df_train['child_id'].unique().tolist() if 'child_id' in df_train.columns else []
        val_ids = df_val['child_id'].unique().tolist() if 'child_id' in df_val.columns else []
        test_ids = df_test['child_id'].unique().tolist() if 'child_id' in df_test.columns else []

        # --- 5. Generate Statistics and Move Files ---
        generate_statistics_file(df, df_train, df_val, df_test, train_ids, val_ids, test_ids, 
                                add_first_minutes=add_first_minutes, minutes=minutes, mode=mode)
        
        for split_name, split_set in [("train", train), ("val", val), ("test", test)]:
            if split_set:
                successful, failed = move_images(
                    image_names=split_set,
                    split_type=split_name,
                    label_path=annotation_folder,
                    input_dir=input_dir,
                    n_workers=4
                )
                logging.info(f"{split_name}: Moved {successful}, Failed {failed}")
            else:
                logging.warning(f"No images for {split_name} split")
    
    except Exception as e:
        logging.error(f"Error processing face detection: {str(e)}")
        raise
    
    logging.info(f"Completed dataset preparation for face detection in mode: {mode}")

# ==============================
# Main
# ==============================

def main():
    parser = argparse.ArgumentParser(description='Create input files for face detection YOLO training')
    parser.add_argument('--mode', choices=["face-only", "age-binary"], default="face-only",
                       help='Select the detection mode')
    parser.add_argument('--add-first-minutes', action='store_true', 
                       help='Add first N minutes of test children recordings to training/validation sets')
    parser.add_argument('--minutes', type=int, default=5, # Added default
                       help='Number of minutes from the beginning to add to train/val')
    parser.add_argument('--fetch-annotations', action='store_true',
                       help='Fetch and save annotations from database (default: False)')
    parser.add_argument('--retrain', action='store_true', default=False,
                       help='Activate retrain mode using fixed IDs and hard negative files defined in FaceConfig.')
    args = parser.parse_args()
    
    data_distribution_file = FaceConfig.DATA_DISTRIBUTION_PATH if args.retrain else None
    hard_neg_file = FaceConfig.RETRAIN_FALSE_POSITIVES_PATH if args.retrain else None
    is_retrain_mode = args.retrain

    if is_retrain_mode:
        logging.info("Retrain mode activated. Using fixed IDs and hard negative files from config.")
        
    if args.add_first_minutes:
        logging.info(f"First minutes mode enabled: adding first {args.minutes} minutes of test children to train/val")
    
    try:            
        if args.fetch_annotations:
            # Output annotations to the correct location (standard or retrain folder)
            labels_output_dir = FaceDetection.LABELS_INPUT_DIR
            anns = fetch_all_annotations(FaceConfig.DATABASE_CATEGORY_IDS)
            save_annotations(anns, output_dir=labels_output_dir, mode=args.mode)
        
        split_data(FaceDetection.LABELS_INPUT_DIR, 
                   add_first_minutes=args.add_first_minutes, 
                   minutes=args.minutes,
                   mode=args.mode,
                   data_distribution_file=data_distribution_file, # Will be Path or None
                   hard_neg_file=hard_neg_file)                 # Will be Path or None
                                   
    except Exception as e:
        logging.error(f"Failed: {e}")
        raise
                                   
if __name__ == "__main__":
    main()