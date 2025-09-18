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

# Configure logging
logging.basicConfig(level=logging.INFO)

# ==============================
# Database Query Functions
# ==============================

def fetch_all_annotations(category_ids: List[int]) -> List[Tuple]:
    """Fetch annotations for given category IDs from the SQLite database.
    
    Retrieves person detection annotations (adults/children) from the database,
    excluding problematic videos defined in the config.
    
    Parameters
    ----------
    category_ids : List[int]
        List of category IDs to filter annotations (typically [1] for person detection).
        
    Returns
    -------
    List[Tuple]       
        List of tuples containing (video_id, frame_id, file_name, category_id, age_group).
    """
    placeholders = ", ".join("?" * len(category_ids))

    # Build SQL exclusion list from config
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
      AND a.outside = 0  -- Only include people inside the frame
      AND v.file_name NOT IN ({excluded_videos_sql})
    ORDER BY i.video_id, i.frame_id
    """
    
    with sqlite3.connect(DataPaths.ANNO_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(query, category_ids)
        results = cursor.fetchall()
        
    logging.info(f"Found {len(results)} annotations after excluding {len(DataConfig.EXCLUDED_VIDEOS)} videos")
    return results

# ==============================
# Data Processing Functions
# ==============================

def build_frame_level_labels(rows: List[Tuple], age_group_mapping: Dict[str, int], model_class_mapping: Dict[int, str]) -> pd.DataFrame:
    """
    Convert individual person annotations into frame-level binary labels.
    
    Aggregates all person detections within each frame into binary presence labels.
    For example, if a frame has 2 children and 1 adult, the result will be:
    {'adult': 1, 'child': 1} indicating both classes are present in the frame.

    Parameters
    ----------
    rows : List[Tuple]
        Raw annotation data: (video_id, frame_id, file_name, category_id, age_group)
    age_group_mapping : Dict[str, int]
        Maps age group strings ('child', 'adult') to class IDs (0, 1)
    model_class_mapping : Dict[int, str]
        Maps class IDs (0, 1) to label strings ('child', 'adult')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['video_id', 'frame_id', 'file_path', 'adult', 'child']
        where 'adult' and 'child' are binary presence indicators (0/1)
    """
    frame_dict = {}

    # Process each annotation
    for video_id, frame_id, file_name, cat_id, age_group in rows:
        # Only process person annotations (category_id = 1)
        if cat_id != 1:
            continue

        # Use (video_id, frame_id, file_name) as unique frame identifier
        key = (video_id, frame_id, file_name)
        if key not in frame_dict:
            # Initialize binary labels for all target classes
            frame_dict[key] = {label: 0 for label in PersonConfig.TARGET_LABELS}

        # Convert age_group to binary label
        if age_group in age_group_mapping:
            label_id = age_group_mapping[age_group]
            label_name = model_class_mapping.get(label_id)

            if label_name:
                frame_dict[key][label_name] = 1  # Mark class as present

    # Convert to DataFrame and verify image files exist
    data = []
    for (video_id, frame_id, file_name), labels in frame_dict.items():
        # Build expected folder path (remove frame number from filename)
        frame_folder = "_".join(file_name.split("_")[:-1])
        
        # Find actual image file (try different extensions)
        image_path = None
        for ext in DataConfig.VALID_EXTENSIONS:
            potential_path = PersonClassification.IMAGES_INPUT_DIR / frame_folder / f"{file_name}{ext}"
            if potential_path.exists():
                image_path = str(potential_path)
                break
        
        # Skip frames where image file is missing
        if image_path is None:
            logging.warning(f"Image not found with any valid extension: {file_name}")
            continue

        # Add frame data to results
        row = {
            "video_id": video_id,
            "frame_id": frame_id,
            "file_path": image_path,
            **labels  # Unpack binary labels (adult: 0/1, child: 0/1)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

# ==============================
# Data Splitting Functions
# ==============================

def split_by_child_id(df: pd.DataFrame, train_ratio: float = PersonConfig.TRAIN_SPLIT_RATIO):
    """
    Split dataset by child IDs to prevent data leakage.
    
    Ensures that all frames from the same child are in the same split (train/val/test).
    Uses a balanced assignment algorithm to maintain similar adult/child ratios across splits.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with frame-level labels
    train_ratio : float
        Proportion of children to assign to training set

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List, List, List]
        (df_train, df_val, df_test, train_ids, val_ids, test_ids)
    """
    logging.info("Starting child ID-based data splitting...")

    # --- Step 1: Map video_id to child_id ---
    conn = sqlite3.connect(DataPaths.ANNO_DB_PATH)
    video_map = pd.read_sql_query("SELECT id as video_id, file_name FROM videos", conn)
    conn.close()

    def extract_child_id(file_name: str) -> str:
        """Extract child ID from video filename (e.g., 'quantex_at_home_id255237_...' -> 'id255237')"""
        try:
            return 'id' + file_name.split('id')[1].split('_')[0]
        except (IndexError, ValueError):
            return None

    video_map['child_id'] = video_map['file_name'].apply(extract_child_id)
    
    # Merge child IDs into main DataFrame
    df = df.merge(video_map.drop(columns='file_name'), on="video_id", how="left")
    df.dropna(subset=['child_id'], inplace=True)
    
    logging.info(f"Found {df['child_id'].nunique()} unique children in dataset")

    # --- Step 2: Calculate class distribution per child ---
    child_group_counts = df.groupby('child_id')[PersonConfig.TARGET_LABELS].sum()

    # Calculate global target ratio to maintain across splits
    global_counts = child_group_counts[PersonConfig.TARGET_LABELS].sum()
    target_adult_ratio = global_counts[PersonConfig.TARGET_LABELS[1]] / global_counts.sum()

    # --- Step 3: Determine split sizes ---
    sorted_child_ids = child_group_counts.index.tolist()
    n_total = len(sorted_child_ids)

    if n_total < 6:
        logging.error(f"Not enough unique children to create balanced splits. Found {n_total}, need at least 6.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []

    # Calculate number of children per split (minimum 2 per split)
    n_train = max(int(n_total * train_ratio), 2)
    remaining_ids = n_total - n_train
    n_val = max(int(remaining_ids / 2), 2)
    n_test = n_total - n_train - n_val

    # Adjust if test set would be too small
    if n_test < 2:
        n_val -= (2 - n_test)
        if n_val < 2:
            logging.error("Could not meet minimum ID requirements for all splits.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], [], []
        n_test = 2

    logging.info(f"Split sizes: Train={n_train}, Val={n_val}, Test={n_test} children")

    # --- Step 4: Initialize splits ---
    split_info = {
        'train': {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
        'val':   {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
        'test':  {'ids': [], 'current_counts': pd.Series([0, 0], index=PersonConfig.TARGET_LABELS)},
    }

    # Randomize assignment order to reduce bias
    random.shuffle(sorted_child_ids)

    def deviation_from_target(current_counts, child_counts):
        """Calculate how much adding this child would deviate from target adult ratio."""
        new_counts = current_counts + child_counts
        if new_counts.sum() == 0:
            return 0
        new_ratio = new_counts[PersonConfig.TARGET_LABELS[1]] / new_counts.sum()
        return abs(new_ratio - target_adult_ratio)

    # --- Step 5: Assign children to splits using balanced algorithm ---
    for child_id in sorted_child_ids:
        child_counts = child_group_counts.loc[child_id, PersonConfig.TARGET_LABELS]

        # Find splits that still need more children
        eligible_splits = []
        for split_name, target_size in zip(['train', 'val', 'test'], [n_train, n_val, n_test]):
            if len(split_info[split_name]['ids']) < target_size:
                eligible_splits.append(split_name)

        if not eligible_splits:
            logging.warning(f"No split has space left for child {child_id}. Skipping.")
            continue

        # Assign to split that maintains best balance
        best_split = min(
            eligible_splits,
            key=lambda s: deviation_from_target(split_info[s]['current_counts'], child_counts)
        )

        # Update split information
        split_info[best_split]['ids'].append(child_id)
        split_info[best_split]['current_counts'] += child_counts

    # --- Step 6: Create final DataFrames ---
    train_ids = split_info['train']['ids']
    val_ids = split_info['val']['ids']
    test_ids = split_info['test']['ids']

    # Filter data by child assignments
    df_train = df[df["child_id"].isin(train_ids)].drop(columns=["child_id"])
    df_val   = df[df["child_id"].isin(val_ids)].drop(columns=["child_id"])
    df_test  = df[df["child_id"].isin(test_ids)].drop(columns=["child_id"])

    return df_train, df_val, df_test, train_ids, val_ids, test_ids

# ==============================
# Statistics and Logging
# ==============================

def generate_statistics_file(df: pd.DataFrame, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, train_ids: List, val_ids: List, test_ids: List):
    """
    Generate detailed statistics file about the dataset splits.
    
    Creates a comprehensive report showing class distributions, split ratios,
    child ID assignments, and validation checks.
    
    Parameters
    ----------
    df : pd.DataFrame
        Original complete dataset
    df_train, df_val, df_test : pd.DataFrame
        Split datasets
    train_ids, val_ids, test_ids : List
        Child IDs assigned to each split
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = BasePaths.LOGGING_DIR / f"split_distribution_person_cls_{timestamp}.txt"
    
    total_images = len(df)
    
    with open(file_path, "w") as f:
        f.write(f"Person Classification Dataset Split Report - {timestamp}\n\n")
        
        # Overall dataset statistics
        total_child = df['child'].sum()
        total_adult = df['adult'].sum()

        f.write(f"=== OVERALL DATASET STATISTICS ===\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Class 0 ({PersonConfig.TARGET_LABELS[0]}): {total_child} images ({total_child / total_images:.2%})\n")
        f.write(f"Class 1 ({PersonConfig.TARGET_LABELS[1]}): {total_adult} images ({total_adult / total_images:.2%})\n\n")

        f.write("=== SPLIT DISTRIBUTION ===\n")
        f.write("-" * 50 + "\n\n")

        def write_split_info(split_name, split_df):
            """Helper function to write statistics for each split."""
            total_split = len(split_df)
            split_percentage = total_split / total_images if total_images > 0 else 0

            count_child = split_df[PersonConfig.TARGET_LABELS[0]].sum()
            count_adult = split_df[PersonConfig.TARGET_LABELS[1]].sum()
            
            # Within-split ratios
            ratio_child = count_child / total_split if total_split > 0 else 0
            ratio_adult = count_adult / total_split if total_split > 0 else 0
            
            f.write(f"{split_name} Set ({split_percentage:.1%} of total):\n")
            f.write(f"  Total Images: {total_split}\n")
            f.write(f"  {PersonConfig.TARGET_LABELS[0]}: {count_child} ({ratio_child:.2%})\n")
            f.write(f"  {PersonConfig.TARGET_LABELS[1]}: {count_adult} ({ratio_adult:.2%})\n\n")

        # Write statistics for each split
        write_split_info("Training", df_train)
        write_split_info("Validation", df_val)
        write_split_info("Test", df_test)

        # Child ID distribution
        f.write("=== CHILD ID ASSIGNMENTS ===\n")
        f.write(f"Training Children ({len(train_ids)}): {sorted(train_ids)}\n")
        f.write(f"Validation Children ({len(val_ids)}): {sorted(val_ids)}\n")
        f.write(f"Test Children ({len(test_ids)}): {sorted(test_ids)}\n\n")

        # Data leakage check
        f.write("=== DATA LEAKAGE CHECK ===\n")
        train_val_overlap = set(train_ids).intersection(set(val_ids))
        train_test_overlap = set(train_ids).intersection(set(test_ids))
        val_test_overlap = set(val_ids).intersection(set(test_ids))
        
        if train_val_overlap or train_test_overlap or val_test_overlap:
            f.write("❌ OVERLAP DETECTED:\n")
            if train_val_overlap:
                f.write(f"  Train-Val overlap: {train_val_overlap}\n")
            if train_test_overlap:
                f.write(f"  Train-Test overlap: {train_test_overlap}\n")
            if val_test_overlap:
                f.write(f"  Val-Test overlap: {val_test_overlap}\n")
        else:
            f.write("✅ No child ID overlap detected - splits are clean\n")
    
    logging.info(f"Statistics report saved to: {file_path}")

# ==============================
# Main Execution
# ==============================

def main():
    """
    Main execution pipeline for person classification data preparation.
    
    Pipeline:
    1. Fetch person annotations from database
    2. Convert to frame-level binary labels (adult/child presence)
    3. Split data by child IDs to prevent leakage
    4. Save train/val/test CSV files
    5. Generate detailed statistics report
    """    
    # Step 1: Fetch annotations from database
    logging.info("Step 1: Fetching annotations from database...")
    rows = fetch_all_annotations(PersonConfig.DATABASE_CATEGORY_IDS)
    
    if not rows:
        logging.error("No annotations found for the specified categories. Exiting.")
        exit(1)

    # Step 2: Build frame-level labels
    logging.info("Step 2: Building frame-level binary labels...")
    df = build_frame_level_labels(
        rows, 
        PersonConfig.AGE_GROUP_TO_CLASS_ID, 
        PersonConfig.MODEL_CLASS_ID_TO_LABEL
    )

    # Validate expected columns exist
    if not all(col in df.columns for col in PersonConfig.TARGET_LABELS):
        logging.error(f"DataFrame is missing expected columns: {PersonConfig.TARGET_LABELS}. "
                     f"Available columns: {df.columns.tolist()}")
        exit(1)

    # Step 3: Split data by child IDs
    logging.info("Step 3: Splitting data by child IDs...")
    df_train, df_val, df_test, train_ids, val_ids, test_ids = split_by_child_id(df)

    # Validate splits were created successfully
    if df_train.empty or df_val.empty or df_test.empty:
        logging.error("Failed to create valid data splits. Check child ID distribution.")
        exit(1)

    # Step 4: Save CSV files
    logging.info("Step 4: Saving train/val/test CSV files...")
    PersonClassification.INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df_train.to_csv(PersonClassification.TRAIN_CSV_PATH, index=False)
    df_val.to_csv(PersonClassification.VAL_CSV_PATH, index=False)
    df_test.to_csv(PersonClassification.TEST_CSV_PATH, index=False)

    # Step 5: Generate statistics report
    logging.info("Step 5: Generating statistics report...")
    generate_statistics_file(df, df_train, df_val, df_test, train_ids, val_ids, test_ids)

    # Final summary
    logging.info(f"✅ Pipeline completed successfully!")
    logging.info(f"📁 CSV files saved to: {PersonClassification.INPUT_DIR}")
    logging.info(f"📊 Train: {len(df_train)} frames, Val: {len(df_val)} frames, Test: {len(df_test)} frames")
    logging.info(f"👥 Children split - Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    
if __name__ == "__main__":
    main()