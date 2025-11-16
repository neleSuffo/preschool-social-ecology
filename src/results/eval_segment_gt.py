import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from results.utils import time_to_seconds, create_second_level_labels
from constants import Evaluation

# Define a placeholder for unannotated time, which is treated as a category
UNANNOTATED_LABEL = 'unannotated'

def process_annotation_file(file_path: Path) -> dict:
    """
    Reads an annotation CSV, converts times to seconds, and groups data by video_name.
    
    The CSV format is expected to be: video_name;interaction_type;start_time_min;end_time_min;...
    
    Parameters
    ----------
    file_path : Path
        Path to the annotation CSV file.
        
    Returns
    -------
    dict
        A dictionary mapping video names to their corresponding DataFrames of segments.
    """
    # 1. Read the CSV using the existing header row (header=0)
    try:
        # Read all columns first, then clean up the column names
        df = pd.read_csv(
            file_path, 
            sep=';', 
            header=0, 
            encoding='utf-8',
            engine='python' 
        )
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return {}

    # 2. Clean and select required columns
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Identify the required columns (must be present after stripping headers)
    required_cols = ['video_name', 'interaction_type', 'start_time_min', 'end_time_min']
    
    # Filter the DataFrame to only keep the required columns and drop rows where all are NaN
    df = df[required_cols].dropna(how='all') 

    # 3. Data Cleaning and Conversion
    
    # Strip whitespace from video_name ---
    df['video_name'] = df['video_name'].astype(str).str.strip()

    # Convert time strings to total seconds using the imported utility    
    df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
    df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)

    # Clean up and lower-case the interaction type
    df['interaction_type'] = df['interaction_type'].astype(str).str.lower().str.strip()
    
    # Filter out invalid segments (missing data or zero/negative duration)
    df = df.dropna(subset=['video_name', 'interaction_type', 'start_time_sec', 'end_time_sec'])

    # 4. Group and Log
    video_data = {name: group.copy() for name, group in df.groupby('video_name')}
    
    # --- LOGGING ADDED ---
    video_names = list(video_data.keys())
    
    return video_data

def compute_second_wise_kappa(annotator1_file: Path, annotator2_file: Path) -> float:
    """
    Computes second-wise Cohen's Kappa between two annotator files.
    
    Parameters
    ----------
    annotator1_file : Path
        Path to the first annotator's CSV file.
    annotator2_file : Path
        Path to the second annotator's CSV file.
        
    Returns
    -------
    float
        The computed Cohen's Kappa score.
    """
    print(f"Loading annotations from: {annotator1_file.name} and {annotator2_file.name}")
    annots_data1 = process_annotation_file(annotator1_file)
    annots_data2 = process_annotation_file(annotator2_file)

    # Find common videos
    common_videos = set(annots_data1.keys()) & set(annots_data2.keys())
    
    if not common_videos:
        print("❌ Error: No common video names found between the two files. Cannot compute reliability.")
        return 0.0

    print(f"Found {len(common_videos)} videos in common for reliability check.")

    ratings_a = []
    ratings_b = []

    for video_name in common_videos:
        df1 = annots_data1[video_name]
        df2 = annots_data2[video_name]
        
        # Determine the maximum duration across both annotators for alignment
        max_duration = max(df1['end_time_sec'].max(), df2['end_time_sec'].max())
        video_duration_seconds = int(max_duration) + 1

        # Use the imported utility to create second-level arrays
        labels_a_raw = create_second_level_labels(df1, video_duration_seconds)
        labels_b_raw = create_second_level_labels(df2, video_duration_seconds)

        # Convert the NumPy arrays (which contain None for unannotated) to lists 
        # using the UNANNOTATED_LABEL placeholder for Kappa computation.
        labels_a = [str(x) if x is not None else UNANNOTATED_LABEL for x in labels_a_raw]
        labels_b = [str(x) if x is not None else UNANNOTATED_LABEL for x in labels_b_raw]
        
        # We only consider seconds where at least one annotator has a label, or 
        # where the max duration dictates the end of the video. 
        # In this second-wise alignment, we take ALL seconds up to max_duration.
        ratings_a.extend(labels_a)
        ratings_b.extend(labels_b)

    # Ensure arrays are the same length before calculating Kappa
    if len(ratings_a) != len(ratings_b):
        print("⚠️ Warning: Rating arrays are misaligned. Skipping calculation.")
        return 0.0

    print(f"Total seconds evaluated: {len(ratings_a):,}")
    
    # Compute Cohen's Kappa on the aggregated second-wise ratings
    kappa = cohen_kappa_score(ratings_a, ratings_b)
    
    return kappa

if __name__ == "__main__":
    kappa_score = compute_second_wise_kappa(Evaluation.GT_1_FILE_PATH, Evaluation.GT_2_FILE_PATH)

    print("\n--- Inter-Rater Reliability Result ---")
    print(f"Cohen's Kappa Score: **{kappa_score:.4f}**")
    
    if 0.61 <= kappa_score <= 0.80:
        print("Interpretation: Substantial Agreement")
    elif kappa_score > 0.80:
        print("Interpretation: Almost Perfect Agreement")
    elif 0.41 <= kappa_score <= 0.60:
        print("Interpretation: Moderate Agreement")
    else:
        print("Interpretation: Fair to Poor Agreement")