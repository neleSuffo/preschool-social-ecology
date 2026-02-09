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

def plot_kappa_comparison(df1: pd.DataFrame, df2: pd.DataFrame, video_name: str, common_duration: int, output_path: Path):
    """
    Plots the segment timelines for two annotators for a specific video 
    up to the common duration.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    
    # Define colors based on your typical categories + a color for unannotated/gaps
    INTERACTION_COLORS = {
        'interacting': '#d62728',  # Red
        'available': '#ff7f0e',    # Orange
        'alone': '#1f77b4',        # Blue
        'unannotated': '#cccccc'   # Grey for gaps/disagreed end
    }
    
    # Ensure both DFs have the 'duration_sec' column for plotting
    for i, df in enumerate([df1, df2]):
        if 'duration_sec' not in df.columns:
            # Recalculate duration if needed (based on your cleaning steps)
            df['duration_sec'] = df['end_time_sec'] - df['start_time_sec']

    fig, ax = plt.subplots(figsize=(15, 3))

    # --- Plot Annotator 1 (GT1) on Y=1.2 ---
    y_pos_1 = 1.2
    for _, row in df1.iterrows():
        # Only plot segments that START before the common duration boundary
        if row['start_time_sec'] < common_duration:
            start = row['start_time_sec']
            end = min(row['end_time_sec'], common_duration)
            duration = end - start
            
            color = INTERACTION_COLORS.get(row['interaction_type'], '#808080')
            ax.barh(y=y_pos_1, 
                    width=duration, 
                    left=start,
                    height=0.2,
                    color=color,
                    edgecolor='black',
                    alpha=0.7)

    # --- Plot Annotator 2 (GT2) on Y=0.8 ---
    y_pos_2 = 0.8
    for _, row in df2.iterrows():
        # Only plot segments that START before the common duration boundary
        if row['start_time_sec'] < common_duration:
            start = row['start_time_sec']
            end = min(row['end_time_sec'], common_duration)
            duration = end - start
            
            color = INTERACTION_COLORS.get(row['interaction_type'], '#808080')
            ax.barh(y=y_pos_2, 
                    width=duration, 
                    left=start,
                    height=0.2,
                    color=color,
                    edgecolor='black',
                    alpha=0.7)
    
    # Y-axis labels
    ax.set_yticks([y_pos_2, y_pos_1])
    ax.set_yticklabels(['Annotator 2', 'Annotator 1'], fontsize=10)
    ax.set_ylim(0.5, 1.5)
    
    # X-axis (Time) configuration
    max_time = np.ceil(common_duration / 60) * 60 # Round up to nearest minute boundary
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xlim(0, max_time if max_time > 0 else 60)
    ax.set_xticks(np.arange(0, max_time + 1, 50))
    # Title
    ax.set_title(f"Inter-Rater Comparison: {video_name}", fontsize=12)

    # Custom Legend
    legend_patches = [
        mpatches.Patch(color=INTERACTION_COLORS.get(cat, '#808080'), alpha=0.7, label=cat.capitalize())
        for cat in sorted(INTERACTION_COLORS.keys()) if cat != 'unannotated'
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.35), ncol=3, frameon=False, fontsize=9)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"🖼️ Plot saved for {video_name} to {output_path}")
    
def compute_second_wise_kappa(annotator1_file: Path, annotator2_file: Path, video_plot_list: list = None, output_folder: Path = None) -> float:
    """
    Computes second-wise Cohen's Kappa between two annotator files and saves the second-wise aligned annotations.
    
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

    common_videos = set(annots_data1.keys()) & set(annots_data2.keys())
    
    if not common_videos:
        print("❌ Error: No common video names found between the two files. Cannot compute reliability.")
        return 0.0

    print(f"Found {len(common_videos)} videos in common for reliability check: {', '.join(sorted(common_videos))}")

    df_annotator1_data = []
    df_annotator2_data = []
    
    ratings_a_aggr = []
    ratings_b_aggr = []

    for video_name in common_videos:
        df1 = annots_data1[video_name]
        df2 = annots_data2[video_name]
        
        # 1. Calculate common maximum duration
        max_end_a = df1['end_time_sec'].max() if not df1.empty else 0
        max_end_b = df2['end_time_sec'].max() if not df2.empty else 0
        common_max_duration = min(max_end_a, max_end_b)
        video_duration_seconds = int(common_max_duration)
        
        if video_duration_seconds <= 0:
            print(f"⚠️ Warning: Video {video_name} has zero or negative common duration, skipping.")
            continue
        
        # Create second-level arrays (constrained by common_max_duration)
        labels_a_raw = create_second_level_labels(df1, video_duration_seconds)
        labels_b_raw = create_second_level_labels(df2, video_duration_seconds)
        
        min_len = len(labels_a_raw) 
        
        ratings_a_video = []
        ratings_b_video = []

        # Filter for mutual Ground Truth (Non-None) AND collect all data for saving
        for sec in range(min_len): 
            label_a_raw = labels_a_raw[sec]
            label_b_raw = labels_b_raw[sec]
            
            # Use UNANNOTATED_LABEL for saving purposes to prevent NaN/None strings in output file
            label_a_save = str(label_a_raw).lower() if label_a_raw is not None else UNANNOTATED_LABEL
            label_b_save = str(label_b_raw).lower() if label_b_raw is not None else UNANNOTATED_LABEL

            # --- NEW: Collect data for output files (all seconds in common duration) ---
            df_annotator1_data.append({
                'video_name': video_name,
                'second': sec,
                'interaction_type': label_a_save
            })
            df_annotator2_data.append({
                'video_name': video_name,
                'second': sec,
                'interaction_type': label_b_save
            })
            
            # Kappa calculation logic (only mutual GT, no UNANNOTATED_LABEL)
            if label_a_raw is not None and label_b_raw is not None:
                ratings_a_video.append(str(label_a_raw).lower())
                ratings_b_video.append(str(label_b_raw).lower())

        ratings_a_aggr.extend(ratings_a_video)
        ratings_b_aggr.extend(ratings_b_video)

        # --- PLOTTING LOGIC ---
        if video_plot_list is not None and output_folder and (video_name in video_plot_list or 'all' in video_plot_list):
            plot_path = output_folder / f"{video_name}_irr.png"
            plot_kappa_comparison(df1, df2, video_name, video_duration_seconds, plot_path)
            
    if len(ratings_a_aggr) == 0:
        print("⚠️ Warning: Zero seconds found with mutual ground truth. Kappa is 0.0.")
        return 0.0

    # Save Second-Wise DataFiles ---
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)
        
        df_gt1 = pd.DataFrame(df_annotator1_data)
        df_gt2 = pd.DataFrame(df_annotator2_data)

        # Save files
        df_gt1.to_csv(Evaluation.GT_1_SECONDWISE_FILE_PATH, index=False)
        df_gt2.to_csv(Evaluation.GT_2_SECONDWISE_FILE_PATH, index=False)
        print(f"✅ Second-wise data saved to {output_folder.name} folder.")
        
    print(f"Total seconds evaluated (mutual agreement only): {len(ratings_a_aggr):,}")
    
    kappa = cohen_kappa_score(ratings_a_aggr, ratings_b_aggr)
    
    return kappa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute second-wise Inter-Rater Reliability (Cohen's Kappa).")
    parser.add_argument('--plot', nargs='*', default=[], help='List of video names to plot (e.g., video1 video2) or use "all" to plot all videos.')
    
    args = parser.parse_args()

# Define your actual paths (replace with your setup)
    ANNOTATOR_1_FILE = Evaluation.GT_1_FILE_PATH
    ANNOTATOR_2_FILE = Evaluation.GT_2_FILE_PATH
    OUTPUT_FOLDER = ANNOTATOR_1_FILE.parent

    if not ANNOTATOR_1_FILE.exists() or not ANNOTATOR_2_FILE.exists():
        print("❌ Error: One or both annotation files not found.")
        sys.exit(1)

    kappa_score = compute_second_wise_kappa(
        ANNOTATOR_1_FILE, 
        ANNOTATOR_2_FILE, 
        video_plot_list=args.plot, 
        output_folder=OUTPUT_FOLDER
    )
    
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