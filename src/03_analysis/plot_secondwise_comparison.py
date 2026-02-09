import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import List, Optional
# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Evaluation

# Placeholder label used in your second-wise saving scripts
UNCLASSIFIED_LABEL = 'unclassified'
UNANNOTATED_LABEL = 'unannotated'

# --- 1. Plotting Function for 3 Timelines ---
def plot_three_timelines(df_gt1: pd.DataFrame, df_gt2: pd.DataFrame, df_pred: pd.DataFrame, 
                         video_name: str, save_path: Path, mode_suffix: str = ""):
    """
    Plots the second-wise label comparison for GT Annotator 1, GT Annotator 2, and Prediction,
    clipping the plot duration to the shortest common maximum end time.
    """
    
    # 1. Filter data for the specific video
    gt1 = df_gt1[df_gt1['video_name'] == video_name].copy()
    gt2 = df_gt2[df_gt2['video_name'] == video_name].copy()
    pred = df_pred[df_pred['video_name'] == video_name].copy()

    if gt1.empty or gt2.empty or pred.empty:
        print(f"Error: Missing data for video '{video_name}' in one or more second-wise files. Skipping plot.")
        return

    # 2. Define Plotting Constants (Remains unchanged)
    INTERACTION_COLORS = {
        'interacting': '#d62728',       # Red
        'available': '#ff7f0e',         # Orange
        'alone': '#1f77b4',             # Blue
        'not interacting': '#33a02c',   # Green (for binary mode)
        UNCLASSIFIED_LABEL: '#cccccc', # Grey
        UNANNOTATED_LABEL: '#cccccc'   # Grey
    }
    
    # Identify all unique categories for the legend
    all_categories = set(gt1['interaction_type'].unique()) | set(gt2['interaction_type'].unique()) | set(pred['interaction_type'].unique())
    plot_categories = [cat for cat in sorted(list(all_categories)) if cat not in [UNCLASSIFIED_LABEL, UNANNOTATED_LABEL]]

    # 3. Determine Duration and Setup Plot
    
    # --- 🛠️ ADJUSTMENT: Use minimum max_time for common plotting range ---
    max_time_gt1 = gt1['second'].max() if not gt1.empty else 0
    max_time_gt2 = gt2['second'].max() if not gt2.empty else 0
    max_time_pred = pred['second'].max() if not pred.empty else 0

    # The plot duration is the shortest end time index (plus 1 second)
    common_max_time_index = min(max_time_gt1, max_time_gt2, max_time_pred)
    common_duration = common_max_time_index + 1
    
    if common_duration <= 1:
        print(f"⚠️ Warning: Video {video_name} has zero or one common second, skipping plot.")
        return
        
    max_time_rounded = np.ceil(common_duration / 60) * 60 # Round up to nearest minute boundary

    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Define Y-positions
    y_pos_gt1 = 1.4
    y_pos_gt2 = 1.0
    y_pos_pred = 0.6
    
    # 4. Plotting (One bar per second, using the second as the left position and width=1)
    
    # Helper to plot a single time series, clipped to common_duration
    def plot_time_series(df, y_pos, label):
        # Clip the DataFrame to include only seconds up to common_max_time_index
        df_clipped = df[df['second'] <= common_max_time_index]
        
        for _, row in df_clipped.iterrows():
            interaction = row['interaction_type']
            color = INTERACTION_COLORS.get(interaction, '#808080')
            
            # The 'second' column represents the start time of the 1-second interval
            ax.barh(y=y_pos, 
                    width=1, 
                    left=row['second'],
                    height=0.3,
                    color=color,
                    edgecolor='none',
                    alpha=0.8)
        
        # Add Y-label
        ax.text(-0.5, y_pos, label, ha='right', va='center', fontsize=10)


    # Plot data
    plot_time_series(gt1, y_pos_gt1, "GT Annotator 1")
    plot_time_series(gt2, y_pos_gt2, "GT Annotator 2")
    plot_time_series(pred, y_pos_pred, "Model Prediction")

    # 5. Final Plot Styling
    
    # Y-axis config (No ticks needed, labels done via ax.text)
    ax.set_yticks([])
    ax.set_ylim(0.4, 1.6)
    
    # X-axis (Time) configuration
    ax.set_xlabel("Time (seconds)", fontsize=12)
    # Set X-limit exactly to the rounded common duration
    ax.set_xlim(0, max_time_rounded) 
    
    # Title
    ax.set_title(f"Second-Wise Comparison: {video_name}{mode_suffix} (Common Duration: {common_duration}s)", fontsize=14)

    # Custom Legend
    legend_patches = [
        mpatches.Patch(color=INTERACTION_COLORS.get(cat, '#808080'), alpha=0.8, label=cat.capitalize())
        for cat in plot_categories
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.45), ncol=len(plot_categories), frameon=False, fontsize=10)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"✅ Plot saved to: {save_path}")

# --- 2. Main Execution Logic ---
def run_plotting(plot_list: List[str]):
    """Loads second-wise data and generates plots."""
    
    # Determine the mode (binary vs tertiary) from the file names
    binary_suffix = "_binary" if Evaluation.GT_1_SECONDWISE_FILE_PATH.with_name("gt_ann1_secondwise_binary.csv").exists() else ""
    mode_suffix = " (Binary Mode)" if binary_suffix else " (Tertiary Mode)"

    # Define file paths based on naming convention
    gt1_file = Evaluation.GT_1_SECONDWISE_FILE_PATH.with_name(f"gt_ann1_secondwise{binary_suffix}.csv")
    gt2_file = Evaluation.GT_2_SECONDWISE_FILE_PATH.with_name(f"gt_ann2_secondwise{binary_suffix}.csv")
    pred_file = Evaluation.PRED_SECONDWISE_FILE_PATH.with_name(f"pred_secondwise{binary_suffix}.csv")
    
    if not gt1_file.exists() or not gt2_file.exists() or not pred_file.exists():
        print(f"❌ Error: Required second-wise files not found in {Evaluation.BASE_OUTPUT_DIR}.")
        print(f"Ensure files like 'gt_ann1_secondwise{binary_suffix}.csv' exist.")
        return

    # Load data
    try:
        df_gt1 = pd.read_csv(gt1_file)
        df_gt2 = pd.read_csv(gt2_file)
        df_pred = pd.read_csv(pred_file)
    except Exception as e:
        print(f"❌ Error reading CSV files: {e}")
        return

    # Identify common videos
    videos_gt1 = set(df_gt1['video_name'].unique())
    videos_gt2 = set(df_gt2['video_name'].unique())
    videos_pred = set(df_pred['video_name'].unique())
    
    all_common_videos = videos_gt1.intersection(videos_gt2, videos_pred)
    
    if not all_common_videos:
        print("❌ Error: No common video names found across all three second-wise files.")
        return

    # Determine which videos to plot
    if 'all' in [p.lower() for p in plot_list]:
        videos_to_plot = sorted(list(all_common_videos))
    else:
        videos_to_plot = sorted(list(set(plot_list) & all_common_videos))

    if not videos_to_plot:
        print("⚠️ Warning: No videos selected or found to plot.")
        return

    print(f"Generating plots for {len(videos_to_plot)} videos{mode_suffix}...")

    # Generate plots
    for video_name in videos_to_plot:
        plot_path = Evaluation.BASE_OUTPUT_DIR / f"{video_name}_3way_comparison{binary_suffix}.png"
        plot_three_timelines(df_gt1, df_gt2, df_pred, video_name, plot_path, mode_suffix)
        
    print("✅ All plots generated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot second-wise comparison of two Ground Truth annotators and Model Prediction.")
    parser.add_argument('--plot', nargs='*', default=['all'], help='List of video names to plot (e.g., video1 video2) or use "all" (default) to plot all common videos.')

    args = parser.parse_args()
    
    run_plotting(args.plot)