import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import os
from typing import List
from utils import time_to_seconds
from constants import Evaluation, Inference
from config import InferenceConfig


def load_data(path: Path, delimiter: str = ',') -> pd.DataFrame:
    """Loads CSV data, attempting to handle common errors."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    
    # Try reading with different delimiters and inferring columns
    try:
        df = pd.read_csv(path, delimiter=delimiter)
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep='\s\s+', engine='python') # Try space-separated
    
    df.columns = df.columns.str.lower()
    return df.copy()

def prepare_data(df: pd.DataFrame, is_ground_truth: bool) -> pd.DataFrame:
    """Standardizes column names and converts time to seconds if necessary.
    
    Parameters:
    ----------
    df : pd.DataFrame
        The input dataframe containing segment data.
    is_ground_truth : bool
        Flag indicating if the dataframe is ground truth data.
    
    Returns:
    -------
    pd.DataFrame
        The prepared dataframe with standardized columns and time in seconds.
    """
    df.columns = df.columns.str.lower().str.strip()

    # --- Time Conversion Logic (Handles user's GT format: start_time_min) ---
    if is_ground_truth and 'start_time_min' in df.columns:
        # Convert min:sec format to seconds
        df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
        df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)

    # Calculate duration
    df['duration'] = df['end_time_sec'] - df['start_time_sec']

    
    # --- Time Conversion Logic (Handles user's GT format: start_time_min) ---
    if is_ground_truth and 'start_time_min' in df.columns:
        # Convert min:sec format to seconds
        df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
        df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)
    
    # Calculate duration
    df['duration'] = df['end_time_sec'] - df['start_time_sec']
    
    # Clean up and standardize interaction type
    df['interaction_type'] = df['interaction_type'].astype(str).str.lower().str.strip()
    
    # Filter for valid segments
    df = df.dropna(subset=['start_time_sec', 'end_time_sec', 'duration'])
    df = df[df['duration'] > 0]
    
    return df[['video_name', 'interaction_type', 'start_time_sec', 'end_time_sec', 'duration']]


def plot_segments_vs_predictions(video_name: str, time_window: List[float] = None, save_path: Path = None) -> None:
    """
    Plot ground truth segments and model predictions for a given video and time window.
    
    Parameters:
    ----------
    video_name : str
        The name or partial name of the video file to plot.
    time_window : List[float]
        A list containing start and end times (in seconds) for the x-axis.
    save_path : Path
        The path where the plot image will be saved.
    """
    print(f"Loading data for video: {video_name}")
    
    # Load and prepare ground truth data (assuming semicolon delimiter from user example)
    try:
        gt_df_raw = load_data(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')
        gt_df = prepare_data(gt_df_raw, is_ground_truth=True)
    except FileNotFoundError:
        print(f"Ground Truth file not found at {Inference.GROUND_TRUTH_SEGMENTS_CSV}.")
        return

    # Load and prepare prediction data (assuming comma delimiter as common for analysis results)
    try:
        pred_df_raw = load_data(Inference.INTERACTION_SEGMENTS_CSV, delimiter=',')
        pred_df = prepare_data(pred_df_raw, is_ground_truth=False)
    except FileNotFoundError:
        print(f"Prediction file not found at {Inference.INTERACTION_SEGMENTS_CSV}.")
        return

    # Filter data for the specific video
    gt = gt_df[gt_df['video_name'].str.contains(video_name, case=False, na=False)]
    pred = pred_df[pred_df['video_name'].str.contains(video_name, case=False, na=False)]

    if gt.empty and pred.empty:
        print(f"Error: No data found for video '{video_name}' in either Ground Truth or Predictions.")
        return

    # --- Plotting Setup ---
    categories = InferenceConfig.INTERACTION_CLASSES
    num_categories = len(categories)

    fig_height_per_category = 0.8
    min_fig_height = 3.0
    figure_height = max(min_fig_height, num_categories * fig_height_per_category)
    
    fig, ax = plt.subplots(figsize=(12, figure_height))
    colors = {'GT': '#1f77b4', 'Pred': '#ff7f0e'} # Blue for GT, Orange for Pred

    # Plot geometry constants
    bar_height = 0.4
    row_step = 1.25
    gt_y_offset_in_category = 0.25 # GT bar centered slightly above the category line
    pred_y_offset_in_category = -0.25 # Pred bar centered slightly below the category line

    yticks_positions = []
    yticklabels_text = []

    # Iterate through categories in reverse for plotting order (top to bottom)
    for i, category in enumerate(categories[::-1]):
        category_center_y = i * row_step
        
        # 1. Plot Ground Truth Segments
        current_gt_bars = gt[gt['interaction_type'] == category]
        for _, row in current_gt_bars.iterrows():
            ax.barh(
                y=category_center_y + gt_y_offset_in_category,
                width=row['duration'],
                left=row['start_time_sec'],
                height=bar_height,
                color=colors['GT'],
                edgecolor='black',
                alpha=0.7,
                label='_nolegend_'
            )
        
        # 2. Plot Prediction Segments
        current_pred_bars = pred[pred['interaction_type'] == category]
        for _, row in current_pred_bars.iterrows():
            ax.barh(
                y=category_center_y + pred_y_offset_in_category,
                width=row['duration'],
                left=row['start_time_sec'],
                height=bar_height,
                color=colors['Pred'],
                edgecolor='black',
                alpha=0.7,
                label='_nolegend_'
            )
        
        # Record tick positions and labels
        yticks_positions.append(category_center_y)
        yticklabels_text.append(category.capitalize())

    # --- Final Plot Configuration ---

    ax.set_yticks(yticks_positions)
    ax.set_yticklabels(yticklabels_text)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Interaction Category", fontsize=12)
    ax.set_title(f"Segment GT vs. Prediction for Video: {video_name}", fontsize=14)

    if num_categories > 0:
        min_bar_edge = (0 * row_step) + pred_y_offset_in_category - bar_height / 2
        max_bar_edge = ((num_categories - 1) * row_step) + gt_y_offset_in_category + bar_height / 2
        padding = 0.1 * row_step
        ax.set_ylim(min_bar_edge - padding, max_bar_edge + padding)
    else:
        ax.set_ylim(-0.5, 0.5)

    # If a valid time_window is provided (start,end), use it. Otherwise plot the full available span.
    if time_window and len(time_window) == 2 and time_window[0] < time_window[1]:
        ax.set_xlim(time_window[0], time_window[1])
    else:
        # Compute data-driven x limits from available ground truth and prediction data
        max_end = 0.0
        if not gt.empty:
            try:
                max_end = max(max_end, float(gt['end_time_sec'].max()))
            except Exception:
                pass
        if not pred.empty:
            try:
                max_end = max(max_end, float(pred['end_time_sec'].max()))
            except Exception:
                pass

        if max_end <= 0:
            # Fallback when no segments exist
            ax.set_xlim(0, 1)
            print("Warning: No segments found; setting X limits to 0-1.")
        else:
            ax.set_xlim(0, max_end + 5)

        if time_window:
            print("Warning: Invalid time window, setting X limits automatically.")
        else:
            print("Info: No time_window provided; plotting full video span based on available data.")

    # Create a simplified legend manually
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=colors['GT'], alpha=0.7, label='Ground Truth (GT)'),
        mpatches.Patch(color=colors['Pred'], alpha=0.7, label='Prediction (PRED)')
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fontsize=10
    )

    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save the plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"\n✅ Plot successfully saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Ground Truth vs. Predicted interaction segments for a specific video.")
    parser.add_argument("--video_name", type=str, required=True, help="The name or partial name of the video file (e.g., 'quantex_at_home_id254922_2022_04_12_01').")
    parser.add_argument("--time_window", type=str, required=False, default=None, help="Optional: Time window for the plot as start,end (e.g. '0,180' for 0 to 180 seconds). If omitted the full video span is used.")
    
    args = parser.parse_args()

    # Parse time window (optional)
    time_window = None
    if args.time_window:
        try:
            time_window = [float(x) for x in args.time_window.split(",")]
            if len(time_window) != 2:
                raise ValueError("Time window must contain exactly two values (start,end).")
        except ValueError as e:
            print(f"Error parsing time window: {e}")
            return

    # Define output path
    output_plot_path = Evaluation.BASE_OUTPUT_DIR / f"{args.video_name}_segment_comparison_{int(time_window[0])}s-{int(time_window[1])}s.png"

    try:
        plot_segments_vs_predictions(args.video_name, time_window, output_plot_path)
    except FileNotFoundError as e:
        print(f"Fatal Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")


if __name__ == "__main__":
    main()