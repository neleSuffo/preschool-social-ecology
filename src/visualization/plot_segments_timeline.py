import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from constants import Analysis, Visualization

def time_to_seconds(time_val) -> float:
    """Converts MM:SS, HH:MM:SS strings, or numeric minutes/seconds to total seconds."""
    if pd.isna(time_val):
        return np.nan
    if isinstance(time_val, (int, float)):
        return float(time_val)
    
    time_str = str(time_val).strip()
    parts = time_str.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(time_str)
    except ValueError:
        return np.nan


def load_and_clean_segments(df: pd.DataFrame, video_name: str, mode: str) -> pd.DataFrame:
    """Filters data for the target video and standardizes columns and labels."""
    df_filtered = df[df["video_name"].astype(str).str.contains(video_name, case=False, na=False)].copy()
    if df_filtered.empty:
        return df_filtered

    if "start_time_sec" not in df_filtered.columns and "start_time_min" in df_filtered.columns:
        df_filtered["start_time_sec"] = df_filtered["start_time_min"].apply(time_to_seconds)
        df_filtered["end_time_sec"] = df_filtered["end_time_min"].apply(time_to_seconds)
    else:
        df_filtered["start_time_sec"] = df_filtered["start_time_sec"].apply(time_to_seconds)
        df_filtered["end_time_sec"] = df_filtered["end_time_sec"].apply(time_to_seconds)

    df_filtered["interaction_type"] = df_filtered["interaction_type"].astype(str).str.lower().str.strip()

    if mode.lower() == "binary":
        binary_map = {
            "interacting": "interacting",
            "available": "not interacting",
            "alone": "not interacting",
            "not interacting": "not interacting"
        }
        df_filtered["interaction_type"] = df_filtered["interaction_type"].map(binary_map).fillna("not interacting")

    df_filtered["duration_sec"] = df_filtered["end_time_sec"] - df_filtered["start_time_sec"]
    df_filtered.dropna(subset=["start_time_sec", "end_time_sec"], inplace=True)
    return df_filtered


def plot_segment_timeline(
    video_name: str,
    output_path: Path,
    mode: str = "tertiary",
    pred_path: Path = Analysis.INTERACTION_SEGMENTS_CSV,
    gt_path: Path = Analysis.GROUND_TRUTH_SEGMENTS_CSV,
):
    """Loads prediction and ground-truth CSVs and plots their temporal alignment."""
    # Load predictions
    pred_df_raw = pd.read_csv(pred_path)

    # Load ground truth (handles comma or semicolon separators)
    try:
        gt_df_raw = pd.read_csv(gt_path, delimiter=";")
        if len(gt_df_raw.columns) <= 1:
            gt_df_raw = pd.read_csv(gt_path)
    except Exception:
        gt_df_raw = pd.read_csv(gt_path)

    # Clean and filter
    pred = load_and_clean_segments(pred_df_raw, video_name, mode)
    gt = load_and_clean_segments(gt_df_raw, video_name, mode)

    if pred.empty and gt.empty:
        print(f"Error: No segments found for video '{video_name}' in either file.")
        return

    is_binary = (mode.lower() == "binary")
    if is_binary:
        interaction_colors = {
            "interacting": "#691633",  # Deep Burgundy
            "not interacting": "#69777B",   # Slate Gray
        }
    else:
        interaction_colors = {
        "interacting": "#691633",  # Deep Burgundy
        "available": "#8E7C3B",    # Muted Olive / Gold
        "alone": "#69777B",        # Slate Gray
        }
        
    # Set timeline limits (rounded up to the next minute)
    max_time = max(
        pred["end_time_sec"].max() if not pred.empty else 0,
        gt["end_time_sec"].max() if not gt.empty else 0
    )
    max_time = np.ceil(max_time / 60) * 60

    fig, ax = plt.subplots(figsize=(15, 4))

    # --- Ground Truth Bar (Y = 1.2) ---
    y_pos_gt = 1.2
    for _, row in gt.iterrows():
        color = interaction_colors.get(row["interaction_type"], "#808080")
        ax.barh(
            y=y_pos_gt,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85
        )

    # --- Prediction Bar (Y = 0.8) ---
    y_pos_pred = 0.8
    for _, row in pred.iterrows():
        color = interaction_colors.get(row["interaction_type"], "#808080")
        ax.barh(
            y=y_pos_pred,
            width=row["duration_sec"],
            left=row["start_time_sec"],
            height=0.22,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85
        )

    # Axis configuration
    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(["Prediction", "Ground Truth"], fontsize=20, fontweight="bold")
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel("Time (seconds)", fontsize=20)
    ax.set_xlim(0, max_time)

    legend_patches = [
        mpatches.Patch(color=interaction_colors[cat], alpha=0.85, label=cat.capitalize())
        for cat in sorted(interaction_colors.keys())
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.30),
        ncol=len(legend_patches),
        frameon=False,
        fontsize=20
    )

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save output image
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Ground Truth vs. Prediction segment timelines for a single video.")
    parser.add_argument("--video", type=str, required=True, help="Video name identifier to plot.")
    parser.add_argument("--mode", type=str, choices=["binary", "tertiary"], default="tertiary", help="Plotting mode.")
    args = parser.parse_args()
    
    # Ensure parent output directory exists
    output_dir = Visualization.SEGMENT_TIMELINE_SUBOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file path with extension
    file_path = output_dir / f"{args.video}_segment_timeline.png"
    
    plot_segment_timeline(
        video_name=args.video,
        output_path=file_path,
        mode=args.mode,
        gt_path=Analysis.GROUND_TRUTH_SEGMENTS_CSV,
        pred_path="/home/nele_pauline_suffo/outputs/quantex_analysis/cv_validation_20260917_191421/combo_0002/interaction_segments.csv"
    )