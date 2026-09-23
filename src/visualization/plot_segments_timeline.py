import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from constants import Analysis, Visualization

def load_and_clean_segments(df: pd.DataFrame, video_name: str, mode: str) -> pd.DataFrame:
    """Filters data for the target video and standardizes interaction labels."""
    df_filtered = df[df["video_name"].astype(str).str.contains(video_name, case=False, na=False)].copy()
    if df_filtered.empty:
        return df_filtered

    df_filtered["start_time_sec"] = pd.to_numeric(df_filtered["start_time_sec"], errors="coerce")
    df_filtered["end_time_sec"] = pd.to_numeric(df_filtered["end_time_sec"], errors="coerce")
    df_filtered["duration_sec"] = pd.to_numeric(df_filtered["duration_sec"], errors="coerce")

    df_filtered["interaction_type"] = df_filtered["interaction_type"].astype(str).str.lower().str.strip()

    if mode.lower() == "binary":
        binary_map = {
            "interacting": "interacting",
            "available": "not interacting",
            "alone": "not interacting",
            "not interacting": "not interacting"
        }
        df_filtered["interaction_type"] = df_filtered["interaction_type"].map(binary_map).fillna("not interacting")

    df_filtered.dropna(subset=["start_time_sec", "end_time_sec", "duration_sec"], inplace=True)
    return df_filtered


def plot_segment_timeline(
    video_name: str,
    output_path: Path,
    pred_df_raw: pd.DataFrame,
    gt_df_raw: pd.DataFrame,
    mode: str = "tertiary",
    show_legend: bool = True,
):
    """Plots temporal alignment between prediction and GT for a specific video."""
    pred = load_and_clean_segments(pred_df_raw, video_name, mode)
    gt = load_and_clean_segments(gt_df_raw, video_name, mode)

    if pred.empty and gt.empty:
        print(f"Warning: No segments found for video '{video_name}' in either file. Skipping.")
        return

    is_binary = (mode.lower() == "binary")
    if is_binary:
        interaction_colors = {
            "interacting": "#691633",
            "not interacting": "#69777B",
        }
    else:
        interaction_colors = {
            "interacting": "#691633",
            "available": "#8E7C3B",
            "alone": "#69777B",
        }

    # Set timeline limits (rounded up to the next minute)
    max_time = max(
        pred["end_time_sec"].max() if not pred.empty else 0,
        gt["end_time_sec"].max() if not gt.empty else 0
    )
    max_time = np.ceil(max_time / 60) * 60

    fig_height = 4 if show_legend else 3.2
    fig, ax = plt.subplots(figsize=(16, fig_height))

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
            alpha=0.85,
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
            alpha=0.85,
        )

    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(["Prediction", "Ground Truth"], fontsize=20, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=20, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlim(0, max_time)

    if show_legend:
        ax.set_ylim(0.5, 1.5)
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
            fontsize=20,
        )
    else:
        ax.set_ylim(0.5, 1.45)

    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Ground Truth vs. Prediction segment timelines for a single video.")
    parser.add_argument("--video", type=str, default="all", help="Video identifier to plot, or 'all' to plot all videos in the ground truth dataset.")
    parser.add_argument("--mode", type=str, choices=["binary", "tertiary"], default="tertiary", help="Plotting mode.")
    parser.add_argument("--no_legend", action="store_true", help="Disable the top legend in the output plot.")
    args = parser.parse_args()
    
    # Ensure parent output directory exists
    output_dir = Visualization.SEGMENT_TIMELINE_SUBOLDER
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data once
    gt_df_raw = pd.read_csv(Analysis.GROUND_TRUTH_SEGMENTS_CSV)
    pred_df_raw = pd.read_csv(Analysis.INTERACTION_SEGMENTS_CSV)

    # Determine video list
    if args.video.strip().lower() == "all":
        video_list = sorted(gt_df_raw["video_name"].dropna().unique().tolist())
        print(f"Found {len(video_list)} unique videos in ground truth. Generating plots...")
    else:
        video_list = [args.video.strip()]

    # Iterate and render
    for vid in video_list:
        clean_name = Path(vid).stem
        out_file = output_dir / f"{clean_name}_segment_timeline.png"
        plot_segment_timeline(
            video_name=vid,
            output_path=out_file,
            pred_df_raw=pred_df_raw,
            gt_df_raw=gt_df_raw,
            mode=args.mode,
            show_legend=not args.no_legend,
        )