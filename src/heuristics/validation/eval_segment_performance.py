import argparse
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Analysis, Inference
from config import AnalysisConfig
from heuristics.utils import time_to_seconds, create_second_level_labels

# Define a label for unclassified/unannotated time for saving purposes
UNCLASSIFIED_LABEL = 'unclassified'

def save_second_wise_labels(df: pd.DataFrame, output_folder: Path, filename: str):
    """
    Generates second-wise labels for all videos in the DataFrame and saves them to a CSV file.
    """
    all_second_wise_data = []

    # Ensure time columns are converted to seconds if not already
    if 'start_time_sec' not in df.columns:
        if 'start_time_min' in df.columns:
            df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
            df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)
        else:
            print(f"⚠️ Warning: Cannot generate second-wise labels for {filename}, time columns missing.")
            return

    for video_name in df['video_name'].unique():
        video_df = df[df['video_name'] == video_name].copy()
        
        # Determine duration based on max end time
        max_end = video_df['end_time_sec'].max() if not video_df.empty else 0
        video_duration_seconds = int(max_end) + 1
        
        if video_duration_seconds <= 1:
            continue
        
        # Create second-level array using the imported utility
        labels_raw = create_second_level_labels(video_df, video_duration_seconds)
        
        # Convert NumPy array to list of dictionaries for the final DataFrame
        for sec in range(video_duration_seconds):
            label = labels_raw[sec]
            
            all_second_wise_data.append({
                'video_name': video_name,
                'second': sec,
                # Use UNCLASSIFIED_LABEL for None/missing time points
                'interaction_type': str(label).lower() if label is not None else UNCLASSIFIED_LABEL
            })

    if all_second_wise_data:
        df_out = pd.DataFrame(all_second_wise_data)
        output_path = output_folder / filename
        df_out.to_csv(output_path, index=False)
        print(f"✅ Second-wise data saved to: {output_path}")
        
def reclassify_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassifies three-class labels to binary 'interacting' vs 'not interacting'."""
    df_copy = df.copy()
    # Ensure labels are lowercase and stripped for robust matching
    df_copy['interaction_type'] = df_copy['interaction_type'].astype(str).str.lower().str.strip()
    
    mapping = {
        'interacting': 'interacting',
        'available': 'not interacting',
        'alone': 'not interacting',
        'not interacting': 'not interacting'
    }
    
    df_copy['interaction_type'] = df_copy['interaction_type'].map(mapping).fillna('not interacting')
    return df_copy

def plot_segment_timeline(predictions_df, ground_truth_df, video_name, save_path, mode='tertiary'):
    """
    Plots the segment timelines (GT vs Prediction) for a specific video.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predicted interaction segments.
    ground_truth_df : pd.DataFrame
        DataFrame containing ground truth interaction segments.
    video_name : str
        The specific video to plot.
    save_path : Path
        Path to save the generated plot.
    binary_mode : bool
        If True, use binary classification color mapping.
    """
    is_binary = (mode == 'binary')

    # Define colors based on mode
    if is_binary:
        INTERACTION_COLORS = {
            'interacting': '#d62728',       # Red
            'not interacting': '#1f77b4',   # Blue
        }
    else:
        INTERACTION_COLORS = {
            'interacting': '#d62728',       # Red
            'available': '#ff7f0e',         # Orange
            'alone': '#1f77b4',             # Blue
        }
    
    # Filter data for the specific video
    pred = predictions_df[predictions_df['video_name'].str.contains(video_name, case=False, na=False)].copy()
    gt = ground_truth_df[ground_truth_df['video_name'].str.contains(video_name, case=False, na=False)].copy()

    if gt.empty and pred.empty:
        print(f"Error: No data found for video '{video_name}' in either Ground Truth or Predictions.")
        return

    # Standardize and clean time columns (assuming time conversion already ran in main/caller)
    for df in [pred, gt]:
        if 'start_time_min' in df.columns:
            df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
            df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)
        df['interaction_type'] = df['interaction_type'].astype(str).str.lower().str.strip()
        df['duration_sec'] = df['end_time_sec'] - df['start_time_sec']
        df.dropna(subset=['start_time_sec', 'end_time_sec'], inplace=True)
        
    if gt.empty and pred.empty:
        print(f"Error: No valid segments found for video '{video_name}' after processing.")
        return

    # Determine plot dimensions
    max_time = max(pred['end_time_sec'].max() if not pred.empty else 0, 
                   gt['end_time_sec'].max() if not gt.empty else 0)
    max_time = np.ceil(max_time / 60) * 60 # Round up to nearest minute
    
    fig, ax = plt.subplots(figsize=(15, 4))

    # --- Plot Ground Truth (GT) on Y=1 ---
    y_pos_gt = 1.2
    for _, row in gt.iterrows():
        color = INTERACTION_COLORS.get(row['interaction_type'], '#808080')
        ax.barh(y=y_pos_gt, 
                width=row['duration_sec'], 
                left=row['start_time_sec'],
                height=0.2,
                color=color,
                edgecolor='black',
                alpha=0.7)

    # --- Plot Predictions (PRED) on Y=0.8 ---
    y_pos_pred = 0.8
    for _, row in pred.iterrows():
        color = INTERACTION_COLORS.get(row['interaction_type'], '#808080')
        ax.barh(y=y_pos_pred, 
                width=row['duration_sec'], 
                left=row['start_time_sec'],
                height=0.2,
                color=color,
                edgecolor='black',
                alpha=0.7)
    
    # Y-axis labels
    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(['Prediction', 'Ground Truth'], fontsize=12)
    ax.set_ylim(0.5, 1.5)
    
    # X-axis (Time) configuration
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xlim(0, max_time)
    
    # Title
    mode_suffix = " (Binary)" if binary_mode else ""
    ax.set_title(f"Segment Timeline Comparison for Video: {video_name}{mode_suffix}", fontsize=14)

    # Custom Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=INTERACTION_COLORS.get(cat, '#808080'), alpha=0.7, label=cat.capitalize())
        for cat in sorted(INTERACTION_COLORS.keys())
    ]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.0, 1.35), ncol=3, frameon=False)
    
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)
    
def evaluate_performance_by_seconds(predictions_df, ground_truth_df, video_subset=None):
    """
    Evaluates model performance by comparing second-by-second classifications,
    excluding the first and last x seconds, as defined in AnalysisConfig.EXCLUSION_SECONDS.
    ...
    """    
    # Identify videos present in both predictions and ground truth
    
    # Identify videos present in both
    videos_with_gt = set(ground_truth_df['video_name'].unique())
    videos_with_pred = set(predictions_df['video_name'].unique())
    videos_to_evaluate = videos_with_gt.intersection(videos_with_pred)
    
    if video_subset:
        # Only evaluate videos that are in our 10% (or 90%) list AND have data
        videos_to_evaluate = set(video_subset).intersection(videos_with_gt).intersection(videos_with_pred)
        print(f"📊 FOLD MODE: Evaluating {len(videos_to_evaluate)} specific videos from the provided list.")
    else:
        videos_to_evaluate = videos_with_gt.intersection(videos_with_pred)
        print(f"📊 FULL MODE: Evaluating all {len(videos_to_evaluate)} available videos.")

    # Determine interaction types based on the data present (will be 2 classes if run in binary mode)
    gt_interaction_types = [str(t).lower() for t in ground_truth_df['interaction_type'].unique()]
    
    total_seconds_all = 0
    correct_seconds_all = 0

    category_stats = {category: {'total': 0, 'correct': 0} for category in gt_interaction_types}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassifications = defaultdict(int)
    video_results = []
    
    # Lists to collect second-by-second vectors for Cohen's Kappa
    all_gt_seconds = []
    all_pred_seconds = []

    # Evaluate each video individually
    for video in videos_to_evaluate:
        pred_video = predictions_df[predictions_df['video_name'] == video].copy()
        gt_video = ground_truth_df[ground_truth_df['video_name'] == video].copy()
        if len(gt_video) == 0 or len(pred_video) == 0:
            continue

        # Standardize formats
        pred_video['interaction_type'] = pred_video['interaction_type'].astype(str).str.lower()
        gt_video['interaction_type'] = gt_video['interaction_type'].astype(str).str.lower()

        # Convert time columns to numeric
        pred_video['start_time_sec'] = pd.to_numeric(pred_video['start_time_sec'], errors='coerce')
        pred_video['end_time_sec'] = pd.to_numeric(pred_video['end_time_sec'], errors='coerce')
        gt_video['start_time_sec'] = pd.to_numeric(gt_video['start_time_sec'], errors='coerce')
        gt_video['end_time_sec'] = pd.to_numeric(gt_video['end_time_sec'], errors='coerce')

        pred_video = pred_video.dropna(subset=['start_time_sec', 'end_time_sec'])
        gt_video = gt_video.dropna(subset=['start_time_sec', 'end_time_sec'])

        if len(gt_video) == 0 or len(pred_video) == 0:
            continue

        max_end_gt = gt_video['end_time_sec'].max()
        video_duration_seconds = int(max_end_gt) + 1
        
        # Skip videos that are too short to have a central evaluation segment
        if video_duration_seconds < AnalysisConfig.EXCLUSION_SECONDS * 2:
            print(f"⚠️ Warning: Video {video} is too short ({video_duration_seconds}s) for {AnalysisConfig.EXCLUSION_SECONDS * 2}s exclusion, skipping evaluation.")
            continue

        # Define the evaluation range: from 30s (inclusive) to end-30s (exclusive)
        start_sec_to_evaluate = AnalysisConfig.EXCLUSION_SECONDS
        end_sec_to_evaluate = video_duration_seconds - AnalysisConfig.EXCLUSION_SECONDS

        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)

        video_total_seconds = 0
        video_correct_seconds = 0

        # Use the adjusted range for evaluation loop
        for sec in range(start_sec_to_evaluate, end_sec_to_evaluate):
            gt_label = gt_labels[sec]
            pred_label = pred_labels[sec]

            if gt_label is not None and gt_label != 'unclassified':
                # Force unpredicted seconds to unclassified/fallback
                effective_pred = pred_label if pred_label is not None else UNCLASSIFIED_LABEL
                
                # Append to vectors for Kappa computation
                all_gt_seconds.append(str(gt_label).lower())
                all_pred_seconds.append(str(effective_pred).lower())
                
                video_total_seconds += 1
                total_seconds_all += 1
                
                # Check if category is already tracked (essential for binary mode)
                if gt_label not in category_stats:
                    category_stats[gt_label] = {'total': 0, 'correct': 0}
                category_stats[gt_label]['total'] += 1

                if pred_label is not None:
                    confusion_matrix[gt_label][pred_label] += 1
                    # Explicitly record misclassification type in confusion matrix
                    if pred_label != gt_label:
                        misclassifications[f"{gt_label} → {pred_label}"] += 1

                if pred_label == gt_label:
                    video_correct_seconds += 1
                    correct_seconds_all += 1
                    category_stats[gt_label]['correct'] += 1

        video_accuracy = video_correct_seconds / video_total_seconds if video_total_seconds > 0 else 0
        video_results.append({
            'video_name': video,
            'total_seconds': video_total_seconds,
            'correct_seconds': video_correct_seconds,
            'accuracy': video_accuracy
        })

    # Compute second-by-second Cohen's Kappa
    if all_gt_seconds and all_pred_seconds:
        overall_kappa = cohen_kappa_score(all_gt_seconds, all_pred_seconds)
    else:
        overall_kappa = 0.0
        
    overall_accuracy = correct_seconds_all / total_seconds_all if total_seconds_all > 0 else 0
    category_accuracies = {
        category: {
            'accuracy': (stats['correct'] / stats['total']) if stats['total'] > 0 else 0,
            'total_seconds': stats['total'],
            'correct_seconds': stats['correct']
        }
        for category, stats in category_stats.items()
    }
    
    # Update interaction_types to reflect the classes actually processed
    final_interaction_types = list(category_stats.keys())

    results = {
        'overall_accuracy': overall_accuracy,
        'overall_kappa': overall_kappa,
        'total_seconds': total_seconds_all,
        'correct_seconds': correct_seconds_all,
        'category_accuracies': category_accuracies,
        'video_results': video_results,
        'confusion_matrix': confusion_matrix,
        'interaction_types': final_interaction_types, # Use the actual categories found
        'misclassifications': misclassifications,
    }

    return results

def calculate_detailed_metrics(results):
  """Calculate precision, recall, and F1-score for each class from confusion matrix.

  This function ensures all ground truth categories are included in metrics,
  even if they were never predicted (resulting in zero precision/recall).

  Parameters
  ----------
  results : dict
      Results dictionary containing confusion_matrix and interaction_types

  Returns
  -------
  dict
      Dictionary with detailed metrics for each class
  """
  confusion_matrix = results['confusion_matrix']
  interaction_types = results['interaction_types']
  detailed_metrics = {}

  for class_name in interaction_types:
    tp = (
        confusion_matrix[class_name].get(class_name, 0)
        if class_name in confusion_matrix
        else 0
    )
    fp = 0
    for gt_class in confusion_matrix:
      if gt_class != class_name:
        fp += confusion_matrix[gt_class].get(class_name, 0)
    fn = 0
    if class_name in confusion_matrix:
      for pred_class in confusion_matrix[class_name]:
        if pred_class != class_name:
          fn += confusion_matrix[class_name][pred_class]
    category_stats = results.get('category_accuracies', {})
    if class_name in category_stats:
      total_gt_instances = category_stats[class_name]['total_seconds']
      fn = total_gt_instances - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    total_actual = tp + fn
    detailed_metrics[class_name] = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'support': total_actual,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
    }

  overall_k = results.get('overall_kappa', 0.0)

  if detailed_metrics and len(detailed_metrics) > 0:
    # Safely select ONLY dictionary entries that are not macro_avg
    non_macro_metrics = [
        m
        for k, m in detailed_metrics.items()
        if isinstance(m, dict) and k not in ['macro_avg', 'overall_kappa']
    ]

    if non_macro_metrics:
      macro_precision = float(
          np.mean([m['precision'] for m in non_macro_metrics])
      )
      macro_recall = float(np.mean([m['recall'] for m in non_macro_metrics]))
      macro_f1 = float(np.mean([m['f1_score'] for m in non_macro_metrics]))
      detailed_metrics['macro_avg'] = {
          'precision': macro_precision,
          'recall': macro_recall,
          'f1_score': macro_f1,
          'kappa': float(overall_k),
      }

  detailed_metrics['overall_kappa'] = float(overall_k)
  return detailed_metrics

def generate_confusion_matrix_plot(results: dict, output_folder: Path):
    confusion_matrix = results["confusion_matrix"]
    interaction_types = results["interaction_types"]

    # Canonical order
    if len(interaction_types) <= 2:
        labels = ["not interacting", "interacting"]
    else:
        labels = ["alone", "available", "interacting"]

    ordered_labels = [l for l in labels if l in interaction_types]
    n_classes = len(ordered_labels)

    # 1. Build original matrix: Rows = True (GT), Cols = Predicted
    cm_counts_gt_rows = np.array(
        [
            [confusion_matrix[gt].get(pred, 0) for pred in ordered_labels]
            for gt in ordered_labels
        ],
        dtype=float,
    )

    # 2. Normalize by row (Ground Truth totals) so percentages reflect class recall
    row_sums = cm_counts_gt_rows.sum(axis=1, keepdims=True)
    cm_pct_gt_rows = np.divide(
        cm_counts_gt_rows * 100.0,
        row_sums,
        out=np.zeros_like(cm_counts_gt_rows),
        where=row_sums != 0,
    )

    # 3. Flip axes so True Label is on X-axis (Cols) and Predicted Label is on Y-axis (Rows)
    cm_pct = cm_pct_gt_rows.T
    cm_counts = cm_counts_gt_rows.T.astype(int)

    # 4. Generate combined annotations: "XX.X%\n(Count)"
    annot_labels = np.empty((n_classes, n_classes), dtype=object)
    for i in range(n_classes):
        for j in range(n_classes):
            pct_val = cm_pct[i, j]
            cnt_val = cm_counts[i, j]
            annot_labels[i, j] = f"{pct_val:.1f}%\n({cnt_val:,})"

    # Display labels capitalized
    display_names = [label.capitalize() for label in ordered_labels]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(9, 7.5))

    sns.heatmap(
        cm_pct,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        cbar_kws={"label": "Percentage (%)"},
        xticklabels=display_names,
        yticklabels=display_names,
        annot_kws={"fontsize": 17, "weight": "bold"},
        ax=ax,
    )

    # Style colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(16)
    cbar.ax.tick_params(labelsize=14)

    # Axes styling (True on X, Predicted on Y)
    ax.set_xlabel("True", fontsize=18, fontweight="bold", labelpad=12)
    ax.set_ylabel("Predicted", fontsize=18, fontweight="bold", labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=15)

    # No headline
    plt.title("")
    plt.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)
    save_path = output_folder / "social_states_confusion_matrix.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Confusion matrix successfully saved to: {save_path}")

def save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename: Path):
    """Save performance summary and detailed metrics to a text file."""
    with open(filename, 'w') as f:
        # Write analysis summary
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total seconds analyzed: {total_seconds:,}\n")
        f.write(f"Total time analyzed: {total_hours:.2f} hours\n\n")

        # Safely grab kappa value
        kappa_val = detailed_metrics.get('overall_kappa', results.get('overall_kappa', 0.0))
        if isinstance(detailed_metrics.get('macro_avg'), dict):
            kappa_val = detailed_metrics['macro_avg'].get('kappa', kappa_val)
            
        # Write overall performance metrics from detailed_metrics
        if 'macro_avg' in detailed_metrics:
            f.write("OVERALL PERFORMANCE METRICS (Macro Average)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Accuracy (second-level):  {results['overall_accuracy']:.4f}\n")
            f.write(f"Cohen's Kappa (κ):        {kappa_val:.4f}\n")
            f.write(f"Macro Average Precision:  {detailed_metrics['macro_avg']['precision']:.4f}\n")
            f.write(f"Macro Average Recall:     {detailed_metrics['macro_avg']['recall']:.4f}\n")
            f.write(f"Macro Average F1-Score:   {detailed_metrics['macro_avg']['f1_score']:.4f}\n")
            f.write(f"Macro Average Kappa:      {kappa_val:.4f}\n\n")

        # Write category-specific performance
        f.write("CATEGORY-SPECIFIC PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")
        for category, metrics in detailed_metrics.items():
          # Skip macro_avg AND any non-dict entries (like overall_kappa)
          if category in ['macro_avg', 'overall_kappa'] or not isinstance(
              metrics, dict
          ):
            continue

          for category, metrics in detailed_metrics.items():
            # Skip macro_avg AND any non-dict entries (like overall_kappa)
            if category in ['macro_avg', 'overall_kappa'] or not isinstance(
                metrics, dict
            ):
                continue

            stats = results['category_accuracies'].get(
                category,
                {'total_seconds': 0, 'correct_seconds': 0, 'accuracy': 0},
            )

            f.write(f"{category.upper()}:\n")
            f.write(f"  Total seconds (GT): {stats['total_seconds']:,}\n")
            f.write(f"  Accuracy (second-level): {stats['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics.get('precision', 0.0):.4f}\n")
            f.write(f"  Recall: {metrics.get('recall', 0.0):.4f}\n")
            f.write(f"  F1-Score: {metrics.get('f1_score', 0.0):.4f}\n")
            f.write(f"  True Positives: {metrics.get('true_positives', 0):,}\n")
            f.write(f"  False Positives: {metrics.get('false_positives', 0):,}\n")
            f.write(f"  False Negatives: {metrics.get('false_negatives', 0):,}\n")
            f.write("\n")
          
def extract_misclassification_segments(predictions_df, ground_truth_df, results_by_seconds):
    """
    Extracts and consolidates continuous misclassified seconds into segments.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predicted interaction segments.
    ground_truth_df : pd.DataFrame
        DataFrame containing ground truth interaction segments.
    results_by_seconds : dict
        Results dictionary from evaluate_performance_by_seconds function.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing misclassified segments with columns:
        ['video_name', 'start_sec', 'end_sec', 'gt_label', 'pred_label', 'duration_sec']
    """
    misclassified_segments = []
    
    # 1. Prepare second-level labels lookup
    videos_to_evaluate = predictions_df['video_name'].unique()
    
    for video in videos_to_evaluate:
        pred_video = predictions_df[predictions_df['video_name'] == video].copy()
        gt_video = ground_truth_df[ground_truth_df['video_name'] == video].copy()
        
        max_end_gt = gt_video['end_time_sec'].max() if not gt_video.empty else 0
        video_duration_seconds = int(max_end_gt) + 1
        
        # Skip exclusion seconds to match evaluation window
        start_sec = AnalysisConfig.EXCLUSION_SECONDS
        end_sec = video_duration_seconds - AnalysisConfig.EXCLUSION_SECONDS
        
        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)
        
        current_segment = None
        
        for sec in range(start_sec, end_sec):
            gt_label = gt_labels[sec]
            pred_label = pred_labels[sec]
            
            # If the prediction is missing, force it to 'unclassified' for comparison
            if pred_label is None:
                pred_label_compare = 'unclassified'
            else:
                pred_label_compare = pred_label
            
            # Condition for misclassification: GT exists but pred is different OR unclassified
            is_misclassified = (gt_label is not None) and (gt_label != pred_label_compare)
            
            if is_misclassified:
                # Store the actual prediction value (None or the class name)
                actual_pred_output = pred_label if pred_label is not None else 'unclassified'

                # Start a new segment or extend the current one
                if current_segment is None:
                    # Start new segment
                    current_segment = {
                        'video_name': video,
                        'start_sec': sec,
                        'end_sec': sec,
                        'gt_label': gt_label,
                        'pred_label': actual_pred_output
                    }
                elif (current_segment['gt_label'] == gt_label and 
                      current_segment['pred_label'] == actual_pred_output):
                    # Extend current segment
                    current_segment['end_sec'] = sec
                else:
                    # Finalize previous segment and start a new one
                    misclassified_segments.append(current_segment)
                    current_segment = {
                        'video_name': video,
                        'start_sec': sec,
                        'end_sec': sec,
                        'gt_label': gt_label,
                        'pred_label': actual_pred_output
                    }
            else:
                # Finalize current segment if it was active
                if current_segment is not None:
                    misclassified_segments.append(current_segment)
                    current_segment = None
        
        # Finalize the last segment of the video
        if current_segment is not None:
            misclassified_segments.append(current_segment)
            
    # 2. Convert to DataFrame and calculate duration
    if misclassified_segments:
        df_miss = pd.DataFrame(misclassified_segments)
        df_miss['duration_sec'] = df_miss['end_sec'] - df_miss['start_sec'] + 1
        return df_miss
    
    return pd.DataFrame()

def run_evaluation(predictions_path: Path, output_folder: Path, mode: str, video_list: list = None):
    """
    Loads data, runs evaluation, and saves outputs in the same folder.
    
    Parameters
    ----------
    predictions_path : Path
        Path to the predictions CSV file.
    output_folder : Path
        Path to the folder where all outputs will be saved.
    mode : str
        Evaluation mode ('binary' or 'tertiary').
    video_list : list, optional
        A specific list of video names to evaluate. If None, evaluates all 
        overlapping videos between GT and Pred.
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    try:
        predictions_df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"❌ Error: Predictions file not found at {predictions_path}")
        sys.exit(1)

    ground_truth_path = Analysis.GROUND_TRUTH_SEGMENTS_CSV
    try:
        ground_truth_df = pd.read_csv(ground_truth_path, delimiter=';')
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    # --- Clean up GT DataFrame for potential malformed columns ---
    ground_truth_df = ground_truth_df.loc[:, ~ground_truth_df.columns.str.contains('^Unnamed')]
    ground_truth_df.dropna(axis=1, how='all', inplace=True)
    ground_truth_df.columns = ground_truth_df.columns.str.strip()
    
    for col in ground_truth_df.columns:
        if ground_truth_df[col].dtype == 'object':
            ground_truth_df[col] = ground_truth_df[col].str.strip()
            
    # --- Apply Binary Reclassification ---
    if mode == 'binary':
        predictions_df = reclassify_to_binary(predictions_df)
        ground_truth_df = reclassify_to_binary(ground_truth_df)

    # Standardize times
    if 'start_time_min' in ground_truth_df.columns and 'end_time_min' in ground_truth_df.columns:
        ground_truth_df['start_time_sec'] = ground_truth_df['start_time_min'].apply(time_to_seconds)
        ground_truth_df['end_time_sec'] = ground_truth_df['end_time_min'].apply(time_to_seconds)

    if 'start_time_min' in predictions_df.columns and 'end_time_min' in predictions_df.columns:
            predictions_df['start_time_sec'] = predictions_df['start_time_min'].apply(time_to_seconds)
            predictions_df['end_time_sec'] = predictions_df['end_time_min'].apply(time_to_seconds)
        
    # --- CORE EVALUATION ---
    # Pass the video_list into the evaluation logic
    results = evaluate_performance_by_seconds(predictions_df, ground_truth_df, video_subset=video_list)
    
    total_seconds = results['total_seconds']
    total_hours = total_seconds / 3600
    detailed_metrics = calculate_detailed_metrics(results)

    # --- Misclassification Analysis ---
    # Extract segments only for the evaluated subset
    df_misclassified = extract_misclassification_segments(predictions_df, ground_truth_df, results)
    
    # Filter misclassifications to only include the requested video_list if provided
    if video_list and not df_misclassified.empty:
        df_misclassified = df_misclassified[df_misclassified['video_name'].isin(video_list)]

    misclassified_path = output_folder / f"misclassified_segments.csv"
    if not df_misclassified.empty:
        df_misclassified.to_csv(misclassified_path, index=False)
        
    # Generate Plots and Results
    generate_confusion_matrix_plot(results, output_folder)
    performance_path = output_folder / (Analysis.PERFORMANCE_RESULTS_TXT.stem + Analysis.PERFORMANCE_RESULTS_TXT.suffix)
    save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename=performance_path)
    
    return predictions_df, ground_truth_df, detailed_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate social interaction predictions against ground truth.")
    parser.add_argument('--plot', nargs='?', const='all', default=None, help=('If omitted: no plotting.\n' 
                                                                              'If specified without value: plots all videos.\n' 
                                                                              'If a video name is given: plots only that video.'))
    parser.add_argument('--mode', type=str, choices=['binary', 'tertiary'], default='tertiary', help="Choose evaluation mode: 'binary' (interacting vs not interacting) or 'tertiary' (interacting, available, alone). Default is 'tertiary'.")
    parser.add_argument('--video_list', type=str, nargs='+', default=None, 
                        help='List of video names to include in this specific evaluation.')
    args = parser.parse_args()
    predictions_path = Analysis.INTERACTION_SEGMENTS_CSV

    # 1. Run evaluation (loads data, runs metrics, prints/saves results)
    output_folder = predictions_path.parent    
    predictions_df, ground_truth_df, _ = run_evaluation(predictions_path, output_folder, args.mode, video_list=args.video_list)

    # 2. Plotting logic
    if args.plot and args.plot.lower() == 'all':
        video_names = ground_truth_df['video_name'].unique()
        print(f"\n📊 Generating plots for all {len(video_names)} videos...")

        for video_name in video_names:
            plot_path = output_folder / f"{video_name}_segment_timeline.png"
            plot_segment_timeline(predictions_df, ground_truth_df, video_name, plot_path, args.mode)

        print("✅ All plots generated.")

    elif args.plot:
        plot_video_name = args.plot
        if plot_video_name not in ground_truth_df['video_name'].unique():
            print(f"⚠️  Video name '{plot_video_name}' not found in ground truth — skipping plot.")
        else:
            plot_path = output_folder / f"{plot_video_name}_segment_timeline.png"
            plot_segment_timeline(predictions_df, ground_truth_df, plot_video_name, plot_path, args.mode)
            print(f"✅ Plot generated for video: {plot_video_name}")