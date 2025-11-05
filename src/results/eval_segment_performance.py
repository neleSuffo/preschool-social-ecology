import argparse
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import defaultdict

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Evaluation, Inference
from config import InferenceConfig
from results.utils import time_to_seconds

def reclassify_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Reclassifies three-class labels to binary 'interacting' vs 'not interacting'."""
    df_copy = df.copy()
    # Define mapping: 'interacting' remains 'interacting', others become 'not interacting'
    mapping = {
        'interacting': 'interacting',
        'available': 'not interacting',
        'alone': 'not interacting'
    }
    # Ensure mapping handles lowercased strings
    df_copy['interaction_type'] = df_copy['interaction_type'].astype(str).str.lower().str.strip().map(mapping).fillna(df_copy['interaction_type'])
    return df_copy

def plot_segment_timeline(predictions_df, ground_truth_df, video_name, save_path, binary_mode=False):
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
    # Define colors based on mode
    if binary_mode:
        INTERACTION_COLORS = {
            'interacting': '#d62728',       # Red
            'not interacting': '#1f77b4',   # Blue
        }
    else:
        INTERACTION_COLORS = {
            'interacting': '#d62728', # Red
            'available': '#ff7f0e',   # Orange
            'alone': '#1f77b4',       # Blue
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
    
def create_second_level_labels(segments_df, video_duration_seconds):
    """
    Create a second-by-second label array for a video based on segments. 
    ...
    """
    labels = np.full(video_duration_seconds, None, dtype=object)
    for _, segment in segments_df.iterrows():
        try:
            start_sec = int(np.round(float(segment['start_time_sec'])))
        except Exception:
            start_sec = 0
        try:
            end_sec = int(np.round(float(segment['end_time_sec'])))
            # Clip end_sec to prevent out-of-bounds indexing
            end_sec = min(end_sec, video_duration_seconds - 1)
        except Exception:
            end_sec = 0
        
        interaction_type = str(segment['interaction_type']).lower()
        
        # Ensure start_sec is valid
        start_sec = max(0, start_sec)

        # Check for valid segment duration after rounding
        # assign interaction type to the range of seconds
        if start_sec <= end_sec: 
            labels[start_sec:end_sec + 1] = interaction_type
           
    return labels

def evaluate_performance_by_seconds(predictions_df, ground_truth_df):
    """
    Evaluates model performance by comparing second-by-second classifications,
    excluding the first and last x seconds, as defined in InferenceConfig.EXCLUSION_SECONDS.
    ...
    """    
    # Identify videos present in both predictions and ground truth
    videos_with_gt = set(ground_truth_df['video_name'].unique())
    videos_with_pred = set(predictions_df['video_name'].unique())
    videos_to_evaluate = videos_with_gt.intersection(videos_with_pred)
    
    print(f"Evaluating {len(videos_to_evaluate)} videos with both GT and Predictions...")

    # Determine interaction types based on the data present (will be 2 classes if run in binary mode)
    gt_interaction_types = [str(t).lower() for t in ground_truth_df['interaction_type'].unique()]
    
    total_seconds_all = 0
    correct_seconds_all = 0

    category_stats = {category: {'total': 0, 'correct': 0} for category in gt_interaction_types}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    misclassifications = defaultdict(int)

    video_results = []

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
        if video_duration_seconds < InferenceConfig.EXCLUSION_SECONDS * 2:
            print(f"⚠️ Warning: Video {video} is too short ({video_duration_seconds}s) for {InferenceConfig.EXCLUSION_SECONDS * 2}s exclusion, skipping evaluation.")
            continue

        # Define the evaluation range: from 30s (inclusive) to end-30s (exclusive)
        start_sec_to_evaluate = InferenceConfig.EXCLUSION_SECONDS
        end_sec_to_evaluate = video_duration_seconds - InferenceConfig.EXCLUSION_SECONDS

        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)

        video_total_seconds = 0
        video_correct_seconds = 0

        # Use the adjusted range for evaluation loop
        for sec in range(start_sec_to_evaluate, end_sec_to_evaluate):
            gt_label = gt_labels[sec]
            pred_label = pred_labels[sec]

            if gt_label is not None:
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
    """
    Calculate precision, recall, and F1-score for each class from confusion matrix.
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
        tp = confusion_matrix[class_name].get(class_name, 0) if class_name in confusion_matrix else 0
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
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        total_actual = tp + fn
        detailed_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': total_actual,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    if detailed_metrics and len(detailed_metrics) > 0:
        # Calculate macro average only over non-macro categories
        non_macro_metrics = [m for k, m in detailed_metrics.items() if k != 'macro_avg']
        if non_macro_metrics:
            macro_precision = np.mean([m['precision'] for m in non_macro_metrics])
            macro_recall = np.mean([m['recall'] for m in non_macro_metrics])
            macro_f1 = np.mean([m['f1_score'] for m in non_macro_metrics])
            detailed_metrics['macro_avg'] = {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1,
            }
    return detailed_metrics

def generate_confusion_matrix_plots(results, output_folder: Path):
    """Generate and save confusion matrix plots inside the specified output folder."""

    confusion_matrix = results['confusion_matrix']
    interaction_types = results['interaction_types']
    
    # Determine the order based on the number of unique interaction types found
    if len(interaction_types) == 2:
        # Binary mode: ['not interacting', 'interacting']
        preferred_order = ['not interacting', 'interacting']
        plot_title_suffix = " (Binary)"
    else:
        # Tertiary mode: ['alone', 'available', 'interacting'] (original three)
        preferred_order = ['alone', 'available', 'interacting']
        plot_title_suffix = ""


    # Filter and sort labels based on current interaction_types and preferred order
    sorted_gt_labels = [label for label in preferred_order if label in interaction_types]
    
    # Reverse the order for the predicted labels (X-axis) for standard visual layout
    sorted_pred_labels = sorted_gt_labels[::-1]

    matrix_array = np.array([
        [confusion_matrix[gt_label].get(pred_label, 0)
         for pred_label in sorted_pred_labels]
        for gt_label in sorted_gt_labels
    ])

    # Convert to percentages
    matrix_percentages = np.zeros_like(matrix_array, dtype=float)
    for i in range(len(sorted_gt_labels)):
        row_sum = np.sum(matrix_array[i])
        if row_sum > 0:
            matrix_percentages[i] = (matrix_array[i] / row_sum) * 100

    # --- Save plots ---
    output_folder.mkdir(parents=True, exist_ok=True)
    conf_matrix_counts_path = output_folder / (Evaluation.CONF_MATRIX_COUNTS.stem + plot_title_suffix.replace(" ", "_").lower() + Evaluation.CONF_MATRIX_COUNTS.suffix)
    conf_matrix_percentages_path = output_folder / (Evaluation.CONF_MATRIX_PERCENTAGES.stem + plot_title_suffix.replace(" ", "_").lower() + Evaluation.CONF_MATRIX_PERCENTAGES.suffix)


    # Absolute counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Number of GT Seconds'})
    plt.title(f'Confusion Matrix (Counts){plot_title_suffix}')
    
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label (Ground Truth)', fontsize=14, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(conf_matrix_counts_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_percentages, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confusion Matrix (Percentages){plot_title_suffix}')
    
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label (Ground Truth)', fontsize=14, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(conf_matrix_percentages_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrices saved to: {conf_matrix_percentages_path}")


def save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename: Path):
    """Save performance summary and detailed metrics to a text file."""
    with open(filename, 'w') as f:
        # Write analysis summary
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total seconds analyzed: {total_seconds:,}\n")
        f.write(f"Total time analyzed: {total_hours:.2f} hours\n\n")

        # Write overall performance metrics from detailed_metrics
        if 'macro_avg' in detailed_metrics:
            f.write("OVERALL PERFORMANCE METRICS (Macro Average)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Accuracy (second-level):  {results['overall_accuracy']:.4f}\n")
            f.write(f"Macro Average Precision:  {detailed_metrics['macro_avg']['precision']:.4f}\n")
            f.write(f"Macro Average Recall:     {detailed_metrics['macro_avg']['recall']:.4f}\n")
            f.write(f"Macro Average F1-Score:   {detailed_metrics['macro_avg']['f1_score']:.4f}\n\n")

        # Write category-specific performance
        f.write("CATEGORY-SPECIFIC PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")
        for category, metrics in detailed_metrics.items():
            if category == 'macro_avg':
                continue
            
            stats = results['category_accuracies'].get(category, {'total_seconds': 0, 'correct_seconds': 0, 'accuracy': 0})

            f.write(f"{category.upper()}:\n")
            f.write(f"  Total seconds (GT): {stats['total_seconds']:,}\n")
            f.write(f"  Accuracy (second-level): {stats['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  True Positives: {metrics['true_positives']:,}\n")
            f.write(f"  False Positives: {metrics['false_positives']:,}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']:,}\n")
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
        start_sec = InferenceConfig.EXCLUSION_SECONDS
        end_sec = video_duration_seconds - InferenceConfig.EXCLUSION_SECONDS
        
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

def run_evaluation(predictions_path: Path, binary_mode: bool):
    """Loads data, runs evaluation, and saves outputs in the same folder."""
    try:
        predictions_df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"❌ Error: Predictions file not found at {predictions_path}")
        sys.exit(1)

    ground_truth_path = Inference.GROUND_TRUTH_SEGMENTS_CSV
    try:
        ground_truth_df = pd.read_csv(ground_truth_path, delimiter=';')
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

    # --- Clean up GT DataFrame for potential malformed columns ---
    # Drop columns that are completely empty and have default Pandas names (Unnamed: X)
    ground_truth_df = ground_truth_df.loc[:, ~ground_truth_df.columns.str.contains('^Unnamed')]
    ground_truth_df.dropna(axis=1, how='all', inplace=True)
    
    # Ensure all column headers are stripped of whitespace
    ground_truth_df.columns = ground_truth_df.columns.str.strip()
    
    # Apply global string strip on data to eliminate invisible chars that prevent matching/parsing
    for col in ground_truth_df.columns:
        if ground_truth_df[col].dtype == 'object':
            ground_truth_df[col] = ground_truth_df[col].str.strip()
            
    # --- Apply Binary Reclassification ---
    if binary_mode:
        predictions_df = reclassify_to_binary(predictions_df)
        ground_truth_df = reclassify_to_binary(ground_truth_df)
        print("📊 Running evaluation in BINARY mode: 'Available' and 'Alone' are mapped to 'Not Interacting'.")
    else:
        print("📊 Running evaluation in TERTIARY mode (Interacting, Available, Alone).")

    # --- Determine dynamic output folder ---
    output_folder = predictions_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # Convert ground truth times to seconds
    if 'start_time_min' in ground_truth_df.columns and 'end_time_min' in ground_truth_df.columns:
        ground_truth_df['start_time_sec'] = ground_truth_df['start_time_min'].apply(time_to_seconds)
        ground_truth_df['end_time_sec'] = ground_truth_df['end_time_min'].apply(time_to_seconds)

    # Evaluate
    results = evaluate_performance_by_seconds(predictions_df, ground_truth_df)
    total_seconds = results['total_seconds']
    total_hours = total_seconds / 3600
    detailed_metrics = calculate_detailed_metrics(results)

    # --- Misclassification Analysis ---
    df_misclassified = extract_misclassification_segments(predictions_df, ground_truth_df, results)
    misclassified_path_suffix = "_binary" if binary_mode else ""
    misclassified_path = output_folder / f"misclassified_segments{misclassified_path_suffix}.csv"
    
    if not df_misclassified.empty:
        df_misclassified.to_csv(misclassified_path, index=False)
        print(f"✅ Misclassified segments saved to: {misclassified_path}")
    else:
        print("✅ No misclassified segments found (or all metrics are zero).")
        
    # Generate outputs
    generate_confusion_matrix_plots(results, output_folder)

    performance_path_suffix = "_binary" if binary_mode else ""
    performance_path = output_folder / (Evaluation.PERFORMANCE_RESULTS_TXT.stem + performance_path_suffix + Evaluation.PERFORMANCE_RESULTS_TXT.suffix)
    
    save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename=performance_path)
    
    # Print F1-scores to console
    print("\n--- F1-Scores by Category ---")
    for category, metrics in detailed_metrics.items():
        if category != 'macro_avg':
            print(f"{category.capitalize()}: F1-Score = {metrics['f1_score']:.4f}")
    if 'macro_avg' in detailed_metrics:
        # print empty line
        print("")
        print("\n--- Overall Macro Average ---")
        print(f"Macro Average: F1-Score = {detailed_metrics['macro_avg']['f1_score']:.4f}")

    return predictions_df, ground_truth_df, detailed_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate social interaction predictions against ground truth.")
    parser.add_argument('--folder_path', type=str, required=True, help='Path to the folder containing the predictions CSV file (e.g. 02_interaction_segments.csv)')
    parser.add_argument('--plot', nargs='?', const='all', default=None, help=('If omitted: no plotting.\n' 
                                                                              'If specified without value: plots all videos.\n' 
                                                                              'If a video name is given: plots only that video.'))
    parser.add_argument('--binary', action='store_true', help='If set, combines "available" and "alone" into "not interacting" for binary classification.')

    args = parser.parse_args()
    predictions_path = Path(args.folder_path) / Inference.INTERACTION_SEGMENTS_CSV

    # 1. Run evaluation (loads data, runs metrics, prints/saves results)
    predictions_df, ground_truth_df, _ = run_evaluation(predictions_path, args.binary)
    output_folder = predictions_path.parent

    # 2. Plotting logic
    if args.plot and args.plot.lower() == 'all':
        video_names = ground_truth_df['video_name'].unique()
        print(f"\n📊 Generating plots for all {len(video_names)} videos...")

        for video_name in video_names:
            plot_path = output_folder / f"{video_name}_segment_timeline{'_binary' if args.binary else ''}.png"
            plot_segment_timeline(predictions_df, ground_truth_df, video_name, plot_path, args.binary)

        print("✅ All plots generated.")

    elif args.plot:
        plot_video_name = args.plot
        if plot_video_name not in ground_truth_df['video_name'].unique():
            print(f"⚠️  Video name '{plot_video_name}' not found in ground truth — skipping plot.")
        else:
            plot_path = output_folder / f"{plot_video_name}_segment_timeline{'_binary' if args.binary else ''}.png"
            plot_segment_timeline(predictions_df, ground_truth_df, plot_video_name, plot_path, args.binary)
            print(f"✅ Plot generated for video: {plot_video_name}")