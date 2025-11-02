import argparse
import sys
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
from utils import time_to_seconds

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

    interaction_types = [str(t).lower() for t in ground_truth_df['interaction_type'].unique()]
    total_seconds_all = 0
    correct_seconds_all = 0

    category_stats = {category: {'total': 0, 'correct': 0} for category in interaction_types}
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

    results = {
        'overall_accuracy': overall_accuracy,
        'total_seconds': total_seconds_all,
        'correct_seconds': correct_seconds_all,
        'category_accuracies': category_accuracies,
        'video_results': video_results,
        'confusion_matrix': confusion_matrix,
        'interaction_types': interaction_types,
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
    if detailed_metrics:
        macro_precision = np.mean([m['precision'] for m in detailed_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in detailed_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in detailed_metrics.values()])
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
    preferred_order = ['interacting', 'available', 'alone']

    sorted_gt_labels = [label for label in preferred_order if label in interaction_types]
    sorted_pred_labels = sorted_gt_labels

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
    conf_matrix_counts_path = output_folder / Evaluation.CONF_MATRIX_COUNTS
    conf_matrix_percentages_path = output_folder / Evaluation.CONF_MATRIX_PERCENTAGES

    # Absolute counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Number of GT Seconds'})
    plt.title('Confusion Matrix (Counts)')
    plt.tight_layout()
    plt.savefig(conf_matrix_counts_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Percentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_percentages, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Confusion Matrix (Percentages)')
    plt.tight_layout()
    plt.savefig(conf_matrix_percentages_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Confusion matrices saved in: {output_folder}")


def save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename: Path):
    """Save performance summary and detailed metrics to a text file."""
    with open(filename, 'w') as f:
        f.write(f"Total evaluated time: {total_seconds:.0f} seconds ({total_hours:.2f} hours)\n\n")
        f.write("=== Performance Metrics ===\n")
        for k, v in results['metrics'].items():
            f.write(f"{k}: {v:.3f}\n")
        f.write("\n=== Detailed Metrics ===\n")
        for label, metrics in detailed_metrics.items():
            f.write(f"\nLabel: {label}\n")
            for m, val in metrics.items():
                f.write(f"  {m}: {val:.3f}\n")

    print(f"✅ Performance results saved to: {filename}")


def run_evaluation(predictions_path: Path):
    """Loads data, runs evaluation, and saves outputs in the same folder."""

    print(f"Loading predictions from: {predictions_path}")
    try:
        predictions_df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"❌ Error: Predictions file not found at {predictions_path}")
        sys.exit(1)

    ground_truth_path = Inference.GROUND_TRUTH_SEGMENTS_CSV
    print(f"Loading ground truth from: {ground_truth_path}")
    try:
        ground_truth_df = pd.read_csv(ground_truth_path, delimiter=';')
    except FileNotFoundError:
        print(f"❌ Error: Ground truth file not found at {ground_truth_path}")
        sys.exit(1)

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

    # Generate outputs
    generate_confusion_matrix_plots(results, output_folder)

    performance_path = output_folder / Evaluation.PERFORMANCE_RESULTS_TXT
    save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename=performance_path)

    return predictions_df, ground_truth_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate social interaction predictions against ground truth.")
    parser.add_argument('--input', type=str, required=True, help='Path to the predictions CSV file (e.g. 02_interaction_segments.csv)')

    args = parser.parse_args()
    predictions_path = Path(args.input)

    run_evaluation(predictions_path)