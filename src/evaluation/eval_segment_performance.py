import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import Inference
from collections import defaultdict
from config import DataConfig

def time_to_seconds(time_str):
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        else:
            return float(time_str)
    except:
        return None

def create_second_level_labels(segments_df, video_duration_seconds):
    """
    Create a second-by-second label array for a video based on segments. 
    Also reports the total number of seconds labeled.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments for a single video, containing 'start_time_sec', 'end_time_sec', 'interaction_type'
    video_duration_seconds : int
        Total number of seconds in the video
    
    Returns
    -------
    np.array
        Array where each index represents a second and contains the interaction type label
        Example: ['alone', 'alone', 'interacting', ...]
    """
    labels = np.full(video_duration_seconds, None, dtype=object)
    for _, segment in segments_df.iterrows():
        try:
            start_sec = int(np.round(float(segment['start_time_sec'])))
        except Exception:
            start_sec = 0
        try:
            end_sec = int(np.round(float(segment['end_time_sec'])))
        except Exception:
            end_sec = 0
        
        interaction_type = str(segment['interaction_type']).lower()
        
        # Clip end_sec to prevent out-of-bounds indexing
        end_sec = min(end_sec, video_duration_seconds - 1)
        
        # Ensure start_sec is valid
        start_sec = max(0, start_sec)

        # Check for valid segment duration after rounding
        # assign interaction type to the range of seconds
        if start_sec <= end_sec: 
            labels[start_sec:end_sec + 1] = interaction_type
           
    return labels

def evaluate_performance_by_seconds(predictions_df, ground_truth_df):
    """
    Evaluates model performance by comparing second-by-second classifications.
    Also captures the types of misclassifications (e.g., alone → interacting).
    """
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

        pred_labels = create_second_level_labels(pred_video, video_duration_seconds)
        gt_labels = create_second_level_labels(gt_video, video_duration_seconds)

        video_total_seconds = 0
        video_correct_seconds = 0

        for sec in range(video_duration_seconds):
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
        'misclassifications': misclassifications,  # new
    }

    return results

def generate_confusion_matrix_plots(results):
    """
    Generate and save confusion matrix plots.
    
    Ensures all ground truth categories appear in both rows and columns,
    even if some categories were never predicted (showing zero counts).
    """
    confusion_matrix = results['confusion_matrix']
    interaction_types = results['interaction_types']
    
    # Define consistent order for labels (interacting, available, alone, unclassified)
    preferred_order = ['interacting', 'available', 'alone']
    
    # Get ground truth labels in preferred order
    sorted_gt_labels = []
    for label in preferred_order:
        if label in interaction_types:
            sorted_gt_labels.append(label)
    # Add any remaining GT labels not in preferred order
    for label in sorted(interaction_types):
        if label not in sorted_gt_labels:
            sorted_gt_labels.append(label)
    
    # Ensure predicted labels include ALL ground truth types (even if count is 0)
    # This ensures symmetric confusion matrix with all GT categories as both rows and columns
    all_pred_labels = set(interaction_types)  # Start with all GT types

    # Add any additional predicted labels that appear in confusion matrix
    for gt_label in confusion_matrix:
        for pred_label in confusion_matrix[gt_label]:
            if pred_label != 'unclassified' and pred_label is not None and str(pred_label).lower() != 'nan':  # Skip unclassified and NaN labels
                all_pred_labels.add(pred_label)

    # Remove None/NaN from all_pred_labels
    all_pred_labels = {lbl for lbl in all_pred_labels if lbl is not None and str(lbl).lower() != 'nan'}

    sorted_pred_labels = []
    for label in preferred_order:
        if label in all_pred_labels:
            sorted_pred_labels.append(label)
    # Add any remaining predicted labels (excluding unclassified and NaN)
    for label in sorted(all_pred_labels):
        if label not in sorted_pred_labels:
            sorted_pred_labels.append(label)
    
    # Reverse the ground truth order to align diagonal properly
    sorted_gt_labels = sorted_gt_labels[::-1]
    
    # Only include seconds with ground truth labels in absolute counts
    matrix_data = []
    gt_counts = []  # Track total seconds with ground truth for each class
    for gt_label in sorted_gt_labels:
        if gt_label is None or str(gt_label).lower() == 'nan':
            continue
        row = []
        gt_total = 0
        for pred_label in sorted_pred_labels:
            if pred_label is None or str(pred_label).lower() == 'nan':
                continue
            count = confusion_matrix[gt_label][pred_label]
            row.append(count)
            gt_total += count
        matrix_data.append(row)
        gt_counts.append(gt_total)

    matrix_array = np.array(matrix_data)

    # Absolute count heatmap (only seconds with ground truth)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_array,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Number of GT Seconds'})

    plt.title('Confusion Matrix - Second-by-Second Classification (Counts, GT Only)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Labels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Inference.CONF_MATRIX_COUNTS, dpi=300, bbox_inches='tight')
    plt.close()

    # Relative count heatmap (row-normalized)
    matrix_percentages = np.zeros_like(matrix_array, dtype=float)
    for i, row in enumerate(matrix_array):
        row_total = row.sum()
        if row_total > 0:
            matrix_percentages[i] = (row / row_total) * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_percentages,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Percentage (%)'})

    plt.title('Confusion Matrix - Second-by-Second Classification (Percentages)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Labels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(Inference.CONF_MATRIX_PERCENTAGES, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {Inference.CONF_MATRIX_COUNTS} and {Inference.CONF_MATRIX_PERCENTAGES}")

def save_performance_results(results, detailed_metrics, total_frames, total_hours, filename=Inference.PERFORMANCE_RESULTS_TXT):
    """
    Save category-specific performance results with detailed metrics to a text file.
    """
    with open(filename, 'w') as f:
        # Write analysis summary
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total seconds analyzed: {total_frames:,}\n")
        f.write(f"Total time analyzed: {total_hours:.2f} hours ({int(total_hours)}h {int((total_frames % 3600) / 60)}m)\n")
        
        # Write detailed metrics summary
        if 'macro_avg' in detailed_metrics:
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Macro Average Precision:  {detailed_metrics['macro_avg']['precision']:.4f} ({detailed_metrics['macro_avg']['precision']*100:.2f}%)\n")
            f.write(f"Macro Average Recall:     {detailed_metrics['macro_avg']['recall']:.4f} ({detailed_metrics['macro_avg']['recall']*100:.2f}%)\n")
            f.write(f"Macro Average F1-Score:   {detailed_metrics['macro_avg']['f1_score']:.4f} ({detailed_metrics['macro_avg']['f1_score']*100:.2f}%)\n\n")
        
        # Write category-specific performance with detailed metrics
        f.write("CATEGORY-SPECIFIC PERFORMANCE\n")
        f.write("=" * 70 + "\n\n")
        for category, stats in results['category_accuracies'].items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Total seconds: {stats['total_seconds']:,}\n")
            f.write(f"  Correctly classified: {stats['correct_seconds']:,}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)\n")
            # Add detailed metrics if available
            if category in detailed_metrics:
                metrics = detailed_metrics[category]
                f.write(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
                f.write(f"  Recall: {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n")
                f.write(f"  True Positives: {metrics['true_positives']:,}\n")
                f.write(f"  False Positives: {metrics['false_positives']:,}\n")
                f.write(f"  False Negatives: {metrics['false_negatives']:,}\n")
            f.write("\n")
        # Add interpretation guide
        f.write("METRIC INTERPRETATION GUIDE\n")
        f.write("=" * 70 + "\n")
        f.write("Accuracy:   Overall percentage of seconds classified correctly\n")
        f.write("Precision:  Of seconds predicted as this class, how many were correct?\n")
        f.write("Recall:     Of actual seconds of this class, how many were detected?\n")
        f.write("F1-Score:   Harmonic mean of precision and recall (balanced metric)\n\n")
        f.write("Macro Avg:    Unweighted average across classes (treats all classes equally)\n")
    print(f"✅ Performance results saved: {filename}")

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

def main(predictions_df, ground_truth_df):
    """
    Main function to evaluate performance using frame-by-frame accuracy.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predicted interaction segments.
    ground_truth_df : pd.DataFrame
        DataFrame containing ground truth interaction segments.
    """
    print("Evaluating performance using frame-by-frame accuracy...")

    pred_videos = set(predictions_df['video_name'].unique())
    gt_videos = set(ground_truth_df['video_name'].unique())
    videos_to_evaluate = pred_videos.intersection(gt_videos)
    print(f"Videos being evaluated (have both GT and predictions): {len(videos_to_evaluate)}")
    
    # convert minute-based times in GT to seconds for consistency using function time_to_seconds
    if 'start_time_min' in ground_truth_df.columns and 'end_time_min' in ground_truth_df.columns:
        ground_truth_df['start_time_sec'] = ground_truth_df['start_time_min'].apply(time_to_seconds)
        ground_truth_df['end_time_sec'] = ground_truth_df['end_time_min'].apply(time_to_seconds)

    # Evaluate performance
    results = evaluate_performance_by_seconds(predictions_df, ground_truth_df)
    # Calculate total seconds and hours analyzed
    total_seconds = results['total_seconds']
    total_hours = total_seconds / 3600
    
    # Calculate detailed metrics (precision, recall, F1)
    detailed_metrics = calculate_detailed_metrics(results)
    
    # Validation: Ensure all ground truth types are in detailed metrics
    gt_types_set = set(results['interaction_types'])
    metrics_types_set = set(detailed_metrics.keys()) - {'macro_avg'}
    if gt_types_set != metrics_types_set:
        print(f"⚠️ Warning: Mismatch between GT types and metrics types")
        print(f"GT types: {sorted(gt_types_set)}")
        print(f"Metrics types: {sorted(metrics_types_set)}")

    # Generate confusion matrix plots
    generate_confusion_matrix_plots(results)
    
    # Save performance results to text file with detailed metrics
    save_performance_results(results, detailed_metrics, total_seconds, total_hours)
    
if __name__ == "__main__":
    # load data
    predictions_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')

    main(predictions_df, ground_truth_df)