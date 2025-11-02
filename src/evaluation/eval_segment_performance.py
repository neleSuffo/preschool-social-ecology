import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from collections import defaultdict
import re

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

# Assuming these are available in your project structure
from constants import Evaluation, Inference
from config import InferenceConfig
from utils import time_to_seconds 

# --- PLOTTING FUNCTION ---

def plot_segment_timeline(predictions_df, ground_truth_df, video_name, save_path):
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
    """
    # Define colors and category order
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
                height=0.4,
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
                height=0.4,
                color=color,
                edgecolor='black',
                alpha=0.7)

    # --- Configuration ---
    
    # Y-axis labels
    ax.set_yticks([y_pos_pred, y_pos_gt])
    ax.set_yticklabels(['Prediction', 'Ground Truth'], fontsize=12)
    ax.set_ylim(0.5, 1.5)
    
    # X-axis (Time) configuration
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_xlim(0, max_time)
    
    # Title
    ax.set_title(f"Segment Timeline Comparison for Video: {video_name}", fontsize=14)

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
    print(f"\n✅ Plot successfully saved to: {save_path}")

# --- EVALUATION CORE FUNCTIONS (RETAINED) ---

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

def evaluate_performance(predictions_df, ground_truth_df, iou_threshold=None):
    """
    Wrapper function for hyperparameter tuning.
    Runs second-by-second evaluation and calculates detailed metrics, 
    matching the expected interface of tune_hyperparameters.py.
    ...
    """
    # Convert GT time columns from min:sec format to seconds if necessary
    if 'start_time_min' in ground_truth_df.columns and 'end_time_min' in ground_truth_df.columns:
        ground_truth_df['start_time_sec'] = ground_truth_df['start_time_min'].apply(time_to_seconds)
        ground_truth_df['end_time_sec'] = ground_truth_df['end_time_min'].apply(time_to_seconds)
    
    # 1. Run the core second-by-second comparison
    results_by_seconds = evaluate_performance_by_seconds(predictions_df, ground_truth_df)

    # 2. Calculate detailed metrics (Precision, Recall, F1, TP, FP, FN)
    detailed_metrics = calculate_detailed_metrics(results_by_seconds)

    # 3. Reformat the output to match tune_hyperparameters.py's expectation
    output = {}
    for class_name, metrics in detailed_metrics.items():
        if class_name != 'macro_avg':
            output[class_name] = {
                # Required for overall metric calculation in tune_hyperparameters.py
                'tp': metrics['true_positives'],
                'fp': metrics['false_positives'],
                'fn': metrics['false_negatives'],
                # Included for completeness
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
            }
            
    return output

def generate_confusion_matrix_plots(results):
    """
    Generate and save confusion matrix plots (absolute counts and relative percentages).
    ...
    """
    confusion_matrix = results['confusion_matrix']
    interaction_types = results['interaction_types']
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
    all_pred_labels = set(interaction_types)  

    for gt_label in confusion_matrix:
        for pred_label in confusion_matrix[gt_label]:
            if pred_label != 'unclassified' and pred_label is not None and str(pred_label).lower() != 'nan': 
                all_pred_labels.add(pred_label)

    # Remove None/NaN from all_pred_labels
    all_pred_labels = {lbl for lbl in all_pred_labels if lbl is not None and str(lbl).lower() != 'nan'}

    sorted_pred_labels = []
    for label in preferred_order:
        if label in all_pred_labels:
            sorted_pred_labels.append(label)
    for label in sorted(all_pred_labels):
        if label not in sorted_pred_labels:
            sorted_pred_labels.append(label)
    
    # Reverse the ground truth order to align diagonal properly
    sorted_gt_labels = sorted_gt_labels[::-1]
    
    # Only include seconds with ground truth labels in absolute counts
    matrix_data = []
    gt_counts = [] 
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

    # Save plots to dynamic folder
    output_folder.mkdir(parents=True, exist_ok=True)
    conf_matrix_counts_path = output_folder / InferenceConfig.CONFUSION_MATRIX_COUNTS
    conf_matrix_percentages_path = output_folder / InferenceConfig.CONFUSION_MATRIX_PERCENTAGES

    # Save absolute matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Number of GT Seconds'})
    plt.title('Confusion Matrix (Counts)')
    plt.tight_layout()
    plt.savefig(conf_matrix_counts_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save percentage matrix
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

def save_performance_results(results, detailed_metrics, total_frames, total_hours, filename=Evaluation.PERFORMANCE_RESULTS_TXT):
    """
    Save category-specific performance results with detailed metrics to a text file.
    """
    # Add timestamp to filename
    filename.parent.mkdir(parents=True, exist_ok=True)
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
    print(f"✅ Performance results saved to: {filename}")


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

def run_evaluation(predictions_path: Path):
    """Loads data, runs evaluation, and prints/saves results."""
    
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

    # --- new: define output folder based on input file ---
    output_folder = predictions_path.parent
    output_folder.mkdir(parents=True, exist_ok=True)

    # convert minute-based times in GT to seconds for consistency
    if 'start_time_min' in ground_truth_df.columns and 'end_time_min' in ground_truth_df.columns:
        ground_truth_df['start_time_sec'] = ground_truth_df['start_time_min'].apply(time_to_seconds)
        ground_truth_df['end_time_sec'] = ground_truth_df['end_time_min'].apply(time_to_seconds)

    results = evaluate_performance_by_seconds(predictions_df, ground_truth_df)
    total_seconds = results['total_seconds']
    total_hours = total_seconds / 3600
    detailed_metrics = calculate_detailed_metrics(results)

    # Generate confusion matrix plots and save to output folder
    generate_confusion_matrix_plots(results, output_folder)

    # Save performance results to output folder
    performance_path = output_folder / InferenceConfig.PERFORMANCE_RESULTS_TXT
    save_performance_results(results, detailed_metrics, total_seconds, total_hours, filename=performance_path)
    
    return predictions_df, ground_truth_df

def main_evaluation_and_plotting(predictions_df, ground_truth_df, args):
    """
    Handles plotting logic if --plot argument is provided.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame containing predicted interaction segments.
    ground_truth_df : pd.DataFrame
        DataFrame containing ground truth interaction segments.
    args: argparse.Namespace
        Parsed command line arguments.
    """
    
    # --- Plotting Logic ---
    if args.plot:
        plot_video_name = args.plot
        
        # Determine the output folder: parent directory of the input CSV file
        input_path = Path(args.input)
        output_folder = input_path.parent
        
        if plot_video_name == '__ALL__':
            video_names = predictions_df['video_name'].unique()
            print(f"Plotting all {len(video_names)} videos...")
            for video_name in video_names:
                # Use the input folder for plots
                plot_path = output_folder / f"{video_name}_segment_timeline.png"
                plot_segment_timeline(predictions_df, ground_truth_df, video_name, plot_path)
        else:
            # Simple check for required format parts: non-empty string
            if not re.match(r'.+', plot_video_name):
                 print(f"❌ Error: Invalid video name '{plot_video_name}' for plotting.")
                 sys.exit(1)
            
            # Use the input folder for the plot
            plot_path = output_folder / f"{plot_video_name}_segment_timeline.png"
            plot_segment_timeline(predictions_df, ground_truth_df, plot_video_name, plot_path)


if __name__ == "__main__":
    # 1. Parse arguments
    parser = argparse.ArgumentParser(description='Segment performance evaluation and plotting utility.')
    parser.add_argument('--input', type=str, required=True, help='Path to the predictions CSV file (e.g., results/interaction_segments.csv).')
    # Updated nargs for plotting argument to handle optional value for plotting all videos
    parser.add_argument('--plot', type=str, nargs='?', const='__ALL__', default=None,
                        help='Video name to plot. If specified without a value, plots all videos found.')
    
    args = parser.parse_args()
    
    # 2. Run evaluation (loads data, runs metrics, prints/saves results)
    predictions_df, ground_truth_df = run_evaluation(Path(args.input))
    
    # 3. Run plotting logic if requested
    main_evaluation_and_plotting(predictions_df, ground_truth_df, args)