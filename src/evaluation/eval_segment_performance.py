import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import Inference
from collections import defaultdict
#from sklearn.metrics import precision_recall_fscore_support, classification_report

def create_frame_level_labels(segments_df, video_duration_frames, fps=30):
    """
    Create a frame-by-frame label array for a video based on segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments for a single video, containing 'start_time_sec', 'end_time_sec', 'interaction_type'
    video_duration_frames : int
        Total number of frames in the video
    fps : int
        Frames per second (default: 30)
        
    Returns
    -------
    np.array
        Array where each index represents a frame and contains the interaction type label
    """
    # Initialize all frames as None (unclassified)
    labels = np.full(video_duration_frames, None, dtype=object)
    
    # Fill in the labels based on segments
    for _, segment in segments_df.iterrows():
        start_frame = int(segment['start_time_sec'] * fps)
        end_frame = int(segment['end_time_sec'] * fps)
        interaction_type = str(segment['interaction_type'])
        
        # Ensure we don't go beyond video duration
        end_frame = min(end_frame, video_duration_frames - 1)
        
        # Label each frame in this segment
        labels[start_frame:end_frame + 1] = interaction_type
    
    return labels

def evaluate_performance_by_frames(predictions_df, ground_truth_df, fps=30):
    """
    Evaluates model performance by comparing frame-by-frame classifications.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    ground_truth_df : pd.DataFrame
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    fps : int
        Frames per second (default: 30)

    Returns
    -------
    dict
        A dictionary with overall accuracy and per-category accuracy metrics.
    """
    # Get all videos that have ground truth and predictions
    videos_with_gt = set(ground_truth_df['video_name'].unique())
    videos_with_pred = set(predictions_df['video_name'].unique())
    
    # Only evaluate videos that have BOTH ground truth AND predictions
    videos_to_evaluate = videos_with_gt.intersection(videos_with_pred)
    
    # Get all interaction types from ground truth (normalize to lowercase)
    interaction_types = [str(t).lower() for t in ground_truth_df['interaction_type'].unique()]
    
    # Initialize counters for overall and per-category accuracy
    total_frames_all = 0
    correct_frames_all = 0
    
    category_stats = {category: {'total': 0, 'correct': 0} for category in interaction_types}
    
    # Initialize confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    video_results = []
    
    for video in videos_to_evaluate:
        # Get predictions and ground truth for this video
        pred_video = predictions_df[predictions_df['video_name'] == video].copy()
        gt_video = ground_truth_df[ground_truth_df['video_name'] == video].copy()

        # Since we're only processing videos with both GT and predictions, this should never happen
        if len(gt_video) == 0 or len(pred_video) == 0:
            continue

        # Normalize interaction types to lowercase for case-insensitive comparison
        pred_video['interaction_type'] = pred_video['interaction_type'].astype(str).str.lower()
        gt_video['interaction_type'] = gt_video['interaction_type'].astype(str).str.lower()

        # Normalize label variations
        pred_video['interaction_type'] = pred_video['interaction_type'].replace('co-present', 'available')

        # Ensure time columns are numeric (convert strings to float if needed)
        pred_video['start_time_sec'] = pd.to_numeric(pred_video['start_time_sec'], errors='coerce')
        pred_video['end_time_sec'] = pd.to_numeric(pred_video['end_time_sec'], errors='coerce')
        
        # Convert MM:SS format to seconds for ground truth
        def time_to_seconds(time_str):
            """Convert MM:SS format to seconds"""
            try:
                parts = str(time_str).split(':')
                if len(parts) == 2:
                    minutes, seconds = map(float, parts)
                    return minutes * 60 + seconds
                else:
                    # If it's already in seconds format, just convert to float
                    return float(time_str)
            except:
                return None
        
        gt_video['start_time_sec'] = gt_video['start_time_sec'].apply(time_to_seconds)
        gt_video['end_time_sec'] = gt_video['end_time_sec'].apply(time_to_seconds)
        
        # Drop any rows with NaN time values
        pred_video = pred_video.dropna(subset=['start_time_sec', 'end_time_sec'])
        gt_video = gt_video.dropna(subset=['start_time_sec', 'end_time_sec'])
        
        # Check again after cleaning
        if len(gt_video) == 0 or len(pred_video) == 0:
            print("Warning: GT or PRED video is empty after cleaning")
            continue
            
        # Determine video duration in frames from ground truth
        max_end_gt = gt_video['end_time_sec'].max()
        video_duration_frames = int(max_end_gt * fps) + 1
        
        # Create frame-by-frame labels
        pred_labels = create_frame_level_labels(pred_video, video_duration_frames, fps)
        gt_labels = create_frame_level_labels(gt_video, video_duration_frames, fps)

        # Compare predictions with ground truth frame by frame
        video_total_frames = 0
        video_correct_frames = 0
        
        for frame in range(video_duration_frames):
            gt_label = gt_labels[frame]
            pred_label = pred_labels[frame]
            
            # Only count frames that have ground truth labels
            if gt_label is not None:
                video_total_frames += 1
                total_frames_all += 1
                
                # Update category-specific counters
                category_stats[gt_label]['total'] += 1
                
                # Update confusion matrix (only if prediction is not None/unclassified)
                if pred_label is not None:
                    confusion_matrix[gt_label][pred_label] += 1
                
                # Check if prediction matches ground truth
                if pred_label == gt_label:
                    video_correct_frames += 1
                    correct_frames_all += 1
                    category_stats[gt_label]['correct'] += 1
        
        # Store video-level results
        video_accuracy = video_correct_frames / video_total_frames if video_total_frames > 0 else 0
        video_results.append({
            'video_name': video,
            'total_frames': video_total_frames,
            'correct_frames': video_correct_frames,
            'accuracy': video_accuracy
        })
    
    # Calculate overall accuracy
    overall_accuracy = correct_frames_all / total_frames_all if total_frames_all > 0 else 0
    
    # Calculate category-specific accuracies
    category_accuracies = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracies[category] = {
            'accuracy': accuracy,
            'total_frames': stats['total'],
            'correct_frames': stats['correct']
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_frames': total_frames_all,
        'correct_frames': correct_frames_all,
        'category_accuracies': category_accuracies,
        'video_results': video_results,
        'confusion_matrix': confusion_matrix,
        'interaction_types': interaction_types,
        'fps': fps
    }


def calculate_detailed_metrics(results):
    """
    Calculate precision, recall, and F1-score for each class from confusion matrix.
    
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
        if class_name not in confusion_matrix:
            continue
            
        # Calculate True Positives, False Positives, False Negatives
        tp = confusion_matrix[class_name].get(class_name, 0)  # Correctly predicted as this class
        
        # False Positives: other classes predicted as this class
        fp = 0
        for gt_class in confusion_matrix:
            if gt_class != class_name:
                fp += confusion_matrix[gt_class].get(class_name, 0)
        
        # False Negatives: this class predicted as other classes
        fn = 0
        for pred_class in confusion_matrix[class_name]:
            if pred_class != class_name:
                fn += confusion_matrix[class_name][pred_class]
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Total actual instances of this class
        total_actual = tp + fn
        
        detailed_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': total_actual,  # Number of actual instances
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    # Calculate macro averages (unweighted average across classes)
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

def generate_confusion_matrix_plots(results):
    """
    Generate and save confusion matrix plots.
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
    
    # Get all predicted labels and sort them in preferred order (exclude unclassified)
    all_pred_labels = set()
    for gt_label in confusion_matrix:
        for pred_label in confusion_matrix[gt_label]:
            if pred_label != 'unclassified':  # Skip unclassified labels
                all_pred_labels.add(pred_label)
    
    sorted_pred_labels = []
    for label in preferred_order:
        if label in all_pred_labels:
            sorted_pred_labels.append(label)
    # Add any remaining predicted labels (excluding unclassified)
    for label in sorted(all_pred_labels):
        if label not in sorted_pred_labels:
            sorted_pred_labels.append(label)
    
    # Reverse the ground truth order to align diagonal properly
    sorted_gt_labels = sorted_gt_labels[::-1]
    
    # Create confusion matrix as numpy array
    matrix_data = []
    for gt_label in sorted_gt_labels:
        row = []
        for pred_label in sorted_pred_labels:
            row.append(confusion_matrix[gt_label][pred_label])
        matrix_data.append(row)
    
    matrix_array = np.array(matrix_data)
    
    # Create absolute count heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_array, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=[label.capitalize() for label in sorted_pred_labels],
                yticklabels=[label.capitalize() for label in sorted_gt_labels],
                cbar_kws={'label': 'Number of Frames'})
    
    plt.title('Confusion Matrix - Frame-by-Frame Classification (Counts)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Labels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save count matrix
    plt.savefig(Inference.CONF_MATRIX_COUNTS, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create percentage matrix (row-normalized)
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
    
    plt.title('Confusion Matrix - Frame-by-Frame Classification (Percentages)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Labels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save percentage matrix
    plt.savefig(Inference.CONF_MATRIX_PERCENTAGES, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {Inference.CONF_MATRIX_PERCENTAGES}")

def save_performance_results(results, detailed_metrics, total_frames, total_hours, filename=Inference.PERFORMANCE_RESULTS_TXT):
    """
    Save category-specific performance results with detailed metrics to a text file.
    """
    fps = results.get('fps', 30)
    total_seconds = total_frames / fps
    
    with open(filename, 'w') as f:
        # Write analysis summary
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total frames analyzed: {total_frames:,}\n")
        f.write(f"Total seconds analyzed: {total_seconds:,.0f}\n")
        f.write(f"Total time analyzed: {total_hours:.2f} hours ({int(total_hours)}h {int((total_seconds % 3600) / 60)}m)\n")
        
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
            f.write(f"  Total frames: {stats['total_frames']:,}\n")
            f.write(f"  Correctly classified: {stats['correct_frames']:,}\n")
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
        f.write("Accuracy:   Overall percentage of frames classified correctly\n")
        f.write("Precision:  Of frames predicted as this class, how many were correct?\n")
        f.write("Recall:     Of actual frames of this class, how many were detected?\n")
        f.write("F1-Score:   Harmonic mean of precision and recall (balanced metric)\n\n")
        f.write("Macro Avg:    Unweighted average across classes (treats all classes equally)\n")
    
    print(f"✅ Performance results saved: {filename}")


def main(predictions_df, ground_truth_df, fps=30):
    """
    Main function to evaluate performance using frame-by-frame accuracy.
    """
    print("Evaluating performance using frame-by-frame accuracy...")
    
    # only keep columns "video_name", "start_time_sec", "end_time_sec", "interaction_type" from gt dataframe
    ground_truth_df = ground_truth_df[['video_name', 'start_time_sec', 'end_time_sec', 'interaction_type']]
    # remove rows with NaN in any of these columns
    ground_truth_df = ground_truth_df.dropna(subset=['video_name', 'start_time_sec', 'end_time_sec', 'interaction_type'])
    
    pred_videos = set(predictions_df['video_name'].unique())
    gt_videos = set(ground_truth_df['video_name'].unique())
    videos_to_evaluate = pred_videos.intersection(gt_videos)
        
    print(f"Videos being evaluated (have both GT and predictions): {len(videos_to_evaluate)}")
    
    # Evaluate performance
    results = evaluate_performance_by_frames(predictions_df, ground_truth_df, fps)
    
    # Calculate total frames and hours analyzed
    total_frames = results['total_frames']
    total_hours = total_frames / (fps * 3600)
    
    # Calculate detailed metrics (precision, recall, F1)
    detailed_metrics = calculate_detailed_metrics(results)

    # Generate confusion matrix plots
    generate_confusion_matrix_plots(results)
    
    # Save performance results to text file with detailed metrics
    save_performance_results(results, detailed_metrics, total_frames, total_hours)
    
    return results

if __name__ == "__main__":
    # load data
    predictions_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')

    results = main(predictions_df, ground_truth_df)