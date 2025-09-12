import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import Inference
from collections import defaultdict

def create_second_level_labels(segments_df, video_duration_seconds):
    """
    Create a second-by-second label array for a video based on segments.
    
    Parameters
    ----------
    segments_df : pd.DataFrame
        DataFrame with segments for a single video, containing 'start_time_sec', 'end_time_sec', 'interaction_type'
    video_duration_seconds : int
        Total duration of the video in seconds
        
    Returns
    -------
    np.array
        Array where each index represents a second and contains the interaction type label
    """
    # Initialize all seconds as None (unclassified)
    labels = np.full(video_duration_seconds, None, dtype=object)
    
    # Fill in the labels based on segments
    for _, segment in segments_df.iterrows():
        start_sec = int(segment['start_time_sec'])
        end_sec = int(segment['end_time_sec'])
        interaction_type = str(segment['interaction_type'])
        
        # Ensure we don't go beyond video duration
        end_sec = min(end_sec, video_duration_seconds - 1)
        
        # Label each second in this segment
        labels[start_sec:end_sec + 1] = interaction_type
    
    return labels

def evaluate_performance_by_seconds(predictions_df, ground_truth_df):
    """
    Evaluates model performance by comparing second-by-second classifications.
    
    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    ground_truth_df : pd.DataFrame
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.

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
    total_seconds_all = 0
    correct_seconds_all = 0
    
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
        
        # Convert HH:MM:SS format to seconds for ground truth
        def time_to_seconds(time_str):
            """Convert HH:MM:SS format to seconds"""
            try:
                parts = str(time_str).split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = map(float, parts)
                    return hours * 3600 + minutes * 60 + seconds
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
            
        # Determine video duration from ground truth only
        max_end_gt = gt_video['end_time_sec'].max()
        video_duration = int(max_end_gt) + 1
        
        # Create second-by-second labels
        pred_labels = create_second_level_labels(pred_video, video_duration)
        gt_labels = create_second_level_labels(gt_video, video_duration)

        # Compare predictions with ground truth second by second
        video_total_seconds = 0
        video_correct_seconds = 0
        
        for second in range(video_duration):
            gt_label = gt_labels[second]
            pred_label = pred_labels[second]
            
            # Only count seconds that have ground truth labels
            if gt_label is not None:
                video_total_seconds += 1
                total_seconds_all += 1
                
                # Update category-specific counters
                category_stats[gt_label]['total'] += 1
                
                # Update confusion matrix
                pred_label_for_matrix = pred_label if pred_label is not None else 'unclassified'
                confusion_matrix[gt_label][pred_label_for_matrix] += 1
                
                # Check if prediction matches ground truth
                if pred_label == gt_label:
                    video_correct_seconds += 1
                    correct_seconds_all += 1
                    category_stats[gt_label]['correct'] += 1
        
        # Store video-level results
        video_accuracy = video_correct_seconds / video_total_seconds if video_total_seconds > 0 else 0
        video_results.append({
            'video_name': video,
            'total_seconds': video_total_seconds,
            'correct_seconds': video_correct_seconds,
            'accuracy': video_accuracy
        })
    
    # Calculate overall accuracy
    overall_accuracy = correct_seconds_all / total_seconds_all if total_seconds_all > 0 else 0
    
    # Calculate category-specific accuracies
    category_accuracies = {}
    for category, stats in category_stats.items():
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        category_accuracies[category] = {
            'accuracy': accuracy,
            'total_seconds': stats['total'],
            'correct_seconds': stats['correct']
        }
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_seconds': total_seconds_all,
        'correct_seconds': correct_seconds_all,
        'category_accuracies': category_accuracies,
        'video_results': video_results,
        'confusion_matrix': confusion_matrix,
        'interaction_types': interaction_types
    }


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
    
    # Get all predicted labels and sort them in preferred order
    all_pred_labels = set()
    for gt_label in confusion_matrix:
        for pred_label in confusion_matrix[gt_label]:
            all_pred_labels.add(pred_label)
    
    sorted_pred_labels = []
    for label in preferred_order:
        if label in all_pred_labels:
            sorted_pred_labels.append(label)
    # Add any remaining predicted labels (like 'unclassified')
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
                cbar_kws={'label': 'Number of Seconds'})
    
    plt.title('Confusion Matrix - Second-by-Second Classification (Counts)', fontsize=16, fontweight='bold')
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
    
    plt.title('Confusion Matrix - Second-by-Second Classification (Percentages)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Labels', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Labels', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Save percentage matrix
    plt.savefig(Inference.CONF_MATRIX_PERCENTAGES, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved: {Inference.CONF_MATRIX_PERCENTAGES}")

def save_performance_results(results, total_seconds, total_hours, filename=Inference.PERFORMANCE_RESULTS_TXT):
    """
    Save category-specific performance results to a text file.
    """
    with open(filename, 'w') as f:
        # Write analysis summary
        f.write("ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total seconds analyzed: {total_seconds:,}\n")
        f.write(f"Total time analyzed: {total_hours:.2f} hours ({int(total_hours)}h {int((total_seconds % 3600) / 60)}m)\n")
        f.write(f"Overall accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)\n\n")
        
        # Write category-specific performance
        f.write("CATEGORY-SPECIFIC PERFORMANCE\n")
        f.write("=" * 60 + "\n\n")
        
        for category, stats in results['category_accuracies'].items():
            f.write(f"{category.upper()}:\n")
            f.write(f"  Total seconds: {stats['total_seconds']:,}\n")
            f.write(f"  Correctly classified: {stats['correct_seconds']:,}\n")
            f.write(f"  Accuracy: {stats['accuracy']:.4f} ({stats['accuracy']*100:.2f}%)\n\n")
    
    print(f"✅ Performance results saved: {filename}")


def main(predictions_df, ground_truth_df):
    """
    Main function to evaluate performance using second-by-second accuracy.
    """
    print("Evaluating performance using second-by-second accuracy...")
    
    pred_videos = set(predictions_df['video_name'].unique())
    gt_videos = set(ground_truth_df['video_name'].unique())
    videos_to_evaluate = pred_videos.intersection(gt_videos)
        
    print(f"Videos being evaluated (have both GT and predictions): {len(videos_to_evaluate)}")
    
    # Evaluate performance
    results = evaluate_performance_by_seconds(predictions_df, ground_truth_df)
    
    # Calculate total frames and hours analyzed
    total_seconds = results['total_seconds']
    total_hours = total_seconds / 3600
    
    # Generate confusion matrix plots
    generate_confusion_matrix_plots(results)
    
    # Save performance results to text file
    save_performance_results(results, total_seconds, total_hours)
    
    return results

if __name__ == "__main__":
    # load data
    predictions_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')

    results = main(predictions_df, ground_truth_df)