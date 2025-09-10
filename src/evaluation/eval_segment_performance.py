import pandas as pd
from constants import Inference
from config import InferenceConfig

def calculate_iou(pred_start, pred_end, gt_start, gt_end):
    """
    Calculates the Intersection over Union (IoU) for two time intervals.
    """
    # Find the start and end of the intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    # Calculate the duration of the intersection
    intersection_duration = max(0, intersection_end - intersection_start)
    
    # Calculate the duration of the union
    union_duration = (pred_end - pred_start) + (gt_end - gt_start) - intersection_duration
    
    if union_duration == 0:
        return 0
    return intersection_duration / union_duration

def evaluate_performance(predictions, ground_truth, iou_threshold):
    """
    Evaluates model performance using IoU and F1-score for each class.
    
    Parameters
    ----------
    predictions (pd.DataFrame)
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    ground_truth (pd.DataFrame)
        DataFrame with 'video_name', 'start_time_sec', 'end_time_sec', and 'interaction_type' columns.
    iou_threshold (float)
        The minimum IoU required to consider a prediction correct.

    Returns
    -------
        dict
            A dictionary with precision, recall, and F1-score for each class.
    """
    classes = ground_truth['interaction_type'].unique()
    videos = ground_truth['video_name'].unique()  # Only evaluate videos with ground truth
    results = {}
    
    for cls in classes:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        
        # Loop through each video
        for video in videos:
            # Filter predictions and ground truth for the current class and video
            pred_cls_video = predictions[
                (predictions['interaction_type'] == cls) & 
                (predictions['video_name'] == video)
            ].copy()
            gt_cls_video = ground_truth[
                (ground_truth['interaction_type'] == cls) & 
                (ground_truth['video_name'] == video)
            ].copy()
            
            if len(gt_cls_video) == 0 and len(pred_cls_video) == 0:
                continue  # Skip videos with no segments for this class
                
            # Mark ground truth segments as 'matched' to avoid double-counting
            gt_cls_video['matched'] = False
            
            # Loop through each prediction for the current class and video
            for _, pred_row in pred_cls_video.iterrows():
                # Find the best IoU match with ground truth segments in the same video
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_row in gt_cls_video.iterrows():
                    # Skip already matched ground truth segments
                    if gt_row['matched']:
                        continue

                    iou = calculate_iou(pred_row['start_time_sec'], pred_row['end_time_sec'], 
                                        gt_row['start_time_sec'], gt_row['end_time_sec'])

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if the best match meets the IoU threshold
                if best_iou >= iou_threshold:
                    tp_total += 1
                    gt_cls_video.loc[best_gt_idx, 'matched'] = True
                else:
                    fp_total += 1
            
            # Count false negatives (unmatched ground truth segments) for this video
            fn_total += (~gt_cls_video['matched']).sum()
        
        # Calculate precision, recall, and F1-score for the class across all videos
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[cls] = {
            'precision': precision, 
            'recall': recall, 
            'f1_score': f1_score,
            'tp': tp_total,
            'fp': fp_total,
            'fn': fn_total
        }
        
    return results


def main(predictions_df, ground_truth_df, iou):
    print(f"Evaluating performance with IoU threshold: {iou}")
    pred_videos = set(predictions_df['video_name'].unique())
    gt_videos = set(ground_truth_df['video_name'].unique())
    
    print(f"Videos in predictions: {len(pred_videos)}")
    print(f"Videos in ground truth: {len(gt_videos)}")
    print(f"Videos being evaluated (have ground truth): {len(gt_videos)}")
    
    # Show videos with predictions but no ground truth (won't be evaluated)
    pred_only_videos = pred_videos - gt_videos
    if pred_only_videos:
        print(f"Videos with predictions but no ground truth (skipped): {len(pred_only_videos)}")
        
    # Show videos with ground truth but no predictions (will contribute to FN)
    gt_only_videos = gt_videos - pred_videos
    if gt_only_videos:
        print(f"Videos with ground truth but no predictions: {len(gt_only_videos)}")
    
    evaluation_results = evaluate_performance(predictions_df, ground_truth_df, iou)

    # Print the results
    print("\nEvaluation Results:")
    for cls, metrics in evaluation_results.items():
        print(f"\nClass: {cls}")
        print(f"  True Positives (TP): {metrics['tp']}")
        print(f"  False Positives (FP): {metrics['fp']}")
        print(f"  False Negatives (FN): {metrics['fn']}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    # load data
    predictions_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV, delimiter=';')

    main(predictions_df, ground_truth_df, InferenceConfig.EVALUATION_IOU)