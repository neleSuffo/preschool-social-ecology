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
        DataFrame with 'start', 'end', and 'class' columns.
    ground_truth (pd.DataFrame)
        DataFrame with 'start', 'end', and 'class' columns.
    iou_threshold (float)
        The minimum IoU required to consider a prediction correct.

    Returns
    -------
        dict
            A dictionary with precision, recall, and F1-score for each class.
    """
    classes = ground_truth['interaction_type'].unique()
    results = {}
    
    for cls in classes:
        # Filter predictions and ground truth for the current class
        pred_cls = predictions[predictions['interaction_type'] == cls].copy()
        gt_cls = ground_truth[ground_truth['interaction_type'] == cls].copy()
        
        tp = 0
        fp = 0
        fn = 0
        
        # Mark ground truth segments as 'matched' to avoid double-counting
        gt_cls['matched'] = False
        
        # Loop through each prediction for the current class
        for _, pred_row in pred_cls.iterrows():
            is_matched = False
            
            # Find the best IoU match with ground truth segments
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_row in gt_cls.iterrows():
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
                tp += 1
                gt_cls.loc[best_gt_idx, 'matched'] = True
            else:
                fp += 1
        
        # Count false negatives (unmatched ground truth segments)
        fn = (~gt_cls['matched']).sum()
        
        # Calculate precision, recall, and F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results[cls] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
        
    return results


def main(predictions_df, ground_truth_df, iou):
    evaluation_results = evaluate_performance(predictions_df, ground_truth_df, iou)

    # Print the results
    print("Evaluation Results:")
    for cls, metrics in evaluation_results.items():
        print(f"\nClass: {cls}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    # load data
    predictions_df = pd.read_csv(Inference.INTERACTION_SEGMENTS_CSV)
    ground_truth_df = pd.read_csv(Inference.GROUND_TRUTH_SEGMENTS_CSV)  
    
    main(predictions_df, ground_truth_df, InferenceConfig.EVALUATION_IOU)