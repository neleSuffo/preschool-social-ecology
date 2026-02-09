import sqlite3
import numpy as np
import json
import pandas as pd

# =========================
# CONFIG
# =========================
GT_DB_PATH = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/annotations.db"
PREDICTIONS_DB_PATH = "/home/nele_pauline_suffo/outputs/quantex_inference/inference.db"
IOU_THRESHOLD = 0.1

VIDEOS_TO_PROCESS = ["quantex_at_home_id255237_2022_05_08_01",
                     "quantex_at_home_id255237_2022_05_08_02",
                     "quantex_at_home_id255237_2022_05_08_04",
                     "quantex_at_home_id257291_2022_03_19_01",
                     "quantex_at_home_id257291_2022_03_19_02",
                     "quantex_at_home_id257291_2022_03_22_01",
                     "quantex_at_home_id257291_2022_03_22_03",
                     "quantex_at_home_id262565_2022_05_08_01", 
                     "quantex_at_home_id262565_2022_05_08_02", 
                     "quantex_at_home_id262565_2022_05_08_03",
                     "quantex_at_home_id262565_2022_05_08_04",
                     "quantex_at_home_id262565_2022_05_26_01"]
# =========================
# HELPERS
# =========================
def extract_frame_id_from_path(frame_path):
    """
    Extract frame ID from frame path by taking the part after the last "_",
    removing trailing zeros and file extension.
    
    Example: "/path/to/video_000123.jpg" -> "123"
    """
    import os
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(frame_path))[0]
    
    # Get the part after the last underscore
    if '_' in filename:
        frame_part = filename.split('_')[-1]
        # Remove leading zeros
        frame_id = str(int(frame_part)) if frame_part.isdigit() else frame_part
        return frame_id
    return filename

def unpack_bbox(bbox_data):
    """
    Unpack bbox data from different possible formats:
    - JSON string: "[x1, y1, x2, y2]"
    - Already unpacked tuple/list
    """
    if isinstance(bbox_data, str):
        try:
            coords = json.loads(bbox_data)
            return tuple(map(int, coords))
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(bbox_data, (tuple, list)) and len(bbox_data) == 4:
        return tuple(map(int, bbox_data))
    return None
# =========================
# DATA EXPORT
# =========================
def save_data_to_csv(db_path_gt, db_path_pred):
    """
    Save ground truth and predictions data to CSV files for inspection.
    """
    import pandas as pd
    
    conn_gt = sqlite3.connect(db_path_gt)
    cursor_gt = conn_gt.cursor()

    conn_pred = sqlite3.connect(db_path_pred)
    cursor_pred = conn_pred.cursor()

    # Save Ground Truth data
    video_filter = "', '".join([video + ".mp4" for video in VIDEOS_TO_PROCESS])
    cursor_gt.execute(f"""
        SELECT a.image_id, v.file_name, a.bbox
        FROM annotations a 
        JOIN videos v ON a.video_id = v.id 
        WHERE a.category_id = 10 AND v.file_name IN ('{video_filter}') AND a.outside = 0
        ORDER BY v.file_name, a.image_id
    """)
    gt_data = cursor_gt.fetchall()
    
    gt_rows = []
    for image_id, video_name, bbox_json in gt_data:
        video_name = video_name.replace('.mp4', '')
        bbox = unpack_bbox(bbox_json)
        if bbox is not None:
            gt_rows.append({
                'video_name': video_name,
                'frame_id': image_id,
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3]
            })
    
    gt_df = pd.DataFrame(gt_rows)
    gt_output_path = '/home/nele_pauline_suffo/outputs/face_retinaface/ground_truth_data.csv'
    gt_df.to_csv(gt_output_path, index=False)
    print(f"Ground truth data saved to '{gt_output_path}' ({len(gt_rows)} bounding boxes)")

    # Save Predictions data
    video_like_conditions = " OR ".join([f"frame_path LIKE '%{video}%'" for video in VIDEOS_TO_PROCESS])
    cursor_pred.execute(f"""
        SELECT frame_path, x1, y1, x2, y2
        FROM RetinaFace 
        WHERE {video_like_conditions}
        ORDER BY frame_path
    """)
    pred_data = cursor_pred.fetchall()
    
    pred_rows = []
    for frame_path, x1, y1, x2, y2 in pred_data:
        frame_id = extract_frame_id_from_path(frame_path)
        video_name = frame_path.split('/')[-2] if '/' in frame_path else 'unknown'
        
        # Only include frames that are multiples of 30 and from specified videos
        if frame_id.isdigit() and int(frame_id) % 30 == 0 and video_name in VIDEOS_TO_PROCESS:
            pred_rows.append({
                'video_name': video_name,
                'frame_id': int(frame_id),
                'x1': int(x1),
                'y1': int(y1),
                'x2': int(x2),
                'y2': int(y2)
            })
    
    pred_df = pd.DataFrame(pred_rows)
    pred_output_path = '/home/nele_pauline_suffo/outputs/face_retinaface/predictions_data.csv'
    pred_df.to_csv(pred_output_path, index=False)
    print(f"Predictions data saved to '{pred_output_path}' ({len(pred_rows)} bounding boxes)")

    conn_gt.close()
    conn_pred.close()
    
    return gt_df, pred_df

# =========================
# EVALUATION
# =========================
def iou_score(boxA, boxB):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    The boxes are expected in the format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def calculate_metrics(gt_df, pred_df, iou_threshold):
    """
    Calculates confusion matrix components, precision, recall, F1 score,
    and generates CSV files for false positive and false negative frames.
    """
    tp = 0
    fp = 0
    fn = 0
    
    # Initialize lists to store false positive and false negative frames
    false_positives_frames = []
    false_negatives_frames = []

    unique_ids = pd.concat([gt_df[['video_name', 'frame_id']], pred_df[['video_name', 'frame_id']]]).drop_duplicates()

    for _, row in unique_ids.iterrows():
        video_name = row['video_name']
        frame_id = row['frame_id']

        gt_boxes = gt_df[(gt_df['video_name'] == video_name) & (gt_df['frame_id'] == frame_id)]
        pred_boxes = pred_df[(pred_df['video_name'] == video_name) & (pred_df['frame_id'] == frame_id)]

        gt_used = np.zeros(len(gt_boxes), dtype=bool)
        pred_used = np.zeros(len(pred_boxes), dtype=bool)

        for i, pred_row in pred_boxes.iterrows():
            best_iou = 0
            best_gt_idx = -1
            pred_box = pred_row[['x1', 'y1', 'x2', 'y2']].values
            
            for j, gt_row in gt_boxes.iterrows():
                gt_box = gt_row[['x1', 'y1', 'x2', 'y2']].values
                current_iou = iou_score(pred_box, gt_box)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and not gt_used[gt_boxes.index.get_loc(best_gt_idx)]:
                tp += 1
                gt_used[gt_boxes.index.get_loc(best_gt_idx)] = True
                pred_used[pred_boxes.index.get_loc(i)] = True

        # Append false positive frames
        if np.sum(~pred_used) > 0:
            false_positives_frames.append({'video_name': video_name, 'frame_id': frame_id})
        
        # Append false negative frames
        if np.sum(~gt_used) > 0:
            false_negatives_frames.append({'video_name': video_name, 'frame_id': frame_id})
        
        # Update FP and FN counts
        fp += np.sum(~pred_used)
        fn += np.sum(~gt_used)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print("\nMetrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    # Create DataFrames and save to CSV
    false_positives_df = pd.DataFrame(false_positives_frames).drop_duplicates()
    false_negatives_df = pd.DataFrame(false_negatives_frames).drop_duplicates()

    false_positives_df.to_csv('/home/nele_pauline_suffo/outputs/face_retinaface/false_positives_frames.csv', index=False)
    false_negatives_df.to_csv('/home/nele_pauline_suffo/outputs/face_retinaface/false_negatives_frames.csv', index=False)

    print(f"Saved false positive frames to {'/home/nele_pauline_suffo/outputs/face_retinaface/false_positives_frames.csv'}")
    print(f"Saved false negative frames to {'/home/nele_pauline_suffo/outputs/face_retinaface/false_negatives_frames.csv'}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # First, save data to CSV files for inspection
    print("Saving ground truth and predictions data to CSV files...")
    gt_df, pred_df = save_data_to_csv(GT_DB_PATH, PREDICTIONS_DB_PATH)
    
    # Then run the evaluation
    print("\n" + "="*50)
    print("Starting evaluation...")
    calculate_metrics(gt_df, pred_df, IOU_THRESHOLD)
