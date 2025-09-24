import sqlite3
import numpy as np
import json

# =========================
# CONFIG
# =========================
GT_DB_PATH = "/home/nele_pauline_suffo/ProcessedData/quantex_annotations/annotations.db"
PREDICTIONS_DB_PATH = "/home/nele_pauline_suffo/outputs/quantex_inference/inference_short_copy.db"
IOU_THRESHOLD = 0.5

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

def compute_iou(box1, box2):
    """
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter_area = inter_w * inter_h

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def match_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Greedy matching between predictions and ground truth boxes.
    """
    matched_gt = set()
    matched_pred = set()

    for pi, pbox in enumerate(pred_boxes):
        for gi, gtbox in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pbox, gtbox)
            if iou >= iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break

    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# =========================
# EVALUATION
# =========================
def evaluate(db_path_gt, db_path_pred, iou_threshold=0.5):
    conn_gt = sqlite3.connect(db_path_gt)
    cursor_gt = conn_gt.cursor()

    conn_pred = sqlite3.connect(db_path_pred)
    cursor_pred = conn_pred.cursor()

    # Get all frame IDs from annotations with category_id = 10
    cursor_gt.execute("""
        SELECT DISTINCT a.image_id, v.file_name
        FROM annotations a 
        JOIN videos v ON a.video_id = v.id 
        WHERE a.category_id = 10
    """)
    gt_data = cursor_gt.fetchall()
    gt_frame_ids = {str(row[0]): row[1] for row in gt_data}  # frame_id -> video_name mapping

    # Get all frame paths from predictions and extract frame IDs
    cursor_pred.execute("SELECT DISTINCT frame_path FROM RetinaFace")
    pred_frame_paths = cursor_pred.fetchall()
    
    pred_frame_ids = {}
    for (frame_path,) in pred_frame_paths:
        frame_id = extract_frame_id_from_path(frame_path)
        pred_frame_ids[frame_id] = frame_path

    # Find common frame IDs
    common_frame_ids = set(gt_frame_ids.keys()) & set(pred_frame_ids.keys())
    print(f"Found {len(common_frame_ids)} frames with both GT and predictions.")
    print(f"GT frames: {len(gt_frame_ids)}, Pred frames: {len(pred_frame_ids)}")

    total_tp, total_fp, total_fn = 0, 0, 0

    for frame_id in common_frame_ids:
        video_name = gt_frame_ids[frame_id]
        frame_path = pred_frame_ids[frame_id]
        
        # Load GT boxes using image_id
        cursor_gt.execute("""
            SELECT a.bbox
            FROM annotations a 
            JOIN videos v ON a.video_id = v.id 
            WHERE a.image_id = ? AND a.category_id = 10
        """, (int(frame_id),))
        
        gt_bbox_data = cursor_gt.fetchall()
        gt_boxes = []
        
        # Unpack bbox data
        for bbox_row in gt_bbox_data:
            bbox = unpack_bbox(bbox_row[0])
            if bbox is not None:
                gt_boxes.append(bbox)
            else:
                print(f"Warning: Could not unpack bbox data: {bbox_row[0]} for frame_id {frame_id}")

        # Load Pred boxes using frame_path
        cursor_pred.execute("""
            SELECT x1, y1, x2, y2 FROM RetinaFace WHERE frame_path = ?
        """, (frame_path,))
        pred_boxes = cursor_pred.fetchall()

        pred_boxes = [tuple(map(int, b)) for b in pred_boxes]

        tp, fp, fn = match_detections(gt_boxes, pred_boxes, iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    conn_gt.close()
    conn_pred.close()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    print("=== Evaluation Results ===")
    print(f"Frames evaluated: {len(common_frames)}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision@IoU={iou_threshold}: {precision:.4f}")
    print(f"Recall@IoU={iou_threshold}: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    evaluate(GT_DB_PATH, PREDICTIONS_DB_PATH, IOU_THRESHOLD)