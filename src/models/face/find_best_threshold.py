import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/nele_pauline_suffo/outputs/face_detections/yolo12l_20250821_155703/weights/best.pt"             # path to your YOLO12 weights
VAL_IMAGES_DIR = Path("/home/nele_pauline_suffo/ProcessedData/face_det_input/images/test")  # validation images
VAL_LABELS_DIR = Path("/home/nele_pauline_suffo/ProcessedData/face_det_input/labels/test")  # YOLO labels
IOU_THRESHOLD = 0.5                  # IoU threshold for TP
CONF_THRESHOLDS = np.arange(0.1, 0.95, 0.05)

# -----------------------------
# UTILITIES
# -----------------------------
def load_yolo_labels(label_path):
    """Load YOLO labels from .txt file"""
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes

def yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format to absolute xyxy coordinates"""
    _, x_c, y_c, w, h = box
    x_c *= img_width
    y_c *= img_height
    w *= img_width
    h *= img_height
    x_min = x_c - w / 2
    y_min = y_c - h / 2
    x_max = x_c + w / 2
    y_max = y_c + h / 2
    return np.array([x_min, y_min, x_max, y_max])

def calculate_iou(boxA, boxB):
    """IoU between two boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0

def evaluate(pred_boxes, gt_boxes, iou_thres=0.5):
    """Compute TP, FP, FN"""
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        ious = [calculate_iou(pred, gt) for gt in gt_boxes]
        max_iou = max(ious) if ious else 0
        max_idx = np.argmax(ious) if ious else -1
        if max_iou >= iou_thres and max_idx not in matched_gt:
            tp += 1
            matched_gt.add(max_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# -----------------------------
# MAIN
# -----------------------------
def main():
    model = YOLO(MODEL_PATH)
    best_f1 = 0
    best_thresh = 0

    image_files = list(VAL_IMAGES_DIR.glob("*.*"))  # all images

    output_file = Path("yolo12_threshold_eval.txt")
    with open(output_file, "w") as f:  # open file for writing
        f.write("Conf\tPrecision\tRecall\tF1\n")  # header

        for conf_thresh in CONF_THRESHOLDS:
            total_tp, total_fp, total_fn = 0, 0, 0

            for img_path in image_files:
                label_path = VAL_LABELS_DIR / (img_path.stem + ".txt")
                if not label_path.exists():
                    continue

                gt_labels = load_yolo_labels(label_path)
                img = cv2.imread(str(img_path))
                img_height, img_width = img.shape[:2]

                gt_boxes = np.array([yolo_to_xyxy(box, img_width, img_height) for box in gt_labels])

                # Run model
                results = model(img, conf=conf_thresh)[0]
                if results.boxes is None or len(results.boxes) == 0:
                    pred_boxes = []
                else:
                    pred_boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]

                tp, fp, fn = evaluate(pred_boxes, gt_boxes, IOU_THRESHOLD)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            # Compute metrics
            precision = total_tp / (total_tp + total_fp + 1e-8)
            recall = total_tp / (total_tp + total_fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            f.write(f"{conf_thresh:.2f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\n")

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = conf_thresh

        f.write(f"\nBest confidence threshold: {best_thresh:.2f} with F1={best_f1:.4f}\n")

    print(f"Evaluation results saved to {output_file}")

if __name__ == "__main__":
    main()