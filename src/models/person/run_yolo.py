import argparse
import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from supervision import Detections
from ultralytics import YOLO

from constants import PersonDetection

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Unified visual styling constants
BOX_THICKNESS = 10
BACKING_OFFSET = 6
PAD = 10
TEXT_THICKNESS = 6


def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """Process image with YOLO model and return raw image and detections."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")

    output = model(Image.open(image_path))
    results = Detections.from_ultralytics(output[0])
    logging.info(f"{len(results.xyxy)} person detection(s)")
    return image, results


def draw_dashed_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 10,
    gap_length: int = 14,
):
    """Draw a dashed rectangle using line segments."""
    x1, y1 = pt1
    x2, y2 = pt2

    lines = [
        ((x1, y1), (x2, y1)),  # Top
        ((x2, y1), (x2, y2)),  # Right
        ((x2, y2), (x1, y2)),  # Bottom
        ((x1, y2), (x1, y1)),  # Left
    ]

    for start, end in lines:
        dist = np.hypot(end[0] - start[0], end[1] - start[1])
        if dist == 0:
            continue
        dashes = int(dist / (dash_length + gap_length))
        for i in range(dashes + 1):
            s = i * (dash_length + gap_length)
            e = s + dash_length
            if s >= dist:
                break
            e = min(e, dist)

            p1 = (
                int(start[0] + (end[0] - start[0]) * (s / dist)),
                int(start[1] + (end[1] - start[1]) * (s / dist)),
            )
            p2 = (
                int(start[0] + (end[0] - start[0]) * (e / dist)),
                int(start[1] + (end[1] - start[1]) * (e / dist)),
            )
            cv2.line(img, p1, p2, color, thickness)


def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0


def load_ground_truth(
    label_path: str, img_width: int, img_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load YOLO format ground truth bounding boxes and class IDs."""
    ground_truth_boxes = []
    ground_truth_classes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            line_vals = line.strip().split()
            if len(line_vals) < 5:
                continue
            class_id, x_center, y_center, width, height = map(float, line_vals[:5])

            x1 = int((x_center - width / 2.0) * img_width)
            y1 = int((y_center - height / 2.0) * img_height)
            x2 = int((x_center + width / 2.0) * img_width)
            y2 = int((y_center + height / 2.0) * img_height)

            ground_truth_boxes.append(np.array([x1, y1, x2, y2]))
            ground_truth_classes.append(int(class_id))

    return np.array(ground_truth_boxes), np.array(ground_truth_classes)


def draw_detections_and_ground_truth(
    image: np.ndarray,
    predictions: Detections,
    ground_truth_boxes: np.ndarray = None,
    ground_truth_classes: np.ndarray = None,
    iou_threshold: float = 0.3,
) -> np.ndarray:
    """Draw matched GT, TP, FP, and FN annotations for person detections."""
    annotated_image = image.copy()
    img_w = annotated_image.shape[1]

    # Unified BGR Color Palette
    COLOR_TP = (40, 160, 40)   # Vivid Green
    COLOR_GT = (180, 140, 0)   # Cyan / Teal
    COLOR_ERR = (0, 70, 230)   # Red-Orange

    font = cv2.FONT_HERSHEY_SIMPLEX
    default_font_scale = 2.5

    # -------------------------------------------------------------
    # 1. GROUND TRUTH (Missed GT = FN -> DASHED BOX)
    # -------------------------------------------------------------
    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for gt_box in ground_truth_boxes:
            x1, y1, x2, y2 = map(int, gt_box)

            max_iou = 0.0
            if predictions is not None and len(predictions.xyxy) > 0:
                iou_scores = [
                    calculate_iou(gt_box, pred_box)
                    for pred_box in predictions.xyxy
                ]
                max_iou = max(iou_scores) if iou_scores else 0.0

            is_fn = max_iou < iou_threshold

            if is_fn:
                label = "FN: GT Person"
                box_color_gt = COLOR_ERR
                # White backing outline
                draw_dashed_rect(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    thickness=BOX_THICKNESS + BACKING_OFFSET,
                    dash_length=10,
                    gap_length=14,
                )
                # Dashed red-orange box
                draw_dashed_rect(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    box_color_gt,
                    thickness=BOX_THICKNESS,
                    dash_length=10,
                    gap_length=14,
                )
            else:
                label = "GT Person"
                box_color_gt = COLOR_GT
                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    BOX_THICKNESS + BACKING_OFFSET,
                )
                cv2.rectangle(
                    annotated_image,
                    (x1, y1),
                    (x2, y2),
                    box_color_gt,
                    BOX_THICKNESS,
                )

            # Badge Text Positioning
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, default_font_scale, TEXT_THICKNESS
            )
            text_x = min(x1 + 4, img_w - text_w - PAD - 6)
            #text_y = y1 - baseline - PAD - BOX_THICKNESS


            # # Flip badge inside if it exceeds the top border
            # if text_y - text_h - PAD < 0:
            #     text_y = y1 + text_h + PAD + BOX_THICKNESS + 4
            img_h, img_w = annotated_image.shape[:2]
            if is_fn:
                # Place FN label below the box (below y2)
                text_y = y2 + text_h + PAD + BOX_THICKNESS + 4

                # Flip inside if hitting the bottom frame
                if text_y + baseline + PAD > img_h:
                    text_y = y2 - baseline - PAD - BOX_THICKNESS - 4
            else:
                # Matched GT remains above the box (above y1)
                text_y = y1 - baseline - PAD - BOX_THICKNESS

                # Flip inside if hitting the top frame
                if text_y - text_h - PAD < 0:
                    text_y = y1 + text_h + PAD + BOX_THICKNESS + 4

            bg_pt1 = (text_x - PAD, text_y - text_h - PAD)
            bg_pt2 = (text_x + text_w + PAD, text_y + baseline + PAD)
            cv2.rectangle(annotated_image, bg_pt1, bg_pt2, (255, 255, 255), -1)
            cv2.putText(
                annotated_image,
                label,
                (text_x, text_y),
                font,
                default_font_scale,
                box_color_gt,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    # -------------------------------------------------------------
    # 2. MODEL PREDICTIONS (Matched = TP -> SOLID GREEN; Unmatched = FP -> SOLID RED)
    # -------------------------------------------------------------
    if predictions is not None and len(predictions.xyxy) > 0:
        for bbox, conf in zip(predictions.xyxy, predictions.confidence):
            x1, y1, x2, y2 = map(int, bbox)

            max_iou = 0.0
            if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
                iou_scores = [
                    calculate_iou(bbox, gt_box) for gt_box in ground_truth_boxes
                ]
                max_iou = max(iou_scores) if iou_scores else 0.0

            is_fp = (
                (ground_truth_boxes is None)
                or (len(ground_truth_boxes) == 0)
                or (max_iou < iou_threshold)
            )

            if is_fp:
                label = f"FP: Person ({conf:.2f})"
                box_color_pred = COLOR_ERR
            else:
                label = f"TP: Person ({conf:.2f})"
                box_color_pred = COLOR_TP

            # Draw white backing then solid prediction box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                BOX_THICKNESS + BACKING_OFFSET,
            )
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                box_color_pred,
                BOX_THICKNESS,
            )

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, default_font_scale, TEXT_THICKNESS
            )
            img_h, img_w = annotated_image.shape[:2]

            # Condition: box is narrow and flush against the left image edge
            box_width = x2 - x1
            is_left_edge_crop = (x1 < 40) and (box_width < text_w)

            if is_left_edge_crop:
                # Place label to the right of the box (outside x2)
                text_x = x2 + BOX_THICKNESS + PAD + 6
                # Vertically center the badge along the box height
                text_y = int((y1 + y2) / 2) + int(text_h / 2)

                # Ensure it doesn't spill past the top or bottom edges of the frame
                text_y = max(text_h + PAD + 4, min(text_y, img_h - baseline - PAD - 4))

            else:
                # Standard horizontal clamp
                text_x = max(PAD + 4, min(x1 + 4, img_w - text_w - PAD - 6))

                # Default placement below the box
                text_y = y2 + text_h + PAD + BOX_THICKNESS + 4

                # Flip inside if hitting the bottom frame
                if text_y + baseline + PAD > img_h:
                    text_y = y2 - baseline - PAD - BOX_THICKNESS - 4

            # Draw white backing and text
            bg_pt1 = (text_x - PAD, text_y - text_h - PAD)
            bg_pt2 = (text_x + text_w + PAD, text_y + baseline + PAD)
            cv2.rectangle(annotated_image, bg_pt1, bg_pt2, (255, 255, 255), -1)
            cv2.putText(
                annotated_image,
                label,
                (text_x, text_y),
                font,
                default_font_scale,
                box_color_pred,
                TEXT_THICKNESS,
                cv2.LINE_AA,
            )

    return annotated_image


def process_and_save(image_path: Path, output_dir: Path, model: YOLO):
    """Run inference, match ground truth, and export annotated output."""
    image, results = process_image(model, image_path)
    img_height, img_width = image.shape[:2]

    label_name = image_path.stem + ".txt"
    label_path = getattr(PersonDetection, "LABELS_INPUT_DIR", Path("labels")) / label_name

    ground_truth_boxes = None
    ground_truth_classes = None

    if label_path.exists():
        ground_truth_boxes, ground_truth_classes = load_ground_truth(
            str(label_path), img_width, img_height
        )
        logging.info(f"Found {len(ground_truth_boxes)} ground truth box(es)")
        for i, detected_bbox in enumerate(results.xyxy):
            iou_scores = [
                calculate_iou(detected_bbox, gt_bbox)
                for gt_bbox in ground_truth_boxes
            ]
            max_iou = max(iou_scores) if iou_scores else 0.0
            logging.info(f"Detection {i+1} - Max IoU: {max_iou:.4f}")
    else:
        logging.warning(
            f"No label file found at {label_path}. Drawing predictions only."
        )

    annotated_image = draw_detections_and_ground_truth(
        image=image,
        predictions=results,
        ground_truth_boxes=ground_truth_boxes,
        ground_truth_classes=ground_truth_classes,
    )

    has_gt = ground_truth_boxes is not None and len(ground_truth_boxes) > 0
    has_pred = len(results.xyxy) > 0

    if has_pred or has_gt:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logging.info(f"Saved annotated image: {output_path}")
    else:
        logging.info(f"Skipped {image_path.name}: No detections and no ground truth.")


def main():
    parser = argparse.ArgumentParser(description="YOLO Person Detection Inference")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Image file or folder"
    )
    args = parser.parse_args()

    model = YOLO(PersonDetection.TRAINED_WEIGHTS_PATH)
    input_path = Path(args.image_path)
    output_dir = PersonDetection.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_path.is_dir():
            folder_output = output_dir / f"{input_path.name}_annotated"
            folder_output.mkdir(parents=True, exist_ok=True)
            image_files = list(input_path.glob("*.jpg")) + list(
                input_path.glob("*.png")
            ) + list(input_path.glob("*.PNG"))
            for img_file in image_files:
                process_and_save(img_file, folder_output, model)
        else:
            process_and_save(input_path, output_dir, model)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())