import os
import cv2
import logging
import argparse
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from supervision import Detections
from pathlib import Path
from PIL import Image
from constants import FaceDetection, DataPaths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """Process image with YOLO model"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
        
    output = model(Image.open(image_path))
    results = Detections.from_ultralytics(output[0])
    logging.info(f"{len(results.xyxy)} face detection(s)")
    return image, results

def draw_detections_and_ground_truth(image: np.ndarray, predictions: Detections, ground_truth_boxes: np.ndarray, ground_truth_classes: np.ndarray) -> np.ndarray:
    """Draw both predictions and ground truth boxes on image"""
    annotated_image = image.copy()
    
    # Draw ground truth boxes in blue
    for i, (gt_box, gt_class) in enumerate(zip(ground_truth_boxes, ground_truth_classes)):
        x1, y1, x2, y2 = map(int, gt_box)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        gt_label = "GT-Child" if gt_class == 0 else "GT-Adult"
        cv2.putText(annotated_image, gt_label, (x1+10, y1+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Draw prediction boxes in green
    for i, (bbox, conf, class_id) in enumerate(zip(predictions.xyxy, predictions.confidence, predictions.class_id)):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Child" if int(class_id) == 0 else f"Adult"
        cv2.putText(annotated_image, f"{label} {conf:.2f}", (x1+10, y2-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_image

def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    
    return interArea / unionArea if unionArea > 0 else 0

def load_ground_truth(label_path: str, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth bounding boxes and class IDs from label file"""
    ground_truth_boxes = []
    ground_truth_classes = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            ground_truth_boxes.append(np.array([x1, y1, x2, y2]))
            ground_truth_classes.append(int(class_id))
    
    return np.array(ground_truth_boxes), np.array(ground_truth_classes)

def main():
    parser = argparse.ArgumentParser(description='YOLO Face Detection Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Image filename (e.g., quantex_at_home_id261609_2022_04_01_01_000000.jpg)')
    args = parser.parse_args()
    
    # Setup paths for face detection
    output_dir = FaceDetection.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Parse image filename to get folder structure
        basename = os.path.basename(args.image)
        name, ext = os.path.splitext(basename)
        parts = name.split('_')
        video_folder = '_'.join(parts[:-1])
        
        # Setup paths
        image_path = DataPaths.IMAGES_INPUT_DIR / video_folder / args.image
        label_path = FaceDetection.LABELS_INPUT_DIR / Path(args.image).with_suffix('.txt')
        
        # Load model and process image
        model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
        logging.info(f"Model loaded from {FaceDetection.TRAINED_WEIGHTS_PATH}")

        image, results = process_image(model, image_path)
        
        # Get image dimensions and load ground truth
        img_height, img_width = image.shape[:2]
        ground_truth_boxes, ground_truth_classes = load_ground_truth(str(label_path), img_width, img_height)
        logging.info(f"Found {len(ground_truth_boxes)} ground truth boxes")
        
        # Calculate IoU for each detection
        for i, (detected_bbox, conf, class_id) in enumerate(zip(results.xyxy, results.confidence, results.class_id)):
            class_name = "Child" if int(class_id) == 0 else "Adult"
            
            if len(ground_truth_boxes) > 0:
                iou_scores = [calculate_iou(detected_bbox, gt_bbox) for gt_bbox in ground_truth_boxes]
                max_iou = max(iou_scores)
                logging.info(f"Detection {i+1} - {class_name} - Max IoU: {max_iou:.4f}")
            else:
                logging.warning(f"No ground truth boxes found for detection {i+1}")
        
        # Draw detections and save
        annotated_image = draw_detections_and_ground_truth(image, results, ground_truth_boxes, ground_truth_classes)
        output_path = output_dir / Path(args.image).name
        cv2.imwrite(str(output_path), annotated_image)
        logging.info(f"Annotated image saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())