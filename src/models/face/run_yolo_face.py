import cv2
import logging
import argparse
import numpy as np
from typing import Tuple
from ultralytics import YOLO
from supervision import Detections
from pathlib import Path
from PIL import Image
from constants import FaceDetection

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

def draw_detections_and_ground_truth(image: np.ndarray, predictions: Detections,
                                     ground_truth_boxes: np.ndarray = None,
                                     ground_truth_classes: np.ndarray = None) -> np.ndarray:
    """Draw predictions and (optionally) ground truth boxes on image"""
    annotated_image = image.copy()
    
    # Draw ground truth boxes in blue (if provided)
    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
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
    # Define input and output directories
    image_dir = Path("/home/nele_pauline_suffo/ProcessedData/face_det_input/images/test")
    
    # Initialize list to store all detection data
    all_detections_data = []
    
    # Load the YOLO model once before the loop
    model = YOLO("/home/nele_pauline_suffo/models/yolov12l-face.pt")
    
    # Get all image file paths in the directory
    # Use glob to handle both .jpg and .PNG extensions
    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.PNG'))
    
    if not image_paths:
        logging.warning(f"No image files found in {image_dir}")
        return 0
    
    logging.info(f"Found {len(image_paths)} images to process.")
    
    for img_path in image_paths:
        try:
            # Extract video_name and frame_id from the filename
            file_name = img_path.stem
            parts = file_name.rsplit('_', 1)
            video_name = parts[0]
            frame_id = int(parts[1])
            
            # Process the image with the YOLO model
            output = model(img_path)
            results = Detections.from_ultralytics(output[0])
            
            # If detections were found, process them
            if len(results.xyxy) > 0:
                for bbox, conf, class_id in zip(results.xyxy, results.confidence, results.class_id):
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Append data to the list
                    all_detections_data.append({
                        'video_name': video_name,
                        'frame_id': frame_id,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
            
        except Exception as e:
            logging.error(f"Failed to process image {img_path}: {e}")
            continue

    # Convert the list to a DataFrame and save to CSV
    if all_detections_data:
        df = pd.DataFrame(all_detections_data)
        
        # Define output directory and path
        output_dir = Path("/home/nele_pauline_suffo/outputs/yolo_face/")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = output_dir / "detections.csv"
        
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Saved all detections to {output_csv_path}")
    else:
        logging.warning("No detections were found to save to CSV.")
    
    return 0

if __name__ == "__main__":
    exit(main())
