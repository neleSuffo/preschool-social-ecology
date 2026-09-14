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
from models.proximity.estimate_proximity import calculate_proximity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
def process_image(model: YOLO, image_path: Path) -> Tuple[np.ndarray, Detections]:
    """Process image with YOLO model
    
    Parameters
    ----------
    model: YOLO
        The YOLO model to use for inference.
    image_path: Path
        Path to the input image file.
    
    Returns
    -------
    Tuple[np.ndarray, Detections]
        The original image as a NumPy array and the detections as a Detections object.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
        
    output = model(Image.open(image_path))
    results = Detections.from_ultralytics(output[0])
    logging.info(f"{len(results.xyxy)} face detection(s)")
    return image, results

def draw_detections_and_ground_truth(image: np.ndarray, 
                                     predictions: Detections,
                                     proximities: list = None,
                                     ground_truth_boxes: np.ndarray = None,
                                     ground_truth_classes: np.ndarray = None) -> np.ndarray:
    """Draw predictions and (optionally) ground truth boxes on image
    
    Parameters
    ----------
    image: np.ndarray
        The original image as a NumPy array.
    predictions: Detections
        The detected bounding boxes, confidence scores, and class IDs.
    ground_truth_boxes: np.ndarray, optional
        Ground truth bounding boxes in the format [x1, y1, x2, y2].
    ground_truth_classes: np.ndarray, optional
        Class IDs corresponding to the ground truth boxes.
        
    Returns
    -------
    np.ndarray
        The annotated image with predictions and ground truth boxes drawn.
    """
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
        #label = f"Child" if int(class_id) == 0 else f"Adult"
        label = f"Face ({conf:.2f})"
        prox_val = None
        if proximities is not None and i < len(proximities):
            prox_val = proximities[i][0] if isinstance(proximities[i], (list, tuple, np.ndarray)) else proximities[i]
            label += f", proximity: {prox_val:.2f}"

        x1, y1, x2, y2 = map(int, bbox)
        box_color = (51, 22, 105)   # #691633 in BGR
        outer_thickness = 2 if (prox_val is not None and prox_val < 0.2) else 6
        inner_thickness = 1 if (prox_val is not None and prox_val < 0.2) else 2

        # 1. Outer thicker white bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 255), outer_thickness)
        # 2. Inner/top thinner red bounding box (BGR: (0, 0, 255))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), box_color, inner_thickness)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1 if (prox_val is not None and prox_val < 0.2) else 1.5
        thickness = 2 if (prox_val is not None and prox_val < 0.2) else 3
        text_color = box_color
        white_color = (255, 255, 255)
        pad = 6

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        text_x = x1 + 5
        text_y = y2 + text_h + pad + 10

        # Draw solid white rectangle behind text with padding
        bg_pt1 = (text_x - pad, text_y - text_h - pad)
        bg_pt2 = (text_x + text_w + pad, text_y + baseline + pad)
        cv2.rectangle(annotated_image, bg_pt1, bg_pt2, white_color, -1)

        # Draw the text on top
        cv2.putText(
            annotated_image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )  
    return annotated_image

def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes
    
    Parameters
    ----------
    boxA: np.ndarray
        Bounding box A in the format [x1, y1, x2, y2].
    boxB: np.ndarray
        Bounding box B in the format [x1, y1, x2, y2].
        
    Returns
    -------
    float
        The IoU score between the two boxes.
    """
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    
    return interArea / unionArea if unionArea > 0 else 0

def load_ground_truth(label_path: str, img_width: int, img_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground truth bounding boxes and class IDs from label file
    
    Parameters    
    ----------
    label_path: str
        Path to the label file.
    img_width: int
        Width of the image.
    img_height: int
        Height of the image.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the ground truth bounding boxes and class IDs.
    """
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

def process_and_save(image_path, output_dir, cut_face, filter_proximity):
    """
    Process the image for face detection and save the results.
    The function can either save cropped face images or an annotated image
    with detected faces, depending on the `cut_face` parameter.
    
    Parameters:
    -----------
    image_path : Path
        Path to the input image file.
    output_dir : Path
        Directory to save the output (annotated image or face crops).
    cut_face : bool
        If True, save each detected face as a PNG in the output directory.
    filter_proximity : bool
        If True, only save faces/images with proximity > 0.3.
    """
    label_name = image_path.stem + ".txt"
    label_path = FaceDetection.LABELS_INPUT_DIR / label_name
    model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    image, results = process_image(model, image_path)
    img_height, img_width = image.shape[:2]
    ground_truth_boxes = None
    ground_truth_classes = None
    if label_path.exists():
        ground_truth_boxes, ground_truth_classes = load_ground_truth(str(label_path), img_width, img_height)
        logging.info(f"Found {len(ground_truth_boxes)} ground truth boxes")
        for i, (detected_bbox, conf, class_id) in enumerate(zip(results.xyxy, results.confidence, results.class_id)):
            iou_scores = [calculate_iou(detected_bbox, gt_bbox) for gt_bbox in ground_truth_boxes]
            max_iou = max(iou_scores) if iou_scores else 0
            logging.info(f"Detection {i+1} - Max IoU: {max_iou:.4f}")
    else:
        logging.warning(f"No label file found for {image_path}. Skipping IoU and GT drawing.")
    saved_any = False
    if cut_face:
        for i, (bbox, class_id) in enumerate(zip(results.xyxy, results.class_id)):
            x1, y1, x2, y2 = map(int, bbox)
            proximity = calculate_proximity([x1, y1, x2, y2], class_id)
            face_crop = image[y1:y2, x1:x2]
            face_filename = f"{image_path.stem}_face_{i+1}.PNG"
            face_path = output_dir / face_filename
            cv2.imwrite(str(face_path), face_crop)
            logging.info(f"Saved face crop: {face_path} (proximity={proximity:.2f})")
            saved_any = True
    else:
        # Only save annotated image if at least one face passes proximity filter
        keep_indices = []
        proximities = []
        for i, (bbox, class_id) in enumerate(zip(results.xyxy, results.class_id)):
            x1, y1, x2, y2 = map(int, bbox)
            proximity = calculate_proximity([x1, y1, x2, y2], class_id)
            keep_indices.append(i)
            proximities.append(proximity)
            if results.xyxy is None or len(results.xyxy) == 0:
                logging.info("No detections in this image.")
                return

            if not keep_indices:
                logging.info(f"No faces found in {image_path}")
                return

            filtered_results = Detections(
                xyxy=np.array([results.xyxy[i] for i in keep_indices]),
                confidence=np.array([results.confidence[i] for i in keep_indices]),
                class_id=np.array([results.class_id[i] for i in keep_indices])
            )
            filtered_proximities = [proximities[i] for i in keep_indices]
            annotated_image = draw_detections_and_ground_truth(image, filtered_results, proximities=filtered_proximities, ground_truth_boxes=ground_truth_boxes, ground_truth_classes=ground_truth_classes)
            output_filename = image_path.stem + "_annotated.jpg"
            output_path = output_dir / output_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not success:
                success = cv2.imwrite(str(output_path), annotated_image)
                if not success:
                    raise RuntimeError(f"Failed to save image to {output_path}")
            logging.info(f"Annotated image saved to: {output_path}")
        else:
            logging.info(f"No faces found in {image_path}")
            
def main():
    parser = argparse.ArgumentParser(description='YOLO Face Detection Inference')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Image filename or folder')
    parser.add_argument('--cut_face', action='store_true', help='Save each detected face as a PNG in the output directory')
    parser.add_argument('--filter_proximity', action='store_true', help='Only save faces/images with proximity > 0.3')
    args = parser.parse_args()
    
    input_path = Path(args.image_path)
    output_dir = FaceDetection.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        if input_path.is_dir():
            folder_name = input_path.name + "_annotated"
            folder_output_dir = output_dir / folder_name
            folder_output_dir.mkdir(parents=True, exist_ok=True)
            image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.PNG"))
            for img_file in image_files:
                process_and_save(img_file, folder_output_dir, args.cut_face, args.filter_proximity)
        else:
            process_and_save(input_path, output_dir, args.cut_face, args.filter_proximity)
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())