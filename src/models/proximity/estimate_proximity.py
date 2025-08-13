import cv2
import os
import argparse
import logging
from proximity_utils import ProximityCalculator
from ultralytics import YOLO
from constants import DetectionPaths, ClassificationPaths

# Configure logging
logging.basicConfig(level=logging.INFO)

def draw_detections(image, detections, output_path):
    """Draw bounding boxes and labels on image and save it"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Colors for different classes
    colors = {
        "child": (0, 255, 0),  # Green
        "adult": (255, 0, 0)   # Blue
    }
    
    for det in detections:
        bbox = det["bbox"]
        age = det["age"]
        proximity = det["proximity"]
        #age_conf = det["age_confidence"]
        
        # Draw bounding box
        cv2.rectangle(image, 
                      (bbox[0], bbox[1]), 
                      (bbox[2], bbox[3]), 
                      colors[age], 
                      thickness)
        
        # Prepare text
        #text = f"{age} Proximity: {proximity:.2f} (Age: {age_conf:.2f})"
        text = f"{age} Proximity: {proximity:.2f}"

        # Calculate text position
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 20
        
        # Draw text background
        cv2.rectangle(image, 
                      (bbox[0], text_y - text_height - 5),
                      (bbox[0] + text_width, text_y + 5),
                      colors[age], 
                      -1)
        
        # Draw text
        cv2.putText(image, text, 
                    (bbox[0]-100, text_y), 
                    font, font_scale, 
                    (255, 255, 255),  # White text
                    thickness)
    
    # Save the annotated image
    cv2.imwrite(output_path, image)
    logging.info(f"Saved annotated image to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Calculate proximity for faces in an image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--age', type=str, required=True, choices=['child', 'adult'], help='Age classification (child/adult)')
    args = parser.parse_args()

    # Initialize models
    detector = YOLO("/home/nele_pauline_suffo/models/yolo11_face_detection.pt")
    #detector = YOLO("/home/nele_pauline_suffo/models/yolo11_person_face_detection.pt")
    #age_classifier = YOLO(ClassificationPaths.face_trained_weights_path)

    # Load image
    videos_folder = "/home/nele_pauline_suffo/ProcessedData/quantex_videos_processed"
    video_base_folder = os.path.basename(args.image_path).rsplit('_', 1)[0]
    full_image_path = os.path.join(videos_folder, video_base_folder, args.image_path)
        
    img = cv2.imread(full_image_path)
    if img is None:
        raise ValueError(f"Could not load image at {full_image_path}")

    # Make a copy for drawing detections
    output_img = img.copy()

    # Detect faces
    detections = detector(img)[0]
    
    # Initialize proximity calculator
    prox_calculator = ProximityCalculator()
    
    results = []
    for box in detections.boxes:
        #if int(box.cls) != 1:  # Skip non-face classes
        #    continue
            
        # Extract face ROI
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_roi = img[y1:y2, x1:x2]
        
        # Classify age
        age_result = args.age
        #age_result = age_classifier(face_roi)[0]
        is_child = True if age_result == "child" else False
        
        # Calculate proximity
        proximity = prox_calculator.calculate((x1, y1, x2, y2), is_child)
        print("proximity", proximity)
        # Format results
        results.append({
            "bbox": [x1, y1, x2, y2],
            "age": "child" if is_child else "adult",
            #"age_confidence": float(age_result.probs.top1conf),
            "proximity": float(proximity),
        })
    
    # Draw detections and save image
    output_path = f"/home/nele_pauline_suffo/outputs/detection_pipeline_results/inference_images/{args.image_path}"
    draw_detections(output_img, results, output_path)

if __name__ == "__main__":
    main()
