import cv2
import logging
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from supervision import Detections
from constants import PersonClassification  # Your paths
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def process_image(model: YOLO, image_path: Path) -> np.ndarray:
    """Run YOLO on an image and return the annotated version."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    output = model(Image.open(image_path))
    detections = Detections.from_ultralytics(output[0])

    annotated_image = image.copy()
    for bbox, conf, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"Person {conf:.2f}"
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_image, label, (x1 + 10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_image

def save_annotated_image(annotated_image: np.ndarray, image_path: Path, output_dir: Path):
    """Save annotated image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    success = cv2.imwrite(str(output_path), annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise RuntimeError(f"Failed to save {output_path}")
    logging.info(f"Saved annotated ima ge: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="YOLO Person Detection Inference")
    parser.add_argument("--image_path", type=str, required=True, help="Image file or folder")
    args = parser.parse_args()

    model = YOLO(PersonClassification.TRAINED_WEIGHTS_PATH)
    input_path = Path(args.image_path)
    output_dir = PersonClassification.OUTPUT_DIR

    if input_path.is_dir():
        folder_output = output_dir / f"{input_path.name}_annotated"
        folder_output.mkdir(parents=True, exist_ok=True)
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        for img_file in image_files:
            annotated_image = process_image(model, img_file)
            save_annotated_image(annotated_image, img_file, folder_output)
    else:
        annotated_image = process_image(model, input_path)
        save_annotated_image(annotated_image, input_path, output_dir)

if __name__ == "__main__":
    main()