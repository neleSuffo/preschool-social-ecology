import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from constants import PersonClassification, DataPaths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_detections_from_labels(
    labels_input_dir: Path,
    rawframe_dir: Path,
    output_dir: Path,
    progress_file: Path,
    missing_frames_file: Path,
    detection_type: str,
):
    """ 
    Reads YOLO annotations and crops faces or persons from rawframes.

    Parameters
    ----------
    labels_input_dir : Path
        Directory containing YOLO annotations in txt format.
    rawframe_dir : Path
        Directory containing rawframes.
    output_dir : Path
        Directory to save cropped detections.
    progress_file : Path
        File to track progress.
    missing_frames_file : Path
        File to log missing frames.
    detection_type : str
        Either 'face', 'gaze', or 'person' to specify the type of detection to crop.
    """
    cv2.setNumThreads(1)

    # Create necessary directories
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = progress_file.parent
    progress_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize progress tracking
    processed_images = set()
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            processed_images = set(line.strip() for line in f)
        logging.info(f"Loaded {len(processed_images)} processed images")

    # Get all annotation files
    annotation_files = list(labels_input_dir.glob('*.txt'))

    for ann_file in tqdm(annotation_files, desc=f"Processing {detection_type} annotations"):
        image_name = ann_file.stem + '.jpg'  

        # Skip if already processed
        if image_name in processed_images:
            continue

        # Construct image path
        video_folder = '_'.join(image_name.split('_')[:8])
        image_path = rawframe_dir / video_folder / image_name

        if not image_path.exists():
            logging.warning(f"Image {image_path} not found")
            with open(missing_frames_file, 'a') as f:
                f.write(f"{image_name}\n")
            continue

        # Read the image
        frame = cv2.imread(str(image_path))
        if frame is None:
            logging.error(f"Failed to load {image_path}")
            with open(missing_frames_file, 'a') as f:
                f.write(f"{image_name}\n")
            continue

        frame_height, frame_width = frame.shape[:2]

        # Read annotations
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                try:
                    # Parse YOLO format: class_id x_center y_center width height
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width / 2) * frame_width)
                    y1 = int((y_center - height / 2) * frame_height)
                    w = int(width * frame_width)
                    h = int(height * frame_height)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    w = min(w, frame_width - x1)
                    h = min(h, frame_height - y1)

                    # Crop the detection
                    cropped_detection = frame[y1:y1 + h, x1:x1 + w]
                    if cropped_detection.size == 0:
                        logging.warning(f"Empty crop for {image_path}")
                        continue

                    # Save the cropped detection
                    output_path = output_dir / f"{ann_file.stem}_{detection_type}_{idx}.jpg"
                    cv2.imwrite(str(output_path), cropped_detection)

                except Exception as e:
                    logging.error(f"Error processing {ann_file}: {e}")
                    continue

        # Update progress
        with open(progress_file, 'a') as f:
            f.write(f"{image_name}\n")

    logging.info(f"Completed {detection_type} extraction. Results saved to {output_dir}")


def main(target: str = None):
    """
    Main function to crop detections.
    
    Parameters
    ----------
    target : str, optional
        Target type to crop ('person_cls', 'face_cls', 'gaze_cls'). If None, it will prompt for input.
    """
    if target is None:
        parser = argparse.ArgumentParser(description='Crop person or face detections from images.')
        parser.add_argument(
            '--target', 
            type=str,
            choices=['person_cls', 'face_cls', 'gaze_cls'],
            required=True,
            help='Target type to crop (person or face)'
        )
        args = parser.parse_args()
        target = args.target
    
    if target == 'person_cls':
        crop_detections_from_labels(
            labels_input_dir=PersonClassification.PERSON_LABELS_INPUT_DIR,
            rawframe_dir=DataPaths.IMAGES_INPUT_DIR,
            output_dir=PersonClassification.PERSON_IMAGES_INPUT_DIR,
            progress_file=PersonClassification.PERSON_EXTRACTION_PROGRESS_FILE_PATH,
            missing_frames_file=PersonClassification.PERSON_MISSING_FRAMES_FILE_PATH,
            detection_type="person"
        )
    else:
        raise ValueError("Invalid target specified. Use 'person_cls', 'face_cls' or 'gaze_cls'.")

if __name__ == "__main__":
    main()