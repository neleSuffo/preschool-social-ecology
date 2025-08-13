import json
import logging
import cv2
from pathlib import Path
from utils import fetch_all_annotations
from multiprocessing import Pool
from constants import DataPaths
from config import FaceConfig, PersonConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_dimensions(image_paths: list) -> dict:
    """Preload image dimensions to avoid multiple cv2.imread() calls.
    
    Parameters
    ----------
    image_paths : list
        List of image file paths.
    
    Returns
    -------
    dict        
        Dictionary mapping image paths to their dimensions (height, width).
    """
    dimensions = {}
    for image_path in image_paths:
        if not image_path.exists():
            logging.warning(f"Image file not found: {image_path}")
            continue
            
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logging.warning(f"Failed to load image: {image_path}")
                continue
            
            dimensions[image_path] = img.shape[:2]
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            continue
            
    if not dimensions:
        logging.error("No valid images found!")
        
    return dimensions

def convert_to_yolo_format(img_width: int, img_height: int, bbox: list) -> tuple:
    """Converts bounding box to YOLO format.
    
    Parameters
    ----------
    img_width : int
        Width of the image.
    img_height : int
        Height of the image
    bbox : list
        Bounding box coordinates [xtl, ytl, xbr, ybr].
        
    Returns
    -------
    tuple
        YOLO formatted bounding box (x_center, y_center, width, height).
    """
    # Convert bounding box to YOLO format
    xtl, ytl, xbr, ybr = bbox

    x_center = (xtl + xbr) / 2.0 / img_width
    y_center = (ytl + ybr) / 2.0 / img_height
    width = (xbr - xtl) / img_width
    height = (ybr - ytl) / img_height

    return (x_center, y_center, width, height)

def map_category_id(target: str, category_id: int, person_age: None, gaze_directed_at_child: None, object_interaction: None) -> int:
    """Maps category ID based on target type and additional attributes.
    
    Parameters
    ----------
    target : str
        The target detection type (e.g., "person", "face", "object").
    category_id : int
        The original category ID.
    person_age : None
        The age of the person (if applicable).
    gaze_directed_at_child : None
        The gaze direction (if applicable).
    object_interaction : None
        The object interaction type (if applicable).
    
    Returns
    -------
    int
        The mapped category ID.
    """
    person_age = person_age.strip().lower() if isinstance(person_age, str) else "unknown"

    mappings = {
        "person_cls": PersonConfig.AGE_GROUP_TO_CLASS_ID.get(person_age, 99),
        "face_det": FaceConfig.AGE_GROUP_TO_CLASS_ID.get(person_age, 99),
    }
    return mappings.get(target, 99)

def write_annotations(txt_file: Path, lines: list) -> None:
    """Write annotation lines to a text file.
    
    Parameters
    ----------
    txt_file : Path
        Path to the output text file
    lines : list
        List of annotation lines to write
    """
    with open(txt_file, "w") as f:
        f.writelines(lines)

def save_annotations_json(annotations, target):
    """Saves annotations in a single JSON file (bounding boxes and labels)."""
    logging.info("Saving annotations in JSON format.")

    image_paths = {DataPaths.IMAGES_INPUT_DIR / ann[3][:-11] / ann[3] for ann in annotations}
    image_dims = get_image_dimensions(image_paths)

    json_data = []
    skipped_count = 0

    for category_id, bbox_json, object_interaction, image_file_name, gaze_directed_at_child, person_age in annotations:
        image_file_path = DataPaths.IMAGES_INPUT_DIR / image_file_name[:-11] / image_file_name

        if image_file_path not in image_dims:
            skipped_count += 1
            continue

        try:
            bbox = json.loads(bbox_json)
            category_id = map_category_id(target, category_id, person_age, gaze_directed_at_child, object_interaction)
            ann_entry = {
                "image_path": str(image_file_path),
                "bbox": bbox,  # [xtl, ytl, xbr, ybr] in pixel format
                "label": category_id
            }
            json_data.append(ann_entry)
        except Exception as e:
            logging.error(f"Error processing annotation for {image_file_path}: {e}")
            skipped_count += 1
            continue

    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)

    logging.info(f"Saved {len(json_data)} annotations to {output_json}. Skipped {skipped_count}.")
                
def save_annotations(annotations, target):
    """Saves annotations in YOLO format using optimized batch writing and multiprocessing."""
    logging.info("Saving annotations in YOLO format.")

    output_dirs = {
        "person_cls": PersonClassification.PERSON_LABELS_INPUT_DIR,
        "face_det": FaceDetection.FACE_LABELS_INPUT_DIR,
    }

    if target not in output_dirs:
        raise ValueError(f"Invalid target: {target}. Must be one of: {', '.join(output_dirs.keys())}")

    output_dir = output_dirs[target]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preload image dimensions
    image_paths = {DataPaths.IMAGES_INPUT_DIR / ann[3][:-11] / ann[3] for ann in annotations}
    image_dims = get_image_dimensions(image_paths)

    if len(image_dims) == 0:
        raise RuntimeError("No valid images found. Please check the image paths and files.")
        
    file_contents = {}
    skipped_count = 0
    processed_count = 0

    for category_id, bbox_json, object_interaction, image_file_name, gaze_directed_at_child, person_age in annotations:
        image_file_path = DataPaths.IMAGES_INPUT_DIR / image_file_name[:-11] / image_file_name

        if image_file_path not in image_dims:
            skipped_count += 1
            continue

        bbox = json.loads(bbox_json)  # Parse JSON only once
        category_id = map_category_id(target, category_id, person_age, gaze_directed_at_child, object_interaction)
        img_height, img_width = image_dims[image_file_path]

        try:
            yolo_bbox = convert_to_yolo_format(img_width, img_height, bbox)
            line = f"{category_id} " + " ".join(map(str, yolo_bbox)) + "\n"

            if image_file_name not in file_contents:
                file_contents[image_file_name] = []
            file_contents[image_file_name].append(line)
            processed_count += 1
        except Exception as e:
            logging.error(f"Error converting bbox for {image_file_path}: {e}")
            skipped_count += 1
            continue

    write_tasks = []
    for img, lines in file_contents.items():
        output_path = output_dir / f"{Path(img).stem}.txt"
        write_tasks.append((output_path, lines))
        
    with Pool(processes=4) as pool:
        pool.starmap(write_annotations, write_tasks)

    logging.info(f"Processed {processed_count} annotations, skipped {skipped_count}.")

def main(target: str):
    """Main function to fetch and save YOLO annotations."""
    logging.info(f"Starting conversion process for YOLO {target} detection.")

    try:
        category_ids = {
            "person_cls": PersonConfig.DATABASE_CATEGORY_IDS,
            "face_det": FaceConfig.DATABASE_CATEGORY_IDS,
        }.get(target)

        if category_ids is None:
            logging.error(f"Invalid target: {target}. Expected one of: {', '.join(category_ids.keys())}.")
            return

        annotations = fetch_all_annotations(category_ids=category_ids)

        logging.info(f"Fetched {len(annotations)} {target} annotations.")
        
        if target == "gaze_cls_vit":
            save_annotations_json(annotations, target)
            logging.info(f"Successfully saved {target} annotations in JSON format.")
            return
        save_annotations(annotations, target)
        logging.info(f"Successfully saved all {target} annotations.")

    except Exception as e:
        logging.error(f"Failed to process annotations: {e}")

if __name__ == "__main__":
    main()
