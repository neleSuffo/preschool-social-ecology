import logging
import sys
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
# Assuming these constants are defined in your environment:
from constants import DataPaths, BasePaths, PersonDetection, PersonClassification 
from config import PersonConfig

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Utility to get the correct constants object ---
def get_data_constants(data_type: str):
    """Returns the correct constants object based on the data_type flag."""
    if data_type == "detection":
        return PersonDetection
    elif data_type == "classification":
        return PersonClassification
    else:
        raise ValueError(f"Unknown data type: {data_type}. Must be 'detection' or 'classification'.")

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for person data')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for evaluation (only used for detection)')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize detection/classification results (default: False)')
    parser.add_argument('--type', choices=["detection", "classification"], default="detection",
                        help='Type of task/data format (detection or classification)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Determine Constants and Paths
    CONSTANTS = get_data_constants(args.type)
    
    # Use specified path or default constant path
    weights_path = CONSTANTS.TRAINED_WEIGHTS_PATH

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(weights_path).parent.parent
    
    # Adjust folder name based on task type
    task_label = "det" if args.type == "detection" else "cls"
    iou_suffix = f"_{args.iou.__str__().replace('.', '_')}" if args.type == "detection" else ""
    folder_name = Path(f"{PersonConfig.MODEL_NAME}_{task_label}_validation_{timestamp}{iou_suffix}")

    # 2. Load the Model
    try:
        model = YOLO(weights_path)
    except FileNotFoundError:
        logging.error(f"Trained weights file not found at: {weights_path}")
        sys.exit(1)

    logging.info(f"Validating model for task: {args.type.upper()}")
    logging.info("-" * 50)

    if args.type == "detection":
        data_path = CONSTANTS.DATA_CONFIG_PATH
    else:
        data_path = CONSTANTS.INPUT_DIR / "images/"
    # 3. Validate the model
    metrics = model.val(
        data=data_path,
        save_json=(args.type == "detection"), # Only save JSON output for detection
        plots=True,
        project=output_dir,
        name=folder_name,
        iou=args.iou,
        visualize=args.visualize,
    )

    # 4. Extract and Log Results based on task type
    
    final_output_path = output_dir / folder_name / "metrics.txt"
    
    # Create the output directory if it wasn't created by model.val()
    (output_dir / folder_name).mkdir(parents=True, exist_ok=True)
    
    with open(final_output_path, "w") as f:
        f.write(f"Task Type: {args.type.upper()}\n")
        f.write(f"Model: {weights_path.name}\n")

        if args.type == "detection":
            # YOLO Detection Metrics
            try:
                precision = metrics.results_dict.get('metrics/precision(B)', 0)
                recall = metrics.results_dict.get('metrics/recall(B)', 0)
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                logging.info(f"Detection Metrics (IoU > {args.iou}):")
                logging.info(f"Precision: {precision:.4f}")
                logging.info(f"Recall: {recall:.4f}")
                logging.info(f"F1 Score: {f1_score:.4f}")
                
                f.write(f"IoU Threshold: {args.iou}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"F1 Score: {f1_score}\n")
                
            except AttributeError:
                logging.error("Could not retrieve detection metrics. Check weights and data config.")

        elif args.type == "classification":
            # YOLO Classification Metrics
            try:
                top1 = metrics.results_dict.get('metrics/top1_acc', 0)
                top5 = metrics.results_dict.get('metrics/top5_acc', 0)

                logging.info("Classification Metrics:")
                logging.info(f"Top-1 Accuracy: {top1:.4f}")
                logging.info(f"Top-5 Accuracy: {top5:.4f}")
                
                f.write(f"Top-1 Accuracy: {top1}\n")
                f.write(f"Top-5 Accuracy: {top5}\n")
                
            except AttributeError:
                logging.error("Could not retrieve classification metrics. Check weights and data config.")
    
    logging.info(f"Results summary saved to {final_output_path}")

if __name__ == '__main__':
    main()