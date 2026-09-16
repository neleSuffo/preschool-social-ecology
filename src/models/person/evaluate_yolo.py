import logging
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from constants import PersonDetection, PersonClassification 
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
    CONSTANTS = get_data_constants(args.type)
    weights_path = CONSTANTS.TRAINED_WEIGHTS_PATH
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(weights_path).parent.parent
    
    task_label = "det" if args.type == "detection" else "cls"
    iou_suffix = f"_{str(args.iou).replace('.', '_')}" if args.type == "detection" else ""
    folder_name = f"{PersonConfig.MODEL_NAME}_{task_label}_validation_{timestamp}_iou_{iou_suffix}"

    model = YOLO(weights_path)

    # 3. Validate (Using 'results' consistently)
    data_path = CONSTANTS.DATA_CONFIG_PATH if args.type == "detection" else CONSTANTS.INPUT_DIR / "images/"
    
    results = model.val(
        data=data_path,
        save_json=(args.type == "detection"),
        plots=True,
        conf = 0.170,
        project=output_dir,
        name=folder_name,
        iou=args.iou,
        visualize=args.visualize,
    )

    # 4. Extract and Save
    final_folder = output_dir / folder_name
    final_folder.mkdir(parents=True, exist_ok=True)
    final_output_path = final_folder / "precision_recall.txt"
    
    with open(final_output_path, "w") as f:
        f.write(f"Task Type: {args.type.upper()}\n")
        f.write(f"Model: {weights_path.name}\n\n")

        if args.type == "detection":
            try:          
                # Standard Metrics
                precision = results.box.mp      
                recall = results.box.mr         
                f1_score = results.box.f1[0]    
                map50 = results.box.map50       
                map50_95 = results.box.map 

                f.write(f"Precision: {precision}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"F1 Score: {f1_score}\n")
                f.write(f"mAP@0.5: {map50}\n")
                f.write(f"mAP@0.5:0.95: {map50_95}\n")
                
            except Exception as e:
                logging.error(f"Error extracting detection metrics: {e}")

        elif args.type == "classification":
            try:
                top1 = results.results_dict.get('metrics/top1_acc', 0)
                top5 = results.results_dict.get('metrics/top5_acc', 0)
                
                f.write(f"Top-1 Accuracy: {top1:.4f}\n")
                f.write(f"Top-5 Accuracy: {top5:.4f}\n")
                logging.info(f"Classification validation complete. Top-1: {top1:.4f}")
            except Exception as e:
                logging.error(f"Error extracting classification metrics: {e}")
    
    logging.info(f"Results summary saved to {final_output_path}")

if __name__ == '__main__':
    main()