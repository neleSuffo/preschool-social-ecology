import logging
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from constants import FaceDetection
from config import FaceConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate YOLO model for face detection')
    parser.add_argument('--config', type=str, default=str(FaceDetection.DATA_CONFIG_PATH),
                      help=f'Path to YOLO data config file (default: {FaceDetection.DATA_CONFIG_PATH})')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold for evaluation (default: 0.7)')
    return parser.parse_args()

def main():
    args = parse_args()
    model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    # Set output directory to parent of trained weights path
    output_dir = Path(FaceDetection.TRAINED_WEIGHTS_PATH).parent.parent
    folder_name = Path(f"{FaceConfig.MODEL_NAME}_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S_") + args.iou.__str__().replace('.', '_'))

    # Determine config file path
    if args.config:
        config_file_path = FaceDetection.DATA_CONFIG_PATH.parent / args.config

    # Validate the model
    metrics = model.val(
        data=config_file_path,
        save_json=True,
        plots=True,
        project=output_dir,
        name=folder_name,
        iou=args.iou,
    )

    # Extract precision and recall
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Log results
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1_score:.4f}")

    # Save precision and recall to a file
    with open(output_dir / folder_name / "precision_recall.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1_score}\n")
            
if __name__ == '__main__':
    main()