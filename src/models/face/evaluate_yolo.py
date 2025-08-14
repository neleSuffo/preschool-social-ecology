import logging
from datetime import datetime
from ultralytics import YOLO
from constants import FaceDetection

def main():
    model = YOLO(FaceDetection.TRAINED_WEIGHTS_PATH)
    folder_name = Path("yolo_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Validate the model
    metrics = model.val(
        data=FaceDetection.DATA_CONFIG_PATH,
        save_json=True,
        iou=0.5,
        plots=True,
        project=FaceDetection.OUTPUT_DIR,
        name=folder_name
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
    with open(FaceDetection.OUTPUT_DIR / folder_name / "precision_recall.txt", "w") as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1_score}\n")
            
if __name__ == '__main__':
    main()