import argparse
import logging
from datetime import datetime
from ultralytics import YOLO
from constants import DetectionPaths, ClassificationPaths

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model.")
    parser.add_argument(
        "--target", 
        type=str, 
        help="What yolo model to evaluate."
    )
    args = parser.parse_args()

    # Load the YOLO model with the supplied target weights
    if args.target == "gaze_cls":
        model = YOLO(ClassificationPaths.gaze_trained_weights_path)
        folder_name = "yolo_gaze_cls_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(ClassificationPaths.gaze_data_config_path)
        project_folder = str(ClassificationPaths.gaze_output_dir)
        output_path = ClassificationPaths.gaze_output_dir / folder_name
    elif args.target == "face_cls":
        model = YOLO(ClassificationPaths.face_trained_weights_path)
        folder_name = "yolo_face_cls_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(ClassificationPaths.face_data_config_path)
        project_folder = str(ClassificationPaths.face_output_dir)
        output_path = ClassificationPaths.face_output_dir / folder_name
    elif args.target == "person_cls":
        model = YOLO(ClassificationPaths.person_trained_weights_path)
        folder_name = "yolo_person_cls_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(ClassificationPaths.person_data_config_path)
        project_folder = str(ClassificationPaths.person_output_dir)
        output_path = ClassificationPaths.person_output_dir / folder_name
    elif args.target == "all":
        model = YOLO(DetectionPaths.all_trained_weights_path)
        folder_name = "yolo_all_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(DetectionPaths.all_data_config_path)
        project_folder = str(DetectionPaths.all_output_dir)
        output_path = DetectionPaths.all_output_dir / folder_name
    elif args.target == "face":
        model = YOLO(DetectionPaths.face_trained_weights_path)
        folder_name = "yolo_face_det_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        data_config = str(DetectionPaths.face_data_config_path)
        project_folder = str(DetectionPaths.face_output_dir)
        output_path = DetectionPaths.face_output_dir / folder_name
        
    # Validate the model
    metrics = model.val(
        data=data_config,
        save_json=True,
        iou=0.5,
        plots=True,
        project=project_folder,
        name=folder_name
    )

    # Extract precision and recall
    if args.target in ["all", "face"]:
        precision = metrics.results_dict['metrics/precision(B)']
        recall = metrics.results_dict['metrics/recall(B)']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Log results
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1_score:.4f}")

        # Save precision and recall to a file
        with open(output_path / "precision_recall.txt", "w") as f:
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"F1 Score: {f1_score}\n")

    elif args.target in ["gaze_cls", "face_cls", "person_cls"]:
        accuracy_top1 = metrics.results_dict['metrics/accuracy_top1']
        accuracy_top5 = metrics.results_dict['metrics/accuracy_top5']
        fitness = metrics.results_dict['fitness']
        
        # Log results
        logging.info(f"Accuracy Top 1: {accuracy_top1:.4f}")
        logging.info(f"Accuracy Top 5: {accuracy_top5:.4f}")
        logging.info(f"Fitness: {fitness:.4f}")
        
        # Save accuracy and fitness to a file
        with open(output_path / "accuracy_fitness.txt", "w") as f:
            f.write(f"Accuracy Top 1: {accuracy_top1}\n")
            f.write(f"Accuracy Top 5: {accuracy_top5}\n")
            f.write(f"Fitness: {fitness}\n")
if __name__ == '__main__':
    main()