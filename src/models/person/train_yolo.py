import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import PersonDetection, PersonClassification 
from config import PersonConfig

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
    parser = argparse.ArgumentParser(description='Train YOLO model for different person data tasks')
    parser.add_argument('--device', type=str, default='1', help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--type', choices=["detection", "classification"], default="detection", help='Type of task/data format (detection or classification)')
    parser.add_argument('--retrain', action='store_true', default=False,
                       help='Activate retrain mode using fixed IDs and hard negative files defined in PersonConfig.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Determine Constants based on --type
    CONSTANTS = get_data_constants(args.type)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '12'
    torch.set_num_threads(12)

    base_output_dir = CONSTANTS.OUTPUT_DIR

    if args.retrain:
        print("Retrain mode activated.")
        model = YOLO(CONSTANTS.TRAINED_WEIGHTS_PATH)
    else:
        # Load appropriate pre-trained model for the task type
        if args.type == "classification":
            # For classification, start from a pre-trained classification model
            model = YOLO('yolo11l-cls.pt')
            model_name = str(PersonConfig.MODEL_NAME) + "_cls"
            data_path = PersonClassification.INPUT_DIR / "images"

        else: 
            model = YOLO('yolo11l.pt')
            model_name = str(PersonConfig.MODEL_NAME)
            data_path = PersonDetection.DATA_CONFIG_PATH

    experiment_name = f"{model_name}_{timestamp}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print("-" * 50)

    # 4. Train the model
    model.train(
        data=data_path,
        epochs=PersonConfig.NUM_EPOCHS,
        imgsz=PersonConfig.IMG_SIZE,
        batch=PersonConfig.BATCH_SIZE,
        project=str(base_output_dir),
        name=experiment_name,
        augment=True,

        # Early stopping with more patience
        patience=20,

        # Training settings
        device=args.device,
        plots=True,

        # Validation settings
        val=True,

        # Workspace settings
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    # Copy the script to the output directory after training starts
    script_copy = output_dir / f"train_{model_name}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()