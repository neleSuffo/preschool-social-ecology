import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import PersonClassification
from config import PersonConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--device', type=str, default='1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--config', type=str, default=str(PersonClassification.DATA_CONFIG_PATH),
                      help=f'Path to YOLO data config file (default: {PersonClassification.DATA_CONFIG_PATH})')
    parser.add_argument('--retrain', action='store_true', default=False,
                       help='Activate retrain mode using fixed IDs and hard negative files defined in PersonConfig.')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '12'
    torch.set_num_threads(12)

    base_output_dir = PersonClassification.OUTPUT_DIR

    # Load the YOLO model - changed default to 'm'
    model_name = PersonConfig.MODEL_NAME
    print(f"Loading model: {model_name}")

    if args.retrain:
        print("Retrain mode activated.")
        model = YOLO(PersonClassification.TRAINED_WEIGHTS_PATH)
    else:
        model = YOLO("yolo12l.pt")
    
    experiment_name = f"{PersonConfig.MODEL_NAME}_{timestamp}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print("-" * 50)

    # Determine config file path
    if args.config:
        config_file_path = PersonClassification.DATA_CONFIG_PATH.parent / args.config

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=config_file_path,
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
    script_copy = output_dir / f"train_{PersonConfig.MODEL_NAME}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()