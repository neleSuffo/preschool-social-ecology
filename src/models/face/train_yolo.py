import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import FaceDetection
from config import FaceConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--device', type=str, default='0,1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '24'
    torch.set_num_threads(24)

    base_output_dir = FaceDetection.OUTPUT_DIR

    # Load the YOLO model - changed default to 'm'
    model_name = FaceConfig.MODEL_NAME
    print(f"Loading model: {model_name}")

    model = YOLO(model_name)

    experiment_name = f"{FaceConfig.MODEL_NAME}_{timestamp}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print("-" * 50)

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=str(FaceDetection.DATA_CONFIG_PATH),
        epochs=FaceConfig.NUM_EPOCHS,
        imgsz=FaceConfig.IMG_SIZE,
        batch=FaceConfig.BATCH_SIZE,
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
    script_copy = output_dir / f"train_{FaceConfig.MODEL_NAME}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()