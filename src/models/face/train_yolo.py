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
    os.environ['OMP_NUM_THREADS'] = '6'
    torch.set_num_threads(6)

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

        # Learning rate settings
        lr0=0.005, # Reduced initial learning rate
        lrf=0.0005, # Reduced final learning rate
        cos_lr=True,

        # Regularization improvements
        weight_decay=0.0005,
        dropout=0.1,

        # Early stopping with more patience
        patience=30,

        # Data augmentation improvements (tuned for face detection)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0, # Reduced to 0 for face detection
        translate=0.1,
        scale=0.5,
        shear=0.0, # Reduced to 0 for face detection
        perspective=0.0, # Reduced to 0
        flipud=0.0, # Set to 0 as vertical flips are unnatural for faces
        fliplr=0.5, # Keep horizontal flips
        mosaic=0.5, # Reduced mosaic probability
        mixup=0.0, # Disabled mixup
        copy_paste=0.0, # Disabled copy-paste

        # Training settings
        device=args.device,
        plots=True,

        # Validation settings
        val=True,

        # Additional settings
        close_mosaic=10,
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,

        # Optimizer settings
        optimizer='AdamW',
        momentum=0.937,

        # Loss function improvements
        box=7.5,
        cls=1.0, # Increased classification loss weight to distinguish classes better
        dfl=1.5,

        # Multi-scale training
        rect=False,
        overlap_mask=True,
        mask_ratio=4,

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