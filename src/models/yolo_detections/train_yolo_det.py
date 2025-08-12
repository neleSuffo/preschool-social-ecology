import os
import sys
import torch
import shutil
import argparse
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from constants import DetectionPaths

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model for different detection tasks')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=640,
                      help='Image size for training')
    parser.add_argument('--target', type=str, default='all',
                      choices=['all', 'face', 'person_face'],
                      help='Target detection task to train on')
    parser.add_argument('--device', type=str, default='0,1',
                      help='Device to use (e.g., "0" for GPU, "cpu" for CPU)')
    parser.add_argument('--model_size', type=str, default='m', # Changed default to 'm' for better generalization
                      choices=['n', 's', 'm', 'l', 'x'],
                      help='YOLO model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    parser.add_argument('--pretrained', type=bool, default=True,
                      help='Use pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume training from checkpoint path')
    return parser.parse_args()

def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set thread limits
    os.environ['OMP_NUM_THREADS'] = '6'
    torch.set_num_threads(6)

    data_config_path = getattr(DetectionPaths, f"{args.target}_data_config_path")
    base_output_dir = getattr(DetectionPaths, f"{args.target}_output_dir")

    # Load the YOLO model - changed default to 'm'
    model_name = f"yolo12{args.model_size}.pt"
    print(f"Loading model: {model_name}")

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        model = YOLO(model_name)

    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Model size: {args.model_size}")
    print(f"Pretrained: {args.pretrained}")

    if args.model_size == 'x':
        print("WARNING: Using YOLO12x - if overfitting, consider using 'l' or 'm' model size")

    experiment_name = f"{timestamp}_yolo12{args.model_size}_{args.target}"
    output_dir = base_output_dir / experiment_name

    print(f"Training will be saved to: {output_dir}")
    print(f"Data config: {data_config_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print("-" * 50)

    # Train the model with improved regularization to reduce overfitting
    model.train(
        data=str(data_config_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
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
    script_copy = output_dir / f"train_yolo12{args.model_size}_{args.target}.py"
    if os.path.exists(__file__):
        shutil.copy(__file__, script_copy)

if __name__ == "__main__":
    main()