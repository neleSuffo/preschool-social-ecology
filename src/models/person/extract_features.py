import torch
import torch.nn as nn
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from person_classifier import CNNEncoder, VideoFrameDataset
from config import PersonConfig
from constants import PersonClassification

def main():
    """
    Main script to extract features from video frames using a pre-trained CNN.
    The features are saved as .pt files, organized by video ID.
    """
    # 1. Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the CNN model with pre-trained weights
    cnn = CNNEncoder(
        backbone=PersonConfig.BACKBONE, 
        pretrained=True, 
        feat_dim=PersonConfig.FEAT_DIM
    ).to(device)
    cnn.eval() # Set the model to evaluation mode
    
    # 2. Setup data transformations and loader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load all data paths from the CSV files
    all_csv_paths = [
        PersonClassification.TRAIN_CSV_PATH,
        PersonClassification.VAL_CSV_PATH,
        PersonClassification.TEST_CSV_PATH
    ]

    for csv_path in all_csv_paths:
        if not Path(csv_path).exists():
            print(f"Warning: CSV file not found at {csv_path}. Skipping.")
            continue
            
        print(f"\nProcessing data from {csv_path}...")
        
        # Create a simplified dataset for feature extraction
        full_dataset = VideoFrameDataset(csv_path, transform=transform, is_feature_extraction=True)
        data_loader = DataLoader(
            full_dataset, 
            batch_size=PersonConfig.BATCH_SIZE_INFERENCE, 
            shuffle=False, 
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        # 3. Create output directories
        split_name = Path(csv_path).stem
        output_dir = PersonClassification.TRAIN_CSV_PATH.parent / "extracted_features" / split_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Extract features
        with torch.no_grad():
            for images, _, _, video_ids, frame_ids in tqdm(data_loader, desc="Extracting features"):
                images = images.to(device)
                
                # Forward pass through the CNN
                features = cnn(images)
                
                # Save features for each frame
                for i in range(len(features)):
                    video_id = video_ids[i]
                    frame_id = frame_ids[i]
                    
                    video_output_dir = output_dir / video_id
                    video_output_dir.mkdir(exist_ok=True)
                    
                    # Save each feature vector
                    feature_path = video_output_dir / f"{frame_id:06d}.pt"
                    torch.save(features[i].cpu(), feature_path)

    print("\nFeature extraction complete for all datasets!")

if __name__ == '__main__':
    main()