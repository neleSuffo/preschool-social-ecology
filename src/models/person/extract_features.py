import torch
import torch.nn as nn
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from person_classifier import YOLOFeatureExtractor
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
    cnn = YOLOFeatureExtractor( # Renamed class
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
        
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} frames to process")
        
        batch_size = PersonConfig.BATCH_SIZE_INFERENCE
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        # 3. Create output directories
        split_name = Path(csv_path).stem
        # Output directory derived from the path of the CSV, ensuring consistency
        output_dir = Path(csv_path).parent / "extracted_features" / split_name 
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. Extract features batch by batch
        with torch.no_grad():
            for batch_idx in tqdm(range(total_batches), desc=f"Extracting features from {split_name}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                
                images = []
                video_names = []
                frame_ids = []
                
                for _, row in batch_df.iterrows():
                    try:
                        file_path = row['file_path']
                        if not Path(file_path).exists():
                            continue
                            
                        video_name = Path(file_path).parent.name
                            
                        img = Image.open(file_path).convert('RGB')
                        img_tensor = transform(img)
                        images.append(img_tensor)
                        video_names.append(video_name)
                        frame_ids.append(row['frame_id'])
                        
                    except Exception as e:
                        print(f"Error loading {row['file_path']}: {e}")
                        continue
                
                if not images: continue
                    
                images_batch = torch.stack(images).to(device)
                features_batch = cnn(images_batch)
                
                # Save features for each frame
                for i, (video_name, frame_id, features) in enumerate(zip(video_names, frame_ids, features_batch)):
                    video_output_dir = output_dir / video_name
                    video_output_dir.mkdir(exist_ok=True)
                    
                    feature_path = video_output_dir / f"{frame_id:06d}.pt"
                    torch.save(features.cpu(), feature_path)

    print("\nFeature extraction complete for all datasets!")

if __name__ == '__main__':
    main()