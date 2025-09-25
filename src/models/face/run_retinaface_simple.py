#!/usr/bin/env python3
"""
Simple RetinaFace detection script that processes all images in a folder
and outputs detections to a CSV file.
"""

import os
import csv
import argparse
from pathlib import Path
from retinaface import RetinaFace
import pandas as pd
from tqdm import tqdm

def extract_video_and_frame_from_filename(filename):
    """
    Extract video name and frame ID from filename.
    Expected format: quantex_at_home_id255237_2022_05_08_01_000240.jpg
    Returns: (video_name, frame_id)
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    
    if len(parts) >= 9:
        # Video name is everything except the last part (frame number)
        video_name = '_'.join(parts[:-1])
        # Frame ID is the last part converted to int to remove leading zeros
        frame_id = int(parts[-1])
        return video_name, frame_id
    else:
        # Fallback: use the full stem as video name and 0 as frame_id
        return stem, 0

def detect_faces_in_folder(input_folder, output_csv, confidence_threshold=0.8):
    """
    Run RetinaFace on all images in a folder and save detections to CSV.
    
    Parameters:
    -----------
    input_folder : str or Path
        Path to folder containing images (.jpg, .png, .jpeg)
    output_csv : str or Path
        Path to output CSV file
    confidence_threshold : float
        Minimum confidence threshold for detections (default: 0.8)
    """
    input_folder = Path(input_folder)
    output_csv = Path(output_csv)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in input_folder.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images in {input_folder}")
    
    # Prepare CSV output
    detections = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Run RetinaFace detection
            img_path = str(image_file)
            faces = RetinaFace.detect_faces(img_path, threshold=confidence_threshold)
            
            # Extract video name and frame ID from filename
            video_name, frame_id = extract_video_and_frame_from_filename(image_file.name)
            
            # Process detections
            if isinstance(faces, dict) and faces:
                for face_key, face_info in faces.items():
                    if 'facial_area' in face_info:
                        # facial_area format: [x1, y1, x2, y2]
                        bbox = face_info['facial_area']
                        
                        # Add detection to list
                        detections.append({
                            'video_name': video_name,
                            'frame_id': frame_id,
                            'x1': int(bbox[0]),
                            'y1': int(bbox[1]),
                            'x2': int(bbox[2]),
                            'y2': int(bbox[3])
                        })
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(detections)
    df.to_csv(output_csv, index=False)
    
    print(f"\nDetection complete!")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total face detections: {len(detections)}")
    print(f"Results saved to: {output_csv}")
    
    # Print some statistics
    if detections:
        unique_videos = df['video_name'].nunique()
        unique_frames = df['frame_id'].nunique()
        print(f"Unique videos: {unique_videos}")
        print(f"Unique frames: {unique_frames}")
        print(f"Average detections per frame: {len(detections) / unique_frames:.2f}")

def main():
    input_folder = Path("/home/nele_pauline_suffo/ProcessedData/face_det_input/images/test")
    output_csv = Path("/home/nele_pauline_suffo/outputs/face_retinaface/predictions_data.csv")
    IOU_THRESHOLD = 0.5
    
    # Create output directory if it doesn't exist
    output_path = output_csv.parent
    output_path.mkdir(parents=True, exist_ok=True)

    # Run detection
    detect_faces_in_folder(input_folder, output_csv, IOU_THRESHOLD)

if __name__ == '__main__':
    main()
