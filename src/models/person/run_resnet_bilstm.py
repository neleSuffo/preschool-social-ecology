"""
Run ResNet-BiLSTM person classification on a single video.
Outputs frame-by-frame predictions in CSV format similar to YOLO face detection output.

Usage:
    python models/person/run_resnet_bilstm.py --video_path /path/to/video.mp4
"""

import argparse
import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PersonConfig
from constants import PersonClassification
from person_classifier import CNNEncoder, FrameRNNClassifier
from utils import sequence_features_from_cnn


def load_trained_model(device):
    """Load the trained ResNet-BiLSTM model from checkpoint.
    
    Parameters
    ----------
    device : torch.device
        Device to load the model on.
        
    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Loaded CNN and RNN models.
    """
    print(f"Loading model from {PersonClassification.TRAINED_WEIGHTS_PATH}")
    if not Path(PersonClassification.TRAINED_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model weights not found at {PersonClassification.TRAINED_WEIGHTS_PATH}")
    
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    # Initialize models with same architecture as training
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)
    
    # Handle compiled models (strip _orig_mod prefix if present)
    def clean_state_dict(state_dict):
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                cleaned[k] = v
        return cleaned
    
    if "cnn_state" in checkpoint and "rnn_state" in checkpoint:
        cnn.load_state_dict(clean_state_dict(checkpoint["cnn_state"]))
        rnn_model.load_state_dict(clean_state_dict(checkpoint["rnn_state"]))
    else:
        raise ValueError(
            "Checkpoint does not contain expected CNN/RNN keys. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    cnn.eval()
    rnn_model.eval()
    
    print(f"✅ Model loaded successfully")
    
    return cnn, rnn_model


def setup_video_reader(video_path):
    """Setup video reader and get video information.
    
    Parameters
    ----------
    video_path : str
        Path to the video file.
        
    Returns
    -------
    Tuple[cv2.VideoCapture, dict]
        Video capture object and video info dictionary.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_info = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': duration
    }
    
    return cap, video_info


def setup_transforms():
    """Setup image transforms for preprocessing frames.
    
    Returns
    -------
    torchvision.transforms.Compose
        Composed transforms for preprocessing.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def extract_frames_batch(cap, start_frame, batch_size, transform, total_frames):
    """Extract a batch of frames from video.
    
    Parameters
    ----------
    cap : cv2.VideoCapture
        Video capture object.
    start_frame : int
        Starting frame number.
    batch_size : int
        Number of frames to extract.
    transform : torchvision.transforms.Compose
        Image transforms.
    total_frames : int
        Total number of frames in video.
        
    Returns
    -------
    Tuple[torch.Tensor, list, int]
        Batch of processed frames, frame numbers, and actual batch size.
    """
    frames = []
    frame_numbers = []
    
    # Set video position to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(batch_size):
        current_frame = start_frame + i
        if current_frame >= total_frames:
            break
            
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        processed_frame = transform(pil_image)
        frames.append(processed_frame)
        frame_numbers.append(current_frame)
    
    if not frames:
        return None, [], 0
    
    # Stack frames into batch tensor
    frames_tensor = torch.stack(frames)
    return frames_tensor, frame_numbers, len(frames)


def run_inference_on_video(video_path, output_dir, device, batch_size, window_size, stride):
    """Run ResNet-BiLSTM inference on video and save results to CSV.
    
    Parameters
    ----------
    video_path : str
        Path to input video file.
    output_dir : str
        Path to output directory.
    device : torch.device
        Device for inference.
    batch_size : int, default=64
        Batch size for processing frames.
    window_size : int, optional
        Window size for sequence processing
    stride : int, optional
        Stride for sliding window.
    """
    # Load trained model
    cnn, rnn = load_trained_model(device)
    
    # Setup video reader
    cap, video_info = setup_video_reader(video_path)
    
    # Setup transforms
    transform = setup_transforms()
    
    # Use config values if not specified
    if window_size is None:
        window_size = PersonConfig.WINDOW_SIZE
    if stride is None:
        stride = PersonConfig.STRIDE
    
    print(f"🎯 Processing video with:")
    print(f"      Window size: {window_size} frames")
    print(f"      Stride: {stride} frames")
    print(f"      Batch size: {batch_size}")
    print(f"      Device: {device}")
    
    total_frames = video_info['total_frames']
    results = []
    
    # Initialize all frames with default values (no person detected)
    all_predictions = {
        'frame_number': list(range(total_frames)),
        'child_person': [0] * total_frames,
        'adult_person': [0] * total_frames,
        'child_confidence': [0.0] * total_frames,
        'adult_confidence': [0.0] * total_frames
    }
    
    # Process video in sliding windows
    print("\n🔄 Processing video frames...")
    
    # Calculate window start positions
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    if window_starts[-1] + window_size < total_frames:
        window_starts.append(total_frames - window_size)  # Ensure we cover all frames
    
    progress_bar = tqdm(window_starts, desc="Processing windows", unit="window")
    
    with torch.no_grad():
        for window_start in progress_bar:
            window_end = min(window_start + window_size, total_frames)
            actual_window_size = window_end - window_start
            
            if actual_window_size < window_size:
                # Pad sequence if necessary
                padding_size = window_size - actual_window_size
            else:
                padding_size = 0
            
            # Extract frames for this window
            frames_tensor, frame_numbers, actual_batch_size = extract_frames_batch(
                cap, window_start, actual_window_size, transform, total_frames
            )
            
            if frames_tensor is None:
                continue
            
            # Add padding if necessary
            if padding_size > 0:
                padding_frames = frames_tensor[-1:].repeat(padding_size, 1, 1, 1)
                frames_tensor = torch.cat([frames_tensor, padding_frames], dim=0)
            
            # Add batch dimension and move to device
            frames_batch = frames_tensor.unsqueeze(0).to(device)  # Shape: (1, seq_len, C, H, W)
            lengths = torch.tensor([actual_window_size]).to(device)
            
            # Run inference
            try:
                # Extract CNN features
                features = sequence_features_from_cnn(cnn, frames_batch, lengths, device)
                
                # Run RNN classification
                logits = rnn(features, lengths)  # Shape: (1, seq_len, 2)
                
                # Convert to probabilities
                probs = torch.sigmoid(logits)  # Shape: (1, seq_len, 2)
                
                # Apply threshold for binary classification
                preds = (probs > PersonConfig.CONFIDENCE_THRESHOLD).float()
                
                # Extract results for this window (excluding padding)
                window_probs = probs[0, :actual_window_size].cpu().numpy()  # (actual_window_size, 2)
                window_preds = preds[0, :actual_window_size].cpu().numpy()   # (actual_window_size, 2)
                
                # Update predictions for frames in this window
                for i, frame_idx in enumerate(frame_numbers):
                    if 0 <= frame_idx < total_frames:
                        # Use max confidence across overlapping windows
                        current_child_conf = all_predictions['child_confidence'][frame_idx]
                        current_adult_conf = all_predictions['adult_confidence'][frame_idx]
                        
                        new_child_conf = float(window_probs[i, 0])
                        new_adult_conf = float(window_probs[i, 1])
                        
                        if new_child_conf > current_child_conf:
                            all_predictions['child_confidence'][frame_idx] = new_child_conf
                            all_predictions['child_person'][frame_idx] = int(window_preds[i, 0])
                        
                        if new_adult_conf > current_adult_conf:
                            all_predictions['adult_confidence'][frame_idx] = new_adult_conf
                            all_predictions['adult_person'][frame_idx] = int(window_preds[i, 1])
                
            except Exception as e:
                print(f"Warning: Error processing window starting at frame {window_start}: {e}")
                continue
            
            # Update progress bar
            progress_bar.set_postfix({
                'frames_processed': min(window_end, total_frames)
            })
    
    # Release video capture
    cap.release()
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(all_predictions)
    
    # Reorder columns to match YOLO face detection style
    df = df[['frame_number', 'child_person', 'adult_person', 'child_confidence', 'adult_confidence']]
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / f"{Path(video_path).stem}_person_classification.csv"
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    total_child_detections = df['child_person'].sum()
    total_adult_detections = df['adult_person'].sum()
    total_no_person = len(df) - total_child_detections - total_adult_detections
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Detection summary:")
    print(f"   Total frames: {len(df)}")
    print(f"   Child detections: {total_child_detections} ({total_child_detections/len(df)*100:.1f}%)")
    print(f"   Adult detections: {total_adult_detections} ({total_adult_detections/len(df)*100:.1f}%)")
    print(f"   No person: {total_no_person} ({total_no_person/len(df)*100:.1f}%)")
    print(f"📁 Results saved to: {output_csv}")
    
    return df


def main():
    """Main function to run ResNet-BiLSTM inference on video."""
    parser = argparse.ArgumentParser(description='Run ResNet-BiLSTM person classification on video')
    parser.add_argument('--video_path', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=PersonClassification.OUTPUT_DIR,
                       help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=PersonConfig.BATCH_SIZE_INFERENCE,
                       help='Batch size for processing frames (default: config value)')
    parser.add_argument('--window_size', type=int, default=PersonConfig.WINDOW_SIZE,
                       help='Window size for sequence processing (default: config value)')
    parser.add_argument('--stride', type=int, default=PersonConfig.STRIDE,
                       help='Stride for sliding window (default: config value)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting ResNet-BiLSTM person classification")
    
    try:
        # Run inference
        results_df = run_inference_on_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride=args.stride
        )
        
        print(f"\n🎉 Successfully processed video!")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
