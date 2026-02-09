"""
Run Person Classification (YOLO Feature-BiLSTM) on a single video.
Generalized for 1 or 2 output classes.
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
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))

from person_classifier import YOLOFeatureExtractor as CNNEncoder, FrameRNNClassifier 
from utils import sequence_features_from_cnn

# --- Conceptual Config (REPLACE WITH REAL IMPORTS) ---
class PersonConfig:
    TARGET_LABELS_AGE_BINARY = ['child', 'adult'] 
    TARGET_LABELS_PERSON_ONLY = ['person']
    CONFIDENCE_THRESHOLD = 0.5
    WINDOW_SIZE = 60
    STRIDE = 30
    BATCH_SIZE_INFERENCE = 64
    BACKBONE = 'efficientnet_b0'
    FEAT_DIM = 512
class PersonClassification:
    OUTPUT_DIR = Path('./output')
    TRAINED_WEIGHTS_PATH = Path('./weights/best.pth')
# ------------------------------------------------------------------


def load_trained_model(device, num_outputs):
    """Load the trained YOLO Feature-BiLSTM model from checkpoint."""
    print(f"Loading model from {PersonClassification.TRAINED_WEIGHTS_PATH}")
    if not Path(PersonClassification.TRAINED_WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model weights not found at {PersonClassification.TRAINED_WEIGHTS_PATH}")
    
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    # Initialize models with same architecture as training
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim, num_outputs=num_outputs).to(device)
    
    def clean_state_dict(state_dict):
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'): cleaned[k[10:]] = v
            else: cleaned[k] = v
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
    """Setup video reader and get video information."""
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
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
    """Setup image transforms for preprocessing frames."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform


def extract_frames_batch(cap, start_frame, batch_size, transform, total_frames):
    """Extract a batch of frames from video."""
    frames = []
    frame_numbers = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for i in range(batch_size):
        current_frame = start_frame + i
        if current_frame >= total_frames: break
            
        ret, frame = cap.read()
        if not ret: break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        processed_frame = transform(pil_image)
        frames.append(processed_frame)
        frame_numbers.append(current_frame)
    
    if not frames: return None, [], 0
    
    frames_tensor = torch.stack(frames)
    return frames_tensor, frame_numbers, len(frames)


def run_inference_on_video(video_path, output_dir, device, batch_size, window_size, stride, class_names):
    """Run YOLO Feature-BiLSTM inference on video and save results to CSV."""
    num_outputs = len(class_names)
    cnn, rnn = load_trained_model(device, num_outputs)
    cap, video_info = setup_video_reader(video_path)
    transform = setup_transforms()
    
    if window_size is None: window_size = PersonConfig.WINDOW_SIZE
    if stride is None: stride = PersonConfig.STRIDE
    
    print(f"🎯 Processing video in {num_outputs}-output mode with:\n      Window size: {window_size} frames\n      Stride: {stride} frames\n      Batch size: {batch_size}\n      Device: {device}")
    
    total_frames = video_info['total_frames']
    
    # Initialize dictionary for predictions
    all_predictions = {'frame_number': list(range(total_frames))}
    for label in class_names:
        all_predictions[f'{label}_present'] = [0] * total_frames
        all_predictions[f'{label}_confidence'] = [0.0] * total_frames
    
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    if window_starts and window_starts[-1] + window_size < total_frames: window_starts.append(total_frames - window_size)
    
    progress_bar = tqdm(window_starts, desc="Processing windows", unit="window")
    
    with torch.no_grad():
        for window_start in progress_bar:
            window_end = min(window_start + window_size, total_frames)
            actual_window_size = window_end - window_start
            padding_size = window_size - actual_window_size if actual_window_size < window_size else 0
            
            frames_tensor, frame_numbers, actual_batch_size = extract_frames_batch(
                cap, window_start, actual_window_size, transform, total_frames
            )
            
            if frames_tensor is None: continue
            
            if padding_size > 0:
                padding_frames = frames_tensor[-1:].repeat(padding_size, 1, 1, 1)
                frames_tensor = torch.cat([frames_tensor, padding_frames], dim=0)
            
            frames_batch = frames_tensor.unsqueeze(0).to(device)
            lengths = torch.tensor([actual_window_size]).to(device)
            
            try:
                features = sequence_features_from_cnn(cnn, frames_batch, lengths, device)
                logits = rnn(features, lengths)  # Shape: (1, seq_len, NUM_OUTPUTS)
                probs = torch.sigmoid(logits)
                preds = (probs > PersonConfig.CONFIDENCE_THRESHOLD).float()
                
                window_probs = probs[0, :actual_window_size].cpu().numpy()
                window_preds = preds[0, :actual_window_size].cpu().numpy()
                
                last_idx = frame_numbers[-1]
                if 0 <= last_idx < total_frames:
                    for i, label in enumerate(class_names):
                        all_predictions[f'{label}_confidence'][last_idx] = float(window_probs[-1, i])
                        all_predictions[f'{label}_present'][last_idx] = int(window_preds[-1, i])
                
            except Exception as e:
                print(f"Warning: Error processing window starting at frame {window_start}: {e}")
                continue
            
            progress_bar.set_postfix({'frames_processed': min(window_end, total_frames)})
    
    cap.release()
    
    df = pd.DataFrame(all_predictions)
    # Keep only frames that match the stride interval
    df = df[df['frame_number'] % stride == (window_size - 1) % stride].reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_path / f"{Path(video_path).stem}_person_classification_{num_outputs}cls.csv"
    
    # Reorder columns: frame_number, person_present, person_confidence, ...
    cols = ['frame_number']
    for label in class_names:
        cols.extend([f'{label}_present', f'{label}_confidence'])
    df = df[cols]

    df.to_csv(output_csv, index=False)
    
    # Print summary statistics
    print(f"\n✅ Processing complete!")
    print(f"📊 Detection summary:")
    print(f"   Total frames: {len(df)}")
    for label in class_names:
        count = df[f'{label}_present'].sum()
        print(f"   {label.capitalize()} detections: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"📁 Results saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run Person Classification on video')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=PersonClassification.OUTPUT_DIR, help='Path to output directory')
    parser.add_argument('--batch_size', type=int, default=PersonConfig.BATCH_SIZE_INFERENCE, help='Batch size for processing frames')
    parser.add_argument('--window_size', type=int, default=PersonConfig.WINDOW_SIZE, help='Window size for sequence processing')
    parser.add_argument('--stride', type=int, default=PersonConfig.STRIDE, help='Stride for sliding window')
    parser.add_argument('--mode', choices=["person-only", "age-binary"], default="age-binary",
                       help='Select the classification mode to load the correct model.')
    
    args = parser.parse_args()
    
    if args.mode == "age-binary":
        class_names = PersonConfig.TARGET_LABELS_AGE_BINARY
    else:
        class_names = PersonConfig.TARGET_LABELS_PERSON_ONLY
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting Person Classification ({args.mode} mode)")
    
    try:
        run_inference_on_video(
            video_path=args.video_path,
            output_dir=args.output_dir,
            device=device,
            batch_size=args.batch_size,
            window_size=args.window_size,
            stride=args.stride,
            class_names=class_names
        )
        
        print(f"\n🎉 Successfully processed video!")
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # main()
    pass