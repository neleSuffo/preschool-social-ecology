import argparse
import logging
import sqlite3
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Import model classes and constants
from constants import DataPaths, PersonClassification
from config import PersonConfig, PersonConfig
from models.person.person_classifier import load_model, CNNEncoder, FrameRNNClassifier
from utils import get_video_id, get_frame_paths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
FINISHED_LOG = Path("finished_videos.txt")

def process_video_frames(video_name: str, cnn: CNNEncoder, rnn_model: FrameRNNClassifier,
                         cursor, conn, device: str,
                         confidence_threshold: float = PersonConfig.CONFIDENCE_THRESHOLD,
                         window_size: int = PersonConfig.WINDOW_SIZE, stride: int = PersonConfig.STRIDE):
    """
    Process a sequence of pre-extracted frames for person classification using CNN + LSTM,
    in a sliding-window batch fashion with overlap averaging.

    Parameters
    ----------
    video_name : str
        Name of the video to process
    cnn : CNNEncoder
        Pretrained CNN encoder.
    rnn_model : FrameRNNClassifier
        LSTM-based classifier for temporal inference.
    cursor : sqlite3.Cursor
        Database cursor for insert operations.
    conn : sqlite3.Connection
        Database connection for commits.
    device : str
        Device to run inference on ('cpu' or 'cuda').
    confidence_threshold : float
        Threshold for binary classification.
    window_size : int
        Number of frames per batch/window.
    stride : int
        Step size to move the window.
    """
    # Get video_id
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        return
    
    # 1. Get all frames for this video (sorted by frame number)
    video_frame_dir = DataPaths.IMAGES_INPUT_DIR / video_name
    frame_paths = get_frame_paths(video_frame_dir)
    if len(frame_paths) == 0:
        logging.warning(f"No frames found for video {video_name}")
        return

    # 2. Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Initialize per-frame storage for averaging overlapping predictions
    frame_indices = [int(fp.stem.split('_')[-1]) for fp in frame_paths]
    frame_preds_sum = np.zeros((len(frame_paths), 2), dtype=float)
    frame_preds_count = np.zeros(len(frame_paths), dtype=int)
    
    # 4. Sliding-window processing
    for start_idx in range(0, len(frame_paths), stride):
        end_idx = min(start_idx + window_size, len(frame_paths))
        batch_paths = frame_paths[start_idx:end_idx]
        batch_indices = frame_indices[start_idx:end_idx]
        
        frames = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert('RGB')
                frames.append(transform(img))
            except Exception as e:
                logging.warning(f"Error loading frame {fp}: {e}")
                continue
        
        if len(frames) == 0:
            continue
        
        frames_tensor = torch.stack(frames).to(device)
        
        # CNN feature extraction
        with torch.no_grad():
            cnn_feats = cnn(frames_tensor).unsqueeze(0)
            lengths = torch.tensor([cnn_feats.size(1)], dtype=torch.long).to(device)
            logits = rnn_model(cnn_feats, lengths)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # shape: (batch_size, 2)
        
        # Accumulate predictions for averaging
        for i, idx in enumerate(batch_indices):
            frame_preds_sum[i] += probs[i]
            frame_preds_count[i] += 1

    # 5. Compute averaged predictions per frame
    averaged_probs = frame_preds_sum / frame_preds_count[:, None]
    preds = (averaged_probs > confidence_threshold).astype(int)
    
    # 6. Insert into database
    for idx, frame_number in enumerate(frame_indices):
        try:
            cursor.execute('''
                INSERT INTO PersonClassifications 
                (video_id, frame_number, model_id, has_adult_person, adult_confidence_score,
                 has_child_person, child_confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                int(video_id),
                int(frame_number),
                int(PersonConfig.MODEL_ID),
                int(preds[idx, 1]),
                float(averaged_probs[idx, 1]),
                int(preds[idx, 0]),
                float(averaged_probs[idx, 0])
            ))
        except Exception as e:
            logging.error(f"DB insert error for frame {frame_number}: {e}")
            
    conn.commit() 
    logging.info(f"Processed video {video_name}: {len(frame_indices)} frames classified")
    
    # Append to finished videos log
    with FINISHED_LOG.open("a") as f:
        f.write(video_name + "\n")


def main(video_list: List[str], frame_step: int = 10, device: str = 'auto'):
    """
    Main function to process videos for person classification
    
    Parameters:
    ----------
    video_list : List[str]
        List of video names to process
    frame_step : int
        Step size for frame processing (default: 10)
    device : str
        Device to use for inference ('cpu', 'cuda', or 'auto')
    """
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else device
    
    cnn, rnn_model = load_model(device, PersonClassification.TRAINED_WEIGHTS_PATH)
    cnn.eval()
    rnn_model.eval()
    
    # Load finished videos
    finished_videos = set()
    if FINISHED_LOG.exists():
        with FINISHED_LOG.open("r") as f:
            finished_videos = set(line.strip() for line in f.readlines())
    
    for video_name in video_list:
        if video_name in finished_videos:
            logging.info(f"Skipping already processed video: {video_name}")
            continue
        try:
            process_video_frames(video_name, cnn, rnn_model, cursor, conn, device)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()
    logging.info("Person classification processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run person classification on extracted video frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                       help="List of video names to process")
    parser.add_argument("--frame_step", type=int, default=10, 
                       help="Frame step for processing (default: 10)")
    parser.add_argument("--device", type=str, default='auto',
                       help="Device to use (cpu, cuda, or auto)")
    
    args = parser.parse_args()
    main(args.video_list, args.frame_step, args.device)