import argparse
import logging
import sqlite3
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import List
from tqdm import tqdm
from constants import DataPaths, PersonClassification
from config import PersonConfig
from models.person.utils import load_model, sequence_features_from_cnn
from models.person.person_classifier import CNNEncoder, FrameRNNClassifier
from utils import get_video_id, get_frame_paths, extract_frame_number

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
FINISHED_LOG = Path("finished_videos.txt")

def process_video(
    video_name: str, 
    cnn: CNNEncoder, 
    rnn_model: FrameRNNClassifier,
    cursor: sqlite3.Cursor, 
    conn: sqlite3.Connection, 
    device: str,
    confidence_threshold: float = PersonConfig.CONFIDENCE_THRESHOLD,
    window_size: int = PersonConfig.WINDOW_SIZE, 
    stride: int = PersonConfig.STRIDE):
    """
    Process a sequence of pre-extracted frames for person classification using CNN + LSTM,
    in a sliding-window batch fashion with overlap averaging.

    Parameters
    ----------
    video_name : str
        Name of the video to process.
    frame_step : int
        Step size for frame processing.
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
    logging.info(f"Processing video: {video_name}")
    
    # Get video_id
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        return
    
    # Get frames directory
    frames_dir = DataPaths.IMAGES_INPUT_DIR / video_name
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return
    
    frame_files = get_frame_paths(frames_dir)
    
    if not frame_files:
        logging.warning(f"No frame files found for video: {video_name}")
        return

    # Extract frame numbers and create mapping
    frames_to_process = []
    for frame_file in frame_files:
        try:
            frame_number = extract_frame_number(frame_file.name)
            frames_to_process.append((frame_file, frame_number))
        except Exception:
            logging.warning(f"Could not extract frame number from: {frame_file.name}")
            continue
    

    # Sort by frame number
    frames_to_process.sort(key=lambda x: x[1])
    
    if not frames_to_process:
        logging.warning(f"No valid frames found for video: {video_name}")
        return

    total_frames = len(frames_to_process)

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize per-frame storage for averaging overlapping predictions
    all_predictions = {}
    for _, frame_number in frames_to_process:
        all_predictions[frame_number] = {
            'child_confidence': 0.0,
            'adult_confidence': 0.0,
            'child_person': 0,
            'adult_person': 0,
            'count': 0
        }
    
    # Calculate window start positions
    window_starts = list(range(0, total_frames - window_size + 1, stride))
    if window_starts and window_starts[-1] + window_size < total_frames:
        window_starts.append(total_frames - window_size)  # Ensure we cover all frames
    
    processed_frames = 0
    
    # Process frames with progress bar
    with tqdm(window_starts, desc=f"Classifying {video_name}", unit="window") as pbar:
        with torch.no_grad():
            for window_start in pbar:
                window_end = min(window_start + window_size, total_frames)
                actual_window_size = window_end - window_start
                
                if actual_window_size < window_size:
                    # Pad sequence if necessary
                    padding_size = window_size - actual_window_size
                else:
                    padding_size = 0
                
                # Get frames for this window
                window_frames = frames_to_process[window_start:window_end]
                
                # Load and preprocess frames
                frames = []
                frame_numbers = []
                for frame_file, frame_number in window_frames:
                    try:
                        img = Image.open(frame_file).convert('RGB')
                        frames.append(transform(img))
                        frame_numbers.append(frame_number)
                    except Exception as e:
                        logging.warning(f"Error loading frame {frame_file}: {e}")
                        continue
                
                if not frames:
                    continue
                
                frames_tensor = torch.stack(frames)
                
                # Add padding if necessary
                if padding_size > 0:
                    padding_frames = frames_tensor[-1:].repeat(padding_size, 1, 1, 1)
                    frames_tensor = torch.cat([frames_tensor, padding_frames], dim=0)
                
                # Add batch dimension and move to device
                frames_batch = frames_tensor.unsqueeze(0).to(device)  # Shape: (1, seq_len, C, H, W)
                lengths = torch.tensor([actual_window_size]).to(device)
                
                try:
                    # Extract CNN features
                    features = sequence_features_from_cnn(cnn, frames_batch, lengths, device)
                    
                    # Run RNN classification
                    logits = rnn_model(features, lengths)  # Shape: (1, seq_len, 2)
                    
                    # Convert to probabilities
                    probs = torch.sigmoid(logits)  # Shape: (1, seq_len, 2)
                    
                    # Apply threshold for binary classification
                    preds = (probs > confidence_threshold).float()
                    
                    # Extract results for this window (excluding padding)
                    window_probs = probs[0, :actual_window_size].cpu().numpy()  # (actual_window_size, 2)
                    window_preds = preds[0, :actual_window_size].cpu().numpy()   # (actual_window_size, 2)
                    
                    # Update predictions for frames in this window (overlap averaging)
                    for i, frame_number in enumerate(frame_numbers):
                        pred_data = all_predictions[frame_number]
                        
                        # Accumulate confidence scores and predictions
                        pred_data['child_confidence'] += float(window_probs[i, 0])
                        pred_data['adult_confidence'] += float(window_probs[i, 1])
                        pred_data['child_person'] += int(window_preds[i, 0])
                        pred_data['adult_person'] += int(window_preds[i, 1])
                        pred_data['count'] += 1
                    
                    processed_frames += len(frame_numbers)
                    
                except Exception as e:
                    logging.warning(f"Error processing window starting at frame {window_start}: {e}")
                    continue
                
                # Update progress bar
                pbar.set_postfix({
                    'frames_processed': processed_frames
                })
                
                # Commit every 100 frames
                if processed_frames % 100 == 0:
                    conn.commit()
    
    # Average predictions and insert into database
    for frame_number, pred_data in all_predictions.items():
        if pred_data['count'] > 0:
            # Average the accumulated values
            avg_child_conf = pred_data['child_confidence'] / pred_data['count']
            avg_adult_conf = pred_data['adult_confidence'] / pred_data['count']
            child_pred = 1 if (pred_data['child_person'] / pred_data['count']) > 0.5 else 0
            adult_pred = 1 if (pred_data['adult_person'] / pred_data['count']) > 0.5 else 0
            
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
                    int(adult_pred),
                    float(avg_adult_conf),
                    int(child_pred),
                    float(avg_child_conf)
                ))
            except Exception as e:
                logging.error(f"DB insert error for frame {frame_number}: {e}")
    
    conn.commit()
    logging.info(f"Processed video {video_name}: {len(all_predictions)} frames classified")
    
    # Append to finished videos log
    with FINISHED_LOG.open("a") as f:
        f.write(video_name + "\n")


def main(video_list: List[str]):
    """
    Main function to process videos for person classification
    
    Parameters:
    ----------
    video_list : List[str]
        List of video names to process
    """
    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # Load model
    try:
        cnn, rnn_model = load_model(device)
        cnn.eval()
        rnn_model.eval()
        logging.info(f"Loaded person classification model from {PersonClassification.TRAINED_WEIGHTS_PATH}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        conn.close()
        return
    
    # Process each video
    for video_name in video_list:        
        try:
            process_video(video_name, cnn, rnn_model, cursor, conn, device)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()
    logging.info("Person classification processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run person classification on extracted video frames")
    parser.add_argument("--video_list", nargs='+', required=True, 
                    help="List of video names to process")
    
    args = parser.parse_args()
    main(args.video_list)