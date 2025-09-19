import argparse
import logging
import sqlite3
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import List, Set
from tqdm import tqdm
from constants import DataPaths, PersonClassification
from config import PersonConfig
from models.person.utils import load_model, sequence_features_from_cnn
from models.person.person_classifier import CNNEncoder, FrameRNNClassifier
from utils import get_video_id, get_frame_paths, extract_frame_number, load_processed_videos, save_processed_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_video(
    video_name: str,
    cnn: CNNEncoder,
    rnn_model: FrameRNNClassifier,
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    device: str,
    confidence_threshold: float = PersonConfig.CONFIDENCE_THRESHOLD,
    window_size: int = PersonConfig.WINDOW_SIZE,
    stride: int = PersonConfig.STRIDE
):
    """
    Process pre-extracted frames for person classification using CNN + BiLSTM,
    keeping temporal context but storing only the last frame per window.
    """
    logging.info(f"Processing video: {video_name}")

    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        logging.warning(f"No video_id found for {video_name}")
        return

    frames_dir = DataPaths.IMAGES_INPUT_DIR / video_name
    if not frames_dir.exists():
        logging.error(f"Frames directory not found: {frames_dir}")
        return

    frame_files = get_frame_paths(frames_dir)
    if not frame_files:
        logging.warning(f"No frame files found for video: {video_name}")
        return

    # Extract frame numbers
    frames_to_process = []
    for frame_file in frame_files:
        try:
            frame_number = extract_frame_number(frame_file.name)
            frames_to_process.append((frame_file, frame_number))
        except Exception:
            logging.warning(f"Could not extract frame number from {frame_file.name}")
            continue

    frames_to_process.sort(key=lambda x: x[1])
    total_frames = len(frames_to_process)
    if total_frames == 0:
        logging.warning(f"No valid frames found for video: {video_name}")
        return

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Initialize storage for only stride-aligned frames (last frame of each window)
    all_predictions = {}

    window_starts = list(range(0, total_frames - window_size + 1, stride))
    if window_starts and window_starts[-1] + window_size < total_frames:
        window_starts.append(total_frames - window_size)  # ensure full coverage

    processed_frames = 0

    with tqdm(window_starts, desc=f"Classifying {video_name}", unit="window") as pbar:
        with torch.no_grad():
            for window_start in pbar:
                window_end = min(window_start + window_size, total_frames)
                actual_window_size = window_end - window_start
                padding_size = max(0, window_size - actual_window_size)

                # Load and preprocess frames in window
                window_frames = frames_to_process[window_start:window_end]
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
                if padding_size > 0:
                    padding_frames = frames_tensor[-1:].repeat(padding_size, 1, 1, 1)
                    frames_tensor = torch.cat([frames_tensor, padding_frames], dim=0)

                frames_batch = frames_tensor.unsqueeze(0).to(device)
                lengths = torch.tensor([actual_window_size]).to(device)

                try:
                    # CNN + BiLSTM
                    features = sequence_features_from_cnn(cnn, frames_batch, lengths, device)
                    logits = rnn_model(features, lengths)
                    probs = torch.sigmoid(logits)
                    preds = (probs > confidence_threshold).float()

                    # Store predictions only for frames that are multiples of stride
                    # This gives us every 30th frame when stride=30
                    for i in range(actual_window_size):
                        frame_idx_in_video = window_start + i
                        frame_number = frame_numbers[i]
                        
                        # Only store if this frame index is a multiple of stride
                        if frame_idx_in_video % stride == 0:
                            frame_prob = probs[0, i].cpu().numpy()   # (2,)
                            frame_pred = preds[0, i].cpu().numpy()   # (2,)
                            
                            all_predictions[frame_number] = {
                                'child_confidence': float(frame_prob[0]),
                                'adult_confidence': float(frame_prob[1]),
                                'child_person': int(frame_pred[0]),
                                'adult_person': int(frame_pred[1])
                            }

                    processed_frames += 1

                except Exception as e:
                    logging.warning(f"Error processing window starting at frame {window_start}: {e}")
                    continue

                pbar.set_postfix({'frames_processed': processed_frames})
                if processed_frames % 100 == 0:
                    conn.commit()

    # Insert only stride-aligned predictions into database
    for frame_number, pred_data in all_predictions.items():
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
                int(pred_data['adult_person']),
                float(pred_data['adult_confidence']),
                int(pred_data['child_person']),
                float(pred_data['child_confidence'])
            ))
        except Exception as e:
            logging.error(f"DB insert error for frame {frame_number}: {e}")

    conn.commit()

def main(video_list: List[str]):
    """
    Main function to process videos for person classification
    """
    # Setup processing log file
    processed_videos = load_processed_videos(Inference.PERSON_LOG_FILE_PATH)
    
    # Filter out already processed videos
    videos_to_process = [v for v in video_list if v not in processed_videos]
    skipped_videos = [v for v in video_list if v in processed_videos]
        
    if not videos_to_process:
        logging.info("All requested videos have already been processed!")
        return
        
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        cnn, rnn_model = load_model(device)
        cnn.eval()
        rnn_model.eval()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        conn.close()
        return

    successfully_processed = 0
    
    for video_name in videos_to_process:
        try:
            process_video(video_name, cnn, rnn_model, cursor, conn, device)
            save_processed_video(log_file_path, video_name)
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue

    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run person classification on extracted video frames")
    parser.add_argument("--video_list", nargs='+', required=True,
                        help="List of video names to process")
    parser.add_argument("--force", action='store_true',
                        help="Force reprocessing of already processed videos")
    args = parser.parse_args()
    
    # Handle force reprocessing
    if args.force:
        logging.info("Force flag enabled - will reprocess all videos")
        log_file_path = Inference.PERSON_LOG_FILE_PATH
        if log_file_path.exists():
            # Create backup of current log
            backup_path = log_file_path.with_suffix('.txt.backup')
            import shutil
            shutil.copy2(log_file_path, backup_path)
            log_file_path.unlink()  # Remove current log

    main(args.video_list)