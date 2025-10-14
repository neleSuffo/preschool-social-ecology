import argparse
import numpy as np
import librosa
import sqlite3
import logging
import shutil
import re
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.preprocessing import MultiLabelBinarizer
from constants import AudioClassification, DataPaths, Inference
from config import AudioConfig, DataConfig
from models.speech_type.audio_classifier import build_model_multi_label 
from models.speech_type.utils import load_thresholds, extract_features 
from utils import get_video_id, load_processed_videos, save_processed_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_PATH = Inference.SPEECH_LOG_FILE_PATH

def parse_rttm_snippets(rttm_path: Path, video_name: str) -> List[Tuple[float, float]]:
    """
    Parses a single large RTTM file, extracts speech segments (start, duration),
    and filters results for the specified video_name (fileID).
    
    Parameters
    ----------
    rttm_path : Path
        Path to the RTTM file.
    video_name : str
        Name of the video to filter snippets for (without extension).
    
    Returns
    -------
    snippets : List[Tuple[float, float]]
        List of (start_time, duration) tuples for the specified video.
    """
    snippets = []
    
    # Patterns to match the video name exactly in the RTTM's fileID field (parts[1])
    match_patterns = {video_name, f"{video_name}.wav", f"{video_name}.mp4", f"{video_name}.MP4"}

    if not rttm_path.exists():
        logging.warning(f"Single large RTTM file not found: {rttm_path}.")
        return snippets
        
    with open(rttm_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # Check for SPEAKER tag and minimum parts (5 for duration)
            if len(parts) >= 5 and parts[0] == 'SPEAKER':
                rttm_file_id = parts[1] # This is the video identifier
                
                # Filter by video name
                if rttm_file_id in match_patterns:
                    try:
                        start = float(parts[3])
                        duration = float(parts[4])
                        if duration > 0:
                            snippets.append((start, duration))
                    except ValueError:
                        logging.warning(f"Skipping malformed RTTM line: {line.strip()}")
    return snippets

def load_inference_model():
    """
    Loads the trained model for inference by rebuilding the architecture and loading weights.

    Returns
    -------
    model : tf.keras.Model
        The loaded Keras model.
    mlb : sklearn.preprocessing.MultiLabelBinarizer
        The fitted MultiLabelBinarizer for class labels.
    """
    try:
        training_classes = AudioConfig.VALID_RTTM_CLASSES
        mlb = MultiLabelBinarizer(classes=training_classes)
        mlb.fit([[]])
        num_classes = len(mlb.classes_)
        model = build_model_multi_label(num_classes=num_classes)
        model.load_weights(AudioClassification.TRAINED_WEIGHTS_PATH)
        return model, mlb
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
    
def classify_audio_snippets(audio_path: Path, snippets: List[Tuple[float, float]], model, mlb, batch_size=32):
    """
    Classifies VTC-extracted audio snippets using the trained model.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file.
    snippets : List[Tuple[float, float]]
        List of (start_time, duration) tuples from the VTC RTTM output.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : MultiLabelBinarizer
        The MultiLabelBinarizer used during training.
    batch_size : int
        Batch size for model prediction.

    Returns
    -------
    results : list of dict
        List of dictionaries containing start_time, duration, and probabilities for each snippet.
    """
    if not snippets:
        logging.info(f"No VTC snippets found for {audio_path.name}.")
        return []
        
    all_snippet_data = []
    
    # 1. Extract features for all snippets
    for start_time, duration in snippets:
        mel = extract_features(
            str(audio_path), start_time, duration, 
            sr=AudioConfig.SR, 
            n_mels=AudioConfig.N_MELS, 
            hop_length=AudioConfig.HOP_LENGTH,
        )
        all_snippet_data.append({
            'start_time': start_time,
            'duration': duration,
            'mel_spec': mel
        })
        
    mel_specs_batch = np.array([d['mel_spec'] for d in all_snippet_data])
    mel_specs_batch = np.expand_dims(mel_specs_batch, -1)
    
    # 2. Predict on batch
    predictions = model.predict(mel_specs_batch, batch_size=batch_size)
    
    # 3. Format results
    results = []
    for i, data_point in enumerate(all_snippet_data):
        results.append({
            'start_time': data_point['start_time'],
            'duration': data_point['duration'],
            'end_time': data_point['start_time'] + data_point['duration'],
            'probabilities': predictions[i]
        })
    return results

def aggregate_and_save_results(predictions: List[Dict], class_names: List[str], db_cursor: sqlite3.Cursor, video_id: int, thresholds: Dict):
    """
    Saves classification results to the database by applying the single prediction
    of each snippet across its entire duration.
    
    Parameters
    ----------
    predictions : List[Dict]
        List of dictionaries containing start_time, end_time, and probabilities for each snippet.
    class_names : List[str]
        List of class names corresponding to the model's output indices.
    db_cursor : sqlite3.Cursor
        Database cursor for saving results.
    video_id : int
        ID of the video in the database.
    thresholds : Dict
        Dictionary of thresholds for each class to determine binary labels.
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Iterate over each classified snippet
    for p in predictions:
        start_sec = p['start_time']
        end_sec = p['end_time']
        avg_probs = p['probabilities']
        
        # Determine binary labels and confidence for the entire snippet
        has_ohs = 1 if avg_probs[class_to_idx['OHS']] > thresholds['OHS'] else 0
        ohs_confidence = float(avg_probs[class_to_idx['OHS']])
        
        has_cds = 1 if avg_probs[class_to_idx['CDS']] > thresholds['CDS'] else 0
        cds_confidence = float(avg_probs[class_to_idx['CDS']])
        
        has_kchi = 1 if avg_probs[class_to_idx['KCHI']] > thresholds['KCHI'] else 0
        kchi_confidence = float(avg_probs[class_to_idx['KCHI']])
        
        # Calculate the frame range covered by this snippet
        start_frame = int(np.floor(start_sec * DataConfig.FPS))
        end_frame = int(np.ceil(end_sec * DataConfig.FPS))

        # Find the first frame that is a multiple of FRAME_STEP_INTERVAL and >= start_frame
        first_frame = ((start_frame + DataConfig.FRAME_STEP_INTERVAL - 1) // DataConfig.FRAME_STEP_INTERVAL) * DataConfig.FRAME_STEP_INTERVAL

        # Insert records for all frames covered by this snippet
        for frame_number in range(first_frame, end_frame, DataConfig.FRAME_STEP_INTERVAL):
            db_cursor.execute("""
                INSERT INTO AudioClassifications (
                    video_id, frame_number, model_id,
                    has_kchi, kchi_confidence_score,
                    has_cds, cds_confidence_score,
                    has_ohs, ohs_confidence_score) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id, 
                frame_number, 
                int(AudioConfig.MODEL_ID),
                has_kchi, 
                kchi_confidence,
                has_cds, 
                cds_confidence,
                has_ohs, 
                ohs_confidence
            ))


def process_audio_file(video_name: str, model, mlb, cursor: sqlite3.Cursor):
    """
    Process a single audio file using VTC snippets and save classification results to database.
    
    Parameters:
    ----------
    video_name : str
        Name of the video (without extension)
    model : tf.keras.Model  
        The trained audio classification model
    mlb : MultiLabelBinarizer
        The MultiLabelBinarizer used during training
    cursor : sqlite3.Cursor
        Database cursor for saving results
    """
    logging.info(f"Processing audio for video: {video_name}")
    
    thresholds = load_thresholds(mlb.classes_)
    video_id = get_video_id(video_name, cursor)
    
    if video_id is None:
        logging.error(f"Video ID not found for {video_name}")
        return
    
    audio_file_path = AudioClassification.QUANTEX_AUDIO_DIR / f"{video_name}.wav"
    
    if not audio_file_path.exists():
        logging.error(f"Audio file not found: {audio_file_path}")
        return
    
    try:
        # 1. Parse RTTM file to get snippet boundaries
        snippets = parse_rttm_snippets(AudioClassification.VTC_RTTM_FILE, video_name)
        
        if not snippets:
            logging.warning(f"No speech snippets found in VTC output for {video_name}. Skipping classification.")
            return

        # 2. Classify RTTM snippets (replaces classify_audio_windows)
        prediction_results = classify_audio_snippets(str(audio_file_path), snippets, model, mlb)
        
        if prediction_results:
            # 3. Save the single snippet prediction across its entire frame range
            aggregate_and_save_results(prediction_results, mlb.classes_, cursor, video_id, thresholds)
        else:
            logging.warning(f"Classification returned no results for VTC snippets in {video_name}")
            
    except Exception as e:
        logging.error(f"❌ Error processing audio for {video_name}: {e}")

def main(video_list: List[str]):
    """
    Main function to process videos for audio classification
    
    Parameters:
    ----------
    video_list : List[str]
        List of video names to process
    """
    # Setup processing log file
    processed_videos = load_processed_videos(LOG_FILE_PATH)
    
    # Filter out already processed videos
    videos_to_process = [v for v in video_list if v not in processed_videos]
    skipped_videos = [v for v in video_list if v in processed_videos]
        
    if not videos_to_process:
        logging.info("All requested videos have already been processed!")
        return
    
    # Connect to database
    conn = sqlite3.connect(DataPaths.INFERENCE_DB_PATH)
    cursor = conn.cursor()
    
    model, mlb = load_inference_model()
    if model is None:
        logging.error("Failed to load model")
        conn.close()
        return
    
    # Process each video
    for video_name in videos_to_process:
        try:
            process_audio_file(video_name, model, mlb, cursor)
            save_processed_video(LOG_FILE_PATH, video_name)
            conn.commit()  # Commit after each video
        except Exception as e:
            logging.error(f"Error processing video {video_name}: {e}")
            continue
    
    conn.close()
    logging.info("Audio classification processing completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio classifier on video list")
    parser.add_argument("--video_list", nargs='+', required=True, 
                    help="List of video names to process")
    parser.add_argument("--force", action='store_true',
                    help="Force reprocessing of already processed videos")
    args = parser.parse_args()
    
        # Handle force reprocessing
    if args.force:
        logging.info("Force flag enabled - will reprocess all videos")
        if LOG_FILE_PATH.exists():
            # Create backup of current log
            backup_path = LOG_FILE_PATH.with_suffix('.txt.backup')
            import shutil
            shutil.copy2(LOG_FILE_PATH, backup_path)
            LOG_FILE_PATH.unlink()  # Remove current log

    main(args.video_list)