import argparse
import numpy as np
import librosa
import sqlite3
import logging
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer
from constants import AudioClassification, DataPaths, Inference
from config import AudioConfig
from models.speech_type.audio_classifier import build_model_multi_label
from models.speech_type.utils import load_thresholds, extract_features
from utils import get_video_id, load_processed_videos, save_processed_video

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOG_FILE_PATH = Inference.SPEECH_LOG_FILE_PATH

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
        fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
        model = build_model_multi_label(
            n_mels=min(AudioConfig.N_MELS, 128),  # Use same capping as training
            fixed_time_steps=fixed_time_steps,
            num_classes=num_classes
        )
        model.load_weights(AudioClassification.TRAINED_WEIGHTS_PATH)
        return model, mlb
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def classify_audio_windows(audio_path, model, mlb, batch_size=32):
    """
    Classifies audio windows using the trained model.

    Parameters
    ----------
    audio_path : Path
        Path to the audio file to be classified.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : sklearn.preprocessing.MultiLabelBinarizer
        MultiLabelBinarizer fitted on the training data.
    batch_size : int, optional
        Batch size for prediction, by default 32

    Returns
    -------
    results : list of dict
        List of dictionaries containing start_time, end_time, and probabilities for each window.
    class_names : list
        List of class names corresponding to the model outputs.
    audio_duration : float
        Duration of the audio file in seconds.
    """
    y_audio, sr_audio = librosa.load(audio_path, sr=AudioConfig.SR)
    audio_duration = librosa.get_duration(y=y_audio, sr=sr_audio)
    all_window_data = []
    current_time = 0.0
    while current_time < audio_duration:
        window_start = current_time
        window_end = min(current_time + AudioConfig.WINDOW_DURATION, audio_duration)
        if (window_end - window_start) > 0.01:
            mel = extract_features(
                audio_path, window_start, AudioConfig.WINDOW_DURATION, 
                sr=AudioConfig.SR, n_mels=min(AudioConfig.N_MELS, 128), hop_length=AudioConfig.HOP_LENGTH
            )
            all_window_data.append({
                'start_time': window_start,
                'end_time': window_end,
                'mel_spec': mel
            })
        current_time += AudioConfig.WINDOW_STEP
    if not all_window_data:
        print("No valid windows extracted for classification.")
        return [], [], []
    mel_specs_batch = np.array([d['mel_spec'] for d in all_window_data])
    mel_specs_batch = np.expand_dims(mel_specs_batch, -1)
    predictions = model.predict(mel_specs_batch, batch_size=batch_size)
    results = []
    for i, data_point in enumerate(all_window_data):
        results.append({
            'start_time': data_point['start_time'],
            'end_time': data_point['end_time'],
            'probabilities': predictions[i]
        })
    return results, mlb.classes_, audio_duration

def aggregate_and_save_results(predictions, class_names, db_cursor, video_id, thresholds):
    """
    Aggregates overlapping window predictions and saves the results to the database.
    For each second of audio, averages the probabilities from all overlapping windows
    and assigns binary labels based on the class-specific optimized thresholds.
    
    Parameters
    ----------
    predictions : list of dict
        List of dictionaries containing start_time, end_time, and probabilities for each window.
    class_names : list
        List of class names corresponding to the model outputs.
    db_cursor : sqlite3.Cursor
        Database cursor for executing SQL commands.
    video_id : int
        ID of the video in the database
    thresholds : dict
        Dictionary mapping class names to their optimized threshold values.
    """
    # Organize predictions by second
    second_predictions = {}
    for p in predictions:
        start_second = int(np.floor(p['start_time']))
        end_second = int(np.ceil(p['end_time']))
        
        for second in range(start_second, end_second):
            if second not in second_predictions:
                second_predictions[second] = {'probabilities': [], 'count': 0}
            second_predictions[second]['probabilities'].append(p['probabilities'])
            second_predictions[second]['count'] += 1

    # Create class name to index mapping for easier access
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # Aggregate and save results using class-specific thresholds
    for second in sorted(second_predictions.keys()):
        avg_probs = np.mean(second_predictions[second]['probabilities'], axis=0)
        
        # Apply class-specific thresholds and extract probabilities
        has_ohs = 1 if avg_probs[class_to_idx['OHS']] > thresholds['OHS'] else 0
        ohs_confidence = float(avg_probs[class_to_idx['OHS']])
        
        has_cds = 1 if avg_probs[class_to_idx['CDS']] > thresholds['CDS'] else 0
        cds_confidence = float(avg_probs[class_to_idx['CDS']])
        
        has_kchi = 1 if avg_probs[class_to_idx['KCHI']] > thresholds['KCHI'] else 0
        kchi_confidence = float(avg_probs[class_to_idx['KCHI']])
        
        # Insert into database
        # Note: Using second as frame_number since we're working with 1-second intervals
        db_cursor.execute("""
            INSERT INTO AudioClassifications (
                video_id, frame_number, model_id,
                has_kchi, kchi_confidence_score,
                has_cds, cds_confidence_score,
                has_ohs, ohs_confidence_score) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, 
            second, 
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
    Process a single audio file and save classification results to database.

    Parameters
    ----------
    video_name : str
        Name of the video (without extension) to process.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : sklearn.preprocessing.MultiLabelBinarizer
        MultiLabelBinarizer fitted on the training data.
    cursor : sqlite3.Cursor
        Database cursor for executing SQL commands.
    """
    logging.info(f"Processing audio for video: {video_name}")
    
    # Load optimized thresholds from training
    thresholds = load_thresholds(mlb.classes_)
    
    # Get video_id from database
    video_id = get_video_id(video_name, cursor)
    if video_id is None:
        logging.error(f"Video ID not found for {video_name}")
        return
    
    # Look for exact audio file match
    audio_file_path = AudioClassification.QUANTEX_AUDIO_DIR / f"{video_name}.wav"
    
    if not audio_file_path.exists():
        logging.error(f"Audio file not found: {audio_file_path}")
        return

    try:
        # Classify audio windows
        prediction_results, _, _ = classify_audio_windows(str(audio_file_path), model, mlb)
        
        if prediction_results:
            aggregate_and_save_results(prediction_results, mlb.classes_, cursor, video_id, thresholds)
        else:
            logging.warning(f"No prediction results for {video_name}")
            
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