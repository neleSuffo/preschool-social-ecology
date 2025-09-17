import argparse
import numpy as np
import librosa
import sqlite3
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path
from constants import AudioClassification, DataPaths
from config import AudioConfig
from models.speech_type.audio_classifier import build_model_multi_label, load_thresholds, extract_features

def load_inference_model(model_path):
    """
    Loads the trained model for inference by rebuilding the architecture and loading weights.

    Parameters
    ----------
    model_path : Path
        The path to the saved model file.

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
        model.load_weights(model_path)
        print("✅ Model loaded successfully by rebuilding architecture and loading weights.")
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

def aggregate_and_save_results(audio_file, predictions, class_names, db_cursor, thresholds, model_id):
    """
    Aggregates overlapping window predictions and saves the results to the database.
    For each second of audio, averages the probabilities from all overlapping windows
    and assigns binary labels based on the class-specific optimized thresholds.
    
    Parameters
    ----------
    audio_file : str
        Path to the audio file being processed.
    predictions : list of dict
        List of dictionaries containing start_time, end_time, and probabilities for each window.
    class_names : list
        List of class names corresponding to the model outputs.
    db_cursor : sqlite3.Cursor
        Database cursor for executing SQL commands.
    thresholds : dict
        Dictionary mapping class names to optimal thresholds
    model_id : int
        ID of the model used for classification
    """
    # Extract video_id from filename (assuming format like "123456.MP4.wav" -> video_id = 123456)
    video_filename = Path(audio_file).stem
    if video_filename.endswith('.MP4'):
        video_id = int(video_filename.replace('.MP4', ''))
    else:
        # Try to extract numeric part from filename
        import re
        numeric_match = re.search(r'(\d+)', video_filename)
        video_id = int(numeric_match.group(1)) if numeric_match else 0
    
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
                has_ohs, ohs_confidence_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            video_id, second, model_id,
            has_kchi, kchi_confidence,
            has_cds, cds_confidence,
            has_ohs, ohs_confidence
        ))

def process_audio_folder(folder_path, model, mlb, db_path, thresholds, model_id):
    """
    Process a folder of audio files and save classification results to database using optimized thresholds.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing audio files.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : sklearn.preprocessing.MultiLabelBinarizer
        MultiLabelBinarizer fitted on the training data.
    db_path : str
        Path to the SQLite database file.
    thresholds : dict
        Dictionary mapping class names to optimal thresholds
    model_id : int, optional
        ID of the model used for classification
    """
    folder_path = Path(folder_path)
    
    audio_files = list(folder_path.glob('*.wav'))
    if not audio_files:
        print(f"⚠️ No WAV audio files found in {folder_path}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files to process.")
    
    # Process files and save to database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        for i, audio_file in enumerate(audio_files):
            print("\n" + "="*50)
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
            print("="*50)
            
            try:
                prediction_results, _, _ = classify_audio_windows(str(audio_file), model, mlb)
                
                if prediction_results:
                    aggregate_and_save_results(
                        str(audio_file), prediction_results, mlb.classes_, 
                        cursor, thresholds, model_id
                    )
                    conn.commit()  # Commit after each file
                    print(f"✅ Saved results for {audio_file.name}")
                else:
                    print(f"No prediction results for {audio_file.name}")
            except Exception as e:
                print(f"❌ An error occurred while processing {audio_file.name}: {e}")

    print(f"\n✅ All processing complete. Results saved to database: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio classifier on a folder of audio files.")
    parser.add_argument('--audio_folder', type=str, required=True, help="Path to folder containing WAV audio files.")
    parser.add_argument('--db_path', type=str, default=DataPaths.INFERENCE_DB_PATH, help="Path to SQLite database file (default: inference.db)")
    args = parser.parse_args()
    
    # Load the trained model and MultiLabelBinarizer once
    results_dir = Path(AudioClassification.RESULTS_DIR)
    model_path = results_dir / "best_model.keras"
    model_id = AudioConfig.MODEL_ID
    
    model, mlb = load_inference_model(model_path)
    if model is None:
        exit()
    
    # Load optimized thresholds from training
    thresholds = load_thresholds(results_dir, mlb.classes_)
    
    print(f"📊 Using thresholds: {thresholds}")
    print(f"💾 Database path: {args.db_path}")

    # Process all audio files in the specified folder and save results to database
    process_audio_folder(args.audio_folder, model, mlb, args.db_path, thresholds, model_id)