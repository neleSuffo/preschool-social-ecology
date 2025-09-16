import argparse
import time
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.metrics import Metric
from pathlib import Path
from constants import AudioClassification
from config import AudioConfig
from audio_classifier import build_model_multi_label

# --- Custom Metrics (from your script) ---
class MacroPrecision(Metric):
    def __init__(self, name='macro_precision', threshold=0.5, **kwargs):
        super(MacroPrecision, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(1,), initializer='zeros')
        self.per_class_fp = self.add_weight(name='per_class_fp', shape=(1,), initializer='zeros')
    def build(self, input_shape):
        self.num_classes = input_shape[-1]
        self.per_class_tp.assign(tf.zeros(self.num_classes))
        self.per_class_fp.assign(tf.zeros(self.num_classes))
        super().build(input_shape)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        self.per_class_tp.assign_add(tp)
        self.per_class_fp.assign_add(fp)
    def result(self):
        precision_per_class = tf.where(tf.math.equal(self.per_class_tp + self.per_class_fp, 0), 0.0, self.per_class_tp / (self.per_class_tp + self.per_class_fp))
        return tf.reduce_mean(precision_per_class)
    def reset_state(self):
        if hasattr(self, 'num_classes'):
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fp.assign(tf.zeros(self.num_classes))
        else: self.per_class_tp.assign(tf.zeros(0)); self.per_class_fp.assign(tf.zeros(0))

class MacroRecall(Metric):
    def __init__(self, name='macro_recall', threshold=0.5, **kwargs):
        super(MacroRecall, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(1,), initializer='zeros')
        self.per_class_fn = self.add_weight(name='per_class_fn', shape=(1,), initializer='zeros')
    def build(self, input_shape):
        self.num_classes = input_shape[-1]
        self.per_class_tp.assign(tf.zeros(self.num_classes))
        self.per_class_fn.assign(tf.zeros(self.num_classes))
        super().build(input_shape)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        self.per_class_tp.assign_add(tp)
        self.per_class_fn.assign_add(fn)
    def result(self):
        recall_per_class = tf.where(tf.math.equal(self.per_class_tp + self.per_class_fn, 0), 0.0, self.per_class_tp / (self.per_class_tp + self.per_class_fn))
        return tf.reduce_mean(recall_per_class)
    def reset_state(self):
        if hasattr(self, 'num_classes'):
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fn.assign(tf.zeros(self.num_classes))
        else: self.per_class_tp.assign(tf.zeros(0)); self.per_class_fn.assign(tf.zeros(0))

# --- Model Architecture (copied from your training script) ---
def multi_head_attention(x, num_heads):
    """
    Implements a simplified multi-head attention mechanism.
    """
    heads = []
    for i in range(num_heads):
        head = Conv1D(filters=128, kernel_size=1, activation='relu')(x)
        attention = Conv1D(filters=1, kernel_size=1, activation='sigmoid')(head)
        head = Multiply()([x, attention])
        head = Lambda(
            lambda xin: tf.reduce_sum(xin, axis=1),
            output_shape=(256,)
        )(head)
        heads.append(head)
    return Concatenate()(heads)

# --- Utility Functions (from your script) ---
def extract_mel_spectrogram_fixed_window(audio_path, start_time, duration, sr, n_mels, hop_length):
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        if sr_loaded != sr:
            y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)
        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), 'constant')
        elif len(y) > expected_samples:
            y = y[:expected_samples]
        if len(y) == 0:
            return np.zeros((n_mels, int(np.ceil(duration * sr / hop_length))), dtype=np.float32)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        fixed_time_steps = int(np.ceil(duration * sr / hop_length))
        if mel_spectrogram_db.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), 'constant')
        elif mel_spectrogram_db.shape[1] > fixed_time_steps:
            mel_spectrogram_db = mel_spectrogram_db[:, :fixed_time_steps]
        return mel_spectrogram_db
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return np.zeros((n_mels, int(np.ceil(duration * sr / hop_length))), dtype=np.float32)

def load_inference_model(model_path):
    """
    Rebuilds the model and loads weights to bypass Lambda layer serialization issues.
    
    Parameters:
    ----------
    model_path : str or Path
        Path to the saved model weights file.
        
    Returns:
    -------
    model : tf.keras.Model
        The reconstructed model with loaded weights.
    mlb : MultiLabelBinarizer
        The MultiLabelBinarizer fitted on the training classes.
    """
    try:
        training_classes = sorted([
            'key_child', 'child_directed_speech', 'overheard_speech'
        ])
        mlb = MultiLabelBinarizer(classes=training_classes)
        mlb.fit([[]])
        num_classes = len(mlb.classes_)
        fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
        model = build_model_multi_label(
            n_mels=AudioConfig.N_MELS,
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
    Classifies a raw audio file using a sliding window approach.
    
    Parameters:
    ----------
    audio_path : str or Path
        Path to the audio file to classify.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : MultiLabelBinarizer
        The MultiLabelBinarizer fitted on the training classes.
    batch_size : int
        Batch size for model prediction.
        
    Returns:
    -------
    results : list of dict
        List of dictionaries containing start_time, end_time, and class probabilities for each window.
    class_names : list of str
        List of class names corresponding to the probabilities.
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
            mel = extract_mel_spectrogram_fixed_window(
                audio_path, window_start, AudioConfig.WINDOW_DURATION, 
                sr=AudioConfig.SR, n_mels=AudioConfig.N_MELS, hop_length=AudioConfig.HOP_LENGTH
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

# --- Main Execution to Process Folder ---
def process_audio_folder(folder_path, model, mlb):
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print(f"❌ Error: Folder not found at {folder_path}")
        return
    
    audio_files = list(folder_path.glob('*.wav'))
    if not audio_files:
        print(f"⚠️ No WAV audio files found in {folder_path}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files to process.")
    
    for i, audio_file in enumerate(audio_files):
        print("\n" + "="*50)
        print(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
        print("="*50)
        
        try:
            prediction_results, class_names, full_audio_duration = \
                classify_audio_windows(str(audio_file), model, mlb)
        except Exception as e:
            print(f"❌ An error occurred while processing {audio_file.name}: {e}")

if __name__ == "__main__":
    # add audio folder path as argparse argument
    parser = argparse.ArgumentParser(description="Run audio classifier on a folder of audio files.")
    parser.add_argument('--audio_folder', type=str, required=True, help="Path to folder containing WAV audio files.")
    args = parser.parse_args()
    
    AUDIO_FOLDER_PATH = args.audio_folder
    # Load the trained model and MultiLabelBinarizer once
    model, mlb = load_inference_model(AudioClassification.TRAINED_WEIGHTS_PATH)
    if model is None:
        exit()

    # Process all audio files in the specified folder
    process_audio_folder(AUDIO_FOLDER_PATH, model, mlb)