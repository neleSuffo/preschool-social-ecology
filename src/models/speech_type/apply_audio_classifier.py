import numpy as np
import tensorflow as tf
import librosa
import os
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import sounddevice as sd
import time

# Import custom objects (your FocalLoss and Metric classes)
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric

# Re-define your custom FocalLoss and Metric classes here or import them from a shared utility file.
# This is crucial for Keras to load the model correctly.
class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_t = (y_true * self.alpha) + ((1 - y_true) * (1 - self.alpha))
        focal_weight = tf.pow(1. - p_t, self.gamma)
        bce = -tf.math.log(p_t)
        loss = alpha_t * focal_weight * bce
        return tf.reduce_mean(loss)
    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

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

class MacroF1Score(Metric):
    def __init__(self, name='macro_f1', threshold=0.5, **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.per_class_tp = self.add_weight(name='per_class_tp', shape=(1,), initializer='zeros')
        self.per_class_fp = self.add_weight(name='per_class_fp', shape=(1,), initializer='zeros')
        self.per_class_fn = self.add_weight(name='per_class_fn', shape=(1,), initializer='zeros')
    def build(self, input_shape):
        self.num_classes = input_shape[-1]
        self.per_class_tp.assign(tf.zeros(self.num_classes))
        self.per_class_fp.assign(tf.zeros(self.num_classes))
        self.per_class_fn.assign(tf.zeros(self.num_classes))
        super().build(input_shape)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        self.per_class_tp.assign_add(tp)
        self.per_class_fp.assign_add(fp)
        self.per_class_fn.assign_add(fn)
    def result(self):
        precision_per_class = tf.where(tf.math.equal(self.per_class_tp + self.per_class_fp, 0), 0.0, self.per_class_tp / (self.per_class_tp + self.per_class_fp))
        recall_per_class = tf.where(tf.math.equal(self.per_class_tp + self.per_class_fn, 0), 0.0, self.per_class_tp / (self.per_class_tp + self.per_class_fn))
        f1_per_class = tf.where(tf.math.equal(precision_per_class + recall_per_class, 0), 0.0, 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class))
        return tf.reduce_mean(f1_per_class)
    def reset_state(self):
        if hasattr(self, 'num_classes'):
            self.per_class_tp.assign(tf.zeros(self.num_classes))
            self.per_class_fp.assign(tf.zeros(self.num_classes))
            self.per_class_fn.assign(tf.zeros(self.num_classes))
        else: self.per_class_tp.assign(tf.zeros(0)); self.per_class_fp.assign(tf.zeros(0)); self.per_class_fn.assign(tf.zeros(0))


# --- Configuration (MUST match training configuration) ---
SR = 16000
N_MELS = 128
HOP_LENGTH = 512
WINDOW_DURATION = 0.5  # Seconds
WINDOW_STEP = 0.25     # Seconds (for sliding window)
FIXED_TIME_STEPS = int(np.ceil(WINDOW_DURATION * SR / HOP_LENGTH))

MODEL_PATH = 'multi_label_speech_type_classifier_focalloss.h5' # Adjust this path

# IMPORTANT: Ensure these classes are in the exact same order as your MLB from training
TRAINING_CLASSES = sorted([
    'key_child', 'child_directed_speech', 'overheard_speech' # Add all unique labels encountered during training
])
mlb = MultiLabelBinarizer(classes=TRAINING_CLASSES)
mlb.fit([[]]) # Fit with an empty list to initialize classes

# --- Feature Extraction Function (same as training) ---
def extract_mel_spectrogram_fixed_window(audio_path, start_time, duration, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fixed_time_steps=FIXED_TIME_STEPS):
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
            return np.zeros((n_mels, fixed_time_steps), dtype=np.float32)

        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if mel_spectrogram_db.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), 'constant')
        elif mel_spectrogram_db.shape[1] > fixed_time_steps:
            mel_spectrogram_db = mel_spectrogram_db[:, :fixed_time_steps]

        return mel_spectrogram_db
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return np.zeros((n_mels, fixed_time_steps), dtype=np.float32)

# --- Model Loading ---
def load_inference_model(model_path):
    custom_objects = {
        'FocalLoss': FocalLoss,
        'MacroPrecision': MacroPrecision,
        'MacroRecall': MacroRecall,
        'MacroF1Score': MacroF1Score
    }
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure custom_objects are correctly defined and match the saved model.")
        return None

# --- Mock Diarization (Replace with your actual diarization output) ---
def get_diarization_segments_mock(audio_duration):
    """
    Mocks diarization segments for demonstration.
    In a real scenario, this would parse an RTTM file.
    """
    segments = []
    # Example: Speaker 00 for first 5 seconds, Speaker 01 for next 3 seconds, etc.
    # Note: For this example, we don't use speaker_id for classification,
    # but it's part of standard diarization output.
    segments.append({'start': 0.0, 'end': min(5.0, audio_duration), 'speaker_id': 'SPEAKER_00'})
    if audio_duration > 5.0:
        segments.append({'start': 5.0, 'end': min(8.0, audio_duration), 'speaker_id': 'SPEAKER_01'})
    if audio_duration > 8.0:
        segments.append({'start': 8.0, 'end': min(12.0, audio_duration), 'speaker_id': 'SPEAKER_00'})
    if audio_duration > 12.0:
        segments.append({'start': 12.0, 'end': audio_duration, 'speaker_id': 'SPEAKER_01'})
    
    print("Mock diarization segments generated.")
    return segments

# --- Classification Pipeline ---
def classify_audio_windows(audio_path, model, mlb, diarization_segments=None, batch_size=32):
    print(f"Processing audio file: {audio_path}")
    y_audio, sr_audio = librosa.load(audio_path, sr=SR)
    audio_duration = librosa.get_duration(y=y_audio, sr=sr_audio)
    print(f"Audio duration: {audio_duration:.2f} seconds")

    all_window_data = [] # Store (start_time, end_time, mel_spectrogram)
    
    if diarization_segments:
        print("Using diarization segments for classification...")
        processed_intervals = [] # To keep track of processed time ranges for visualization
        
        for i, segment in enumerate(diarization_segments):
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Ensure segment is within audio bounds
            segment_start = max(0.0, segment_start)
            segment_end = min(audio_duration, segment_end)
            segment_duration = segment_end - segment_start

            if segment_duration <= 0:
                continue

            print(f"  Processing diarization segment {i+1}: {segment_start:.2f}s - {segment_end:.2f}s (Speaker: {segment.get('speaker_id', 'N/A')})")

            current_time = segment_start
            while current_time < segment_end:
                window_start = current_time
                window_end = min(current_time + WINDOW_DURATION, segment_end)
                
                # Only process if window has sufficient duration
                if (window_end - window_start) > 0.01: # Check for non-zero duration
                    mel = extract_mel_spectrogram_fixed_window(
                        audio_path, window_start, WINDOW_DURATION, 
                        sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fixed_time_steps=FIXED_TIME_STEPS
                    )
                    all_window_data.append({
                        'start_time': window_start,
                        'end_time': window_end,
                        'mel_spec': mel,
                        'diarization_speaker': segment.get('speaker_id') # Store speaker for visualization
                    })
                current_time += WINDOW_STEP
                processed_intervals.append({'start': window_start, 'end': window_end, 'speaker': segment.get('speaker_id')})
        
    else: # Process entire audio with sliding windows if no diarization provided
        print("No diarization segments provided. Classifying entire audio with sliding windows...")
        current_time = 0.0
        while current_time < audio_duration:
            window_start = current_time
            window_end = min(current_time + WINDOW_DURATION, audio_duration)
            
            if (window_end - window_start) > 0.01:
                mel = extract_mel_spectrogram_fixed_window(
                    audio_path, window_start, WINDOW_DURATION, 
                    sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fixed_time_steps=FIXED_TIME_STEPS
                )
                all_window_data.append({
                    'start_time': window_start,
                    'end_time': window_end,
                    'mel_spec': mel,
                    'diarization_speaker': 'N/A' # No speaker info without diarization
                })
            current_time += WINDOW_STEP
        processed_intervals = None # No specific diarization intervals to visualize

    if not all_window_data:
        print("No valid windows extracted for classification.")
        return [], [], [], None

    # Prepare data for batch prediction
    mel_specs_batch = np.array([d['mel_spec'] for d in all_window_data])
    mel_specs_batch = np.expand_dims(mel_specs_batch, -1) # Add channel dimension

    print(f"Predicting on {len(mel_specs_batch)} windows...")
    predictions = model.predict(mel_specs_batch, batch_size=batch_size)
    print("Prediction complete.")

    # Collect results
    results = []
    for i, data_point in enumerate(all_window_data):
        results.append({
            'start_time': data_point['start_time'],
            'end_time': data_point['end_time'],
            'probabilities': predictions[i],
            'diarization_speaker': data_point['diarization_speaker']
        })
    
    return results, mlb.classes_, audio_duration, processed_intervals


# --- Visualization ---
def visualize_probabilities(audio_path, results, class_names, audio_duration, diarization_intervals=None):
    if not results:
        print("No results to visualize.")
        return

    # Load audio for playback
    y, sr = librosa.load(audio_path, sr=SR)

    time_points = [r['start_time'] for r in results]
    
    # Initialize a dictionary to store probabilities per class over time
    class_probabilities = {class_name: [] for class_name in class_names}
    
    for r in results:
        for i, class_name in enumerate(class_names):
            class_probabilities[class_name].append(r['probabilities'][i])

    plt.figure(figsize=(15, 8))

    # Plot probabilities for each class
    for class_name in class_names:
        plt.plot(time_points, class_probabilities[class_name], label=f'Prob. {class_name}', alpha=0.8)

    # Add vertical lines for diarization segments if available
    if diarization_intervals:
        for i, interval in enumerate(diarization_intervals):
            plt.axvline(interval['start'], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            # Add speaker label only for the first line of the segment or at intervals
            if i % 2 == 0 or i == len(diarization_intervals) -1: # Avoid too many labels
                plt.text(interval['start'] + 0.1, plt.ylim()[1] * 0.95, f"Sp: {interval['speaker']}", 
                         rotation=90, verticalalignment='top', color='dimgray', fontsize=8)


    plt.title('Class Probabilities Over Time with Audio Playback')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.ylim(-0.05, 1.05) # Extend y-axis slightly for better visualization
    plt.xlim(0, audio_duration) # Set x-axis limit to audio duration
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # Play audio and synchronize a red vertical line
    print("Playing audio and visualizing. Close plot to stop audio.")
    
    # Store the audio stream for potential stopping later
    # This setup for sounddevice.play is blocking. For non-blocking, use sd.Stream
    
    # To make the plot interactive with playback, we need animation or real-time plotting.
    # For simplicity, we'll plot everything first, then play audio with a static marker.
    # A true real-time synchronized plot is more complex (matplotlib animation, GUI toolkits).
    
    # Simple playback (blocking)
    sd.play(y, sr)
    
    # Add a red line indicating playback progress (this will not move dynamically)
    # For dynamic update, you'd need matplotlib animation or a more advanced GUI.
    # Let's add a simple static line at the beginning as a visual cue.
    current_playback_line = plt.axvline(0, color='red', linestyle='-', linewidth=2, label='Current Playback')
    
    # Keep plot open while audio plays
    plt.show(block=False) # Non-blocking show
    
    # Update the playback line in a loop (still not perfect synchronization, but better than static)
    start_play_time = time.time()
    while sd.get_status().output_running:
        elapsed_time = time.time() - start_play_time
        if current_playback_line is not None:
            current_playback_line.set_xdata([elapsed_time])
        plt.draw()
        plt.pause(0.1) # Update every 100ms

    sd.stop()
    print("Audio playback finished.")
    plt.close() # Close plot after audio finishes
    
# --- Main Execution for Inference and Visualization ---
if __name__ == "__main__":
    # --- Paths and Configuration ---
    SAMPLE_AUDIO_PATH = 'path/to/your/sample_audio.wav' # <--- Set your sample audio file here!

    if not os.path.exists(SAMPLE_AUDIO_PATH):
        print(f"Error: Sample audio file not found at {SAMPLE_AUDIO_PATH}")
        print("Please update SAMPLE_AUDIO_PATH to a valid .wav file.")
        exit()

    # Load the trained model
    model = load_inference_model(MODEL_PATH)
    if model is None:
        exit()

    # Get diarization segments (mocked or from your diarization tool)
    audio_duration_loaded = librosa.get_duration(filename=SAMPLE_AUDIO_PATH, sr=SR)
    diarization_segments = get_diarization_segments_mock(audio_duration_loaded)
    # If you have an RTTM file, you'd parse it here:
    # diarization_segments = parse_rttm_file_to_list('your_diarization.rttm')

    # Classify the audio segments/windows
    prediction_results, class_names, full_audio_duration, diarization_display_intervals = \
        classify_audio_windows(SAMPLE_AUDIO_PATH, model, mlb, diarization_segments=diarization_segments)

    # Visualize the results
    if prediction_results:
        visualize_probabilities(SAMPLE_AUDIO_PATH, prediction_results, class_names, full_audio_duration, 
                                diarization_intervals=diarization_display_intervals)
    else:
        print("No prediction results to visualize.")