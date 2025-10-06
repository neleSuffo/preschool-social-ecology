import json
import librosa
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import csv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from config import AudioConfig
from constants import AudioClassification
from models.speech_type.audio_classifier import build_model_multi_label, ThresholdOptimizer
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
    
# --- Data Generator ---
class AudioSegmentDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for audio classification with enhanced features and augmentation.
    Now loads precomputed features from cache_dir if provided.
    
    Generates batches of audio features (mel-spectrogram + MFCC) with corresponding
    multi-label targets for training and evaluation. Supports data augmentation
    during training to improve model generalization.
    
    Parameters
    ----------
    segments_file_path (str): 
        Path to JSONL file containing segment metadata
    mlb (MultiLabelBinarizer): 
        Fitted multi-label binarizer for encoding
    n_mels (int): 
        Number of mel-frequency bands
    hop_length (int): 
        Hop length for feature extraction
    sr (int): 
        Sample rate for audio processing
    window_duration (float): 
        Duration of each audio segment
    fixed_time_steps (int): 
        Fixed number of time steps for consistent input shape
    batch_size (int): 
        Number of samples per batch (default: 32)
    shuffle (bool): 
        Whether to shuffle data at epoch end (default: True)
    augment (bool): 
        Whether to apply data augmentation (default: False)

    Features
    --------
    Lazy loading: 
        Features extracted on-demand for memory efficiency
    Data augmentation:
        Time/frequency masking, noise addition, pitch shifting
    Multi-label support: 
        Handles overlapping voice type labels
    Consistent batching: 
        Fixed input dimensions across all samples
    """
    def __init__(self, segments_file_path, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, 
                batch_size=32, shuffle=True, augment=False, cache_dir=None):
        self.segments_file_path = segments_file_path
        self.mlb = mlb
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.window_duration = window_duration
        self.fixed_time_steps = fixed_time_steps
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.cache_dir = cache_dir
        self.segments_data = self._load_segments_metadata()
        self.on_epoch_end()

    def __len__(self):
        return (len(self.segments_data) + self.batch_size - 1) // self.batch_size

    def _load_segments_metadata(self):
        segments = []
        with open(self.segments_file_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line.strip()))
        return segments

    def augment_spectrogram(self, mel_spec):
        """Apply data augmentation to spectrograms for better generalization."""
        if not self.augment:
            return mel_spec
            
        augmented = mel_spec.copy()
        
        # Time masking: mask random time segments (simulates missing audio)
        if np.random.random() < 0.5:
            time_mask_width = np.random.randint(1, min(20, mel_spec.shape[1] // 4))
            time_mask_start = np.random.randint(0, mel_spec.shape[1] - time_mask_width)
            augmented[:, time_mask_start:time_mask_start + time_mask_width] = -1
            
        # Frequency masking: mask random frequency bands (simulates filtering effects)  
        if np.random.random() < 0.5:
            freq_mask_width = np.random.randint(1, min(15, mel_spec.shape[0] // 4))
            freq_mask_start = np.random.randint(0, mel_spec.shape[0] - freq_mask_width)
            augmented[freq_mask_start:freq_mask_start + freq_mask_width, :] = -1
            
        # Additive noise: simulate background noise conditions
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, mel_spec.shape)
            augmented = np.clip(augmented + noise, -1, 1)
            
        # Pitch shift: simulate speaker variation (changes fundamental frequency)
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)  # Shift by up to 2 semitones
            augmented = librosa.effects.pitch_shift(augmented, sr=self.sr, n_steps=n_steps)
            
        return augmented

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_segments = [self.segments_data[k] for k in indexes]
        X_batch = []
        y_batch = []
        for segment in batch_segments:
            # Load cached feature if cache_dir is set
            if self.cache_dir is not None:
                print("Loading cached feature...")
                segment_id = segment.get('id') or f"{Path(segment['audio_path']).stem}_{segment['start']}_{segment['duration']}"
                cache_path = os.path.join(self.cache_dir, f"{segment_id}.npy")
                if os.path.exists(cache_path):
                    mel = np.load(cache_path)
                else:
                    # Fallback to on-the-fly extraction if cache missing
                    mel = extract_features(
                        segment['audio_path'], segment['start'], segment['duration'],
                        sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
                        fixed_time_steps=self.fixed_time_steps
                    )
            else:
                mel = extract_features(
                    segment['audio_path'], segment['start'], segment['duration'],
                    sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
                    fixed_time_steps=self.fixed_time_steps
                )
            mel = self.augment_spectrogram(mel)
            if mel.ndim != 2:
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    mel = mel.squeeze(axis=-1)
                else:
                    continue
            X_batch.append(mel)
            multi_hot_labels = self.mlb.transform([segment['labels']])[0]
            y_batch.append(multi_hot_labels)
        if not X_batch:
            return np.array([]), np.array([])
        X_batch_np = np.array(X_batch)
        X_batch_final = np.expand_dims(X_batch_np, -1)
        return X_batch_final, np.array(y_batch)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.segments_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# --- Cosine Annealing LR ---
def compute_cosine_annealing_lr(epoch):
    """
    Calculate learning rate using cosine annealing schedule.
    
    Implements cosine annealing with warm restarts every 20 epochs,
    cycling between maximum and minimum learning rates. This helps
    the model escape local minima and achieve better convergence.

    Parameters
    ----------
    epoch (int): 
        Current training epoch (0-indexed)
    
    Returns
    -------
    float:
        Learning rate for the current epoch
        
    Schedule details
    ---------------
    max_lr: 
        0.001 (peak learning rate at start of each cycle)
    min_lr: 
        0.00001 (minimum learning rate at end of each cycle)  
    cycle_length: 
        20 epochs per complete cosine cycle
    Formula: 
        min_lr + (max_lr - min_lr) * (1 + cos(π * epoch_in_cycle / cycle_length)) / 2
    
    The cosine shape provides:
    - Fast initial learning (high LR)
    - Gradual slowdown for fine-tuning (decreasing LR)  
    - Periodic restarts to escape local minima
    """
    max_lr = 0.001
    min_lr = 0.00001
    epochs_per_cycle = 20
    
    # Calculate position within current cycle (0 to epochs_per_cycle-1)
    epoch_in_cycle = epoch % epochs_per_cycle
    
    # Compute cosine annealing: starts at max_lr, decreases to min_lr
    cos_inner = (np.pi * epoch_in_cycle) / epochs_per_cycle
    lr = min_lr + (max_lr - min_lr) * (1 + np.cos(cos_inner)) / 2
    return lr

# --- Custom History and Plotting Callback ---
class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Custom callback for logging training metrics and generating plots.
    
    Logs detailed training metrics to CSV file and creates visualization plots
    after training completion. Tracks loss, accuracy, precision, recall, 
    macro F1-score, and learning rate for both training and validation.
    
    Parameters
    ----------
    log_dir (str): 
        Directory to save CSV logs and plot images
    mlb_classes (list): 
        List of multi-label class names

    Attributes
    ----------
    csv_file_path (str): 
        Path to CSV file containing training metrics
    history (dict): 
        Dictionary storing all training metrics
    start_time (float): 
        Training start timestamp

    Features
    --------
    Real-time CSV logging of all metrics per epoch
    Automatic plot generation (loss and macro F1-score)
    Elapsed time tracking
    Learning rate monitoring
    """
    def __init__(self, log_dir, mlb_classes):
        super().__init__()
        self.log_dir = log_dir
        self.csv_file_path = os.path.join(log_dir, 'results.csv')
        self.start_time = 0
        self.epoch_times = []
        self.mlb_classes = mlb_classes
        self.history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'macro_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_macro_f1': [],
            'lr': []
        }
        
        self.csv_headers = [
            'epoch', 'time_sec', 
            'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_macro_f1',
            'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_macro_f1',
            'learning_rate'
        ]
        
        # Initialize CSV file with headers
        with open(self.csv_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        print(f"Training started. Logs will be saved to: {self.log_dir}")

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.epoch_times.append(elapsed_time)

        # Get learning rate from optimizer
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # Store metrics in history
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['precision'].append(logs.get('precision'))
        self.history['recall'].append(logs.get('recall'))
        self.history['macro_f1'].append(logs.get('macro_f1'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        self.history['val_precision'].append(logs.get('val_precision'))
        self.history['val_recall'].append(logs.get('val_recall'))
        self.history['val_macro_f1'].append(logs.get('val_macro_f1'))
        self.history['lr'].append(current_lr)

        # Write to CSV
        row = [
            epoch + 1, # epoch starts from 0 in callback, but 1 in CSV
            elapsed_time,
            logs.get('loss', 0.0), logs.get('accuracy', 0.0), logs.get('precision', 0.0), logs.get('recall', 0.0), logs.get('macro_f1', 0.0),
            logs.get('val_loss', 0.0), logs.get('val_accuracy', 0.0), logs.get('val_precision', 0.0), logs.get('val_recall', 0.0), logs.get('val_macro_f1', 0.0),
            current_lr
        ]
        with open(self.csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"Epoch {epoch+1} completed. Elapsed time: {elapsed_time:.2f}s")

    def on_train_end(self, logs=None):
        self.plot_metrics()

    def plot_metrics(self):
        epochs = range(1, len(self.history['loss']) + 1)
        
        # Plot Loss
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close()

        # Plot Macro F1
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, self.history['macro_f1'], label='Training Macro F1')
        plt.plot(epochs, self.history['val_macro_f1'], label='Validation Macro F1')
        plt.title('Training and Validation Macro F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Macro F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'macro_f1_plot.png'))
        plt.close()

def create_data_generators(segment_files, mlb):
    """
    Create data generators for training, validation, and testing.
    
    Parameters:
    ----------
    segment_files (dict): 
        Paths to segment files for each split
    mlb: 
        Fitted MultiLabelBinarizer
        
    Returns:
    -------
    tuple: 
        (train_generator, val_generator, test_generator)
    """
    # Use centralized time steps calculation to ensure consistency
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
    
    train_generator = None
    if segment_files.get('train') is not None:
        train_generator = AudioSegmentDataGenerator(
            segment_files['train'], mlb, 
            AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR, 
            AudioConfig.WINDOW_DURATION, fixed_time_steps,
            batch_size=32, shuffle=True, augment=True, cache_dir=(AudioClassification.CACHE_DIR/"train")
        )
    
    val_generator = None
    if segment_files.get('val') is not None:
        val_generator = AudioSegmentDataGenerator(
            segment_files['val'], mlb,
            AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
            AudioConfig.WINDOW_DURATION, fixed_time_steps,
            batch_size=32, shuffle=False, augment=False, cache_dir=(AudioClassification.CACHE_DIR/"val")
        )
    
    test_generator = None
    if segment_files.get('test') is not None:
        test_generator = AudioSegmentDataGenerator(
            segment_files['test'], mlb,
            AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
            AudioConfig.WINDOW_DURATION, fixed_time_steps,
            batch_size=32, shuffle=False, augment=False, cache_dir=(AudioClassification.CACHE_DIR/"test")
        )
    
    return train_generator, val_generator, test_generator

def create_training_callbacks(run_dir, val_generator, mlb_classes):
    """
    Create and configure training callbacks for model optimization.
    
    Parameters:
    ----------
    run_dir (str): 
        Directory to save model checkpoints and logs
    val_generator: 
        Validation data generator for threshold optimization
    mlb_classes (list): 
        List of multi-label class names
        
    Returns:
    -------
    list: 
        Configured Keras callbacks for training
    """
    callbacks = [
        EarlyStopping(monitor='val_macro_f1', patience=30, mode='max', restore_best_weights=True, verbose=1),
        LearningRateScheduler(compute_cosine_annealing_lr, verbose=1),
        ModelCheckpoint(
            filepath=(run_dir / 'best_model.keras'),
            monitor='val_macro_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ThresholdOptimizer(val_generator, mlb_classes),
        TrainingLogger(run_dir, mlb_classes)
    ]
    return callbacks

def load_thresholds(mlb_classes):
    """
    Load class-specific optimized thresholds from training run output.
    
    During training, the ThresholdOptimizer callback searches for optimal 
    decision thresholds per class to maximize macro F1-score. These thresholds
    are typically different from the default 0.5 due to class imbalance and
    varying prediction confidence distributions.
    
    Threshold Optimization Process:
    1. During training, validation set predictions are analyzed
    2. For each class, thresholds from 0.1 to 0.9 are tested
    3. The threshold maximizing F1-score per class is selected
    4. Results are saved to thresholds.json in the training run directory
    
    Fallback Strategy:
    If optimized thresholds are not available (e.g., interrupted training),
    the function defaults to 0.5 for all classes with appropriate warnings.
    
    Parameters:
    ----------
    mlb_classes (list): 
        List of class names in the same order as model outputs
    
    Returns:
    -------
    dict: 
        Dictionary mapping class names to their optimal thresholds
        
    File Format:
    -----------
    thresholds.json contains:
    {
        "class_name_1": 0.3,
        "class_name_2": 0.7,
        ...
    }
    """
    thresholds_file = AudioClassification.RESULTS_DIR / 'thresholds.json'
    
    if thresholds_file.exists():
        try:
            with open(thresholds_file, 'r') as f:
                thresholds_dict = json.load(f)
            
            # Return the dictionary directly, filtering out non-class keys
            thresholds = {class_name: thresholds_dict.get(class_name, 0.5) for class_name in mlb_classes}
                        
            # Validate threshold ranges
            if any(t < 0.1 or t > 0.9 for t in thresholds.values()):
                print("⚠️ Warning: Some thresholds are outside typical range [0.1, 0.9]")
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Error reading thresholds file: {e}")
            print("🔄 Falling back to default thresholds (0.5)")
            thresholds = {class_name: 0.5 for class_name in mlb_classes}
    else:
        thresholds = {class_name: 0.5 for class_name in mlb_classes}
        print(f"⚠️ Optimized thresholds not found at: {thresholds_file}")
        print("🔄 Using default thresholds (0.5 for all classes)")
    
    return thresholds

# --- Feature Extraction ---
def extract_features(audio_path, start_time, duration, sr=16000, n_mels=256, hop_length=512, fixed_time_steps=None):
    """
    Extract audio features (mel-spectrogram + MFCC) from audio segment.

    Processes audio segments to create combined feature representations suitable
    for deep learning models. Applies preprocessing, normalization, and padding
    to ensure consistent output dimensions.
    
    Parameters
    ----------
    audio_path (str):
        Path to audio file (.wav format)
    start_time (float): 
        Start time of segment in seconds
    duration (float): 
        Duration of segment in seconds
    sr (int): 
        Target sample rate for audio loading (default: 16000)
    n_mels (int): 
        Number of mel filter banks (default: 256)
    hop_length (int): 
        Hop length for STFT computation (default: 512)
    fixed_time_steps (int): 
        Fixed number of time steps for output padding

    Returns
    -------
    np.ndarray: 
        Combined feature matrix of shape (n_mels + 13, fixed_time_steps) where 13 is the number of MFCC coefficients

    Features extracted
    -----------------
    Mel-spectrogram: 
        Perceptually-relevant frequency representation
    MFCC: 
        Compact spectral features for speech/audio
    Preprocessing: 
        Normalization, pre-emphasis filtering
    Post-processing: 
        Padding/truncation to fixed dimensions
    """
    if fixed_time_steps is None:
        fixed_time_steps = int(np.ceil(duration * sr / hop_length))
    
    try:
        # Load audio segment with resampling if needed
        y, sr_loaded = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        if sr_loaded != sr:
            y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)
        
        # Ensure consistent audio length through padding/truncation
        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), 'constant')  # Zero-padding
        elif len(y) > expected_samples:
            y = y[:expected_samples]  # Truncation
        
        # Handle empty audio segments
        if len(y) == 0:
            effective_n_mels = min(n_mels, 128)  # Cap at 128 for 16kHz to avoid empty filters
            return np.zeros((effective_n_mels + 13, fixed_time_steps), dtype=np.float32)  # +13 for MFCC
        
        # Audio preprocessing: normalization and pre-emphasis filtering
        y = y / (np.max(np.abs(y)) + 1e-6)  # Amplitude normalization
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])  # Pre-emphasis filter (removes DC bias)
        
        # Mel-spectrogram computation with frequency limits for 16kHz audio
        effective_n_mels = min(n_mels, 128)  # Cap at 128 for 16kHz to avoid empty filters
        fmax_safe = min(sr // 2, 8000)  # Use Nyquist frequency or 8kHz, whichever is lower
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=effective_n_mels, hop_length=hop_length, n_fft=2048, 
            fmin=80, fmax=fmax_safe  # Focus on speech-relevant frequencies
        )
        # Convert to dB scale and normalize to [-1, 1] range
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_spectrogram_db = 2 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-6) - 1
        
        # MFCC computation for complementary spectral features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6) - 1  # Normalize to [-1, 1]
        
        # Concatenate mel and MFCC features along frequency axis
        combined = np.concatenate([mel_spectrogram_db, mfcc], axis=0)
        
        # Ensure consistent time dimension through padding/truncation
        if combined.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
        elif combined.shape[1] > fixed_time_steps:
            combined = combined[:, :fixed_time_steps]
        
        return combined
    
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        effective_n_mels = min(n_mels, 128)  # Cap at 128 for 16kHz to avoid empty filters
        return np.zeros((effective_n_mels + 13, fixed_time_steps), dtype=np.float32)

# ---- Evaluation Functions ----
def aggregate_windows_to_seconds(window_predictions, window_true_labels, window_metadata):
    """
    Aggregate sliding window predictions to second-level predictions.
    
    For each second in the test set, this function:
    1. Finds all overlapping windows
    2. Averages their probability predictions
    3. Aggregates their ground truth labels (union of all labels in overlapping windows)
    
    Parameters:
    ----------
    window_predictions (ndarray): 
        Raw probability predictions from model (n_windows, n_classes)
    window_true_labels (ndarray): 
        Ground truth binary labels for windows (n_windows, n_classes)
    window_metadata (list): 
        List of dictionaries with 'start_time', 'end_time' for each window
        
    Returns:
    -------
    tuple: (second_predictions, second_true_labels, second_metadata)
        - second_predictions: Averaged probabilities per second
        - second_true_labels: Aggregated ground truth per second  
        - second_metadata: List of {'second': int} for each output second
    """
    if len(window_predictions) != len(window_true_labels) or len(window_predictions) != len(window_metadata):
        raise ValueError("Predictions, labels, and metadata must have same length")
    
    # Find the range of seconds covered by all windows
    all_starts = [meta['start_time'] for meta in window_metadata]
    all_ends = [meta['end_time'] for meta in window_metadata]
    min_second = int(np.floor(min(all_starts)))
    max_second = int(np.ceil(max(all_ends)))
    
    # Initialize aggregated data structures
    second_data = {}
    
    for second in range(min_second, max_second):
        second_data[second] = {
            'predictions': [],
            'true_labels': [],
            'window_count': 0
        }
    
    # Aggregate data from overlapping windows
    for i, meta in enumerate(window_metadata):
        window_start = meta['start_time']
        window_end = meta['end_time']
        
        # Find which seconds this window overlaps with
        start_second = int(np.floor(window_start))
        end_second = int(np.ceil(window_end))
        
        for second in range(start_second, min(end_second, max_second)):
            # Calculate overlap between window and current second
            second_start = second
            second_end = second + 1
            
            overlap_start = max(window_start, second_start)
            overlap_end = min(window_end, second_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Only include windows with meaningful overlap (>0.1 seconds)
            if overlap_duration > 0.1:
                second_data[second]['predictions'].append(window_predictions[i])
                second_data[second]['true_labels'].append(window_true_labels[i])
                second_data[second]['window_count'] += 1
    
    # Convert aggregated data to arrays
    valid_seconds = []
    second_predictions = []
    second_true_labels = []
    second_metadata = []
    
    for second in sorted(second_data.keys()):
        data = second_data[second]
        
        if data['window_count'] > 0:  # Only include seconds with overlapping windows
            # Average predictions across overlapping windows
            avg_predictions = np.mean(data['predictions'], axis=0)
            
            # Aggregate ground truth: union of all labels (max across windows)
            # This means if any overlapping window has a label, the second gets that label
            agg_true_labels = np.max(data['true_labels'], axis=0).astype(int)
            
            valid_seconds.append(second)
            second_predictions.append(avg_predictions)
            second_true_labels.append(agg_true_labels)
            second_metadata.append({'second': second})
    
    return (np.array(second_predictions), 
            np.array(second_true_labels), 
            second_metadata)

def evaluate_model(model, test_generator, mlb, thresholds, output_dir, generate_confusion_matrices=False, aggregate_to_seconds=True):
    """
    Perform comprehensive multi-label classification evaluation with detailed analysis. 
    
    Parameters
    ----------
    model (tf.keras.Model): 
        Trained multi-label audio classification model
    test_generator (EvaluationDataGenerator): 
        Test data generator with deterministic ordering
    mlb (MultiLabelBinarizer): 
        Fitted label encoder from training pipeline
    thresholds (dict): 
        Dictionary mapping class names to decision thresholds for binary classification
    output_dir (str or Path): 
        Directory to save evaluation results and visualizations
    aggregate_to_seconds (bool):
        If True, aggregate sliding windows to second-level predictions before evaluation
    """    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Running comprehensive model evaluation...")
    print("=" * 60)
    
    # Stage 1: Generate probability predictions for all test samples
    print("📊 Generating predictions for test set...")
    test_predictions = model.predict(test_generator, verbose=1)
    
    # Stage 2: Collect true labels and metadata with progress tracking
    test_true_labels = []
    test_metadata = []
    for i in tqdm(range(len(test_generator)), desc="Processing batches"):
        _, labels = test_generator[i]
        if len(labels) > 0:  # Skip empty batches
            test_true_labels.extend(labels)
            
            # Extract metadata from test generator if available
            batch_segments = test_generator.segments_data[
                i * test_generator.batch_size:(i + 1) * test_generator.batch_size
            ]
            for segment in batch_segments[:len(labels)]:  # Match actual batch size
                test_metadata.append({
                    'start_time': segment.get('start', 0.0),
                    'end_time': segment.get('start', 0.0) + segment.get('duration', AudioConfig.WINDOW_DURATION),
                    'audio_path': segment.get('audio_path', 'unknown')
                })
    
    test_true_labels = np.array(test_true_labels)
    
    # Stage 3: Handle potential shape mismatches between predictions and labels
    if test_predictions.shape[0] != test_true_labels.shape[0]:
        print(f"⚠️ Shape mismatch detected:")
        print(f"   Predictions: {test_predictions.shape[0]} samples")
        print(f"   True labels: {test_true_labels.shape[0]} samples")
        
        # Use minimum available samples to ensure valid comparison
        min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
        test_predictions = test_predictions[:min_samples]
        test_true_labels = test_true_labels[:min_samples]
        test_metadata = test_metadata[:min_samples]
        
        print(f"✂️ Adjusted evaluation set to {min_samples} samples")
        
        if min_samples == 0:
            raise ValueError("No samples available for evaluation after shape adjustment")
    
    # Stage 4: Aggregate to second-level predictions if requested
    if aggregate_to_seconds:
        print("🔄 Aggregating sliding windows to second-level predictions...")
        test_predictions, test_true_labels, test_metadata = aggregate_windows_to_seconds(
            test_predictions, test_true_labels, test_metadata
        )
        evaluation_level = "second"
    else:
        evaluation_level = "window"
    
    # Stage 5: Apply class-specific thresholds to convert probabilities to predictions
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[mlb.classes_[i]]).astype(int) 
        for i in range(len(mlb.classes_))
    ]).T
    
    # Stage 6: Calculate comprehensive evaluation metrics
    if test_true_labels.sum() > 0 or len(test_true_labels) > 0: # Include silent frames
        
        # Create expanded labels to include "no speech" class for consistent metrics
        # Add "no speech" as the first column (index 0)
        expanded_true_labels = np.zeros((len(test_true_labels), len(test_true_labels[0]) + 1), dtype=int)
        expanded_pred_labels = np.zeros((len(test_pred_binary), len(test_pred_binary[0]) + 1), dtype=int)
        
        # Fill in the speech classes (columns 1, 2, 3, ...)
        expanded_true_labels[:, 1:] = test_true_labels
        expanded_pred_labels[:, 1:] = test_pred_binary
        
        # Fill in the "no speech" class (column 0)
        for i in range(len(test_true_labels)):
            # True "no speech" if no other classes are active
            if test_true_labels[i].sum() == 0:
                expanded_true_labels[i, 0] = 1
                
            # Predicted "no speech" if no other classes were predicted
            if test_pred_binary[i].sum() == 0:
                expanded_pred_labels[i, 0] = 1
        
        # Calculate metrics on expanded labels (including "no speech")
        class_names_expanded = ['no speech'] + list(mlb.classes_)
        
        # Per-class detailed metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            expanded_true_labels, expanded_pred_labels, average=None, zero_division=0
        )
        
        # Macro-averaged metrics (equal weight per class)
        macro_precision = precision_score(expanded_true_labels, expanded_pred_labels, average='macro', zero_division=0)
        macro_recall = recall_score(expanded_true_labels, expanded_pred_labels, average='macro', zero_division=0)
        macro_f1 = f1_score(expanded_true_labels, expanded_pred_labels, average='macro', zero_division=0)
        
        # Micro-averaged metrics (global performance)
        micro_precision = precision_score(expanded_true_labels, expanded_pred_labels, average='micro', zero_division=0)
        micro_recall = recall_score(expanded_true_labels, expanded_pred_labels, average='micro', zero_division=0)
        micro_f1 = f1_score(expanded_true_labels, expanded_pred_labels, average='micro', zero_division=0)
        
        # Subset accuracy (exact multi-label match)
        subset_accuracy = accuracy_score(expanded_true_labels, expanded_pred_labels)
        
        # Stage 7: Save comprehensive results to files
        save_evaluation_results(
            output_dir, class_names_expanded, thresholds,
            expanded_true_labels, expanded_pred_labels, test_predictions,
            precision_per_class, recall_per_class, f1_per_class, support_per_class,
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1, subset_accuracy,
            generate_confusion_matrices, evaluation_level, test_metadata
        )
    else:
        print("⚠️ Warning: No positive instances found in test set")
        print("❌ Cannot compute meaningful evaluation metrics")
    
    print(f"\n✅ Evaluation completed at {evaluation_level} level!")
    print(f"📁 Results saved to: {output_dir}")

def save_evaluation_results(output_dir, class_names, thresholds,
                        test_true_labels, test_pred_binary, test_predictions,
                        precision_per_class, recall_per_class, f1_per_class, support_per_class,
                        macro_precision, macro_recall, macro_f1,
                        micro_precision, micro_recall, micro_f1, subset_accuracy, 
                        generate_confusion_matrices=False, evaluation_level="window", metadata=None):
    """
    Save comprehensive evaluation results in multiple formats for analysis and reporting.
    
    This function creates both machine-readable (JSON) and human-readable (CSV) outputs
    containing detailed evaluation metrics.
    
    Parameters:
    ----------
    output_dir (Path): 
        Target directory for result files
    class_names (list): 
        Names of classification classes
    thresholds (dict): 
        Dictionary mapping class names to decision thresholds
    test_true_labels (ndarray): 
        Ground truth binary labels (n_samples, n_classes)
    test_pred_binary (ndarray): 
        Binary predictions (n_samples, n_classes)
    test_predictions (ndarray): 
        Probability predictions (n_samples, n_classes)
    precision_per_class (ndarray): 
        Per-class precision scores
    recall_per_class (ndarray): 
        Per-class recall scores  
    f1_per_class (ndarray): 
        Per-class F1 scores
    support_per_class (ndarray): 
        Per-class positive sample counts
    macro_precision (float): 
        Macro-averaged precision
    macro_recall (float): 
        Macro-averaged recall
    macro_f1 (float): 
        Macro-averaged F1 score
    micro_precision (float): 
        Micro-averaged precision
    micro_recall (float):   
        Micro-averaged recall
    micro_f1 (float): 
        Micro-averaged F1 score
    subset_accuracy (float): 
        Subset accuracy (exact match rate)
    evaluation_level (str):
        Level of evaluation ("window" or "second")
    metadata (list):
        Optional metadata for each sample
    """    
    output_dir = Path(output_dir)
    
    # Create comprehensive metrics summary for programmatic analysis
    summary = {
        'evaluation_metadata': {
            'test_set_size': len(test_true_labels),
            'num_classes': len(class_names),
            'class_names': list(class_names),
            'evaluation_level': evaluation_level,
            'evaluation_timestamp': datetime.now().isoformat()
        },
        'overall_metrics': {
            'subset_accuracy': float(subset_accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1)
        },
        'per_class_metrics': {
            str(class_names[i]): {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i]),
                'threshold': float(thresholds[str(class_names[i])]),
                'positive_rate': float(support_per_class[i] / len(test_true_labels))
            } for i in range(len(class_names))
        },
        'threshold_configuration': {
            'method': 'optimized_from_validation',
            'fallback': 0.5,
            'per_class_thresholds': {str(class_name): float(thresholds[str(class_name)]) for class_name in class_names}
        }
    }
    
    # Save structured JSON summary for automated analysis
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Create detailed per-sample results for error analysis
    class_names_list = list(class_names)
    
    # Probability predictions (raw model outputs)
    predictions_df = pd.DataFrame(
        test_predictions, 
        columns=[f'{name}_prob' for name in class_names_list]
    )
    
    # Binary predictions (after threshold application)
    predictions_binary_df = pd.DataFrame(
        test_pred_binary, 
        columns=[f'{name}_pred' for name in class_names_list]
    )
    
    # True labels (ground truth)
    true_labels_df = pd.DataFrame(
        test_true_labels, 
        columns=[f'{name}_true' for name in class_names_list]
    )
    
    # Combine all information for comprehensive per-sample analysis
    detailed_results = pd.concat([
        predictions_df, 
        predictions_binary_df, 
        true_labels_df
    ], axis=1)
    
    # Add derived columns for quick analysis
    if evaluation_level == "second" and metadata:
        detailed_results['sample_id'] = [meta.get('second', i) for i, meta in enumerate(metadata)]
        detailed_results['sample_type'] = 'second'
    else:
        detailed_results['sample_id'] = range(len(detailed_results))
        detailed_results['sample_type'] = evaluation_level
        
    detailed_results['num_true_labels'] = test_true_labels.sum(axis=1)
    detailed_results['num_pred_labels'] = test_pred_binary.sum(axis=1)
    detailed_results['exact_match'] = (test_true_labels == test_pred_binary).all(axis=1).astype(int)
    
    # Save detailed predictions for manual inspection and error analysis
    predictions_path = output_dir / f'detailed_predictions_{evaluation_level}_level.csv'
    detailed_results.to_csv(predictions_path, index=False, float_format='%.4f')
    
    # Generate confusion matrices if requested
    if generate_confusion_matrices:
        from sklearn.metrics import confusion_matrix
        print("📊 Generating multi-class confusion matrix...")
        
        confusion_matrices_dir = output_dir / 'confusion_matrices'
        confusion_matrices_dir.mkdir(exist_ok=True)
        
        # Define class order for confusion matrix (actual classes + no speech)
        class_names_list = list(class_names)  # ['KCHI', 'CDS', 'OHS'] or similar
        matrix_classes = ['no speech'] + class_names_list  # ['no speech', 'KCHI', 'CDS', 'OHS']
        
        # Create confusion matrix by counting frame occurrences for each label
        # Each frame can contribute multiple times if it has multiple labels
        cm = np.zeros((len(matrix_classes), len(matrix_classes)), dtype=int)
        
        for i in range(len(test_true_labels)):
            true_labels = test_true_labels[i]
            pred_labels = test_pred_binary[i]
            
            # Handle frames with no ground truth labels first (true "no speech" frames)
            if true_labels.sum() == 0:
                no_speech_true_idx = matrix_classes.index('no speech')
                
                # Check what was predicted for this "no speech" frame
                if pred_labels.sum() == 0:
                    # Correctly predicted no speech
                    no_speech_pred_idx = matrix_classes.index('no speech')
                    cm[no_speech_true_idx, no_speech_pred_idx] += 1
                else:
                    # Incorrectly predicted some speech class for a silent frame
                    for pred_idx, pred_val in enumerate(pred_labels):
                        if pred_val == 1:
                            pred_class_name = class_names_list[pred_idx]
                            pred_matrix_idx = matrix_classes.index(pred_class_name)
                            cm[no_speech_true_idx, pred_matrix_idx] += 1
            else:
                # Handle frames with ground truth labels
                for true_idx, true_val in enumerate(true_labels):
                    if true_val == 1:  # This class is present in ground truth
                        true_class_name = class_names_list[true_idx]
                        true_matrix_idx = matrix_classes.index(true_class_name)
                        
                        # Check what was predicted for this frame
                        if pred_labels.sum() == 0:
                            # Nothing was predicted, count as "no speech" prediction
                            no_speech_idx = matrix_classes.index('no speech')
                            cm[true_matrix_idx, no_speech_idx] += 1
                        else:
                            # Something was predicted
                            for pred_idx, pred_val in enumerate(pred_labels):
                                if pred_val == 1:  # This class was predicted
                                    pred_class_name = class_names_list[pred_idx]
                                    pred_matrix_idx = matrix_classes.index(pred_class_name)
                                    cm[true_matrix_idx, pred_matrix_idx] += 1
        
        # Calculate percentages row-wise (based on ground truth counts for each class)
        cm_percent = np.zeros_like(cm, dtype=float)
        for i in range(cm.shape[0]):
            row_sum = cm[i].sum()  # Total ground truth frames for this class
            if row_sum > 0:
                cm_percent[i] = (cm[i] / row_sum) * 100
            else:
                cm_percent[i] = 0.0
        
        # Create DataFrame for counts
        cm_counts_df = pd.DataFrame(cm, index=matrix_classes, columns=matrix_classes)
        cm_counts_df.index.name = 'Ground Truth'
        cm_counts_df.columns.name = 'Predicted'
        
        # Create DataFrame for percentages
        cm_percent_df = pd.DataFrame(cm_percent, index=matrix_classes, columns=matrix_classes)
        cm_percent_df.index.name = 'Ground Truth'
        cm_percent_df.columns.name = 'Predicted'
        
        # Save confusion matrices to CSV
        counts_path = confusion_matrices_dir / f'confusion_matrix_counts_{evaluation_level}_level.csv'
        percent_path = confusion_matrices_dir / f'confusion_matrix_percentages_{evaluation_level}_level.csv'
        
        cm_counts_df.to_csv(counts_path)
        cm_percent_df.to_csv(percent_path, float_format='%.2f')
        
        # Create and save visualization
        plt.figure(figsize=(12, 5))
        
        # Counts subplot
        plt.subplot(1, 2, 1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Counts')
        plt.colorbar()
        tick_marks = np.arange(len(matrix_classes))
        plt.xticks(tick_marks, matrix_classes, rotation=45)
        plt.yticks(tick_marks, matrix_classes)
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        
        # Add text annotations for counts
        for i_cm, j_cm in np.ndindex(cm.shape):
            plt.text(j_cm, i_cm, f'{cm[i_cm, j_cm]}',
                    ha="center", va="center", color="white" if cm[i_cm, j_cm] > cm.max() / 2 else "black")
        
        # Percentages subplot
        plt.subplot(1, 2, 2)
        plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Percentages')
        plt.colorbar()
        plt.xticks(tick_marks, matrix_classes, rotation=45)
        plt.yticks(tick_marks, matrix_classes)
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted')
        
        # Add text annotations for percentages
        for i_cm, j_cm in np.ndindex(cm_percent.shape):
            plt.text(j_cm, i_cm, f'{cm_percent[i_cm, j_cm]:.1f}%',
                    ha="center", va="center", color="white" if cm_percent[i_cm, j_cm] > cm_percent.max() / 2 else "black")
        
        plt.tight_layout()
        viz_path = confusion_matrices_dir / f'confusion_matrix_{evaluation_level}_level.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Confusion matrix: {confusion_matrices_dir}")
    
    print(f"✅ Results saved:")
    print(f"  📊 Summary: {summary_path}")
    print(f"  📋 Detailed: {predictions_path}")
    print(f"  🎯 Evaluation level: {evaluation_level}")

def evaluate_model_both_levels(model, test_generator, mlb, thresholds, output_dir, generate_confusion_matrices=False):
    """
    Evaluate model at both window-level and second-level for comparison.
    
    This function runs the evaluation twice:
    1. At the original window level (no aggregation)
    2. At the second level (with sliding window aggregation)
    
    This allows for direct comparison of how aggregation affects performance metrics.
    
    Parameters:
    ----------
    model: Trained classification model
    test_generator: Test data generator  
    mlb: MultiLabelBinarizer
    thresholds: Optimized decision thresholds
    output_dir: Base output directory
    generate_confusion_matrices: Whether to generate confusion matrices
    """
    base_output_dir = Path(output_dir)
    
    print("🔍 Running evaluation at both window and second levels...")
    print("=" * 70)
    
    # Evaluate at window level
    print("\n📊 WINDOW-LEVEL EVALUATION")
    print("=" * 40)
    window_output_dir = base_output_dir / 'window_level'
    evaluate_model(
        model=model,
        test_generator=test_generator,
        mlb=mlb, 
        thresholds=thresholds,
        output_dir=window_output_dir,
        generate_confusion_matrices=generate_confusion_matrices,
        aggregate_to_seconds=False
    )
    
    # Evaluate at second level  
    print("\n📊 SECOND-LEVEL EVALUATION")
    print("=" * 40)
    second_output_dir = base_output_dir / 'second_level'
    evaluate_model(
        model=model,
        test_generator=test_generator,
        mlb=mlb,
        thresholds=thresholds, 
        output_dir=second_output_dir,
        generate_confusion_matrices=generate_confusion_matrices,
        aggregate_to_seconds=True
    )
    
    print("\n✅ EVALUATION COMPARISON COMPLETE")
    print("=" * 70)
    print(f"📁 Window-level results: {window_output_dir}")
    print(f"📁 Second-level results: {second_output_dir}")
    print("\n💡 Compare the evaluation_summary.json files to see the difference in metrics!")

# ---- Inference Functions ----
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

def aggregate_and_save_results(audio_file, predictions, class_names, output_writer, thresholds):
    """
    Aggregates overlapping window predictions and writes the final results to a CSV file.
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
    output_writer : csv.DictWriter
        CSV writer object to write the results.
    thresholds : dict
        Dictionary mapping class names to optimal thresholds
    """    
    video_id = Path(audio_file).stem
    
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

    # Aggregate and write results using class-specific thresholds
    for second in sorted(second_predictions.keys()):
        avg_probs = np.mean(second_predictions[second]['probabilities'], axis=0)
        
        # Apply class-specific thresholds
        binary_labels = []
        for i, class_name in enumerate(class_names):
            threshold = thresholds[str(class_name)]
            binary_labels.append(1 if avg_probs[i] > threshold else 0)
        
        row = {'video_id': video_id, 'second': second}
        for i, class_name in enumerate(class_names):
            row[class_name] = binary_labels[i]
        
        output_writer.writerow(row)

def process_audio_folder(folder_path, model, mlb, thresholds):
    """
    Process a folder of audio files and save classification results using optimized thresholds.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing audio files.
    model : tf.keras.Model
        The trained audio classification model.
    mlb : sklearn.preprocessing.MultiLabelBinarizer
        MultiLabelBinarizer fitted on the training data.
    thresholds : dict
        Dictionary mapping class names to optimal thresholds
    """    
    output_file = AudioClassification.OUTPUT_DIR / 'classification_results.csv'
    
    audio_files = list(folder_path.glob('*.wav'))
    if not audio_files:
        print(f"⚠️ No WAV audio files found in {folder_path}")
        return
    
    print(f"✅ Found {len(audio_files)} audio files to process.")
    
    # Prepare CSV file
    class_names = mlb.classes_
    fieldnames = ['video_id', 'second'] + list(class_names)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, audio_file in enumerate(audio_files):
            print("\n" + "="*50)
            print(f"Processing file {i+1}/{len(audio_files)}: {audio_file.name}")
            print("="*50)
            
            try:
                prediction_results, _, _ = classify_audio_windows(str(audio_file), model, mlb)
                
                if prediction_results:
                    aggregate_and_save_results(str(audio_file), prediction_results, class_names, writer, thresholds)
                else:
                    print(f"No prediction results for {audio_file.name}")
            except Exception as e:
                print(f"❌ An error occurred while processing {audio_file.name}: {e}")

    print(f"\n✅ All processing complete. Results saved to {output_file}")

def load_model(model_path: Path = AudioClassification.TRAINED_WEIGHTS_PATH):
    """
    Load trained model for inference by building the architecture and loading weights.
    
    Parameters:
    ----------
    model_path (Path): 
        Path to the saved model weights file (.keras)
        
    Returns:
    -------
    tuple: (model, mlb) where:
        - model: Loaded Keras model ready for inference
        - mlb: Fitted MultiLabelBinarizer for decoding predictions
    """
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("💡 Please train the model first using train_audio_classifier.py")
        return None, None
        
    # Setup multi-label binarizer consistent with training
    mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
    mlb.fit([[]])  # Initialize with empty list to set up classes
    num_classes = len(mlb.classes_)

    # Use centralized calculation for consistent time steps across all components
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    try:
        # Build the model architecture first
        model = build_model_multi_label(
            n_mels=AudioConfig.N_MELS,
            fixed_time_steps=fixed_time_steps,
            num_classes=num_classes
        )
        
        # Load only the weights from the saved .keras file
        model.load_weights(model_path)
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None
    
    return model, mlb

# ---- GPU Configuration ----
def setup_gpu_config():
    """
    Configure GPU settings with proper error handling.
    """
    try:
        # Check for GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        
        if not physical_devices:
            print("⚠️ No GPU devices found")
            return False
        
        # Configure GPU memory growth
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        
        # Test GPU functionality
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0, 2.0, 3.0])
            test_result = tf.reduce_sum(test_tensor)
        
        return True
        
    except Exception as e:
        print(f"⚠️ GPU configuration failed: {e}")
        return False