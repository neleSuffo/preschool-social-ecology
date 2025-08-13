import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import json
import csv
import time
import shutil
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tqdm import tqdm
from contants import AudioClassification
from config import AudioClsConfig
import argparse

# Environment setup
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Custom Focal Loss ---
class FocalLoss(tf.keras.losses.Loss):
    """
    Custom Focal Loss implementation for handling class imbalance in multi-label classification.
    
    Focal loss addresses class imbalance by down-weighting easy examples and focusing
    training on hard negatives. Particularly effective for scenarios with significant
    class imbalance or when easier examples dominate the loss.
    
    Parameters
    ---------- 
    gamma (float): 
        Focusing parameter. Higher values down-weight easy examples more
    alpha (float): 
        Weighting factor for rare class. Values in [0,1] for class 1, 1-alpha for class 0
    name (str): 
        Name for the loss function
        
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        where p_t = p if y=1, else 1-p
    """
    def __init__(self, gamma=3.0, alpha=0.5, name='improved_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        loss = alpha_t * focal_weight * ce
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

# --- Custom Macro F1 Metric ---
class MacroF1Score(tf.keras.metrics.Metric):
    """
    Custom Keras metric for computing macro-averaged F1-score in multi-label classification.
    
    Calculates F1-score for each class independently using precision and recall,
    then averages across all classes. Particularly useful for multi-label scenarios
    where equal weight should be given to each class regardless of frequency.
    
    Parameters:
    ----------
    num_classes (int): 
        Number of classes in the multi-label problem
    threshold (float): 
        Decision threshold for binary predictions (default: 0.5)
    name (str): 
        Name for the metric (default: 'macro_f1')

    Attributes:
        precisions (list): List of Precision metrics, one per class
        recalls (list): List of Recall metrics, one per class
        
    Returns:
        float: Macro-averaged F1-score across all classes
        
    Formula:
        F1_i = 2 * (precision_i * recall_i) / (precision_i + recall_i)
        Macro_F1 = (1/N) * Σ F1_i
    """
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        self.precisions = [Precision(thresholds=threshold, name=f'precision_{i}') for i in range(num_classes)]
        self.recalls = [Recall(thresholds=threshold, name=f'recall_{i}') for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        for i in range(self.num_classes):
            y_true_class = y_true[..., i]
            y_pred_class = y_pred[..., i]
            self.precisions[i].update_state(y_true_class, y_pred_class, sample_weight)
            self.recalls[i].update_state(y_true_class, y_pred_class, sample_weight)

    def result(self):
        f1_scores = []
        for i in range(self.num_classes):
            p = self.precisions[i].result()
            r = self.recalls[i].result()
            f1 = 2 * (p * r) / (p + r + tf.keras.backend.epsilon())
            f1_scores.append(f1)
        return tf.reduce_mean(f1_scores) if f1_scores else tf.constant(0.0, dtype=tf.float32)

    def reset_state(self):
        for i in range(self.num_classes):
            self.precisions[i].reset_state()
            self.recalls[i].reset_state()

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)

# --- Threshold Optimizer ---
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    """
    Keras callback for optimizing classification thresholds during training.
    
    Automatically finds optimal decision thresholds for each class to maximize F1-score.
    Runs threshold optimization every 5 epochs on validation data.
    
    Args:
        validation_generator: Validation data generator for threshold optimization
        mlb_classes (list): List of multi-label class names
        
    Attributes:
        best_thresholds (list): Current best thresholds for each class
        best_f1 (float): Best macro F1-score achieved
        
    Note:
        Saves optimized thresholds to 'thresholds.json' in model log directory.
        Uses F1-score as optimization criterion for each class independently.
    """
    def __init__(self, validation_generator, mlb_classes):
        super().__init__()
        self.validation_generator = validation_generator
        self.mlb_classes = mlb_classes
        self.best_thresholds = [0.5] * len(mlb_classes)
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.optimize_thresholds()

    def optimize_thresholds(self):
        predictions = self.model.predict(self.validation_generator, verbose=0)
        true_labels = []
        for i in range(len(self.validation_generator)):
            _, labels = self.validation_generator[i]
            true_labels.extend(labels)
        true_labels = np.array(true_labels)
        
        best_thresholds = []
        for class_idx in range(len(self.mlb_classes)):
            best_threshold = 0.5
            best_class_f1 = 0.0
            for threshold in np.arange(0.1, 0.9, 0.01):  # Finer granularity
                pred_binary = (predictions[:, class_idx] > threshold).astype(int)
                f1 = f1_score(true_labels[:, class_idx], pred_binary, zero_division=0)
                if f1 > best_class_f1:
                    best_class_f1 = f1
                    best_threshold = threshold
            best_thresholds.append(best_threshold)
        
        self.best_thresholds = best_thresholds
        print(f"Optimized thresholds: {dict(zip(self.mlb_classes, best_thresholds))}")
        # Save thresholds
        with open(os.path.join(self.model.log_dir, 'thresholds.json'), 'w') as f:
            json.dump(dict(zip(self.mlb_classes, best_thresholds)), f)

# --- Feature Extraction ---
def extract_enhanced_features(audio_path, start_time, duration, sr=16000, n_mels=256, hop_length=512, fixed_time_steps=None):
    """
    Extract enhanced audio features (mel-spectrogram + MFCC) from audio segment.
    
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

    Returns:
        np.ndarray: Combined feature matrix of shape (n_mels + 13, fixed_time_steps)
                   where 13 is the number of MFCC coefficients
                   
    Features extracted:
        - Mel-spectrogram: Perceptually-relevant frequency representation
        - MFCC: Compact spectral features for speech/audio
        - Preprocessing: Normalization, pre-emphasis filtering
        - Post-processing: Padding/truncation to fixed dimensions
    """
    if fixed_time_steps is None:
        fixed_time_steps = int(np.ceil(duration * sr / hop_length))
    
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
            return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)  # +13 for MFCC
        
        y = y / (np.max(np.abs(y)) + 1e-6)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048, fmin=20, fmax=10000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_spectrogram_db = 2 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-6) - 1
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6) - 1
        
        # Concatenate features
        combined = np.concatenate([mel_spectrogram_db, mfcc], axis=0)
        
        if combined.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
        elif combined.shape[1] > fixed_time_steps:
            combined = combined[:, :fixed_time_steps]
        
        return combined
    
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)

# --- RTTM Processing ---
def convert_rttm_to_training_segments(rttm_path, audio_files_dir, valid_rttm_classes, window_duration, window_step, sr, n_mels, hop_length, output_segments_path):
    """
    Parse RTTM annotation files into windowed multi-label training segments.
    
    Converts continuous speaker/activity annotations into fixed-duration overlapping
    windows suitable for supervised multi-label audio classification training.
    
    Parameters:
    ----------
    rttm_path (str):
        Path to RTTM annotation file with speaker/activity labels
    audio_files_dir (str):
        Directory containing corresponding audio files (.wav)
    valid_rttm_classes (list):
        List of valid speaker/activity IDs to include (e.g., ['OHS', 'CDS', 'KCHI'])
    window_duration (float):
        Duration of each training segment in seconds (e.g., 3.0)
    window_step (float):
        Step size between windows in seconds (e.g., 1.0 for overlap)
    sr (int):
        Sample rate for audio processing (used for validation)
    n_mels (int):
        Number of mel filters (used for validation)
    hop_length (int):
        Hop length for feature extraction (used for validation)
    output_segments_path (str):
        Path where segment metadata will be saved (JSONL format)
    
    Returns:
    -------
    tuple: (segment_count, unique_labels_found)
        segment_count (int): Number of valid segments created
        unique_labels_found (list): Sorted list of unique labels found in data
        
    Output format (JSONL):
        Each line contains: {"audio_path": str, "start": float, "duration": float, "labels": list}
    
    Processing logic:
        1. Load RTTM file with speaker/activity time annotations
        2. For each audio file, create sliding windows of specified duration
        3. For each window, find all overlapping speaker/activity labels
        4. Keep only windows with at least one valid label
        5. Save segment metadata for training data generators
        
    RTTM format expected:
        SPEAKER file_id channel start duration NA1 NA2 speaker_id NA3 NA4
        
    Note:
        Handles audio duration mismatches and missing files gracefully.
        Progress bar shows processing status across all audio files.
    """
    try:
        rttm_df = pd.read_csv(rttm_path, sep=' ', header=None, 
                              names=['type', 'file_id', 'channel', 'start', 'duration', 'NA1', 'NA2', 'speaker_id', 'NA3', 'NA4'])
    except Exception as e:
        print(f"Error reading RTTM file {rttm_path}: {e}")
        return [], []
    
    all_unique_labels = set()
    unique_file_ids = rttm_df['file_id'].unique()
    print(f"Processing {len(unique_file_ids)} audio files from RTTM: {os.path.basename(rttm_path)}...")
    
    segment_counter = 0
    with open(output_segments_path, 'w', newline='') as f_out:
        for file_id in tqdm(unique_file_ids):
            audio_path = os.path.join(audio_files_dir, f"{file_id}.wav")
            if not os.path.exists(audio_path):
                print(f"Warning: Audio file not found: {audio_path}. Skipping.")
                continue
            
            file_segments = rttm_df[rttm_df['file_id'] == file_id].copy()
            file_segments['end'] = file_segments['start'] + file_segments['duration']
            
            try:
                audio_duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_path}: {e}. Skipping.")
                continue
                
            max_time_rttm = file_segments['end'].max() if not file_segments.empty else 0
            analysis_end_time = min(audio_duration, max_time_rttm)
            
            current_time = 0.0
            while current_time < analysis_end_time:
                window_start = current_time
                window_end = min(current_time + window_duration, audio_duration, analysis_end_time)
                
                active_speaker_ids = set()
                for _, row in file_segments.iterrows():
                    segment_start = row['start']
                    segment_end = row['end']
                    if max(window_start, segment_start) < min(window_end, segment_end):
                        active_speaker_ids.add(row['speaker_id'])
                
                active_labels = {sid for sid in active_speaker_ids if sid in valid_rttm_classes}
                all_unique_labels.update(active_labels)
                
                if active_labels and (window_end - window_start) > 0:
                    segment_data = {
                        'audio_path': audio_path,
                        'start': window_start,
                        'duration': window_duration,
                        'labels': sorted(list(active_labels))
                    }
                    f_out.write(json.dumps(segment_data) + '\n')
                    segment_counter += 1
                
                current_time += window_step
                
    return segment_counter, sorted(list(all_unique_labels))

# --- Deep Learning Model Architecture ---
def build_model_multi_label(n_mels, fixed_time_steps, num_classes):
    """
    Build a deep learning model for multi-label voice type classification.
    
    Creates a CNN-RNN hybrid architecture with multi-head attention for 
    classifying overlapping voice types in audio segments. The model combines:
    - ResNet-style CNN blocks for feature extraction from spectrograms
    - Bidirectional GRU layers for temporal modeling
    - Multi-head attention mechanism for focus on important features
    - Independent output heads per class for multi-label prediction
    
    Parameters:
    ----------
    n_mels (int): 
        Number of mel-frequency bins in input spectrograms
    fixed_time_steps (int): 
        Fixed number of time steps in input spectrograms
    num_classes (int): 
        Number of voice type classes to predict
        
    Returns
    -------
    Model
        Compiled Keras model ready for training
        
    Architecture details:
        - Input: (n_mels + 13, fixed_time_steps, 1) - mel + MFCC features
        - CNN: 4 ResNet blocks with [32, 64, 128, 256] filters
        - RNN: 2 bidirectional GRU layers with [256, 128] units
        - Attention: 4-head attention mechanism
        - Output: Sigmoid activation per class for multi-label prediction
        - Loss: Focal loss for handling class imbalance
        - Metrics: Accuracy, precision, recall, macro F1-score
    """
    input_mel = Input(shape=(n_mels + 13, fixed_time_steps, 1), name='mel_spectrogram_input')
    x = input_mel

    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
        shortcut = x
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        if shortcut.shape[-1] != filters or strides != (1, 1):
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    x = conv_block(x, 32)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = conv_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    x = conv_block(x, 256)  # Added deeper block
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    reduced_n_mels = (n_mels + 13) // 16
    reduced_time_steps = fixed_time_steps // 16
    channels_after_cnn = 256

    x = Permute((2, 1, 3))(x)
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3))(x)  # Increased units
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)

    def multi_head_attention(x, num_heads=4):
        head_size = x.shape[-1] // num_heads
        heads = []
        for i in range(num_heads):
            attention = Dense(1, activation='tanh')(x)
            attention = Flatten()(attention)
            attention = Activation('softmax')(attention)
            attention = RepeatVector(x.shape[-1])(attention)
            attention = Permute((2, 1))(attention)
            head = multiply([x, attention])
            head = Lambda(lambda xin: tf.reduce_sum(xin, axis=1))(head)
            heads.append(head)
        return Concatenate()(heads) if len(heads) > 1 else heads[0]

    x = multi_head_attention(x)

    dense_input = x
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    if dense_input.shape[-1] == 256:
        x = Add()([x, dense_input])
    
    class_outputs = []
    for i in range(num_classes):
        class_branch = Dense(128, activation='relu', name=f'class_{i}_dense')(x)
        class_branch = Dropout(0.3)(class_branch)
        class_output = Dense(1, activation='sigmoid', name=f'class_{i}_output')(class_branch)
        class_outputs.append(class_output)
    
    output = Concatenate(name='combined_output')(class_outputs)

    model = Model(inputs=input_mel, outputs=output)
    
    macro_f1 = MacroF1Score(num_classes=num_classes, name='macro_f1')
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), macro_f1]
    )
    model.log_dir = None  # For ThresholdOptimizer
    return model

# --- Class Weights ---
def calculate_balanced_class_weights(train_segments, mlb):
    """
    Calculate balanced class weights for handling class imbalance in training.
    
    Computes inverse frequency weights for each class to balance training
    when dealing with imbalanced datasets. Uses total sample count normalized
    by class frequency.
    
    Parameters
    -----------
    training_segments (list):
        List of training segment dictionaries containing 'labels' key
    mlb (MultiLabelBinarizer):
        Fitted multi-label binarizer with class information
        
    Returns
    --------
    dict:
        Dictionary mapping class indices to computed weights
              
    Weight calculation:
        weight_i = total_samples / (num_classes * class_i_count)
        
    Note:
        For ID-based split balanced datasets, this typically returns near-uniform weights.
        Consider using uniform weights (None) for already balanced data.
    """
    class_counts = {label: 0 for label in mlb.classes_}
    for seg in train_segments:
        for label in seg['labels']:
            if label in class_counts:
                class_counts[label] += 1
    
    class_weights_for_keras = {}
    total_samples = len(train_segments)
    for i, class_name in enumerate(mlb.classes_):
        count = class_counts.get(class_name, 1)
        weight = (total_samples / (len(mlb.classes_) * count))  # Balanced weighting
        class_weights_for_keras[i] = weight
    
    return class_weights_for_keras

# --- Data Generator ---
class EnhancedAudioSegmentDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for audio classification with enhanced features and augmentation.
    
    Generates batches of audio features (mel-spectrogram + MFCC) with corresponding
    multi-label targets for training and evaluation. Supports data augmentation
    during training to improve model generalization.
    
    Args:
        segments_file_path (str): Path to JSONL file containing segment metadata
        mlb (MultiLabelBinarizer): Fitted multi-label binarizer for encoding
        n_mels (int): Number of mel-frequency bands
        hop_length (int): Hop length for feature extraction
        sr (int): Sample rate for audio processing
        window_duration (float): Duration of each audio segment
        fixed_time_steps (int): Fixed number of time steps for consistent input shape
        batch_size (int): Number of samples per batch (default: 32)
        shuffle (bool): Whether to shuffle data at epoch end (default: True)
        augment (bool): Whether to apply data augmentation (default: False)
        
    Features:
        - Lazy loading: Features extracted on-demand for memory efficiency
        - Data augmentation: Time/frequency masking, noise addition, pitch shifting
        - Multi-label support: Handles overlapping voice type labels
        - Consistent batching: Fixed input dimensions across all samples
    """
    def __init__(self, segments_file_path, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, 
                 batch_size=32, shuffle=True, augment=False):
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
        if not self.augment:
            return mel_spec
        augmented = mel_spec.copy()
        if np.random.random() < 0.5:
            time_mask_width = np.random.randint(1, min(20, mel_spec.shape[1] // 4))
            time_mask_start = np.random.randint(0, mel_spec.shape[1] - time_mask_width)
            augmented[:, time_mask_start:time_mask_start + time_mask_width] = -1
        if np.random.random() < 0.5:
            freq_mask_width = np.random.randint(1, min(15, mel_spec.shape[0] // 4))
            freq_mask_start = np.random.randint(0, mel_spec.shape[0] - freq_mask_width)
            augmented[freq_mask_start:freq_mask_start + freq_mask_width, :] = -1
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.05, mel_spec.shape)
            augmented = np.clip(augmented + noise, -1, 1)
        # Pitch shift (new augmentation)
        if np.random.random() < 0.3:
            n_steps = np.random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(augmented, sr=self.sr, n_steps=n_steps)
        return augmented

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_segments = [self.segments_data[k] for k in indexes]
        X_batch = []
        y_batch = []
        for segment in batch_segments:
            mel = extract_enhanced_features(
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
def compute_cosine_annealing_lr(epoch=int):
    """
    Calculate learning rate using cosine annealing schedule.
    
    Implements cosine annealing with warm restarts every 20 epochs,
    cycling between maximum and minimum learning rates.
    
    Args:
        epoch (int): Current training epoch (0-indexed)
    
    Returns:
        float: Learning rate for the current epoch
        
    Schedule details:
        - max_lr: 0.001 (peak learning rate)
        - min_lr: 0.00001 (minimum learning rate)
        - cycle_length: 20 epochs per complete cosine cycle
        - Formula: min_lr + (max_lr - min_lr) * (1 + cos(π * epoch_in_cycle / cycle_length)) / 2
    """
    """
    Calculate learning rate using cosine annealing schedule.
    
    Implements cosine annealing with warm restarts every 20 epochs,
    cycling between maximum and minimum learning rates.
    
    Parameters
    ---------
        epoch (int): Current training epoch (0-indexed)
    
    Returns:
        float: Learning rate for the current epoch
        
    Schedule details:
        - max_lr: 0.001 (peak learning rate)
        - min_lr: 0.00001 (minimum learning rate)
        - cycle_length: 20 epochs per complete cosine cycle
        - Formula: min_lr + (max_lr - min_lr) * (1 + cos(π * epoch_in_cycle / cycle_length)) / 2
    """
    max_lr = 0.001
    min_lr = 0.00001
    epochs_per_cycle = 20
    cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
    lr = min_lr + (max_lr - min_lr) * (1 + np.cos(cos_inner)) / 2
    return lr

# --- Training Callbacks ---
def create_training_callbacks(run_dir, val_generator, mlb_classes):
    """
    Create and configure training callbacks for model optimization.
    
    Args:
        run_dir (str): Directory to save model checkpoints and logs
        val_generator: Validation data generator for threshold optimization
        mlb_classes (list): List of multi-label class names
        
    Returns:
        list: Configured Keras callbacks for training
    """
    callbacks = [
        EarlyStopping(monitor='val_macro_f1', patience=30, mode='max', restore_best_weights=True, verbose=1),
        LearningRateScheduler(compute_cosine_annealing_lr, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(run_dir, 'best_model.h5'),
            monitor='val_macro_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ThresholdOptimizer(val_generator, mlb_classes),
        TrainingLogger(run_dir, mlb_classes)
    ]
    return callbacks

# --- Custom History and Plotting Callback ---
class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Custom callback for logging training metrics and generating plots.
    
    Logs detailed training metrics to CSV file and creates visualization plots
    after training completion. Tracks loss, accuracy, precision, recall, 
    macro F1-score, and learning rate for both training and validation.
    
    Args:
        log_dir (str): Directory to save CSV logs and plot images
        mlb_classes (list): List of multi-label class names
        
    Attributes:
        csv_file_path (str): Path to CSV file containing training metrics
        history (dict): Dictionary storing all training metrics
        start_time (float): Training start timestamp
        
    Features:
        - Real-time CSV logging of all metrics per epoch
        - Automatic plot generation (loss and macro F1-score)
        - Elapsed time tracking
        - Learning rate monitoring
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
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))

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
        print("Training finished. Generating plots...")
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

# --- Setup Functions ---
def setup_training_environment():
    """
    Set up training environment with directories and configuration.
    
    Returns:
        tuple: (run_dir, segment_files_dict) containing training directory 
               and paths to segment files for train/val/test splits
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(AudioClassification.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Training run output directory: {run_dir}")
    
    # Generate segment file paths
    window_duration_str = str(AudioClsConfig.window_duration).replace('.', 'p')
    window_step_str = str(AudioClsConfig.window_step).replace('.', 'p')
    
    segment_files = {
        'train': os.path.join(os.path.dirname(run_dir), f'train_segments_w{window_duration_str}_s{window_step_str}.jsonl'),
        'val': os.path.join(os.path.dirname(run_dir), f'val_segments_w{window_duration_str}_s{window_step_str}.jsonl'),
        'test': os.path.join(os.path.dirname(run_dir), f'test_segments_w{window_duration_str}_s{window_step_str}.jsonl')
    }
    
    # Copy current script for reproducibility
    current_script_path = os.path.abspath(__file__)
    shutil.copy(current_script_path, os.path.join(run_dir, 'train_audio_classifier.py'))
    
    return run_dir, segment_files

def process_rttm_data_splits(segment_files):
    """
    Process RTTM files for all data splits (train/val/test) with ID-based splitting already applied.
    
    Args:
        segment_files (dict): Dictionary with keys 'train', 'val', 'test' and paths as values
        
    Returns:
        tuple: (segment_counts, unique_labels) where segment_counts is dict with counts per split
               and unique_labels contains all unique labels found across splits
    """
    # RTTM file paths (already ID-split)
    rttm_files = {
        'train': AudioClassification.train_rttm_file,
        'val': AudioClassification.val_rttm_file,  # Assuming this exists in config
        'test': AudioClassification.test_rttm_file  # Assuming this exists in config
    }
    
    segment_counts = {}
    all_unique_labels = set()
    
    for split_name in ['train', 'val', 'test']:
        print(f"--- Processing {split_name.title()} Data ---")
        
        num_segments, unique_labels = convert_rttm_to_training_segments(
            rttm_files[split_name], 
            AudioClassification.audio_files_dir, 
            AudioClassification.valid_rttm_classes,
            AudioClsConfig.window_duration, 
            AudioClsConfig.window_step, 
            AudioClsConfig.sr, 
            AudioClsConfig.n_mels, 
            AudioClsConfig.hop_length, 
            segment_files[split_name]
        )
        
        segment_counts[split_name] = num_segments
        all_unique_labels.update(unique_labels)
        
        if num_segments == 0:
            raise ValueError(f"No segments found for {split_name} split. Check RTTM file: {rttm_files[split_name]}")
    
    return segment_counts, sorted(list(all_unique_labels))

def setup_multilabel_encoder(unique_labels):
    """
    Set up multi-label binarizer for voice type classification.
    
    Args:
        unique_labels (list): List of unique voice type labels found in data
        
    Returns:
        tuple: (mlb, num_classes) where mlb is fitted MultiLabelBinarizer 
               and num_classes is the number of classes
    """
    mlb = MultiLabelBinarizer(classes=unique_labels)
    mlb.fit([[]])  # Fit with empty list to initialize
    num_classes = len(mlb.classes_)
    
    print(f"\nDetected {num_classes} unique target classes: {mlb.classes_}")
    return mlb, num_classes

def determine_class_weights(mlb, use_balanced_weights=False):
    """
    Determine class weights for training. Since ID-based splitting is used,
    data should be balanced, so uniform weights are typically sufficient.
    
    Args:
        mlb: Fitted MultiLabelBinarizer
        use_balanced_weights (bool): Whether to calculate balanced weights or use uniform
        
    Returns:
        dict or None: Class weights for Keras training, None for uniform weighting
    """
    if use_balanced_weights:
        # Calculate weights if needed (typically not necessary with ID-split balanced data)
        print("Calculating balanced class weights...")
        # Implementation would go here if needed
        return {i: 1.0 for i in range(len(mlb.classes_))}
    else:
        print("Using uniform class weighting (recommended for ID-split balanced data)")
        return None

def create_data_generators(segment_files, mlb):
    """
    Create data generators for training, validation, and testing.
    
    Args:
        segment_files (dict): Paths to segment files for each split
        mlb: Fitted MultiLabelBinarizer
        
    Returns:
        tuple: (train_generator, val_generator, test_generator)
    """
    # Calculate fixed time steps for consistent input shape
    fixed_time_steps = int(np.ceil(AudioClsConfig.window_duration * AudioClsConfig.sr / AudioClsConfig.hop_length))
    
    train_generator = EnhancedAudioSegmentDataGenerator(
        segment_files['train'], mlb, 
        AudioClsConfig.n_mels, AudioClsConfig.hop_length, AudioClsConfig.sr, 
        AudioClsConfig.window_duration, fixed_time_steps,
        batch_size=32, shuffle=True, augment=True
    )
    
    val_generator = EnhancedAudioSegmentDataGenerator(
        segment_files['val'], mlb,
        AudioClsConfig.n_mels, AudioClsConfig.hop_length, AudioClsConfig.sr,
        AudioClsConfig.window_duration, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False
    )
    
    test_generator = EnhancedAudioSegmentDataGenerator(
        segment_files['test'], mlb,
        AudioClsConfig.n_mels, AudioClsConfig.hop_length, AudioClsConfig.sr,
        AudioClsConfig.window_duration, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False
    )
    
    return train_generator, val_generator, test_generator

def train_model_with_callbacks(model, train_generator, val_generator, callbacks, class_weights=None, epochs=100):
    """
    Train the model with specified callbacks and data generators.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        val_generator: Validation data generator
        callbacks: List of Keras callbacks
        class_weights: Class weights for imbalanced data (None for uniform)
        epochs (int): Number of training epochs
        
    Returns:
        History: Keras training history object
    """
    print(f"\nStarting training for {epochs} epochs...")
    
    if len(train_generator) == 0 or len(val_generator) == 0:
        raise ValueError("No batches available for training/validation. Check your RTTM files and data paths.")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

def evaluate_model_performance(model, val_generator, test_generator, mlb, run_dir):
    """
    Evaluate trained model on validation and test sets with detailed metrics.
    
    Args:
        model: Trained Keras model
        val_generator: Validation data generator
        test_generator: Test data generator  
        mlb: MultiLabelBinarizer for label handling
        run_dir (str): Directory containing saved thresholds and logs
    """
    print("\n" + "="*60)
    print("FINAL VALIDATION RESULTS (from best restored weights)")
    print("="*60)
    
    val_results = model.evaluate(val_generator, verbose=0)
    val_metrics_dict = dict(zip(model.metrics_names, val_results))
    for name, value in val_metrics_dict.items():
        print(f"Validation {name}: {value:.4f}")
    
    if test_generator and len(test_generator) > 0:
        print("\n" + "="*60)
        print("DETAILED TEST SET EVALUATION")
        print("="*60)
        
        # Get predictions and true labels
        test_predictions = model.predict(test_generator, verbose=1)
        test_true_labels = []
        for i in tqdm(range(len(test_generator)), desc="Collecting true labels"):
            _, labels = test_generator[i]
            test_true_labels.extend(labels)
        test_true_labels = np.array(test_true_labels)
        
        # Handle shape mismatches
        if test_predictions.shape[0] != test_true_labels.shape[0]:
            print(f"Warning: Test predictions ({test_predictions.shape[0]}) and true labels ({test_true_labels.shape[0]}) mismatch.")
            min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
            test_predictions = test_predictions[:min_samples]
            test_true_labels = test_true_labels[:min_samples]
            print(f"Adjusted to {min_samples} samples for evaluation.")
        
        # Load optimized thresholds
        try:
            with open(os.path.join(run_dir, 'thresholds.json'), 'r') as f:
                thresholds_dict = json.load(f)
            thresholds = [thresholds_dict[class_name] for class_name in mlb.classes_]
        except:
            thresholds = [0.5] * len(mlb.classes_)
            print("Using default thresholds (0.5) - optimized thresholds not found.")
        
        # Apply thresholds and calculate metrics
        test_pred_binary = np.array([(test_predictions[:, i] > thresholds[i]).astype(int) for i in range(len(mlb.classes_))]).T
        
        if test_true_labels.sum() > 0:
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                test_true_labels, test_pred_binary, average=None, zero_division=0
            )
            macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
            subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)
            
            print(f"\nTest Metrics (optimized thresholds: {[f'{t:.3f}' for t in thresholds]}):")
            print(f"  Subset Accuracy (exact match): {subset_accuracy:.4f}")
            print(f"  Macro Precision: {macro_precision:.4f}")
            print(f"  Macro Recall: {macro_recall:.4f}")
            print(f"  Macro F1-score: {macro_f1:.4f}")
            
            print(f"\nPer-Class Results:")
            print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            print("-" * 65)
            for i, class_name in enumerate(mlb.classes_):
                print(f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
                      f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
            print("-" * 65)
            print(f"{'MACRO AVG':<15} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
        else:
            print("Warning: No positive instances in test set for metrics calculation.")
    
    print("\nTraining completed successfully! Check the run directory for logs and plots.")

def print_dataset_summary(segment_counts):
    """
    Print summary of dataset splits and segment counts.
    
    Args:
        segment_counts (dict): Dictionary with segment counts per split
    """
    print(f"\nDataset Summary (ID-based splits):")
    print(f"  Training segments: {segment_counts['train']:,}")
    print(f"  Validation segments: {segment_counts['val']:,}")
    print(f"  Test segments: {segment_counts['test']:,}")
    print(f"  Total segments: {sum(segment_counts.values()):,}")

def create_model_and_setup(unique_labels):
    """
    Create and compile the multi-label classification model.
    
    Args:
        unique_labels (list): List of unique voice type labels
        
    Returns:
        tuple: (model, mlb, num_classes, fixed_time_steps) where:
               - model: Compiled Keras model
               - mlb: Fitted MultiLabelBinarizer
               - num_classes: Number of classes
               - fixed_time_steps: Fixed time steps for model input
    """
    # Setup multi-label encoder
    mlb, num_classes = setup_multilabel_encoder(unique_labels)
    
    # Calculate fixed time steps for consistent input shape
    fixed_time_steps = int(np.ceil(AudioClsConfig.window_duration * AudioClsConfig.sr / AudioClsConfig.hop_length))
    
    # Build model architecture
    model = build_model_multi_label(
        n_mels=AudioClsConfig.n_mels, 
        fixed_time_steps=fixed_time_steps, 
        num_classes=num_classes
    )
    
    return model, mlb, num_classes, fixed_time_steps

# --- Main Execution ---
if __name__ == "__main__":
    """
    Main execution entry point for audio classification training.
    
    Runs the complete training pipeline with proper error handling.
    """
    try:
        print("🚀 Starting Audio Classification Training Pipeline")
        print("=" * 60)
        
        # 1. Setup training environment and paths
        print("📁 Setting up training environment...")
        run_dir, segment_files = setup_training_environment()
        
        # 2. Process RTTM data for all splits (ID-based splitting already applied)
        print("📊 Processing RTTM data splits...")
        segment_counts, unique_labels = process_rttm_data_splits(segment_files)
        print_dataset_summary(segment_counts)
        
        # 3. Create model and setup encoders
        print("🧠 Creating model and setting up encoders...")
        model, mlb, num_classes, fixed_time_steps = create_model_and_setup(unique_labels)
        model.log_dir = run_dir  # For ThresholdOptimizer callback
        
        # 4. Determine class weights (uniform for ID-split balanced data)
        print("⚖️ Determining class weights...")
        class_weights = determine_class_weights(mlb, use_balanced_weights=False)
        
        # 5. Create data generators
        print("🔄 Creating data generators...")
        train_generator, val_generator, test_generator = create_data_generators(segment_files, mlb)
        
        # 6. Setup training callbacks
        print("🎯 Setting up training callbacks...")
        callbacks = create_training_callbacks(run_dir, val_generator, mlb.classes_)
        
        # 7. Train the model
        print("🏋️ Starting model training...")
        history = train_model_with_callbacks(
            model, train_generator, val_generator, callbacks, 
            class_weights=class_weights, epochs=100
        )
        
        # 8. Evaluate model performance
        print("📈 Evaluating model performance...")
        evaluate_model_performance(model, val_generator, test_generator, mlb, run_dir)
        
        print("✅ Training pipeline completed successfully!")
    except Exception as e:
        print(f"❌ Training pipeline failed with error: {e}")
        raise