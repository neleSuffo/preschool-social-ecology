import librosa
import numpy as np
import datetime
import json
import csv
import time
import shutil
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Environment setup - MUST BE BEFORE TensorFlow import
os.environ["OMP_NUM_THREADS"] = "6"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Configuration - Try GPU first, fallback to CPU if needed
print("� Configuring GPU mode...")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

import tensorflow as tf

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
        
        print(f"🚀 GPU acceleration enabled - Found {len(physical_devices)} GPU(s)")
        for i, device in enumerate(physical_devices):
            print(f"   GPU {i}: {device}")
        return True
        
    except Exception as e:
        print(f"⚠️ GPU configuration failed: {e}")
        print("💡 To fix CUDA library issues, try:")
        print("   1. conda install -c conda-forge cudatoolkit=11.8 cudnn")
        print("   2. pip install tensorflow[and-cuda]")
        print("   3. Check NVIDIA driver: nvidia-smi")
        return False

# Setup GPU configuration
gpu_available = setup_gpu_config()

if not gpu_available:
    print("\n🔄 GPU setup failed. To continue anyway, press Enter.")
    print("   To force CPU mode, set CUDA_VISIBLE_DEVICES='' before running.")
    input("   Press Enter to continue with current configuration...")
    print()

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tqdm import tqdm
from constants import AudioClassification
from config import AudioConfig
from audio_classifier import FocalLoss, MacroF1Score, ThresholdOptimizer

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
        
        # Mel-spectrogram with optimized parameters to avoid empty filters
        # For 16kHz audio, use fmax = 8000 (Nyquist frequency)
        # Reduce n_mels if needed to avoid empty filters
        effective_n_mels = min(n_mels, 128)  # Cap at 128 for 16kHz audio
        fmax_safe = min(sr // 2, 8000)  # Safe maximum frequency
        
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=effective_n_mels, hop_length=hop_length, n_fft=2048, 
            fmin=80, fmax=fmax_safe
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
        Number of mel-frequency bins in input spectrograms (will be capped at 128)
    fixed_time_steps (int): 
        Fixed number of time steps in input spectrograms
    num_classes (int): 
        Number of voice type classes to predict
        
    Returns
    -------
    Model
        Compiled Keras model ready for training
        
    Architecture details:
        - Input: (effective_n_mels + 13, fixed_time_steps, 1) - mel + MFCC features
        - CNN: 4 ResNet blocks with [32, 64, 128, 256] filters
        - RNN: 2 bidirectional GRU layers with [256, 128] units
        - Attention: 4-head attention mechanism
        - Output: Sigmoid activation per class for multi-label prediction
        - Loss: Focal loss for handling class imbalance
        - Metrics: Accuracy, precision, recall, macro F1-score
    """
    # Use effective mel count (capped at 128 for 16kHz audio)
    effective_n_mels = min(n_mels, 128)
    input_shape = (effective_n_mels + 13, fixed_time_steps, 1)
    
    input_mel = Input(shape=input_shape, name='mel_spectrogram_input')
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

    reduced_n_mels = (effective_n_mels + 13) // 16
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

# --- Data Generator ---
class AudioSegmentDataGenerator(tf.keras.utils.Sequence):
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
    max_lr = 0.001
    min_lr = 0.00001
    epochs_per_cycle = 20
    cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
    lr = min_lr + (max_lr - min_lr) * (1 + np.cos(cos_inner)) / 2
    return lr

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

# --- Setup and Training Functions --
def setup_training_environment():
    """
    Set up training environment with directories and configuration.
    
    Returns:
    -------
    str: Path to training run directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = AudioClassification.OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training run output directory: {run_dir}")
    
    # Copy current script for reproducibility
    current_script_path = Path(__file__)
    shutil.copy(current_script_path, run_dir / 'train_audio_classifier.py')
    
    return run_dir

def setup_multilabel_encoder(unique_labels):
    """
    Set up multi-label binarizer for voice type classification.
    
    Parameters:
    ----------
    unique_labels (list): List of unique voice type labels found in data
        
    Returns:
    -------
    tuple: (mlb, num_classes) where mlb is fitted MultiLabelBinarizer 
           and num_classes is the number of classes
    """
    mlb = MultiLabelBinarizer(classes=unique_labels)
    mlb.fit([[]])  # Fit with empty list to initialize
    num_classes = len(mlb.classes_)
    
    print(f"\nDetected {num_classes} unique target classes: {mlb.classes_}")
    return mlb, num_classes

def create_data_generators(segment_files, mlb):
    """
    Create data generators for training, validation, and testing.
    
    Parameters:
    ----------
    segment_files (dict): Paths to segment files for each split
    mlb: Fitted MultiLabelBinarizer
        
    Returns:
    -------
    tuple: (train_generator, val_generator, test_generator)
    """
    # Calculate fixed time steps for consistent input shape
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
    
    train_generator = AudioSegmentDataGenerator(
        segment_files['train'], mlb, 
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR, 
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=True, augment=True
    )
    
    val_generator = AudioSegmentDataGenerator(
        segment_files['val'], mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False
    )
    
    test_generator = AudioSegmentDataGenerator(
        segment_files['test'], mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False
    )
    
    return train_generator, val_generator, test_generator

def create_training_callbacks(run_dir, val_generator, mlb_classes):
    """
    Create and configure training callbacks for model optimization.
    
    Parameters:
    ----------
    run_dir (str): Directory to save model checkpoints and logs
    val_generator: Validation data generator for threshold optimization
    mlb_classes (list): List of multi-label class names
        
    Returns:
    -------
    list: Configured Keras callbacks for training
    """
    callbacks = [
        EarlyStopping(monitor='val_macro_f1', patience=30, mode='max', restore_best_weights=True, verbose=1),
        LearningRateScheduler(compute_cosine_annealing_lr, verbose=1),
        ModelCheckpoint(
            filepath=(run_dir / 'best_model.h5'),
            monitor='val_macro_f1',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ThresholdOptimizer(val_generator, mlb_classes),
        TrainingLogger(run_dir, mlb_classes)
    ]
    return callbacks

def create_model_and_setup(unique_labels):
    """
    Create and compile the multi-label classification model.
    
    Parameters:
    ----------
    unique_labels (list): List of unique voice type labels
        
    Returns:
    -------
    tuple: (model, mlb, num_classes, fixed_time_steps) where:
           - model: Compiled Keras model
           - mlb: Fitted MultiLabelBinarizer
           - num_classes: Number of classes
           - fixed_time_steps: Fixed time steps for model input
    """
    # Setup multi-label encoder
    mlb, num_classes = setup_multilabel_encoder(unique_labels)
    
    # Calculate fixed time steps for consistent input shape
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    # Build model architecture
    model = build_model_multi_label(
        n_mels=AudioConfig.N_MELS,
        fixed_time_steps=fixed_time_steps,
        num_classes=num_classes
    )
    
    return model, mlb, num_classes, fixed_time_steps

def train_model_with_callbacks(model, train_generator, val_generator, callbacks, epochs):
    """
    Train the model with specified callbacks and data generators.
    
    Parameters:
    ----------
    model: Compiled Keras model
    train_generator: Training data generator
    val_generator: Validation data generator
    callbacks: List of Keras callbacks
    epochs (int): Number of training epochs
        
    Returns:
    -------
    History: Keras training history object
    """
    print(f"\nStarting training for {epochs} epochs...")
    
    if len(train_generator) == 0 or len(val_generator) == 0:
        raise ValueError("No batches available for training/validation. Check your segment files and data paths.")
    
    # Using uniform class weights for ID-split balanced data
    class_weight = None
    print("Using uniform class weighting (recommended for ID-split balanced data)")
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )
    
    return history

def main():
    """
    Main function for audio classification training.
    """   
    unique_labels = AudioConfig.VALID_RTTM_CLASSES
    segment_files = {
        'train': AudioClassification.TRAIN_SEGMENTS_FILE,
        'val': AudioClassification.VAL_SEGMENTS_FILE,
        'test': AudioClassification.TEST_SEGMENTS_FILE
    }

    try:
        print("🚀 Starting Audio Classification Training Pipeline")
        print("=" * 60)
                
        # 1. Setup training environment
        print("🏗️ Setting up training environment...")
        run_dir = setup_training_environment()
        
        # 2. Create model and setup encoders
        print("🧠 Creating model and setting up encoders...")
        model, mlb, num_classes, fixed_time_steps = create_model_and_setup(unique_labels)
        model.log_dir = run_dir  # For ThresholdOptimizer callback

        # 3. Create data generators
        print("🔄 Creating data generators...")
        train_generator, val_generator, test_generator = create_data_generators(segment_files, mlb)

        # 4. Setup training callbacks
        print("🎯 Setting up training callbacks...")
        callbacks = create_training_callbacks(run_dir, val_generator, mlb.classes_)

        # 5. Train the model
        print("🏋️ Starting model training...")
        history = train_model_with_callbacks(
            model, train_generator, val_generator, callbacks, AudioConfig.EPOCHS
        )
        
        # 6. Evaluate model performance
        print("📈 Evaluating model on validation set...")
        val_results = model.evaluate(val_generator, verbose=0)
        val_metrics_dict = dict(zip(model.metrics_names, val_results))
        
        print("\n" + "="*60)
        print("FINAL VALIDATION RESULTS (from best restored weights)")
        print("="*60)
        for name, value in val_metrics_dict.items():
            print(f"Validation {name}: {value:.4f}")
        
        print("✅ Training pipeline completed successfully!")
        print(f"📁 Results saved to: {run_dir}")
        print(f"\n🔍 To evaluate on test set, run:")
        print(f"   python evaluate_audio_classifier.py --model_path {run_dir}/best_model.h5 --run_dir {run_dir}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()