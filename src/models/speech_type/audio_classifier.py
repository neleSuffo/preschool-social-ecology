import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.metrics import Precision, Recall, Metric
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from sklearn.metrics import f1_score

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
        
    Architecture details
    -------------------
    Input: 
        (effective_n_mels + 13, fixed_time_steps, 1) - mel + MFCC features
    CNN: 
        4 ResNet blocks with [32, 64, 128, 256] filters
    RNN: 
        2 bidirectional GRU layers with [256, 128] units
    Attention: 
        4-head attention mechanism
    Output: 
        Sigmoid activation per class for multi-label prediction
    Loss: 
        Focal loss for handling class imbalance
    Metrics: 
        Accuracy, precision, recall, macro F1-score
    """
    # Use effective mel count (capped at 128 for 16kHz audio)
    effective_n_mels = min(n_mels, 128)
    input_shape = (effective_n_mels + 13, fixed_time_steps, 1)
    
    input_mel = Input(shape=input_shape, name='mel_spectrogram_input')
    x = input_mel

    def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
        """ResNet-style convolutional block with skip connections."""
        shortcut = x  # Store input for skip connection
        
        # First convolution + batch norm + activation
        x = Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second convolution + batch norm (no activation yet)
        x = Conv2D(filters, kernel_size, padding='same')(x)
        x = BatchNormalization()(x)
        
        # Adjust shortcut dimensions if needed for skip connection
        if shortcut.shape[-1] != filters or strides != (1, 1):
            shortcut = Conv2D(filters, (1, 1), strides=strides, padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add skip connection and final activation
        x = Add()([x, shortcut])  # Element-wise addition enables gradient flow
        x = Activation('relu')(x)
        return x
    
    # Build CNN backbone: 4 ResNet blocks with progressive filter sizes
    x = conv_block(x, 32)      # First block: learn basic patterns
    x = MaxPooling2D((2, 2))(x)  # Downsample by 2x
    x = Dropout(0.25)(x)       # Prevent overfitting
    
    x = conv_block(x, 64)      # Second block: more complex features  
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = conv_block(x, 128)     # Third block: high-level patterns
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = conv_block(x, 256)     # Fourth block: abstract representations
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Calculate dimensions after CNN layers for reshaping
    reduced_n_mels = (effective_n_mels + 13) // 16     # Divided by 2^4 from 4 pooling layers
    reduced_time_steps = fixed_time_steps // 16
    channels_after_cnn = 256

    # Prepare for RNN: reshape (freq, time, channels) -> (time, freq * channels)
    x = Permute((2, 1, 3))(x)  # Change from (freq, time, channels) to (time, freq, channels)
    x = Reshape((reduced_time_steps, reduced_n_mels * channels_after_cnn))(x)
    
    # RNN layers for temporal modeling of audio sequences
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.3))(x)  # Bidirectional: see past and future context
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(x)  # Smaller layer for refinement

    def multi_head_attention(x, num_heads=4):
        """Multi-head attention mechanism to focus on important time steps."""
        head_size = x.shape[-1] // num_heads
        heads = []
        
        # Create multiple attention heads to focus on different aspects
        for i in range(num_heads):
            # Compute attention weights: which time steps are most important?
            attention = Dense(1, activation='tanh')(x)     # Score each time step
            attention = Flatten()(attention)               # Flatten to 1D
            attention = Activation('softmax')(attention)   # Convert to probabilities (sum=1)
            attention = RepeatVector(x.shape[-1])(attention)  # Broadcast to feature dimension
            attention = Permute((2, 1))(attention)         # Align dimensions for multiplication
            
            # Apply attention: weight the features by importance
            head = multiply([x, attention])                # Element-wise multiplication
            head = Lambda(lambda xin: tf.reduce_sum(xin, axis=1), output_shape=(256,))(head)  # Specify the expected output shape
            heads.append(head)
        
        # Combine all attention heads
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
    """
    def __init__(self, gamma=3.0, alpha=0.5, name='improved_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Convert inputs to float32 for consistent computation
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0) which would cause NaN
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Compute standard binary cross-entropy loss
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate probability of correct classification (p_t in focal loss formula)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Apply alpha weighting (addresses class imbalance)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        
        # Apply focal weight (1-p_t)^gamma (focuses on hard examples)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        
        # Combine all components: FL(p_t) = -α_t * (1-p_t)^γ * CE
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

    Returns
    -------
    float: 
        Macro-averaged F1-score across all classes
    """
    def __init__(self, num_classes, threshold=0.5, name='macro_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.threshold = threshold
        
        # Use single confusion matrix variables instead of per-class metrics
        self.true_positives = self.add_weight(name='tp', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='fp', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to float and apply threshold for binary predictions
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
        
        # Compute confusion matrix components for all classes simultaneously
        # True Positives: both true and predicted are 1
        tp = tf.reduce_sum(y_true * y_pred, axis=0)
        # False Positives: true is 0 but predicted is 1  
        fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
        # False Negatives: true is 1 but predicted is 0
        fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)
        
        # Accumulate counts across batches
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        # Compute precision and recall for all classes using confusion matrix
        # Precision = TP / (TP + FP) - "Of all positive predictions, how many were correct?"
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        # Recall = TP / (TP + FN) - "Of all actual positives, how many did we find?"
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        
        # F1-score = 2 * (precision * recall) / (precision + recall) - harmonic mean
        f1_scores = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        
        # Return macro average: equal weight to each class regardless of frequency
        return tf.reduce_mean(f1_scores)

    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))

    def get_config(self):
        config = super().get_config()
        config.update({'num_classes': self.num_classes, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        num_classes = config.pop('num_classes')
        threshold = config.pop('threshold', 0.5)
        return cls(num_classes=num_classes, threshold=threshold, **config)
    
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
        else: self.per_class_tp.assign(tf.zeros(0)); self.per_class_fn.assign(tf.zeros(0))

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
        
# --- Threshold Optimizer ---
class ThresholdOptimizer(tf.keras.callbacks.Callback):
    """
    Keras callback for optimizing classification thresholds during training.
    
    Automatically finds optimal decision thresholds for each class to maximize F1-score.
    Runs threshold optimization every 5 epochs on validation data.

    Parameters
    ----------
    validation_generator: 
        Validation data generator for threshold optimization
    mlb_classes (list): 
        List of multi-label class names

    Attributes
    ----------
    best_thresholds (list): 
        Current best thresholds for each class
    best_f1 (float): 
        Best macro F1-score achieved
    """
    def __init__(self, validation_generator, mlb_classes):
        super().__init__()
        self.validation_generator = validation_generator
        self.mlb_classes = mlb_classes
        self.best_thresholds = [0.5] * len(mlb_classes)
        self.best_f1 = 0.0
        # Pre-compute threshold candidates for efficiency
        self.threshold_candidates = np.arange(0.1, 0.91, 0.05)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            self.optimize_thresholds()

    def optimize_thresholds(self):
        """Optimized threshold search using vectorized operations."""
        # Get all predictions and true labels at once to avoid repeated model calls
        predictions = self.model.predict(self.validation_generator, verbose=0)
        true_labels = []
        for i in range(len(self.validation_generator)):
            _, labels = self.validation_generator[i]
            true_labels.extend(labels)
        true_labels = np.array(true_labels)
        
        # Handle potential mismatch in sample counts (edge case)
        if len(predictions) != len(true_labels):
            min_samples = min(len(predictions), len(true_labels))
            predictions = predictions[:min_samples]
            true_labels = true_labels[:min_samples]
        
        best_thresholds = []
        
        # Find optimal threshold for each class independently
        for class_idx in range(len(self.mlb_classes)):
            y_true_class = true_labels[:, class_idx]
            y_pred_class = predictions[:, class_idx]
            
            # Skip optimization if no positive samples exist (F1 undefined)
            if np.sum(y_true_class) == 0:
                best_thresholds.append(0.5)  # Use default threshold
                continue
            
            # Test each threshold candidate and compute F1-score
            f1_scores = []
            for threshold in self.threshold_candidates:
                pred_binary = (y_pred_class > threshold).astype(int)
                f1 = f1_score(y_true_class, pred_binary, zero_division=0)
                f1_scores.append(f1)
            
            # Select threshold that maximizes F1-score
            best_idx = np.argmax(f1_scores)
            best_threshold = self.threshold_candidates[best_idx]
            best_thresholds.append(float(best_threshold))
        
        self.best_thresholds = best_thresholds
        
        # Compute overall macro F1 with optimized thresholds
        macro_f1 = self._compute_macro_f1(predictions, true_labels, best_thresholds)
        
        print(f"Optimized thresholds (Macro F1: {macro_f1:.4f}): {dict(zip(self.mlb_classes, best_thresholds))}")
        
        # Save thresholds and performance for later use
        threshold_dict = dict(zip(self.mlb_classes, best_thresholds))
        threshold_dict['macro_f1'] = float(macro_f1)
        
        with open(os.path.join(self.model.log_dir, 'thresholds.json'), 'w') as f:
            json.dump(threshold_dict, f, indent=2)
    
    def _compute_macro_f1(self, predictions, true_labels, thresholds):
        """Compute macro F1-score with given thresholds.
        
        Macro F1: Average of per-class F1 scores (treats all classes equally).
        This is different from micro F1 which would pool all predictions together.
        """
        f1_scores = []
        for class_idx, threshold in enumerate(thresholds):
            y_true = true_labels[:, class_idx]
            y_pred = (predictions[:, class_idx] > threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        return np.mean(f1_scores)  # Equal weight to each class
    
def load_thresholds(run_dir, mlb_classes):
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
    run_dir (str or Path): 
        Training run directory containing threshold optimization results
    mlb_classes (list): 
        List of class names in the same order as model outputs
    
    Returns:
    -------
    list: 
        Per-class thresholds in the same order as mlb_classes
        
    File Format:
    -----------
    thresholds.json contains:
    {
        "class_name_1": 0.3,
        "class_name_2": 0.7,
        ...
    }
    """
    run_dir = Path(run_dir)
    thresholds_file = run_dir / 'thresholds.json'
    
    if thresholds_file.exists():
        try:
            with open(thresholds_file, 'r') as f:
                thresholds_dict = json.load(f)
            
            # Map class names to thresholds, using 0.5 as fallback for missing classes
            thresholds = [thresholds_dict.get(class_name, 0.5) for class_name in mlb_classes]
            
            print(f"📊 Class thresholds: {dict(zip(mlb_classes, thresholds))}")
            
            # Validate threshold ranges
            if any(t < 0.1 or t > 0.9 for t in thresholds):
                print("⚠️ Warning: Some thresholds are outside typical range [0.1, 0.9]")
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Warning: Error reading thresholds file: {e}")
            print("🔄 Falling back to default thresholds (0.5)")
            thresholds = [0.5] * len(mlb_classes)
    else:
        thresholds = [0.5] * len(mlb_classes)
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
            return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)  # +13 for MFCC
        
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
        return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)