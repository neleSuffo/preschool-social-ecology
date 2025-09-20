import json
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import AudioConfig
from audio_classifier import build_model_multi_label
from sklearn.preprocessing import MultiLabelBinarizer

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
    run_dir = Path(run_dir)
    thresholds_file = run_dir / 'thresholds.json'
    
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

def load_model_and_setup(model_path):
    """
    Load trained model by building the architecture and loading weights.
    
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
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Setup multi-label binarizer consistent with training
    mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
    mlb.fit([[]])  # Initialize with empty list to set up classes
    num_classes = len(mlb.classes_)

    # Use centralized calculation for consistent time steps across all components
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    try:
        # Build the model architecture first, with the corrected Lambda layer
        model = build_model_multi_label(
            n_mels=AudioConfig.N_MELS,
            fixed_time_steps=fixed_time_steps,
            num_classes=num_classes
        )
        
        # Load only the weights from the saved .keras file
        model.load_weights(model_path)
    
    except Exception as e:
        raise ValueError(f"Failed to load model: {e}")
    
    return model, mlb

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
   
# --- Data Generator for Evaluation ---
class EvaluationDataGenerator(tf.keras.utils.Sequence):
    """
    Deterministic data generator for model evaluation with consistent batch loading.
    
    This generator provides reproducible evaluation by maintaining deterministic
    ordering and eliminating data augmentation. It ensures consistent evaluation
    metrics across multiple runs and fair comparison with training performance.    
    """
    def __init__(self, segments_file_path, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, batch_size=32):
        """
        Initialize evaluation data generator with configuration parameters.
        
        Parameters:
        ----------
        segments_file_path (str or Path): 
            Path to JSONL file containing evaluation segments
        mlb (MultiLabelBinarizer): 
            Fitted multi-label binarizer from training
        n_mels (int): 
            Number of mel frequency bands for spectrograms
        hop_length (int): 
            Hop length for STFT computation
        sr (int): 
            Audio sample rate for loading and processing
        window_duration (float): 
            Fixed window duration for segment extraction
        fixed_time_steps (int): 
            Standardized time steps for model input consistency
        batch_size (int, default=32): 
            Number of samples per batch for inference
        """
        self.segments_file_path = segments_file_path
        self.mlb = mlb
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.window_duration = window_duration
        self.fixed_time_steps = fixed_time_steps
        self.batch_size = batch_size
        # Load all segment metadata for deterministic access patterns
        self.segments_data = self._load_segments_metadata()

    def __len__(self):
        """Calculate total number of batches for complete evaluation."""
        return (len(self.segments_data) + self.batch_size - 1) // self.batch_size

    def _load_segments_metadata(self):
        """
        Load all segment metadata from JSONL file for deterministic ordering.
        
        This method preloads segment paths and labels while keeping audio data
        lazy-loaded to manage memory usage during evaluation.
        
        Returns:
        -------
        list: Segment metadata dictionaries with paths and labels
        """
        segments = []
        with open(self.segments_file_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line.strip()))
        return segments

    def __getitem__(self, index):
        """
        Generate batch of features and labels for evaluation.
        
        This method handles batch boundary conditions gracefully and ensures
        consistent feature dimensions across all samples in the batch.
        
        Parameters:
        ----------
        index (int): Batch index for sequential access
            
        Returns:
        -------
        tuple: (X_batch, y_batch) where:
            - X_batch: Feature array (batch_size, effective_n_mels+13, time_steps, 1)
            - y_batch: Multi-hot label array (batch_size, n_classes)
        """
        # Calculate batch boundaries with overflow protection
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.segments_data))
        batch_segments = self.segments_data[start_idx:end_idx]
        
        # Initialize batch collections
        X_batch = []
        y_batch = []
        
        # Process each segment in the batch
        for segment in batch_segments:
            # Extract features using consistent pipeline
            mel = extract_enhanced_features(
                segment['audio_path'], segment['start'], segment['duration'],
                sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
                fixed_time_steps=self.fixed_time_steps
            )
            
            # Validate and correct feature dimensions if necessary
            if mel.ndim != 2:
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    # Remove singleton dimension
                    mel = mel.squeeze(axis=-1)
                else:
                    # Skip malformed features
                    continue
                    
            X_batch.append(mel)
            
            # Convert string labels to multi-hot encoding
            multi_hot_labels = self.mlb.transform([segment['labels']])[0]
            y_batch.append(multi_hot_labels)
        
        # Handle empty batch case gracefully
        if not X_batch:
            return np.array([]), np.array([])
            
        # Convert to numpy arrays with correct dimensions for model input
        X_batch_np = np.array(X_batch)
        X_batch_final = np.expand_dims(X_batch_np, -1)  # Add channel dimension
        return X_batch_final, np.array(y_batch)

def create_empty_feature_matrix(n_mels, fixed_time_steps):
    """
    Create standardized empty feature matrix for error cases.
    
    This helper ensures consistent fallback behavior when audio processing
    fails, maintaining expected input dimensions for the neural network.
    Uses effective n_mels calculation to match model architecture.
    
    Parameters:
    ----------
    n_mels (int): Number of mel frequency bands (will be capped at 128)
    fixed_time_steps (int): Number of time steps
        
    Returns:
    -------
    np.ndarray: Zero-filled feature matrix with correct dimensions
    """
    # Use effective mel count to match model architecture 
    effective_n_mels = min(n_mels, 128)
    return np.zeros((effective_n_mels + 13, fixed_time_steps), dtype=np.float32)

# --- Feature Extraction ---
def extract_enhanced_features(audio_path, start_time, duration, sr, n_mels, hop_length, fixed_time_steps=None):
    """
    Extract enhanced mel-spectrogram and MFCC features from audio segments.
    
    This function maintains exact consistency with the training pipeline to ensure
    evaluation accuracy. It combines mel-spectrogram and MFCC features with 
    identical preprocessing steps including normalization, pre-emphasis filtering,
    and dimensional standardization.
    
    Feature Processing Pipeline:
    1. Audio loading with precise timing and resampling validation
    2. Duration padding/truncation to exact expected sample count
    3. Amplitude normalization to prevent clipping artifacts
    4. Pre-emphasis filtering to balance spectral energy (α=0.97)
    5. Mel-spectrogram extraction with perceptually relevant frequency range
    6. Power-to-dB conversion with controlled dynamic range (80dB)
    7. Feature normalization to [-1, 1] range for stable training
    8. MFCC computation for complementary cepstral features
    9. Feature concatenation and temporal dimension standardization
    
    Audio Configuration:
    - Sample Rate: Configured in AudioConfig.SR
    - Frequency Range: 20 Hz - 10 kHz (human speech range)
    - Window: 2048-point FFT with hop_length overlap
    - Mel Bands: Perceptually spaced frequency bins (capped at 128 for 16kHz)
    - MFCC: 13 coefficients capturing spectral envelope
    
    Parameters:
    ----------
    audio_path (str): 
        Full path to audio file for segment extraction
    start_time (float): 
        Segment start time in seconds from audio beginning
    duration (float): 
        Segment duration in seconds (fixed window size)
    sr (int): 
        Target sample rate for audio loading and processing
    n_mels (int): 
        Number of mel frequency bands for spectrogram (will be capped at 128)
    hop_length (int): 
        Hop length in samples for STFT computation
    fixed_time_steps (int, optional): 
        Fixed number of time steps for consistent model input dimensions

    Returns:
    -------
    np.ndarray: Combined feature matrix (effective_n_mels + 13, fixed_time_steps)
        Shape represents concatenated mel-spectrogram and MFCC features
        normalized to [-1, 1] range with standardized temporal dimension
    """
    # Use effective mel count (capped at 128 for 16kHz audio) to match model architecture
    effective_n_mels = min(n_mels, 128)
    
    # Use centralized time steps calculation if not provided
    if fixed_time_steps is None:
        fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    try:
        # Load audio segment with precise timing control
        y, sr_loaded = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        
        # Validate and correct sample rate if necessary
        if sr_loaded != sr:
            y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)
        
        # Ensure exact sample count for consistent processing
        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            # Zero-pad short segments to expected length
            y = np.pad(y, (0, expected_samples - len(y)), 'constant')
        elif len(y) > expected_samples:
            # Truncate long segments to expected length
            y = y[:expected_samples]
        
        # Handle empty audio gracefully with appropriate fallback
        if len(y) == 0:
            return create_empty_feature_matrix(n_mels, fixed_time_steps)
        
        # Amplitude normalization to prevent numerical instability
        y = y / (np.max(np.abs(y)) + 1e-6)
        
        # Apply pre-emphasis filter to balance spectral energy
        # Formula: y[n] = x[n] - α*x[n-1] with α=0.97
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # Extract mel-spectrogram with perceptually relevant frequency range (using effective n_mels)
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=effective_n_mels, hop_length=hop_length, 
            n_fft=2048,    # 2048-point FFT for good frequency resolution
            fmin=20,       # Lower frequency bound (human hearing)
            fmax=10000     # Upper frequency bound (speech content)
        )
        
        # Convert power spectrogram to decibel scale with controlled dynamic range
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        
        # Normalize mel-spectrogram to [-1, 1] range for stable neural network input
        mel_spectrogram_db = 2 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-6) - 1
        
        # Extract MFCC features for complementary cepstral representation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Normalize MFCC features to [-1, 1] range matching mel-spectrogram
        mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6) - 1
        
        # Concatenate mel-spectrogram and MFCC for enhanced feature representation
        combined = np.concatenate([mel_spectrogram_db, mfcc], axis=0)
        
        # Standardize temporal dimension for consistent model input
        if combined.shape[1] < fixed_time_steps:
            # Pad short sequences with silence marker (-1)
            pad_width = fixed_time_steps - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
        elif combined.shape[1] > fixed_time_steps:
            # Truncate long sequences to fixed length
            combined = combined[:, :fixed_time_steps]
        
        return combined
    
    except Exception as e:
        # Graceful error handling with informative logging
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return create_empty_feature_matrix(n_mels, fixed_time_steps)

# ---- Evaluation Data Generator ----
def create_evaluation_generator(test_segments_file, mlb):
    """
    Create deterministic data generator for comprehensive model evaluation.
    
    This function creates a specialized data generator for evaluation that maintains
    consistency with the training pipeline while optimizing for inference performance.
    The generator eliminates randomness and augmentation to ensure reproducible 
    evaluation results across multiple runs.
    
    Generator Configuration:
    - Deterministic ordering (no shuffling) for reproducible results
    - No data augmentation to preserve original audio characteristics  
    - Consistent feature extraction pipeline matching training
    - Fixed batch size for optimal inference throughput
    - Lazy loading strategy for memory efficiency
    
    Parameters:
    ----------
    test_segments_file (Path): 
        Path to JSONL file containing test segment metadata
    mlb (MultiLabelBinarizer): 
        Fitted multi-label binarizer from training run
    
    Returns:
    -------
    EvaluationDataGenerator: 
        Configured test data generator ready for model evaluation
        
    Features:
    --------
    - Batch size: 32 samples (optimal for most GPUs)
    - Memory efficient: Lazy loading with small memory footprint
    - Consistent preprocessing: Identical to training feature extraction
    - Reproducible: Deterministic sample ordering for consistent metrics
    """
    # Use centralized time steps calculation to ensure consistency with training
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
    
    test_generator = EvaluationDataGenerator(
        test_segments_file, mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32
    )
    return test_generator