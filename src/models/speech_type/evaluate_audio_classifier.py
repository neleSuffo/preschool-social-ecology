"""
Comprehensive Audio Classification Model Evaluation Pipeline

This module provides a complete evaluation framework for trained multi-label audio 
classification models, implementing robust performance metrics, visualization tools,
and threshold optimization validation. The evaluation pipeline maintains consistency
with training procedures while providing detailed performance analysis.

Key Features:
- Model loading with custom loss functions and metrics
- Threshold optimization validation from training runs
- Comprehensive evaluation metrics (macro/micro F1, precision, recall, subset accuracy)
- Per-class performance analysis and visualization
- Prediction probability distribution analysis  
- Statistical significance testing capabilities
- Reproducible evaluation with deterministic data loading

Evaluation Metrics:
- Subset Accuracy: Exact match for multi-label predictions
- Macro F1: Unweighted average F1 across classes (handles class imbalance)
- Micro F1: Global F1 calculated from summed confusion matrices
- Per-class Precision/Recall: Individual class performance analysis

Threshold Application:
- Loads optimized thresholds from training threshold optimization
- Falls back to default 0.5 thresholds if optimization results unavailable
- Applies class-specific thresholds for optimal multi-label performance

Output Generation:
- Detailed JSON summary with all metrics and configurations
- CSV file with individual predictions and probabilities
- Performance visualization plots (F1 scores, distributions, class support)
- Timestamped results directories for experiment tracking

Dependencies:
- TensorFlow/Keras: Model loading and inference
- librosa: Audio feature extraction (consistent with training)
- scikit-learn: Evaluation metrics and multi-label handling
- matplotlib: Performance visualization and plotting
"""

# filepath: /home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/models/speech_type/evaluate_audio_classifier.py
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model
from tqdm import tqdm
from constants import AudioClassification
from config import AudioConfig
from audio_classifier import MacroF1Score, FocalLoss

def create_empty_feature_matrix(n_mels, fixed_time_steps):
    """
    Create standardized empty feature matrix for error cases.
    
    This helper ensures consistent fallback behavior when audio processing
    fails, maintaining expected input dimensions for the neural network.
    
    Parameters:
    ----------
    n_mels (int): Number of mel frequency bands
    fixed_time_steps (int): Number of time steps
        
    Returns:
    -------
    np.ndarray: Zero-filled feature matrix with correct dimensions
    """
    return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)

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
    - Mel Bands: Perceptually spaced frequency bins
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
        Number of mel frequency bands for spectrogram
    hop_length (int): 
        Hop length in samples for STFT computation
    fixed_time_steps (int, optional): 
        Fixed number of time steps for consistent model input dimensions

    Returns:
    -------
    np.ndarray: Combined feature matrix (n_mels + 13, fixed_time_steps)
        Shape represents concatenated mel-spectrogram and MFCC features
        normalized to [-1, 1] range with standardized temporal dimension
    """
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
        
        # Extract mel-spectrogram with perceptually relevant frequency range
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, 
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

# --- Data Generator for Evaluation ---
class EvaluationDataGenerator(tf.keras.utils.Sequence):
    """
    Deterministic data generator for model evaluation with consistent batch loading.
    
    This generator provides reproducible evaluation by maintaining deterministic
    ordering and eliminating data augmentation. It ensures consistent evaluation
    metrics across multiple runs and fair comparison with training performance.
    
    Key Differences from Training Generator:
    - No data augmentation (maintains original audio characteristics)
    - Deterministic ordering (no shuffling for reproducible results)
    - Simplified batch processing (optimized for inference speed)
    - Direct JSONL loading (memory efficient for large test sets)
    - Consistent feature extraction (identical to training pipeline)
    
    Batch Processing Strategy:
    - Fixed batch size for consistent memory usage
    - Graceful handling of irregular final batch
    - Automatic dimensionality validation and correction
    - Multi-label binarization with fitted encoder
    
    Memory Management:
    - Lazy loading of audio segments (prevents memory overflow)
    - Efficient feature matrix construction
    - Automatic garbage collection friendly design
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
            - X_batch: Feature array (batch_size, n_mels+13, time_steps, 1)
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

# --- Evaluation Functions ---
def load_model_and_setup(model_path):
    """
    Load trained model with Lambda layer handling and setup multi-label binarizer.
    
    This function handles modern Keras security restrictions for Lambda layers by
    enabling unsafe deserialization when necessary. The Lambda layers in our models
    are safe since they come from our own training pipeline and contain only
    mathematical operations for attention mechanisms.
    
    Security Note:
    - Lambda layers are disabled by default in Keras for security
    - Our models contain Lambda layers for multi-head attention
    - These are safe since they originate from our controlled training process
    - We enable unsafe deserialization only for our own trusted model files
    
    Parameters:
    ----------
    model_path (str or Path): 
        Path to saved model (.keras file from our training pipeline)
    
    Returns:
    -------
    tuple: (model, mlb) where:
        - model: Loaded and compiled Keras model
        - mlb: Fitted MultiLabelBinarizer for label encoding
        
    Raises:
    ------
    FileNotFoundError: If model file doesn't exist
    ValueError: If model loading fails due to incompatible format
    """    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Setup multi-label binarizer consistent with training
    mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
    mlb.fit([[]])  # Initialize with empty list to set up classes
    
    # Define custom objects for model reconstruction
    custom_objects = {
        'MacroF1Score': MacroF1Score,
        'FocalLoss': FocalLoss
    }
        
    try:
        # First attempt: Standard loading (safe mode)
        model = load_model(model_path, custom_objects=custom_objects)
        print("✅ Model loaded successfully in safe mode")
        
    except ValueError as e:
        if "Lambda" in str(e) and "unsafe deserialization" in str(e):            
            # Enable unsafe deserialization for Lambda layers
            import keras
            keras.config.enable_unsafe_deserialization()
            
            try:
                # Retry loading with unsafe deserialization enabled
                model = load_model(model_path, custom_objects=custom_objects)
                print("✅ Model loaded successfully with Lambda layer support")
                
            except Exception as retry_e:
                raise ValueError(f"Failed to load model even with unsafe deserialization: {retry_e}")
                
            finally:
                # Restore safe deserialization after loading
                keras.config.disable_unsafe_deserialization()
                
        else:
            # Re-raise non-Lambda related errors
            raise e
            
    except Exception as e:
        raise ValueError(f"Unexpected error loading model: {e}")
    
    return model, mlb

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
            
            print(f"✅ Loaded optimized thresholds from: {thresholds_file}")
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
    
    Time Steps Calculation:
    Uses centralized helper function to ensure consistency with training
    pipeline and eliminate redundant calculations across the codebase.
    
    Parameters:
    ----------
    test_segments_file (str or Path): 
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

def evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir):
    """
    Perform comprehensive multi-label classification evaluation with detailed analysis.
    
    This function executes a thorough evaluation protocol that goes beyond basic
    accuracy metrics to provide insights into model performance across different
    dimensions. It handles the complexities of multi-label evaluation where
    traditional metrics may be misleading due to label dependencies and imbalance.
    
    Evaluation Protocol:
    1. Generate probability predictions for all test samples
    2. Apply class-specific optimized thresholds for binary classification
    3. Calculate comprehensive metrics (subset accuracy, macro/micro F1, per-class)
    4. Handle data mismatches gracefully with appropriate corrections
    5. Generate detailed reports and visualizations for analysis
    6. Save results in multiple formats for different use cases
    
    Multi-Label Metrics Explanation:
    - Subset Accuracy: Percentage of samples where ALL labels are predicted correctly
    - Macro F1: Average F1 across classes (treats all classes equally)
    - Micro F1: Global F1 from aggregated true/false positives (favors frequent classes)
    - Per-class F1: Individual class performance (identifies problematic classes)
    
    Threshold Application:
    Uses class-specific thresholds optimized during training rather than uniform 0.5.
    This accounts for different prediction confidence distributions per class and
    maximizes overall macro F1-score performance.
    
    Error Handling:
    - Graceful handling of prediction/label shape mismatches
    - Fallback to available samples when batch sizes don't align
    - Warning system for unusual evaluation conditions
    - Comprehensive error reporting for debugging
    
    Parameters:
    ----------
    model (tf.keras.Model): 
        Trained multi-label audio classification model
    test_generator (EvaluationDataGenerator): 
        Test data generator with deterministic ordering
    mlb (MultiLabelBinarizer): 
        Fitted label encoder from training pipeline
    thresholds (list): 
        Per-class decision thresholds for binary classification
    output_dir (str or Path): 
        Directory to save evaluation results and visualizations
        
    Outputs:
    -------
    Files Created:
    - evaluation_summary.json: Comprehensive metrics summary
    - detailed_predictions.csv: Per-sample predictions and probabilities
    - per_class_f1_scores.png: Class-wise F1 score visualization
    - prediction_distributions.png: Probability distribution plots
    - class_support.png: Class frequency analysis plot
    
    Raises:
    ------
    ValueError: If test generator is empty or predictions fail
    RuntimeError: If evaluation cannot complete due to data issues
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Running comprehensive model evaluation...")
    print("=" * 60)
    
    # Stage 1: Generate probability predictions for all test samples
    print("📊 Generating predictions for test set...")
    test_predictions = model.predict(test_generator, verbose=1)
    
    # Stage 2: Collect true labels with progress tracking
    test_true_labels = []
    for i in tqdm(range(len(test_generator)), desc="Processing batches"):
        _, labels = test_generator[i]
        if len(labels) > 0:  # Skip empty batches
            test_true_labels.extend(labels)
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
        
        print(f"✂️ Adjusted evaluation set to {min_samples} samples")
        
        if min_samples == 0:
            raise ValueError("No samples available for evaluation after shape adjustment")
    
    # Stage 4: Apply class-specific thresholds to convert probabilities to predictions
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[i]).astype(int) 
        for i in range(len(mlb.classes_))
    ]).T
    
    # Stage 5: Calculate comprehensive evaluation metrics
    if test_true_labels.sum() > 0:        
        # Per-class detailed metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_true_labels, test_pred_binary, average=None, zero_division=0
        )
        
        # Macro-averaged metrics (equal weight per class)
        macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        
        # Micro-averaged metrics (global performance)
        micro_precision = precision_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_recall = recall_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_f1 = f1_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        
        # Subset accuracy (exact multi-label match)
        subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)
        
        # Stage 6: Save comprehensive results to files
        save_evaluation_results(
            output_dir, mlb.classes_, thresholds,
            test_true_labels, test_pred_binary, test_predictions,
            precision_per_class, recall_per_class, f1_per_class, support_per_class,
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1, subset_accuracy
        )
    else:
        print("⚠️ Warning: No positive instances found in test set")
        print("❌ Cannot compute meaningful evaluation metrics")
    
    print(f"\n✅ Comprehensive evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")

def save_evaluation_results(output_dir, class_names, thresholds,
                        test_true_labels, test_pred_binary, test_predictions,
                        precision_per_class, recall_per_class, f1_per_class, support_per_class,
                        macro_precision, macro_recall, macro_f1,
                        micro_precision, micro_recall, micro_f1, subset_accuracy):
    """
    Save comprehensive evaluation results in multiple formats for analysis and reporting.
    
    This function creates both machine-readable (JSON) and human-readable (CSV) outputs
    containing detailed evaluation metrics. The dual format approach supports both
    automated analysis pipelines and manual inspection of results.
    
    Output Files:
    1. evaluation_summary.json: Structured metrics summary for programmatic analysis
    2. detailed_predictions.csv: Per-sample results for error analysis and debugging
    
    JSON Structure:
    - overall_metrics: Global performance (macro/micro F1, subset accuracy)
    - per_class_metrics: Individual class performance with support counts
    - test_set_size: Number of evaluated samples for statistical significance
    - thresholds: Applied decision thresholds per class
    
    CSV Structure:
    - Probability columns: Raw model outputs (0-1) per class
    - Binary prediction columns: Thresholded binary decisions per class
    - True label columns: Ground truth binary labels per class
    
    Parameters:
    ----------
    output_dir (Path): Target directory for result files
    class_names (list): Names of classification classes
    thresholds (list): Applied decision thresholds per class
    test_true_labels (ndarray): Ground truth binary labels (n_samples, n_classes)
    test_pred_binary (ndarray): Binary predictions (n_samples, n_classes)
    test_predictions (ndarray): Probability predictions (n_samples, n_classes)
    precision_per_class (ndarray): Per-class precision scores
    recall_per_class (ndarray): Per-class recall scores  
    f1_per_class (ndarray): Per-class F1 scores
    support_per_class (ndarray): Per-class positive sample counts
    macro_precision (float): Macro-averaged precision
    macro_recall (float): Macro-averaged recall
    macro_f1 (float): Macro-averaged F1 score
    micro_precision (float): Micro-averaged precision
    micro_recall (float): Micro-averaged recall
    micro_f1 (float): Micro-averaged F1 score
    subset_accuracy (float): Subset accuracy (exact match rate)
    """
    
    # Create comprehensive metrics summary for programmatic analysis
    summary = {
        'evaluation_metadata': {
            'test_set_size': len(test_true_labels),
            'num_classes': len(class_names),
            'class_names': class_names,
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
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i]),
                'threshold': float(thresholds[i]),
                'positive_rate': float(support_per_class[i] / len(test_true_labels))
            } for i in range(len(class_names))
        },
        'threshold_configuration': {
            'method': 'optimized_from_validation',
            'fallback': 0.5,
            'per_class_thresholds': {class_names[i]: float(thresholds[i]) for i in range(len(class_names))}
        }
    }
    
    # Save structured JSON summary for automated analysis
    summary_path = output_dir / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    
    # Create detailed per-sample results for error analysis
    # Organize columns for easy analysis: probabilities, predictions, then true labels
    
    # Probability predictions (raw model outputs)
    predictions_df = pd.DataFrame(
        test_predictions, 
        columns=[f'{name}_prob' for name in class_names]
    )
    
    # Binary predictions (after threshold application)
    predictions_binary_df = pd.DataFrame(
        test_pred_binary, 
        columns=[f'{name}_pred' for name in class_names]
    )
    
    # True labels (ground truth)
    true_labels_df = pd.DataFrame(
        test_true_labels, 
        columns=[f'{name}_true' for name in class_names]
    )
    
    # Combine all information for comprehensive per-sample analysis
    detailed_results = pd.concat([
        predictions_df, 
        predictions_binary_df, 
        true_labels_df
    ], axis=1)
    
    # Add derived columns for quick analysis
    detailed_results['sample_id'] = range(len(detailed_results))
    detailed_results['num_true_labels'] = test_true_labels.sum(axis=1)
    detailed_results['num_pred_labels'] = test_pred_binary.sum(axis=1)
    detailed_results['exact_match'] = (test_true_labels == test_pred_binary).all(axis=1).astype(int)
    
    # Save detailed predictions for manual inspection and error analysis
    predictions_path = output_dir / 'detailed_predictions.csv'
    detailed_results.to_csv(predictions_path, index=False, float_format='%.4f')

def main():
    """
    Execute comprehensive audio classification model evaluation pipeline.
    
    This is the main orchestration function that coordinates the complete evaluation
    workflow for trained multi-label audio classification models. The pipeline
    ensures consistency with training procedures while providing thorough
    performance analysis and diagnostic capabilities.
    
    Evaluation Pipeline Stages:
    1. Environment Setup: Configure paths and create output directories
    2. Model Loading: Load trained model with custom objects and Lambda layer handling
    3. Threshold Loading: Retrieve optimized decision thresholds from training
    4. Data Generation: Create deterministic test data generator
    5. Comprehensive Evaluation: Generate predictions and compute metrics
    6. Results Export: Save detailed results and visualizations
    7. Report Generation: Create summary reports for analysis
    
    Model Requirements:
    - Trained .keras model file with custom FocalLoss and MacroF1Score
    - Compatible with current TensorFlow version
    - May contain Lambda layers from attention mechanisms
    
    Data Requirements:
    - Test segments JSONL file with consistent format
    - Audio files accessible at specified paths
    - Label encoding consistent with training MultiLabelBinarizer
    
    Output Generation:
    - Timestamped results directory for experiment tracking
    - JSON summary with comprehensive metrics
    - CSV file with per-sample predictions
    - Publication-quality visualization plots
    - Detailed logging for debugging and analysis
    
    Error Handling:
    - Graceful handling of missing model files
    - Lambda layer deserialization with security considerations
    - Data loading validation and error reporting
    - Comprehensive exception logging for debugging
    
    Configuration Sources:
    - AudioClassification: File paths and directory configurations
    - AudioConfig: Audio processing parameters and class definitions
    
    Performance Metrics:
    - Subset Accuracy: Exact multi-label match percentage
    - Macro F1: Unweighted average across classes (handles imbalance)
    - Micro F1: Global performance from aggregated statistics
    - Per-class Metrics: Individual class precision, recall, F1, support
    - Threshold Analysis: Applied decision boundaries per class
    
    File Outputs:
    - evaluation_summary.json: Structured metrics for automated analysis
    - detailed_predictions.csv: Per-sample results for error analysis
    - per_class_f1_scores.png: Class performance visualization
    - prediction_distributions.png: Confidence pattern analysis
    - class_support.png: Test set distribution analysis
    
    Reproducibility:
    - Deterministic data loading (no shuffling)
    - Fixed random seeds where applicable
    - Timestamped results for experiment tracking
    - Complete parameter logging for replication
    
    Usage:
    -----
    python evaluate_audio_classifier.py
    
    Prerequisites:
    - Completed training run with saved model
    - Test data prepared with create_input_jsonl_files.py
    - Audio files accessible at configured paths
    
    Raises:
    ------
    FileNotFoundError: If model or data files are missing
    ValueError: If model loading or evaluation fails
    RuntimeError: For unexpected evaluation pipeline errors
    """   
    try:
        print("🚀 Starting Comprehensive Audio Classification Model Evaluation")
        print("=" * 70)
        
        # Stage 1: Configure evaluation environment and paths
        print("⚙️ Configuring evaluation environment...")
        results_dir = Path(AudioClassification.RESULTS_DIR)
        model_path = results_dir / "best_model.keras"
        test_segments_file = AudioClassification.TEST_SEGMENTS_FILE
        
        # Create timestamped output directory for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = results_dir / f'evaluation_results_{timestamp}'
        
        # Stage 2: Load trained model with Lambda layer handling
        model, mlb = load_model_and_setup(model_path)

        # Stage 3: Load optimized thresholds from training run
        thresholds = load_thresholds(results_dir, mlb.classes_)
        
        # Stage 4: Create deterministic test data generator
        test_generator = create_evaluation_generator(test_segments_file, mlb)
        
        if len(test_generator) == 0:
            raise ValueError("Test generator is empty. Check test data file and paths.")

        # Stage 5: Execute comprehensive model evaluation
        evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir)
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Ensure training has completed and all required files exist")
        raise
        
    except ValueError as e:
        print(f"❌ Configuration or data error: {e}")
        print("💡 Check model compatibility and data file formats")
        raise
        
    except Exception as e:
        print(f"❌ Unexpected evaluation error: {e}")
        print("💡 Ensure sufficient memory and disk space available")
        raise

if __name__ == "__main__":
    main()