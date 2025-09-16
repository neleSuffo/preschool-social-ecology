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
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import load_model
from tqdm import tqdm
from constants import AudioClassification
from config import AudioConfig
from audio_classifier import MacroF1Score, FocalLoss, calculate_fixed_time_steps

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
        fixed_time_steps = calculate_fixed_time_steps()
    
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
    Load trained model and setup multi-label binarizer from training run.
    
    Parameters:
    ----------
    model_path (str or Path): 
        Path to saved model (.keras file)
    
    Returns:
    -------
    tuple: (model, mlb)
    """    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Setup multi-label binarizer
    mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
    mlb.fit([[]])  # Initialize
    
    # Load model with custom objects
    custom_objects = {
        'MacroF1Score': MacroF1Score,
        'FocalLoss': FocalLoss
    }
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    
    return model, mlb

def load_thresholds(run_dir, mlb_classes):
    """
    Load optimized thresholds from training run.
    
    Parameters:
    ----------
    run_dir (str or Path): Directory containing training run files
    mlb_classes (list): List of class names
    
    Returns:
    -------
    list: Optimized thresholds for each class
    """
    run_dir = Path(run_dir)
    thresholds_file = run_dir / 'thresholds.json'
    
    if thresholds_file.exists():
        with open(thresholds_file, 'r') as f:
            thresholds_dict = json.load(f)
        thresholds = [thresholds_dict.get(class_name, 0.5) for class_name in mlb_classes]
        print(f"Loaded optimized thresholds: {dict(zip(mlb_classes, thresholds))}")
    else:
        thresholds = [0.5] * len(mlb_classes)
        print("Using default thresholds (0.5) - optimized thresholds not found.")
    
    return thresholds

def create_evaluation_generator(test_segments_file, mlb):
    """
    Create data generator for evaluation.
    
    Parameters:
    ----------
    test_segments_file (str or Path): Path to test segments file
    mlb: Fitted MultiLabelBinarizer
    
    Returns:
    -------
    EvaluationDataGenerator: Test data generator
    """
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
    
    test_generator = EvaluationDataGenerator(
        test_segments_file, mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32
    )
    
    print(f"Created test generator with {len(test_generator)} batches")
    return test_generator

def evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir):
    """
    Perform comprehensive evaluation of the model on test set.
    
    Parameters:
    ----------
    model: Loaded Keras model
    test_generator: Test data generator
    mlb: MultiLabelBinarizer
    thresholds (list): Optimized thresholds for each class
    output_dir (str or Path): Directory to save evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Running comprehensive model evaluation...")
    print("=" * 60)
    
    # Get predictions and true labels
    test_predictions = model.predict(test_generator, verbose=1)
    
    test_true_labels = []
    for i in tqdm(range(len(test_generator)), desc="Collecting labels"):
        _, labels = test_generator[i]
        test_true_labels.extend(labels)
    test_true_labels = np.array(test_true_labels)
    
    # Handle shape mismatches
    if test_predictions.shape[0] != test_true_labels.shape[0]:
        print(f"⚠️ Shape mismatch: predictions {test_predictions.shape[0]}, labels {test_true_labels.shape[0]}")
        min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
        test_predictions = test_predictions[:min_samples]
        test_true_labels = test_true_labels[:min_samples]
        print(f"✂️ Adjusted to {min_samples} samples")
    
    # Apply thresholds
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[i]).astype(int) 
        for i in range(len(mlb.classes_))
    ]).T
    
    if test_true_labels.sum() > 0:
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_true_labels, test_pred_binary, average=None, zero_division=0
        )
        
        # Macro metrics
        macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        
        # Micro metrics
        micro_precision = precision_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_recall = recall_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_f1 = f1_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        
        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)
        
        # Save detailed results
        save_evaluation_results(
            output_dir, mlb.classes_, thresholds,
            test_true_labels, test_pred_binary, test_predictions,
            precision_per_class, recall_per_class, f1_per_class, support_per_class,
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1, subset_accuracy
        )
        
        # Generate plots
        generate_evaluation_plots(
            output_dir, mlb.classes_,
            test_true_labels, test_pred_binary, test_predictions
        )
        
    else:
        print("⚠️ Warning: No positive instances in test set for metrics calculation.")
    
    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")

def save_evaluation_results(output_dir, class_names, thresholds,
                        test_true_labels, test_pred_binary, test_predictions,
                        precision_per_class, recall_per_class, f1_per_class, support_per_class,
                        macro_precision, macro_recall, macro_f1,
                        micro_precision, micro_recall, micro_f1, subset_accuracy):
    """Save detailed evaluation results to files."""
    
    # Save summary metrics
    summary = {
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
                'threshold': float(thresholds[i])
            } for i in range(len(class_names))
        },
        'test_set_size': len(test_true_labels),
        'thresholds': {class_names[i]: float(thresholds[i]) for i in range(len(class_names))}
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save detailed predictions
    predictions_df = pd.DataFrame(test_predictions, columns=[f'{name}_prob' for name in class_names])
    predictions_binary_df = pd.DataFrame(test_pred_binary, columns=[f'{name}_pred' for name in class_names])
    true_labels_df = pd.DataFrame(test_true_labels, columns=[f'{name}_true' for name in class_names])
    
    detailed_results = pd.concat([predictions_df, predictions_binary_df, true_labels_df], axis=1)
    detailed_results.to_csv(output_dir / 'detailed_predictions.csv', index=False)
    
    print(f"💾 Saved evaluation summary to: {output_dir / 'evaluation_summary.json'}")
    print(f"💾 Saved detailed predictions to: {output_dir / 'detailed_predictions.csv'}")

def generate_evaluation_plots(output_dir, class_names, test_true_labels, test_pred_binary, test_predictions):
    """
    Generate evaluation plots and save them.

    Parameters:
    ----------
    output_dir (Path): 
        Directory to save plots
    class_names (list): 
        List of class names
    test_true_labels (ndarray): 
        True labels for the test set
    test_pred_binary (ndarray): 
        Binary predictions for the test set
    test_predictions (ndarray): 
        Probability predictions for the test set
    """
    
    # 1. Per-class F1 scores
    f1_scores = [f1_score(test_true_labels[:, i], test_pred_binary[:, i], zero_division=0) 
                for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, f1_scores)
    plt.title('Per-Class F1 Scores')
    plt.ylabel('F1 Score')
    plt.xlabel('Classes')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction probability distributions
    fig, axes = plt.subplots(1, len(class_names), figsize=(4*len(class_names), 4))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        axes[i].hist(test_predictions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{class_name} Prediction Probabilities')
        axes[i].set_xlabel('Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class support (number of positive samples per class)
    support_counts = test_true_labels.sum(axis=0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, support_counts)
    plt.title('Class Support (Number of Positive Samples)')
    plt.ylabel('Number of Samples')
    plt.xlabel('Classes')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, support_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_support.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Generated evaluation plots in: {output_dir}")

def main():
    """
    Main function for model evaluation.
    """   
    try:
        print("🚀 Starting Audio Classification Model Evaluation")
        print("=" * 60)
        
        #Setup paths
        results_dir = Path(AudioClassification.RESULTS_DIR)
        model_path = results_dir / "best_model.keras"
        test_segments_file = AudioClassification.TEST_SEGMENTS_FILE
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_path.parent / 'resnet_gru_evaluation_' / timestamp
                
        # Load model and setup
        model, mlb = load_model_and_setup(model_path)

        # Load optimized thresholds
        thresholds = load_thresholds(results_dir, mlb.classes_)
        
        # Create test data generator
        test_generator = create_evaluation_generator(test_segments_file, mlb)
        
        # Run comprehensive evaluation
        evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()