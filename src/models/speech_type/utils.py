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
            mel = extract_features(
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

# ---- Evaluation Functions ----
def evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir):
    """
    Perform comprehensive multi-label classification evaluation with detailed analysis. 
    
    Parameters:
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
        
    Outputs:
    -------
    Files Created:
    - evaluation_summary.json: Comprehensive metrics summary
    - detailed_predictions.csv: Per-sample predictions and probabilities
    
    Raises:
    ------
    ValueError: If test generator is empty or predictions fail
    RuntimeError: If evaluation cannot complete due to data issues
    """
    from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
    from tqdm import tqdm
    from pathlib import Path
    
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
        (test_predictions[:, i] > thresholds[mlb.classes_[i]]).astype(int) 
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
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved to: {output_dir}")

def save_evaluation_results(output_dir, class_names, thresholds,
                        test_true_labels, test_pred_binary, test_predictions,
                        precision_per_class, recall_per_class, f1_per_class, support_per_class,
                        macro_precision, macro_recall, macro_f1,
                        micro_precision, micro_recall, micro_f1, subset_accuracy):
    """
    Save comprehensive evaluation results in multiple formats for analysis and reporting.
    
    This function creates both machine-readable (JSON) and human-readable (CSV) outputs
    containing detailed evaluation metrics.
    
    Parameters:
    ----------
    output_dir (Path): Target directory for result files
    class_names (list): Names of classification classes
    thresholds (dict): Dictionary mapping class names to decision thresholds
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
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    
    output_dir = Path(output_dir)
    
    # Create comprehensive metrics summary for programmatic analysis
    summary = {
        'evaluation_metadata': {
            'test_set_size': len(test_true_labels),
            'num_classes': len(class_names),
            'class_names': list(class_names),
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
    detailed_results['sample_id'] = range(len(detailed_results))
    detailed_results['num_true_labels'] = test_true_labels.sum(axis=1)
    detailed_results['num_pred_labels'] = test_pred_binary.sum(axis=1)
    detailed_results['exact_match'] = (test_true_labels == test_pred_binary).all(axis=1).astype(int)
    
    # Save detailed predictions for manual inspection and error analysis
    predictions_path = output_dir / 'detailed_predictions.csv'
    detailed_results.to_csv(predictions_path, index=False, float_format='%.4f')