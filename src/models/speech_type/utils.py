import json
import librosa
import numpy as np
import tensorflow as tf
from pathlib import Path
from config import AudioConfig
from models.speech_type.audio_classifier import build_model_multi_label
from sklearn.preprocessing import MultiLabelBinarizer

# --- Data Generator ---
class AudioSegmentDataGenerator(tf.keras.utils.Sequence):
    """
    Keras data generator for audio classification with enhanced features and augmentation.
    
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
    
    train_generator = AudioSegmentDataGenerator(
        segment_files['train'], mlb, 
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR, 
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=True, augment=True  # Enable augmentation for training
    )
    
    val_generator = AudioSegmentDataGenerator(
        segment_files['val'], mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False  # No augmentation for validation
    )
    
    test_generator = AudioSegmentDataGenerator(
        segment_files['test'], mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32, shuffle=False, augment=False  # No augmentation for testing
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

def create_model_and_setup(unique_labels):
    """
    Create and compile the multi-label classification model.
    
    Parameters:
    ----------
    unique_labels (list): List of unique voice type labels
        
    Returns:
    -------
    tuple: 
        (model, mlb, num_classes, fixed_time_steps) where:
            - model: Compiled Keras model
            - mlb: Fitted MultiLabelBinarizer
            - num_classes: Number of classes
            - fixed_time_steps: Fixed time steps for model input
    """
    # Setup multi-label encoder
    mlb, num_classes = setup_multilabel_encoder(unique_labels)
    
    # Use centralized calculation for consistent time steps across all components
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))

    # Build model architecture
    model = build_model_multi_label(
        n_mels=AudioConfig.N_MELS,
        fixed_time_steps=fixed_time_steps,
        num_classes=num_classes
    )
    
    return model, mlb, num_classes, fixed_time_steps

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

# ---- Inference Functions ----
def classify_audio_windows(audio_path, model, mlb, batch_size=32):
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
    import librosa
    
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
    from pathlib import Path
    
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

def process_audio_folder(folder_path, model, mlb, output_dir, thresholds):
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
    output_dir : str
        Path to the directory where output files will be saved.
    thresholds : dict
        Dictionary mapping class names to optimal thresholds
    """
    import csv
    from pathlib import Path
    
    folder_path = Path(folder_path)
    output_file = output_dir / 'classification_results.csv'
    
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

def load_inference_model(model_path):
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
        print(f"✅ Model loaded successfully from {model_path}")
    
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
