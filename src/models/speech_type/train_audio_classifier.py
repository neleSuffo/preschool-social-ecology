import sys
import os
# --- Mixed Precision Setup ---
from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
import tensorflow as tf

num_threads = 24

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

# TensorFlow threading config
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)  # few parallel ops at once

# GPU: select only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU

# Check if we need to restart with correct LD_LIBRARY_PATH
lib_path = "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cublas/lib/"

# If LD_LIBRARY_PATH doesn't contain our path, restart the script
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if lib_path not in current_ld_path:
    print("Setting LD_LIBRARY_PATH and restarting...")
    new_ld_path = f"{lib_path}:{current_ld_path}" if current_ld_path else lib_path
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    os.execv(sys.executable, ['python'] + sys.argv)
    
    
import datetime
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import *
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from constants import AudioClassification
from config import AudioConfig
from utils import create_data_generators, create_training_callbacks, setup_gpu_config, load_model
from audio_classifier import build_model_multi_label, FocalLoss, MacroF1Score, ThresholdOptimizer
# Setup GPU configuration
gpu_available = setup_gpu_config()

if not gpu_available:
    print("\n🔄 GPU setup failed. To continue anyway, press Enter.")
    print("   To force CPU mode, set CUDA_VISIBLE_DEVICES='' before running.")
    input("   Press Enter to continue with current configuration...")
    print()
    
# --- Setup and Training Functions --
def setup_training_environment():
    """
    Set up training environment with directories and configuration.
    
    Returns:
    -------
    str: 
        Path to training run directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = AudioClassification.OUTPUT_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training run output directory: {run_dir}")
    
    # Copy current script for reproducibility
    current_script_path = Path(__file__)
    shutil.copy(current_script_path, run_dir / 'train_audio_classifier.py')
    
    return run_dir

def train_model_with_callbacks(model, train_generator, val_generator, callbacks, epochs, phase):
    """
    Train the model with given data generators and callbacks.
    
    This function orchestrates the training process with:
    - Validation data monitoring for overfitting detection
    - Callback mechanisms for early stopping and model checkpointing
    - Uniform class weighting suitable for balanced ID-split data
    - Comprehensive training history tracking
    
    Parameters:
    ----------
    model: tf.keras.Model
        Compiled multi-label audio classification model
    train_generator: AudioSegmentDataGenerator
        Training data generator with augmentation enabled
    val_generator: AudioSegmentDataGenerator  
        Validation data generator (no augmentation)
    callbacks: list
        Keras callbacks (EarlyStopping, ModelCheckpoint, LRScheduler)
    epochs (int): 
        Maximum number of training epochs
    phase (int):
        Training phase identifier (1 or 2)

    Returns:
    -------
    tf.keras.callbacks.History:
        Training history with metrics per epoch
        
    Raises:
    ------
    ValueError:
        If generators contain no data batches
    """
    print(f"\nStarting training for {epochs} epochs (Phase {phase})...")
    if hasattr(train_generator, 'cardinality') and train_generator.cardinality().numpy() == 0:
        raise ValueError("No batches available for training. Check your segment files and data paths.")
    if hasattr(val_generator, 'cardinality') and val_generator.cardinality().numpy() == 0:
        raise ValueError("No batches available for validation. Check your segment files and data paths.")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def main():
    """
    Main training pipeline for multi-label audio voice type classification.
    
    This function orchestrates the complete training workflow:
    1. Data preparation with train/validation/test split
    2. Model architecture construction and compilation  
    3. Training with callbacks (early stopping, checkpointing, LR scheduling)
    4. Model evaluation with threshold optimization
    5. Results saving and reporting
    
    The pipeline uses:
    - CNN-RNN hybrid architecture with multi-head attention
    - Focal loss for handling class imbalance
    - Data augmentation for training robustness
    - Macro F1-score optimization for multi-label performance
    - Cosine annealing learning rate scheduling
    
    Configuration:
    -------------
    - Labels: Defined in AudioConfig.VALID_RTTM_CLASSES
    - Data paths: Defined in AudioClassification segment files
    - Audio parameters: Defined in AudioConfig
    - Model parameters: Defined in build_model_multi_label()
    
    Outputs:
    -------
    - Trained model saved as .keras file
    - Training history plots and metrics
    - Threshold optimization results
    - Test set performance evaluation
    """   
    # Define voice type classes and data split file paths
    unique_labels = AudioConfig.VALID_RTTM_CLASSES
    segment_files = {
        'train': AudioClassification.TRAIN_SEGMENTS_FILE,
        'val': AudioClassification.VAL_SEGMENTS_FILE,
        'test': None
    }

    try:
        print("🚀 Starting Audio Classification Training Pipeline")
        print("=" * 60)
                
        # 1. Setup training environment
        run_dir = setup_training_environment()
        
        # 2. Create model and setup encoders
        mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
        mlb.fit([[]])  # Initialize with empty list to set up classes

        # 3. Create data generators
        train_generator, val_generator, _ = create_data_generators(segment_files, mlb)

        # --- PHASE 1: Train the classifier head with frozen CNN ---
        print("\n🧠 PHASE 1: Building and training model with frozen CNN layers...")
        fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
        model_phase1 = build_model_multi_label(
            n_mels=min(AudioConfig.N_MELS, 128),
            fixed_time_steps=fixed_time_steps,
            num_classes=len(mlb.classes_),
            freeze_cnn=True
        )
        model_phase1.log_dir = run_dir
        
        callbacks_phase1 = create_training_callbacks(run_dir, val_generator, mlb.classes_)
        history_phase1 = train_model_with_callbacks(
            model_phase1, train_generator, val_generator, callbacks_phase1, AudioConfig.EPOCHS_PHASE1, 1
        )
        # Instead of saving the full model, save only the weights
        model_phase1.save_weights(run_dir / 'best_weights_phase1.h5')

        # --- PHASE 2: Fine-tune the entire model ---
        print("\n🧠 PHASE 2: Building a new model and fine-tuning it...")

        # 1. Build a new, unfrozen model from scratch
        model_phase2 = build_model_multi_label(
            n_mels=min(AudioConfig.N_MELS, 128),
            fixed_time_steps=train_generator.fixed_time_steps,
            num_classes=len(mlb.classes_),
            freeze_cnn=False # This model is fully trainable
        )

        # 2. Load the weights from the saved file
        model_phase2.load_weights(run_dir / 'best_weights_phase1.h5')

        # 3. Re-compile the new model with the lower learning rate
        model_phase2.compile(
            optimizer=Adam(learning_rate=AudioConfig.LR_PHASE2),
            loss=FocalLoss(gamma=2.0, alpha=0.25),
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), MacroF1Score(num_classes=len(mlb.classes_), name='macro_f1')]
        )

        model_phase2.log_dir = run_dir

        callbacks_phase2 = create_training_callbacks(run_dir, val_generator, mlb.classes_)
        history_phase2 = train_model_with_callbacks(
            model_phase2, train_generator, val_generator, callbacks_phase2, AudioConfig.EPOCHS_PHASE2, 2
        )
        model_phase2.save(run_dir / 'final_model_phase2.keras')
        
        print("✅ Training pipeline completed successfully!")
        print(f"📁 Results saved to: {run_dir}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()