import sys
import os
# --- Mixed Precision Setup ---
from tensorflow.keras.mixed_precision import policy
policy.set_global_policy(policy.Policy('mixed_float16'))
import tensorflow as tf

num_threads = 24

os.environ["OMP_NUM_THREADS"] = str(nuAudioSegmentDataGeneratorm_threads)
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
from pathlib import Path
from constants import AudioClassification
from config import AudioConfig
from utils import create_data_generators, create_training_callbacks, setup_gpu_config, load_model

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

def train_model_with_callbacks(model, train_generator, val_generator, callbacks, epochs):
    """
    Train the audio classification model using data generators and callbacks.
    
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

    Returns:
    -------
    tf.keras.callbacks.History:
        Training history with metrics per epoch
        
    Raises:
    ------
    ValueError:
        If generators contain no data batches
    """
    print(f"\nStarting training for {epochs} epochs...")
    
    # Validate data availability before training
    if len(train_generator) == 0 or len(val_generator) == 0:
        raise ValueError("No batches available for training/validation. Check your segment files and data paths.")
    
    # Use uniform class weights for balanced ID-split data
    # Note: For time-split data, consider computing class weights from training data
    class_weight = None
    print("Using uniform class weighting (recommended for ID-split balanced data)")
    
    # Execute training with validation monitoring
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,  # Early stopping, checkpointing, LR scheduling
        class_weight=class_weight,
        verbose=1  # Show progress bar and metrics per epoch
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
        print("🏗️ Setting up training environment...")
        run_dir = setup_training_environment()
        
        # 2. Create model and setup encoders
        print("🧠 Creating model and setting up encoders...")
        model, mlb = load_model()
        model.log_dir = run_dir  # For ThresholdOptimizer callback

        # 3. Create data generators
        print("🔄 Creating data generators...")
        train_generator, val_generator, _ = create_data_generators(segment_files, mlb)

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
        
    except Exception as e:
        print(f"❌ Training pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()