import os
import sys

# Set up CUDA library paths directly
cuda_lib_paths = [
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cublas/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cudnn/lib", 
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cufft/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/curand/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cusolver/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cusparse/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_runtime/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_cupti/lib",
    "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cuda_nvrtc/lib"
]

existing_path = os.environ.get("LD_LIBRARY_PATH", "")
new_path = ":".join(cuda_lib_paths)

if existing_path:
    os.environ["LD_LIBRARY_PATH"] = new_path + ":" + existing_path
else:
    os.environ["LD_LIBRARY_PATH"] = new_path

print("🔧 CUDA library paths configured")

import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from constants import AudioClassification
from utils import load_thresholds, load_model_and_setup, create_evaluation_generator, save_evaluation_results

# --- Evaluation Functions ---
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
    - per_class_f1_scores.png: Class-wise F1 score visualization
    - prediction_distributions.png: Probability distribution plots
    - class_support.png: Class frequency analysis plot
    
    Raises:
    ------
    ValueError: If test generator is empty or predictions fail
    RuntimeError: If evaluation cannot complete due to data issues
    """
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

def main():
    """
    Execute comprehensive audio classification model evaluation pipeline.
    
    This is the main orchestration function that coordinates the complete evaluation
    workflow for the trained multi-label audio classification model.

    Performance Metrics:
    - Subset Accuracy: Exact multi-label match percentage
    - Macro F1: Unweighted average across classes (handles imbalance)
    - Micro F1: Global performance from aggregated statistics
    - Per-class Metrics: Individual class precision, recall, F1, support
    - Threshold Analysis: Applied decision boundaries per class   
    """   
    try:
        print("🚀 Starting Comprehensive Audio Classification Model Evaluation")
        print("=" * 70)
        
        # Stage 1: Configure evaluation environment and paths
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
        raise

if __name__ == "__main__":
    main()