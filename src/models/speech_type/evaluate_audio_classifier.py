import os
import sys

# Only set environment if running as subprocess with --direct flag
if len(sys.argv) > 1 and sys.argv[1] == "--direct":
    print("🔧 Running in subprocess with CUDA environment configured")
    
    # Verify critical library exists
    critical_lib = "/home/nele_pauline_suffo/projects/naturalistic-social-analysis/.venv/lib/python3.8/site-packages/nvidia/cublas/lib/libcublasLt.so.12"
    if os.path.exists(critical_lib):
        print(f"✅ Critical library found: {critical_lib}")
    else:
        print(f"❌ Critical library missing: {critical_lib}")
        sys.exit(1)

import subprocess
import sys
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
from constants import AudioClassification
from utils import load_thresholds

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
    # Organize columns for easy analysis: probabilities, predictions, then true labels
    
    # Convert class_names to list to ensure proper column naming
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

def run_with_cuda_environment():
    """
    Run the evaluation with proper CUDA environment setup.
    
    This function sets up the required environment variables before running
    the evaluation, similar to what would be done in a shell script.
    """    
    # Set up CUDA library paths
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
    
    # Prepare environment
    env = os.environ.copy()
    existing_path = env.get("LD_LIBRARY_PATH", "")
    new_path = ":".join(cuda_lib_paths)
    
    if existing_path:
        env["LD_LIBRARY_PATH"] = new_path + ":" + existing_path
    else:
        env["LD_LIBRARY_PATH"] = new_path
    
    print("🚀 Running evaluation with CUDA environment setup...")
    print(f"🔧 LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    
    # Get the path to the current script
    current_script = __file__
    
    # Run the script in a subprocess with the proper environment
    try:
        result = subprocess.run([
            sys.executable, current_script, "--direct"
        ], env=env, check=True)
        print("✅ Evaluation completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with return code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
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
    import sys
    
    # Check if this is a direct call (from subprocess) or initial call
    if len(sys.argv) > 1 and sys.argv[1] == "--direct":
        # This is the subprocess call with environment already set
        main()
    else:
        # This is the initial call, set up environment and restart
        exit_code = run_with_cuda_environment()
        sys.exit(exit_code)