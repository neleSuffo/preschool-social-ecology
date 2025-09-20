import os

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

from pathlib import Path
from datetime import datetime
from constants import AudioClassification
from utils import load_thresholds, load_model, create_data_generators, evaluate_model_comprehensive

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
        test_segments_file = AudioClassification.TEST_SEGMENTS_FILE
        
        # Create timestamped output directory for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(AudioClassification.RESULTS_DIR) / f'evaluation_results_{timestamp}'

        # Stage 2: Load trained model with Lambda layer handling
        model, mlb = load_model()

        # Stage 3: Load optimized thresholds from training run
        thresholds = load_thresholds(mlb.classes_)
        
        # Stage 4: Create test data generator (reuse existing function)
        segment_files = {
            'train': None,  # Not needed for evaluation
            'val': None,    # Not needed for evaluation
            'test': test_segments_file
        }
        _, _, test_generator = create_data_generators(segment_files, mlb)
        
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