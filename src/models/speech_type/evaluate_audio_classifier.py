import tensorflow as tf
from pathlib import Path
from datetime import datetime
from config import AudioConfig
from constants import AudioClassification
from utils import load_thresholds, load_model, create_data_generators, evaluate_model_comprehensive, evaluate_model_at_both_levels, setup_gpu_config

# Setup GPU configuration
gpu_available = setup_gpu_config()
if gpu_available:
    print("✅ GPU configuration completed successfully")
else:
    print("⚠️ GPU configuration failed, will use CPU")

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
        output_dir = AudioClassification.TRAINED_WEIGHTS_PATH.parent
        folder_name = output_dir / f'evaluation_{timestamp}'

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
        
        if test_generator is None or len(test_generator) == 0:
            raise ValueError("Test generator is empty. Check test data file and paths.")

        # Stage 5: Execute comprehensive model evaluation at both levels
        print("\n🎯 EVALUATION OPTIONS:")
        print("1. Window-level only (original sliding windows)")
        print("2. Second-level only (aggregated predictions)")  
        print("3. Both levels for comparison (recommended)")
        
        evaluation_choice = input("\nChoose evaluation level (1/2/3, default=3): ").strip() or "3"
        
        if evaluation_choice == "1":
            evaluate_model_comprehensive(
                model=model, 
                test_generator=test_generator, 
                mlb=mlb, 
                thresholds=thresholds, 
                output_dir=folder_name,
                generate_confusion_matrices=True,
                aggregate_to_seconds=False
            )
        elif evaluation_choice == "2":
            evaluate_model_comprehensive(
                model=model,
                test_generator=test_generator, 
                mlb=mlb,
                thresholds=thresholds,
                output_dir=folder_name,
                generate_confusion_matrices=True, 
                aggregate_to_seconds=True
            )
        else:
            evaluate_model_at_both_levels(
                model=model,
                test_generator=test_generator,
                mlb=mlb, 
                thresholds=thresholds,
                output_dir=folder_name,
                generate_confusion_matrices=True
            )
        
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