"""
Hyperparameter Tuning for Social Interaction Analysis

This script systematically tests different combinations of hyperparameters
for the social interaction analysis pipeline and evaluates their performance
against ground truth data.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from itertools import product
import json
import subprocess
import tempfile
import time
from datetime import datetime
import argparse 
import shutil
import inspect

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference, Evaluation
from config import InferenceConfig 
from results.eval_segment_performance import run_evaluation
# Import the main functions from the analysis scripts
from results.rq_01_frame_level_analysis import main as frame_analysis_main
from results.rq_01_video_level_analysis import main as segment_analysis_main


class HyperparameterConfig:
    """Configuration class for hyperparameter ranges."""
    
    # Define the hyperparameter search space 
    HYPERPARAMETER_RANGES= {
    'MIN_INTERACTING_SEGMENT_DURATION_SEC': [0.75, 1, 1.25],  # Centered on best: 1
    'MIN_ALONE_SEGMENT_DURATION_SEC': [4, 5, 6],              # Centered on best: 5
    'MIN_AVAILABLE_SEGMENT_DURATION_SEC': [4, 5, 6],          # Centered on best: 5
    'MIN_KCDS_DURATION_SEC': [1.5, 2, 2.5],                   # Centered on best: 2
    'MIN_RECLASSIFY_DURATION_SEC': [4, 5, 6],                 # Centered on best: 5
    'SUSTAINED_KCDS_SEC': [0.75, 1, 1.25],                    # Centered on best: 1
    'PROXIMITY_THRESHOLD': [0.75, 0.8, 0.85],                # Centered on best: 0.8
    'PERSON_AVAILABLE_WINDOW_SEC': [8, 9, 10],               # Centered on best: 9
    'MIN_PRESENCE_FRACTION': [0.35, 0.4, 0.45],              # Centered on best: 0.4
    'KCHI_PERSON_BUFFER_FRAMES': [10, 11, 12],               # Centered on best: 11
    'MAX_SAME_SPEAKER_GAP_SEC': [1.5, 2, 2.5],               # Centered on best: 2
    'MAX_TURN_TAKING_GAP_SEC': [5, 6, 7],                    # Centered on best: 6
    'GAP_MERGE_DURATION_SEC': [9, 10, 11],                   # Centered on best: 10
    'KCHI_ONLY_FRACTION_THRESHOLD': [0.75, 0.8, 0.85],       # Centered on best: 0.8
    'MIN_PERSON_PRESENCE_FRACTION': [0.03, 0.05, 0.07]       # Centered on best: 0.05
}

def generate_hyperparameter_combinations(max_combinations=None, random_sample=False):
    """
    Generate combinations of hyperparameters to test.
    
    Parameters
    ----------
    max_combinations : int, optional
        Maximum number of combinations to generate. If None, generates all combinations.
        Use this to limit computational cost.
    random_sample : bool
        If True and max_combinations is set, randomly sample combinations.
        If False, use systematic sampling.
        
    Returns
    -------
    list of dict
        List of hyperparameter dictionaries to test
    """
    param_names = list(HyperparameterConfig.HYPERPARAMETER_RANGES.keys())
    param_values = list(HyperparameterConfig.HYPERPARAMETER_RANGES.values())
    all_combinations = list(product(*param_values))
    valid_combinations = []
    
    for combo in all_combinations:
        params = dict(zip(param_names, combo))            
        valid_combinations.append(params)
        
    if max_combinations and len(valid_combinations) > max_combinations:
        if random_sample:
            np.random.seed(42)
            indices = np.random.choice(len(valid_combinations), max_combinations, replace=False)
            valid_combinations = [valid_combinations[i] for i in indices]
        else:
            step = len(valid_combinations) // max_combinations
            valid_combinations = valid_combinations[::step][:max_combinations]
        
    
    return valid_combinations

def run_pipeline_for_combo(hyperparameters, combo_dir):
    """
    Runs the full analysis pipeline by temporarily setting InferenceConfig 
    attributes, executing the analysis functions, and resetting the config.
    """
    
    # Backup original config state
    original_config = {}
    
    # List of all hyperparams (tuned + required fixed)
    all_params = hyperparameters

    try:
        # 1. Temporarily override InferenceConfig attributes
        for key, value in all_params.items(): # Iterate only over tuned params
            if hasattr(InferenceConfig, key):
                original_config[key] = getattr(InferenceConfig, key)
                setattr(InferenceConfig, key, value)
            else:
                setattr(InferenceConfig, key, value)
                
        # Assuming all rules are active for consistency in tuning
        rules = [1, 2, 3, 5]
        rule_suffix = "_1_2_3_5"
        
        frame_output_path = combo_dir / f"01_frame_level_social_interactions{rule_suffix}.csv"
        segment_output_path = combo_dir / "01_interaction_segments.csv"
        
        # 2. Execute Frame-Level Analysis
        frame_analysis_main(
            db_path=DataPaths.INFERENCE_DB_PATH,
            output_dir=combo_dir, # Pass combo_dir as the output directory
            hyperparameter_tuning=True,
            included_rules=rules,
        )

        # 3. Execute Segment-Level Analysis
        segment_analysis_main(
            output_file_path=segment_output_path, 
            frame_data_path=frame_output_path
        )
        
        return True, frame_output_path, segment_output_path, None

    except Exception as e:
        return False, None, None, f"Pipeline execution failed: {str(e)}"
        
    finally:
        # 4. Reset InferenceConfig to original state
        for key, value in original_config.items():
            setattr(InferenceConfig, key, value)


def run_analysis_with_config(hyperparameters, combo_id, output_base_dir):
    """
    Wrapper function to integrate the new direct pipeline runner with the existing loop structure.
    """
    combo_dir = output_base_dir / f"combo_{combo_id:04d}"
    combo_dir.mkdir(exist_ok=True)
    
    # Save hyperparameters for reference
    params_file = combo_dir / "hyperparameters.json"
    with open(params_file, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
        
    # Execute the full pipeline directly
    success, frame_output_path, segment_output_path, error = run_pipeline_for_combo(hyperparameters, combo_dir)
        
    # Final checks (must be done outside the temporary config scope)
    if success and not segment_output_path.exists():
        success = False
        error = "Pipeline ran successfully but final segment file was not generated."

    return success, frame_output_path, segment_output_path, error


def evaluate_combination(segment_output_path):
    """
    Evaluate a single hyperparameter combination using segment performance and save results.
    
    Parameters:
    ----------
    segment_output_path : Path
        Path to the segment output CSV file generated by the analysis pipeline.
    """
    try:
        # Load data
        save_path = segment_output_path.parent / "performance_results.txt"
        # Run evaluation (assumes eval_segment_performance.evaluate_performance is available)
        _, _, detailed_metrics = run_evaluation(segment_output_path, binary_mode=False)

        # Calculate overall metrics directly from results
        if detailed_metrics:
            overall_metrics = {
                'macro_avg_precision': detailed_metrics['macro_avg']['precision'],
                'macro_avg_recall': detailed_metrics['macro_avg']['recall'],
                'macro_avg_f1_score': detailed_metrics['macro_avg']['f1_score'],
            }

        else:
            overall_metrics = {}

        return {
            'success': True,
            'overall_metrics': overall_metrics,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'overall_metrics': None,
            'error': str(e)
        }

def main(max_combinations=None): 
    """
    Main hyperparameter tuning pipeline.
    
    Parameters:
    ----------
    max_combinations : int or None
        Maximum number of hyperparameter combinations to test. If None, tests all valid combinations.
    """        
    print(f"Running Hyperparameter Tuning Pipeline for {max_combinations} combinations")
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(f"{Evaluation.HYPERPARAMETER_OUTPUT_DIR}_{timestamp}")
    output_base_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate hyperparameter combinations
    print("\n📊 Generating hyperparameter combinations...")
    combinations = generate_hyperparameter_combinations(
        max_combinations=max_combinations, 
        random_sample=getattr(InferenceConfig, 'RANDOM_SAMPLING', False)
    )
    
    print(f"Will test {len(combinations)} combinations")
    
    # Ground truth path
    ground_truth_path = Inference.GROUND_TRUTH_SEGMENTS_CSV
    if not ground_truth_path.exists():
        print(f"❌ Ground truth file not found: {ground_truth_path}")
        return
    
    # Results storage
    all_results = []
    successful_runs = 0
    failed_runs = 0
    
    # Process each combination
    print("\n🚀 Running analysis for each combination...")
    start_time = time.time()
    
    for i, hyperparams in enumerate(combinations):
        combo_id = i + 1
        print(f"\n[{combo_id}/{len(combinations)}] Testing combination {combo_id}")
        print(f"Parameters: {hyperparams}")
        
        # Run analysis
        success, frame_output_path, segment_output_path, error = run_analysis_with_config(
            hyperparams, combo_id, output_base_dir
        )
        
        if not success:
            print(f"❌ Analysis failed: {error}")
            failed_runs += 1
            continue
        
        # Evaluate performance
        evaluation_result = evaluate_combination(segment_output_path)
        
        if not evaluation_result['success']:
            print(f"❌ Evaluation failed: {evaluation_result['error']}")
            failed_runs += 1
            continue
        
        # Store results
        result_record = {
            'combo_id': combo_id,
            'hyperparameters': hyperparams,
            'evaluation': evaluation_result,
            'frame_output_path': str(frame_output_path),
            'segment_output_path': str(segment_output_path)
        }
        all_results.append(result_record)
        
        # Print key metrics
        overall = evaluation_result['overall_metrics']
        print(f"✅ Success! Macro F1: {overall['macro_avg_f1_score']:.4f}")
        
        successful_runs += 1
        
        # Save intermediate results periodically
        if combo_id % 10 == 0:
            save_intermediate_results(all_results, output_base_dir)
    
    # Final results processing
    elapsed_time = time.time() - start_time
    print(f"\n📈 Hyperparameter tuning completed!")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    
    if successful_runs == 0:
        print("❌ No successful runs to analyze")
        return
    
    # Find best configuration
    best_config = find_best_configuration(all_results)
    
    # Save final results
    save_final_results(all_results, best_config, output_base_dir)
    
    # Print summary
    print_results_summary(all_results, best_config)


def save_intermediate_results(results, output_dir):
    """Save intermediate results to prevent data loss."""
    results_path = output_dir / "intermediate_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def find_best_configuration(results):
    """
    Find the best performing hyperparameter configuration.
    """
    # Sort by macro F1 score (primary), then micro F1 (secondary)
    sorted_results = sorted(
        results,
        key=lambda x: (
            x['evaluation']['overall_metrics']['macro_f1'],
            x['evaluation']['overall_metrics']['micro_f1']
        ),
        reverse=True
    )
    
    return sorted_results[0]

def save_final_results(all_results, best_config, output_dir):
    """
    Save final results and best configuration.
    """
    # Save all results
    results_path = output_dir / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Save best configuration
    best_config_path = output_dir / "best_configuration.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2, default=str)
    
    # Create summary CSV
    summary_data = []
    for result in all_results:
        combo_id = result['combo_id']
        hyperparams = result['hyperparameters']
        overall = result['evaluation']['overall_metrics']
        
        row = {
            'combo_id': combo_id,
            'macro_f1': overall['macro_f1'],
            'micro_f1': overall['micro_f1'],
            'macro_precision': overall['macro_precision'],
            'macro_recall': overall['macro_recall'],
            **hyperparams
        }
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_dir / "results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"  - All results: {results_path}")
    print(f"  - Best config: {best_config_path}")
    print(f"  - Summary CSV: {summary_path}")

def print_results_summary(all_results, best_config):
    """
    Print a summary of the hyperparameter tuning results.
    """
    print("\n" + "=" * 80)
    print("🏆 HYPERPARAMETER TUNING RESULTS SUMMARY")
    print("=" * 80)
    
    # Best configuration
    print(f"\n🥇 BEST CONFIGURATION (Combo {best_config['combo_id']}):")
    best_params = best_config['hyperparameters']
    best_metrics = best_config['evaluation']['overall_metrics']
    
    print("\nOptimal Hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nPerformance Metrics:")
    print(f"  Macro F1:        {best_metrics['macro_f1']:.4f}")
    print(f"  Micro F1:        {best_metrics['micro_f1']:.4f}")
    print(f"  Macro Precision: {best_metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:    {best_metrics['macro_recall']:.4f}")
    
    # Performance distribution
    f1_scores = [r['evaluation']['overall_metrics']['macro_f1'] for r in all_results]
    print(f"\n📊 Performance Distribution (Macro F1):")
    print(f"  Best:    {max(f1_scores):.4f}")
    print(f"  Worst:   {min(f1_scores):.4f}")
    print(f"  Mean:    {np.mean(f1_scores):.4f}")
    print(f"  Std:     {np.std(f1_scores):.4f}")
    
    # Top 5 configurations
    sorted_results = sorted(
        all_results,
        key=lambda x: x['evaluation']['overall_metrics']['macro_f1'],
        reverse=True
    )
    
    print(f"\n🔝 TOP 5 CONFIGURATIONS:")
    for i, result in enumerate(sorted_results[:5]):
        combo_id = result['combo_id']
        f1_score = result['evaluation']['overall_metrics']['macro_f1']
        print(f"  {i+1}. Combo {combo_id}: F1 = {f1_score:.4f}")
    
    print("\n" + "=" * 80)
    print("Hyperparameter tuning completed successfully!")
    print("Use the best configuration parameters in your config.py file.")
    
def parse_args():
    """
    Parse command line arguments to determine the maximum number of combinations.
    """
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning for social interaction analysis')
    parser.add_argument('--quick', action='store_true', 
                    help='Quick test with 5 combinations (~30 minutes)')
    parser.add_argument('--full', action='store_true',
                    help='Full search with all valid combinations (can take hours)')
    parser.add_argument('--max-combos', type=int, default=None,
                    help='Maximum number of combinations to test (overrides quick/full)')
    
    args = parser.parse_args()
    
    # Determine number of combinations based on flags
    if args.quick:
        max_combinations = 5
        print("🚀 Running QUICK hyperparameter tuning (5 combinations)")
    elif args.full:
        max_combinations = None 
        print("🚀 Running FULL hyperparameter tuning (all valid combinations)")
    elif args.max_combos is not None:
        max_combinations = args.max_combos
        print(f"🚀 Running hyperparameter tuning with {max_combinations} combinations (custom)")
    else:
        # Default behavior: uses InferenceConfig.MAX_COMBINATIONS_TUNING
        try:
            max_combinations = InferenceConfig.MAX_COMBINATIONS_TUNING 
        except AttributeError:
            max_combinations = 20 # Fallback default

        print(f"🚀 Running hyperparameter tuning with {max_combinations} combinations (default)")

    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING FOR SOCIAL INTERACTION ANALYSIS")
    print("="*80)
    
    return max_combinations

if __name__ == "__main__":
    max_combos = parse_args()
    
    # Run the hyperparameter tuning
    try:
        main(max_combos)
        print("\n🎉 Hyperparameter tuning completed successfully!")
        print("\nCheck the 'hyperparameter_tuning_results' directory for:")
        print("  - best_configuration.json: Optimal hyperparameters")
        print("  - results_summary.csv: Performance of all combinations")
        print("  - Individual combo directories with detailed results")
        exit(0)
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
        print("Partial results may be available in 'hyperparameter_tuning_results' directory")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during hyperparameter tuning: {e}")
        exit(1)