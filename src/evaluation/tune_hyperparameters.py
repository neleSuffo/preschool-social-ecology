#!/usr/bin/env python3
"""
Hyperparameter Tuning for Social Interaction Analysis

This script systematically tests different combinations of hyperparameters
for the social interaction analysis pipeline and evaluates their performance
against ground truth data.

The script:
1. Generates hyperparameter combinations
2. Runs frame-level and video-level analysis with each combination
3. Evaluates performance using IoU-based segment matching
4. Identifies the best performing configuration

Usage:
    python hyperparameter_tuning.py

Output:
    - Individual result files for each combination
    - Performance evaluation results
    - Best hyperparameter configuration
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

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Inference
from config import InferenceConfig
from eval_segment_performance import evaluate_performance


class HyperparameterConfig:
    """Configuration class for hyperparameter ranges."""
    
    # Define the hyperparameter search space (reduced for computational efficiency)
    HYPERPARAMETER_RANGES = {
        'PROXIMITY_THRESHOLD': [0.6, 0.7, 0.8],
        'MIN_SEGMENT_DURATION_SEC': [3, 5, 7],
        'MIN_CHANGE_DURATION_SEC': [2, 3, 4],
        'TURN_TAKING_BASE_WINDOW_SEC': [8, 10, 12],
        'TURN_TAKING_EXT_WINDOW_SEC': [12, 15, 18],
        'PERSON_AUDIO_WINDOW_SEC': [8, 10, 12],
        'GAP_MERGE_DURATION_SEC': [3, 5, 7],
        'VALIDATION_SEGMENT_DURATION_SEC': [8, 10, 12]
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
    # Get all parameter names and their possible values
    param_names = list(HyperparameterConfig.HYPERPARAMETER_RANGES.keys())
    param_values = list(HyperparameterConfig.HYPERPARAMETER_RANGES.values())
    
    # Generate all possible combinations
    all_combinations = list(product(*param_values))
        
    # Filter combinations based on logical constraints
    valid_combinations = []
    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        
        # Constraint 1: Extended window should be >= base window
        if params['TURN_TAKING_EXT_WINDOW_SEC'] < params['TURN_TAKING_BASE_WINDOW_SEC']:
            continue
            
        # Constraint 2: Min change duration should be <= min segment duration
        if params['MIN_CHANGE_DURATION_SEC'] > params['MIN_SEGMENT_DURATION_SEC']:
            continue
        
        # Constraint 3: Validation duration should be >= min segment duration
        if params['VALIDATION_SEGMENT_DURATION_SEC'] < params['MIN_SEGMENT_DURATION_SEC']:
            continue
            
        valid_combinations.append(params)
        
    # Limit combinations if requested
    if max_combinations and len(valid_combinations) > max_combinations:
        if random_sample:
            # Random sampling
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(valid_combinations), max_combinations, replace=False)
            valid_combinations = [valid_combinations[i] for i in indices]
        else:
            # Systematic sampling - take every nth combination
            step = len(valid_combinations) // max_combinations
            valid_combinations = valid_combinations[::step][:max_combinations]
        
    
    return valid_combinations

def create_temp_config(hyperparameters, temp_dir):
    """
    Create a temporary config file with the specified hyperparameters.
    
    Parameters
    ----------
    hyperparameters : dict
        Dictionary of hyperparameter values
    temp_dir : Path
        Temporary directory to store config file
        
    Returns
    -------
    Path
        Path to the temporary config file
    """
    config_content = f"""
# Temporary config file for hyperparameter tuning
# Generated at: {datetime.now()}

class InferenceConfig:
    # Hyperparameters being tuned
    PROXIMITY_THRESHOLD = {hyperparameters['PROXIMITY_THRESHOLD']}
    MIN_SEGMENT_DURATION_SEC = {hyperparameters['MIN_SEGMENT_DURATION_SEC']}
    MIN_CHANGE_DURATION_SEC = {hyperparameters['MIN_CHANGE_DURATION_SEC']}
    TURN_TAKING_BASE_WINDOW_SEC = {hyperparameters['TURN_TAKING_BASE_WINDOW_SEC']}
    TURN_TAKING_EXT_WINDOW_SEC = {hyperparameters['TURN_TAKING_EXT_WINDOW_SEC']}
    MAX_TURN_TAKING_GAP_SEC = {hyperparameters['MAX_TURN_TAKING_GAP_SEC']}
    PERSON_AUDIO_WINDOW_SEC = {hyperparameters['PERSON_AUDIO_WINDOW_SEC']}
    GAP_MERGE_DURATION_SEC = {hyperparameters['GAP_MERGE_DURATION_SEC']}
    VALIDATION_SEGMENT_DURATION_SEC = {hyperparameters['VALIDATION_SEGMENT_DURATION_SEC']}
    
    # Fixed parameters
    SPEECH_CLASSES = {HyperparameterConfig.FIXED_PARAMS['SPEECH_CLASSES']}
    
    # Other parameters from original config (keep defaults)
    EVALUATION_IOU = 0.5
    
class DataConfig:
    FPS = 30

class DataPaths:
    INFERENCE_DB_PATH = "{DataPaths.INFERENCE_DB_PATH}"
    SUBJECTS_CSV_PATH = "{DataPaths.SUBJECTS_CSV_PATH if hasattr(DataPaths, 'SUBJECTS_CSV_PATH') else 'N/A'}"
"""
    
    config_path = temp_dir / "temp_config.py"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def create_modified_script(original_script_path, temp_dir, hyperparameters):
    """
    Create a modified version of an analysis script with hardcoded hyperparameters.
    
    Parameters
    ----------
    original_script_path : Path
        Path to the original script
    temp_dir : Path
        Temporary directory for modified script
    hyperparameters : dict
        Hyperparameter values to hardcode
        
    Returns
    -------
    Path
        Path to the modified script
    """
    # Read original script
    with open(original_script_path, 'r') as f:
        content = f.read()
    
    # Create hyperparameter replacement strings
    replacements = []
    for param, value in hyperparameters.items():
        if isinstance(value, str):
            replacements.append(f"InferenceConfig.{param} = '{value}'")
        else:
            replacements.append(f"InferenceConfig.{param} = {value}")
    
    # Add fixed parameters
    for param, value in HyperparameterConfig.FIXED_PARAMS.items():
        if isinstance(value, list):
            replacements.append(f"InferenceConfig.{param} = {value}")
        else:
            replacements.append(f"InferenceConfig.{param} = '{value}'")
    
    # Insert hyperparameter overrides after imports
    import_end = content.find("# Constants")
    if import_end == -1:
        import_end = content.find("def ")
    
    if import_end != -1:
        # Insert overrides after imports
        override_code = "\n# Hyperparameter overrides for tuning\n"
        for replacement in replacements:
            override_code += f"{replacement}\n"
        override_code += "\n"
        
        content = content[:import_end] + override_code + content[import_end:]
    
    # Write modified script
    modified_script_path = temp_dir / original_script_path.name
    with open(modified_script_path, 'w') as f:
        f.write(content)
    
    return modified_script_path

def run_analysis_with_config(hyperparameters, combo_id, output_base_dir):
    """
    Run the analysis pipeline with specified hyperparameters.
    
    Parameters
    ----------
    hyperparameters : dict
        Dictionary of hyperparameter values
    combo_id : int
        Unique identifier for this combination
    output_base_dir : Path
        Base directory for outputs
        
    Returns
    -------
    tuple
        (success, frame_output_path, segment_output_path, error_message)
    """
    combo_dir = output_base_dir / f"combo_{combo_id:04d}"
    combo_dir.mkdir(exist_ok=True)
    
    # Save hyperparameters for reference
    params_file = combo_dir / "hyperparameters.json"
    with open(params_file, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
        
    frame_output = combo_dir / "frame_level_interactions.csv"
    segment_output = combo_dir / "interaction_segments.csv"
    
    try:
        # Create temporary directory for modified scripts
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create modified scripts with hardcoded hyperparameters
            original_frame_script = src_path / "results" / "rq_01_frame_level_analysis.py"
            original_segment_script = src_path / "results" / "rq_01_video_level_analysis.py"
            
            frame_script_path = create_modified_script(original_frame_script, temp_dir_path, hyperparameters)
            segment_script_path = create_modified_script(original_segment_script, temp_dir_path, hyperparameters)
            
            # Run frame-level analysis
            try:
                result = subprocess.run([
                    sys.executable, str(frame_script_path),
                    "--output", str(frame_output),
                    "--db_path", str(DataPaths.INFERENCE_DB_PATH)
                ], capture_output=True, text=True, timeout=600, cwd=str(src_path))  # 10 minute timeout
                
                if result.returncode != 0:
                    error_msg = f"Frame analysis failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    return False, None, None, error_msg
                    
            except subprocess.TimeoutExpired:
                return False, None, None, "Frame analysis timed out"
            
            # Check if frame output was created
            if not frame_output.exists():
                return False, None, None, "Frame analysis completed but no output file was created"
            
            # Run segment-level analysis
            try:
                result = subprocess.run([
                    sys.executable, str(segment_script_path),
                    "--input", str(frame_output),
                    "--output", str(segment_output)
                ], capture_output=True, text=True, timeout=300, cwd=str(src_path))  # 5 minute timeout
                
                if result.returncode != 0:
                    error_msg = f"Segment analysis failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
                    return False, None, None, error_msg
                    
            except subprocess.TimeoutExpired:
                return False, None, None, "Segment analysis timed out"
            
            # Check if segment output was created
            if not segment_output.exists():
                return False, None, None, "Segment analysis completed but no output file was created"
        
        return True, frame_output, segment_output, None
        
    except Exception as e:
        return False, None, None, f"Unexpected error: {str(e)}"



def evaluate_combination(segment_output_path, ground_truth_path, iou_threshold):
    """
    Evaluate a single hyperparameter combination using segment performance.
    
    Parameters
    ----------
    segment_output_path : Path
        Path to the generated segments CSV
    ground_truth_path : Path
        Path to the ground truth segments CSV
    iou_threshold : float
        IoU threshold for evaluation
        
    Returns
    -------
    dict
        Evaluation metrics for this combination
    """
    try:
        # Load data
        predictions_df = pd.read_csv(segment_output_path)
        ground_truth_df = pd.read_csv(ground_truth_path, delimiter=';')
                
        # Run evaluation
        results = evaluate_performance(predictions_df, ground_truth_df, iou_threshold)
        
        # Calculate overall metrics directly from results
        if results:
            # Extract metrics for each class
            precisions = [metrics['precision'] for metrics in results.values()]
            recalls = [metrics['recall'] for metrics in results.values()]
            f1_scores = [metrics['f1_score'] for metrics in results.values()]
            
            # Calculate totals for micro averaging
            total_tp = sum(metrics['tp'] for metrics in results.values())
            total_fp = sum(metrics['fp'] for metrics in results.values())
            total_fn = sum(metrics['fn'] for metrics in results.values())
            
            # Micro averages
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            
            # Macro averages
            macro_precision = np.mean(precisions) if precisions else 0
            macro_recall = np.mean(recalls) if recalls else 0
            macro_f1 = np.mean(f1_scores) if f1_scores else 0
            
            overall_metrics = {
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'micro_precision': micro_precision,
                'micro_recall': micro_recall,
                'micro_f1': micro_f1,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            }
        else:
            overall_metrics = {}
        
        return {
            'success': True,
            'class_metrics': results,
            'overall_metrics': overall_metrics,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'class_metrics': None,
            'overall_metrics': None,
            'error': str(e)
        }

def main():
    """
    Main hyperparameter tuning pipeline.
    """        
    # Setup output directories
    output_base_dir = Inference.HYPERPARAMETER_OUTPUT_DIR
    output_base_dir.mkdir(exist_ok=True)
    
    # Generate hyperparameter combinations
    print("\n📊 Generating hyperparameter combinations...")
    combinations = generate_hyperparameter_combinations(
        max_combinations=InferenceConfig.MAX_COMBINATIONS_TUNING,
        random_sample=InferenceConfig.RANDOM_SAMPLING
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
        success, frame_output, segment_output, error = run_analysis_with_config(
            hyperparams, combo_id, output_base_dir
        )
        
        if not success:
            print(f"❌ Analysis failed: {error}")
            failed_runs += 1
            continue
        
        # Evaluate performance
        evaluation_result = evaluate_combination(
            segment_output, ground_truth_path, InferenceConfig.EVALUATION_IOU
        )
        
        if not evaluation_result['success']:
            print(f"❌ Evaluation failed: {evaluation_result['error']}")
            failed_runs += 1
            continue
        
        # Store results
        result_record = {
            'combo_id': combo_id,
            'hyperparameters': hyperparams,
            'evaluation': evaluation_result,
            'frame_output_path': str(frame_output),
            'segment_output_path': str(segment_output)
        }
        all_results.append(result_record)
        
        # Print key metrics
        overall = evaluation_result['overall_metrics']
        print(f"✅ Success! Macro F1: {overall['macro_f1']:.4f}, Micro F1: {overall['micro_f1']:.4f}")
        
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
    
    Parameters
    ----------
    results : list
        List of result records
        
    Returns
    -------
    dict
        Best configuration record
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
    
    Parameters
    ----------
    all_results : list
        All experiment results
    best_config : dict
        Best configuration found
    output_dir : Path
        Output directory
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
    
    Parameters
    ----------
    all_results : list
        All experiment results
    best_config : dict
        Best configuration found
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

if __name__ == "__main__":
    main()
