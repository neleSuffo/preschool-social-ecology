"""
Hyperparameter Tuning for Social Interaction Analysis

This script systematically tests different combinations of hyperparameters
for the social interaction analysis pipeline and evaluates their performance
against ground truth data. It supports selective processing of video subsets
to facilitate cross-validation and prevent data leakage.
"""

import pandas as pd
import sys
import random
import json
import time
import random
import argparse 
from datetime import datetime
from pathlib import Path
from itertools import product

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from constants import DataPaths, Analysis, Inference
from config import AnalysisConfig, HyperparameterConfig
from heuristics.validation.eval_segment_performance import run_evaluation
from heuristics.pipeline_frame_level_analysis import main as frame_analysis_main
from heuristics.pipeline_video_level_analysis import main as segment_analysis_main


def generate_hyperparameter_combinations(max_combinations=20, 
                                         random_sample=True):
    """
    Generates a list of hyperparameter combinations to evaluate. 
    
    Parameters
    ----------
    max_combinations: int
        Maximum number of combinations to generate. If random_sample is False, this is ignored and all combinations are generated.
    random_sample: bool
        If True, generates a random sample of combinations. If False, generates the full Cartesian product of the hyperparameter ranges.
        
    Returns    
    -------
    list of dict
        Each dict contains a unique combination of hyperparameters.
    """
    ranges = HyperparameterConfig.HYPERPARAMETER_RANGES
    param_names = list(ranges.keys())
    
    # If you aren't doing a random sample and want the full grid, 
    # only then should you use the product (and only if the grid is small).
    if not random_sample:
        # Standard Cartesian product (only use if you know the grid is small!)
        from itertools import product
        all_combos = [dict(zip(param_names, c)) for c in product(*[ranges[p] for p in param_names])]
        return all_combos[:max_combinations] if max_combinations else all_combos

    # --- MEMORY EFFICIENT RANDOM SEARCH ---
    combinations = []
    seen = set() # To ensure uniqueness
    
    # Safety break to avoid infinite loops if max_combos > possible combos
    total_possible = 1
    for r in ranges.values(): total_possible *= len(r)
    limit = min(max_combinations, total_possible)

    while len(combinations) < limit:
        # Pick one random value for every parameter
        combo = {p: random.choice(ranges[p]) for p in param_names}
        
        # Convert to a frozenset of items to make it hashable for the 'seen' check
        combo_frozen = tuple(sorted(combo.items()))
        
        if combo_frozen not in seen:
            seen.add(combo_frozen)
            combinations.append(combo)
            
    return combinations

def run_pipeline_for_combo(hyperparameters: dict,
                           combo_dir: Path, 
                           social_state_mode: str,
                           hyperparameter_tuning=False,
                           video_list=None):
    """
    Runs the full analysis pipeline by temporarily setting AnalysisConfig 
    attributes and executing the analysis functions.
    
    Parameters
    ----------
    hyperparameters: dict
        Hyperparameters to set for this run.
    combo_dir: Path
        Directory where outputs for this combination will be stored.
    social_state_mode: str
        tertiary or binary evaluation mode (affects evaluation metrics and thresholds)
    hyperparameter_tuning: bool
        Whether this run is part of hyperparameter tuning (affects configuration settings).
    video_list: list, optional
        Specific videos to process. If provided, Step 1 will only query these videos.
    """
    original_config = {}

    try:
        # 1. Temporarily override AnalysisConfig attributes
        for key, value in hyperparameters.items():
            if hasattr(AnalysisConfig, key):
                original_config[key] = getattr(AnalysisConfig, key)
                setattr(AnalysisConfig, key, value)
                
        frame_output_path = combo_dir / Analysis.FRAME_LEVEL_INTERACTIONS_CSV.name
        segment_output_path = combo_dir / Analysis.INTERACTION_SEGMENTS_CSV.name
        
        # 2. Execute Frame-Level Analysis (Passing the video filter)
        frame_analysis_main(
            db_path=DataPaths.INFERENCE_DB_PATH,
            output_dir=combo_dir,
            video_list=video_list,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning
        )

        # 3. Execute Segment-Level Analysis
        segment_analysis_main(
            output_file_path=segment_output_path, 
            frame_data_path=frame_output_path,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning
        )
        
        return True, frame_output_path, segment_output_path, None

    except Exception as e:
        return False, None, None, f"Pipeline execution failed: {str(e)}"
        
    finally:
        # 4. Reset AnalysisConfig to original state
        for key, value in original_config.items():
            setattr(AnalysisConfig, key, value)

def run_analysis_with_config(hyperparameters: dict,
                             combo_id: int,
                             output_base_dir: Path,
                             social_state_mode: str,
                             hyperparameter_tuning: bool,
                             video_list=None):
    """
    Wrapper function to manage directory creation and pipeline execution for a combination.
    
    Parameters
    ----------
    hyperparameters: dict
        Hyperparameters to set for this run.
    combo_id: int
        Unique identifier for this combination (used for directory naming).
    output_base_dir: Path
        Base directory where results for this combination will be stored.
    social_state_mode: str
        tertiary or binary evaluation mode (affects evaluation metrics and thresholds)
    hyperparameter_tuning: bool
        Whether this run is part of hyperparameter tuning (affects configuration settings).
    video_list: list, optional
        Specific videos to process. If provided, Step 1 will only query these videos.
        
    Returns
    -------
    success: bool
        Whether the pipeline ran successfully.
    frame_output_path: Path or None
        Path to frame-level output CSV if successful, else None.
    segment_output_path: Path or None
        Path to segment-level output CSV if successful, else None.
    error: str or None
        Error message if failed, else None.
    """
    combo_dir = output_base_dir / f"combo_{combo_id:04d}"
    combo_dir.mkdir(exist_ok=True, parents=True)
    
    with open(combo_dir / "hyperparameters.json", 'w') as f:
        json.dump(hyperparameters, f, indent=2)
        
    # Run the pipeline with the specified hyperparameters and video filter
    success, frame_out, seg_out, error = run_pipeline_for_combo(
        hyperparameters, combo_dir, social_state_mode, hyperparameter_tuning=hyperparameter_tuning, video_list=video_list
    )
        
    return success, frame_out, seg_out, error

def evaluate_combination(segment_output_path: Path, 
                         social_state_mode: str,
                         video_list=None):
    """
    Evaluates the performance of a specific hyperparameter combination by comparing
    the generated segments against ground truth annotations for the specified videos.
    
    Parameters    
    ----------
    segment_output_path: Path
        Path to the CSV file containing the predicted interaction segments for this combination.
    social_state_mode: str
        "tertiary" or "binary" evaluation mode (affects evaluation metrics and thresholds).
    video_list: list, optional
        Specific videos to evaluate. If None, evaluates on all videos in the segment_output_path.
        
    Returns
    -------
    dict containing:
        success: bool
            Whether the evaluation ran successfully.
        detailed_metrics: dict or None
            Dictionary of overall performance metrics (e.g., macro_avg_f1_score) if successful, else None.
        error: str or None
            Error message if evaluation failed, else None.
    """
    try:
        _, _, detailed_metrics = run_evaluation(
            predictions_path=segment_output_path, 
            output_folder=segment_output_path.parent,
            mode=social_state_mode, 
            video_list=video_list
        )

        if detailed_metrics:
            # We return the whole dictionary now so the CV script can see all metrics
            return {'success': True, 'detailed_metrics': detailed_metrics, 'error': None}
        
        return {'success': False, 'detailed_metrics': None, 'error': "No metrics returned"}
    except Exception as e:
        return {'success': False, 'detailed_metrics': None, 'error': str(e)}

def find_best_configuration(results: list):
    """
    Finds the configuration with the highest Macro F1 score.
    
    Parameters
    ----------
    results: list of dict
        Each dict should contain 'combo_id', 'hyperparameters', and 'evaluation' with '
        detailed_metrics' that includes 'macro_f1'.
        
    Returns
    -------
    dict
        The result dict corresponding to the best configuration.
    """
    return max(results, key=lambda x: x['evaluation']['detailed_metrics']['macro_avg']['f1_score'])

def save_final_results(all_results: list,
                       best_config: dict,
                       output_dir: Path):
    """
    Saves result summary CSV and best configuration JSON.
    
    Parameters
    ----------
    all_results: list of dict
        List of all evaluated combinations with their hyperparameters and evaluation metrics.
    best_config: dict
        The best configuration found.
    output_dir: Path
        Directory where the results will be saved.
    """
    summary_data = []
    for res in all_results:
        metrics = res['evaluation']['detailed_metrics']
        
        # Safely grab kappa from macro_avg or top-level detailed_metrics
        kappa_val = metrics.get('overall_kappa', metrics.get('macro_avg', {}).get('kappa', None))
        
        row = {
            'combo_id': res['combo_id'],
            'macro_f1': res['evaluation']['detailed_metrics']['macro_avg']['f1_score'],
            'cohen_kappa': kappa_val,
            **res['hyperparameters']
        }
        summary_data.append(row)
    
    pd.DataFrame(summary_data).to_csv(output_dir / "results_summary.csv", index=False)
    with open(output_dir / "best_configuration.json", 'w') as f:
        json.dump(best_config, f, indent=2)

def print_results_summary(all_results: list, 
                          best_config: dict):
    """
    Prints best performance metrics to console.
    
    Parameters
    ----------
    all_results: list of dict
        List of all evaluated combinations with their hyperparameters and evaluation metrics.
    best_config: dict
        The best configuration found.
    """
    metrics = best_config['evaluation']['detailed_metrics']
    f1_val = metrics['macro_avg']['f1_score']
    kappa_val = metrics.get('overall_kappa', metrics.get('macro_avg', {}).get('kappa', 0.0))
    
    print("\n" + "=" * 30 + "\n🏆 WINNER: Combo", best_config['combo_id'])
    for k, v in best_config['hyperparameters'].items():
        print(f"  {k}: {v}")
    print(f"  F1: {best_config['evaluation']['detailed_metrics']['macro_avg']['f1_score']:.4f}")
    
def main(max_combinations=None, 
         video_list=None, 
         social_state_mode="tertiary",
         hyperparameter_tuning=True):
    """
    Main hyperparameter tuning loop.
    
    Parameters
    ----------
    max_combinations: int or None
        Maximum number of hyperparameter combinations to evaluate. If None, evaluates all possible combinations.
    video_list: list or None
        Optional list of specific video filenames to evaluate (e.g. ['video1.mp4', 'video2.mp4']). If None, evaluates on all videos in the segment output.
    social_state_mode: str
        "tertiary" or "binary" evaluation mode (affects evaluation metrics and thresholds).
    hyperparameter_tuning: bool
        Whether this run is part of hyperparameter tuning (affects configuration settings and logging).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = Path(f"{Inference.HYPERPARAMETER_OUTPUT_DIR}_{timestamp}")
    output_base_dir.mkdir(exist_ok=True, parents=True)
    
    combinations = generate_hyperparameter_combinations(
        max_combinations=max_combinations, 
        random_sample=getattr(AnalysisConfig, 'RANDOM_SAMPLING', True)
    )
    
    all_results = []
    
    for i, hyperparams in enumerate(combinations):
        combo_id = i + 1
        print(f"\n[{combo_id}/{len(combinations)}] Testing Combo {combo_id}")
        
        # Run Analysis
        success, _, seg_out, error = run_analysis_with_config(
            hyperparameters=hyperparams,
            combo_id=combo_id,
            output_base_dir=output_base_dir,
            social_state_mode=social_state_mode,
            hyperparameter_tuning=hyperparameter_tuning,
            video_list=video_list)
        
        if not success:
            print(f"❌ Failed: {error}")
            continue
        
        # Evaluate performance for the specific video subset
        eval_res = evaluate_combination(seg_out, social_state_mode=social_state_mode, video_list=video_list)
        
        if eval_res['success']:
            result_record = {
                'combo_id': combo_id,
                'hyperparameters': hyperparams,
                'evaluation': eval_res
            }
            all_results.append(result_record)
            print(f"✅ Macro F1: {eval_res['detailed_metrics']['macro_avg']['f1_score']:.4f}")

    # Final Summaries
    if all_results:
        best_config = find_best_configuration(all_results)
        save_final_results(all_results, best_config, output_base_dir)
        print_results_summary(all_results, best_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-combos', type=int, default=20)
    parser.add_argument('--video_list', type=str, nargs='+', default=None)
    parser.add_argument('--social_state_mode', type=str, default='tertiary', choices=['binary', 'tertiary'])
    args = parser.parse_args()
    
    main(max_combinations=args.max_combos, video_list=args.video_list, social_state_mode=args.social_state_mode)