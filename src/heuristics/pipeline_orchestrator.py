import argparse
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupKFold
from datetime import datetime
from tune_hyperparameters import generate_hyperparameter_combinations, run_analysis_with_config, evaluate_combination
from constants import Analysis
from config import AnalysisConfig
from utils import extract_child_id

def get_folds():
    """
    Splits all available annotated videos into 5 groups by Child ID.
    
    Returns:
    --------
    folds: list of tuples
        List of (train_indices, test_indices) for each fold
    df: pandas DataFrame
        DataFrame containing video names and corresponding Child IDs
    """
    # Load video list and extract Child IDs
    with open(Analysis.QUANTEX_VIDEOS_LIST_FILE, "r") as f:
        videos = [line.strip() for line in f if line.strip()]

    df = pd.DataFrame({"video_name": videos})
    df["child_id"] = df["video_name"].apply(lambda x: extract_child_id(x) or "unknown")
    
    # Split into folds using GroupKFold to ensure all videos of the same child are in the same fold
    gkf = GroupKFold(n_splits=AnalysisConfig.NUM_FOLDS)
    return list(gkf.split(df, groups=df['child_id'])), df

def run_cross_validation(mode="validation", social_state_mode="tertiary", max_combos=20):  
    """
    Runs NUM_FOLDS-fold cross-validation for the analysis pipeline.
    
    Parameters:
    -----------
    mode: str
        "validation" for standard CV, "tuning" for hyperparameter tuning (evaluates on a smaller subset of combinations)
    social_state_mode: str
        "tertiary" or "binary" evaluation mode (affects evaluation metrics and thresholds)
    max_combos: int
        Maximum number of hyperparameter combinations to evaluate during tuning (only applicable in "tuning")
    """
    # get correct mode applied to config (tertiary vs binary) for the rest of the pipeline and evaluation
    AnalysisConfig.apply_mode(social_state_mode)
    
    folds, df = get_folds()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Analysis.BASE_OUTPUT_DIR / f"cv_{mode}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    fold_results = []

    def flatten_metrics(detailed_metrics):
        flat = {}
        for category, scores in detailed_metrics.items():
            cat_clean = category.lower().replace(" ", "_")

            # If scores is a float/scalar (e.g. overall_kappa = 0.48)
            if not isinstance(scores, dict):
                flat[cat_clean] = scores
                continue

            # If scores is a dict (e.g. 'interacting': {'precision': ..., 'recall': ...})
            for metric, value in scores.items():
                metric_clean = metric.replace("_score", "")
                flat[f"{cat_clean}_{metric_clean}"] = value
        return flat

    # --- CROSS-VALIDATION LOOP ---
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id = i + 1
        train_videos = df.iloc[train_idx]['video_name'].tolist()
        test_videos = df.iloc[test_idx]['video_name'].tolist()
        
        print(f"\n🚀 FOLD {fold_id}/{AnalysisConfig.NUM_FOLDS}")

        if mode == "tuning":
            # --- NESTED TUNING: Find best params using ONLY Train Videos ---
            print(f"🔎 Phase 1: Tuning on {len(train_videos)} training videos...")
            
            # Generate combinations (Random Search)
            combos = generate_hyperparameter_combinations(max_combinations=max_combos, random_sample=True)
            
            best_f1 = -1
            best_params = None
            
            tuning_dir = output_root / f"fold_{fold_id}_tuning_search"

            for c_idx, combo in enumerate(combos):
                # Run the pipeline on TRAINING videos only
                success, _, seg_path, error = run_analysis_with_config(
                    combo, c_idx, tuning_dir, 
                    social_state_mode=social_state_mode,
                    hyperparameter_tuning=True,
                    video_list=train_videos
                )
                
                if success:
                    # Score the combo on TRAINING videos
                    res = evaluate_combination(seg_path, video_list=train_videos, social_state_mode=social_state_mode)
                    current_f1 = res['detailed_metrics']['macro_avg'].get('f1_score', 0)
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_params = combo
            
            print(f"✅ Best Train F1: {best_f1:.4f}. Now validating on unseen test videos...")

            # --- PHASE 2: Evaluate 'Winner' on TEST Videos ---
            success, _, seg_path, error = run_analysis_with_config(
                best_params, fold_id, output_root / f"fold_{fold_id}_final_test", 
                social_state_mode=social_state_mode,
                hyperparameter_tuning=True,
                video_list=test_videos
            )
            
            if success:
                final_res = evaluate_combination(seg_path, video_list=test_videos, social_state_mode=social_state_mode)
                fold_results.append(flatten_metrics(final_res['detailed_metrics']))
        else:
            print(f"📊 Validating on: {len(test_videos)} videos")

            if social_state_mode == "binary":
                print("⚠️ Running in BINARY mode: Social states will be evaluated as 'Interacting' vs 'Not Interacting' (Available + Alone combined).")
                AnalysisConfig.apply_mode("binary")
            else:
                AnalysisConfig.apply_mode("tertiary")

            # now read all active parameters
            current_params = {
                item: getattr(AnalysisConfig, item)
                for item in dir(AnalysisConfig)
                if not item.startswith("__")
                and not callable(getattr(AnalysisConfig, item))
            }
            
            success, _, seg_path, error = run_analysis_with_config(
                current_params, 
                fold_id, 
                output_root, 
                social_state_mode=social_state_mode,
                hyperparameter_tuning=False,
                video_list=test_videos
            )
            
            if success:
                # 2. Evaluate
                eval_res = evaluate_combination(seg_path, video_list=test_videos, social_state_mode=social_state_mode)
                if eval_res['success']:
                    # Flatten detailed metrics into the fold results
                    flat_row = flatten_metrics(eval_res['detailed_metrics'])
                    fold_results.append(flat_row)
                else:
                    print(f"⚠️ Eval failed for fold {fold_id}: {eval_res['error']}")
            else:
                print(f"❌ Pipeline failed for fold {fold_id}: {error}")
            
    # --- AGGREGATE RESULTS ---
    if not fold_results:
        print("❌ No results collected. Check if database query returned 0 rows.")
        return

    results_df = pd.DataFrame(fold_results)
    print("\n" + "="*40 + "\n🏁 SUMMARY\n" + "="*40)
    
    # print cv summary and store results
    stats = results_df.describe()

    if not stats.empty and "mean" in stats.index:
        summary_rows = stats.loc[["mean", "std", "min", "max"]]
        print(summary_rows)

        # Append summary rows to fold results
        output_df = pd.concat([results_df, summary_rows])
    else:
        print(results_df)
        output_df = results_df

    # also add what videos were in each fold for reference
    fold_video_info = []
    for i, (train_idx, test_idx) in enumerate(folds):
        fold_id = i + 1
        test_videos = df.iloc[test_idx]['video_name'].tolist()
        fold_video_info.append({
            "fold_id": fold_id,
            "num_test_videos": len(test_videos),
            "test_videos": ", ".join(test_videos)
        })
    fold_video_df = pd.DataFrame(fold_video_info)
    fold_video_df.to_csv(output_root / "fold_video_info.csv", index=False)
    output_df.to_csv(output_root / "cv_summary.csv", index=True)
    
    # save copy of pipeline_frame_level_analysis.py and pipeline_video_level_analysis.py for reference
    shutil.copy(Analysis.FRAME_ANALYSIS_SCRIPT, output_root / Analysis.FRAME_ANALYSIS_SCRIPT.name)
    shutil.copy(Analysis.SEGMENT_CREATION_SCRIPT, output_root / Analysis.SEGMENT_CREATION_SCRIPT.name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_combos', type=int, default=20, help='Maximum number of hyperparameter combinations to evaluate during tuning (only applicable in "tuning" mode)')
    parser.add_argument('--social_state_mode', type=str, default='tertiary', help='Evaluation mode for social state analysis')
    parser.add_argument('--mode', type=str, default='validation', choices=['validation', 'tuning'], help='Whether to run standard cross-validation or hyperparameter tuning')
    args = parser.parse_args()
    
    run_cross_validation(mode=args.mode, social_state_mode=args.social_state_mode, max_combos=args.max_combos)