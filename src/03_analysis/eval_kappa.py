import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

# Add the src directory to path for imports
src_path = Path(__file__).parent.parent.parent if '__file__' in globals() else Path.cwd().parent.parent
sys.path.append(str(src_path))

from constants import Evaluation, Inference
from config import InferenceConfig
from results.utils import time_to_seconds, create_second_level_labels

# Label for unannotated or missing model output
UNCLASSIFIED_LABEL = 'unclassified'

def load_and_clean_data(file_path: Path, is_gt: bool = False) -> pd.DataFrame:
    """Loads CSV, handles delimiters, and ensures time columns are in seconds."""
    sep = ';' if is_gt else ','
    df = pd.read_csv(file_path, sep=sep).dropna(how='all')
    df.columns = df.columns.str.strip()
    
    # Strip whitespace from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip()

    # Time conversion: if min columns exist, convert to sec
    if 'start_time_min' in df.columns:
        df['start_time_sec'] = df['start_time_min'].apply(time_to_seconds)
        df['end_time_sec'] = df['end_time_min'].apply(time_to_seconds)
    
    df['interaction_type'] = df['interaction_type'].str.lower()
    return df

def generate_second_wise_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Converts segment DataFrame into a second-wise flat DataFrame."""
    all_data = []
    for video_name, video_df in df.groupby('video_name'):
        max_end = int(video_df['end_time_sec'].max()) + 1
        labels = create_second_level_labels(video_df, max_end)
        
        for sec, label in enumerate(labels):
            all_data.append({
                'video_name': video_name,
                'second': sec,
                'interaction_type': str(label).lower() if label is not None else UNCLASSIFIED_LABEL
            })
    return pd.DataFrame(all_data)

def run_evaluation(pred_path: Path, gt_path: Path, output_dir: Path):
    """Core evaluation logic: Aligns data, calculates Kappa/Accuracy, and saves outputs."""
    print(f"📊 Loading GT: {gt_path.name} | PRED: {pred_path.name}")
    
    df_pred = load_and_clean_data(pred_path, is_gt=False)
    df_gt = load_and_clean_data(gt_path, is_gt=True)

    y_true_all = []
    y_pred_all = []
    
    common_videos = set(df_gt['video_name'].unique()) & set(df_pred['video_name'].unique())
    
    for video in common_videos:
        v_gt = df_gt[df_gt['video_name'] == video]
        v_pred = df_pred[df_pred['video_name'] == video]
        
        duration = int(min(v_gt['end_time_sec'].max(), v_pred['end_time_sec'].max()))
        
        # Apply exclusion buffer
        start_eval = InferenceConfig.EXCLUSION_SECONDS
        end_eval = duration - InferenceConfig.EXCLUSION_SECONDS
        
        if end_eval <= start_eval:
            continue

        labels_gt = create_second_level_labels(v_gt, duration)
        labels_pred = create_second_level_labels(v_pred, duration)

        for sec in range(start_eval, end_eval):
            if labels_gt[sec] is not None:
                y_true_all.append(str(labels_gt[sec]).lower())
                y_pred_all.append(str(labels_pred[sec]).lower() if labels_pred[sec] else UNCLASSIFIED_LABEL)

    # --- 1. Statistics Calculation ---
    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    
    # Calculate Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Calculate Overall Accuracy (%)
    correct_matches = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    accuracy_pct = (correct_matches / total_samples) * 100 if total_samples > 0 else 0
    
    report = classification_report(y_true, y_pred)
    
    # --- 2. Saving Second-wise Files ---
    output_dir.mkdir(parents=True, exist_ok=True)
    sw_gt = generate_second_wise_labels(df_gt)
    sw_pred = generate_second_wise_labels(df_pred)
    
    sw_gt.to_csv(output_dir / "gt_secondwise_labels.csv", index=False)
    sw_pred.to_csv(output_dir / "pred_secondwise_labels.csv", index=False)

    # --- 3. Output to Console ---
    print("\n" + "="*40)
    print(f"OVERALL ACCURACY:   **{accuracy_pct:.2f}%**")
    print(f"COHEN'S KAPPA SCORE: **{kappa:.4f}**")
    print("="*40)
    
    # Interpretation
    if kappa > 0.80: print("Interpretation: Almost Perfect Agreement")
    elif kappa > 0.60: print("Interpretation: Substantial Agreement")
    elif kappa > 0.40: print("Interpretation: Moderate Agreement")
    else: print("Interpretation: Fair to Poor Agreement")
    
    print("\nDetailed Classification Report:")
    print(report)

    # --- 4. Save Text Summary ---
    with open(output_dir / "kappa_evaluation_summary.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy_pct:.2f}%\n")
        f.write(f"Kappa Score: {kappa:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Second-Wise Kappa for Model Evaluation")
    parser.add_argument('--folder_path', type=str, required=True, help='Path containing 01_interaction_segments.csv')
    
    args = parser.parse_args()
    
    PRED_FILE = Path(args.folder_path) / Inference.INTERACTION_SEGMENTS_CSV.name
    GT_FILE = Evaluation.GROUND_TRUTH_SEGMENTS_CSV
    
    run_evaluation(PRED_FILE, GT_FILE, Path(args.folder_path))