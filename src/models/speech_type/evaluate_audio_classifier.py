import json
import numpy as np
from datetime import datetime
from constants import AudioClassification
from utils import load_thresholds, load_model, get_tf_dataset, setup_gpu_config
from tqdm import tqdm
from config import AudioConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_fscore_support
from pathlib import Path # Ensure Path is imported for file manipulation

# Setup GPU configuration
gpu_available = setup_gpu_config()
if gpu_available:
    print("✅ GPU configuration completed successfully")
else:
    print("⚠️ GPU configuration failed, will use CPU")


def evaluate_and_report(model, test_dataset, mlb, thresholds, output_dir):
    """
    Generates model predictions for all test snippets, reports classification metrics,
    and generates the multi-label confusion matrices (absolute and percentage).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = mlb.classes_
    num_classes = len(classes)
    
    # --- 1. Generate Predictions and Collect GT Labels (Snippet Level) ---
    print("📊 Generating predictions and collecting true labels...")
    test_predictions = model.predict(test_dataset, verbose=1)
    
    test_true_labels = []
    # Collect all true labels from the dataset batches
    for batch in tqdm(test_dataset, desc="Collecting GT Labels"):
        _, labels = batch
        test_true_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)
    
    test_true_labels = np.array(test_true_labels)
    
    # Handle sample mismatch
    min_samples = min(test_predictions.shape[0], len(test_true_labels))
    test_predictions = test_predictions[:min_samples]
    test_true_labels = test_true_labels[:min_samples]
    
    # --- 2. Apply Thresholds (Per-Snippet Binary Prediction) ---
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[classes[i]]).astype(int)
        for i in range(num_classes)
    ]).T

    # --- 3. Save Predictions to File (NEW STEP) ---
    predictions_file = output_dir / "detailed_predictions.jsonl"
    print(f"\n💾 Saving detailed predictions to {predictions_file}...")
    
    # Load segment metadata (audio_path, start, duration) from the original file
    segment_metadata = []
    try:
        with open(AudioClassification.TEST_SEGMENTS_FILE, 'r') as f:
            for line in f:
                segment_metadata.append(json.loads(line.strip()))
    except Exception as e:
        print(f"⚠️ Warning: Could not load segment metadata from {AudioClassification.TEST_SEGMENTS_FILE}: {e}")
        # Create dummy metadata if file reading fails
        segment_metadata = [{'audio_path': 'unknown', 'start': i, 'duration': 0} for i in range(min_samples)]

    # Truncate metadata to match the number of predictions made
    segment_metadata = segment_metadata[:min_samples]

    records = []
    for i in range(min_samples):
        meta = segment_metadata[i]
        
        # Convert numpy arrays to lists for JSON serialization
        true_labels_list = [classes[j] for j, val in enumerate(test_true_labels[i]) if val == 1]
        pred_labels_list = [classes[j] for j, val in enumerate(test_pred_binary[i]) if val == 1]
        
        record = {
            'audio_path': meta.get('audio_path'),
            'start': meta.get('start'),
            'duration': meta.get('duration'),
            'labels_true': sorted(true_labels_list),
            'labels_pred_binary': sorted(pred_labels_list),
            'probabilities': test_predictions[i].tolist() # Raw model output
        }
        records.append(record)
    
    with open(predictions_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"✅ Detailed predictions saved.")
    
    # --- 4. Calculate Core Multi-Label Metrics (OvR Methodology) ---
    
    # Calculate Macro F1 (average F1 score per class)
    macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
    
    # Per-class detailed metrics (Precision, Recall, F1, Support)
    metrics_per_class = precision_recall_fscore_support(
        test_true_labels, test_pred_binary, average=None, zero_division=0
    )
    
    # Calculate Per-Class TP, FP, FN counts from the metrics (OvR methodology)
    support = metrics_per_class[3]
    recall = metrics_per_class[1]
    precision = metrics_per_class[0]

    # Calculate TP (rounded from recall * support)
    TP_counts_ovr = np.round(support * recall).astype(int)
    # Calculate FN
    FN_counts_ovr = support - TP_counts_ovr
    # Calculate FP (handle division by zero where Precision is zero)
    FP_counts_ovr = np.zeros_like(TP_counts_ovr)
    non_zero_precision_indices = precision > 0
    FP_counts_ovr[non_zero_precision_indices] = (
        TP_counts_ovr[non_zero_precision_indices] / precision[non_zero_precision_indices]
    ) - TP_counts_ovr[non_zero_precision_indices]
    FP_counts_ovr = np.round(FP_counts_ovr).astype(int)
    
    # --- 5. Generate Misclassification/Miss Matrix (ALIGNED TO F1) ---
    cm_miss = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(min_samples):
        for gt_idx in np.where(test_true_labels[i] == 1)[0]:
            
            # --- Diagonal (True Positives for this class) ---
            if test_pred_binary[i, gt_idx] == 1:
                cm_miss[gt_idx, gt_idx] += 1
            
            # --- Off-Diagonal (Misclassifications/False Negatives) ---
            else:
                # If the primary GT class (gt_idx) was MISSED (FN), 
                # count what spurious prediction (pred_idx) took its place.
                for pred_idx in np.where(test_pred_binary[i] == 1)[0]:
                    cm_miss[gt_idx, pred_idx] += 1
                
    # Calculate percentages (row-normalized: percentage of the true class's total support)
    cm_percent = cm_miss.astype(float)
    row_sums = cm_miss.sum(axis=1, keepdims=True)
    # Avoid division by zero
    cm_percent = np.divide(cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums!=0) * 100

    # --- 6. Save Metrics Summary to Text File ---
    metrics_file = output_dir / "classification_metrics_snippet_level.txt"
    with open(metrics_file, "w") as f:
        f.write("Snippet-Level Classification Metrics (Trained on GT Events):\n")
        f.write("="*60 + "\n")
        
        f.write(f"Total Snippets Evaluated: {min_samples}\n")
        f.write("\nOverall Metrics:\n")
        f.write(f"  Macro F1 Score (Average per class): {macro_f1:.4f}\n")
        
        f.write("\nPer-Class Metrics (One-vs-Rest, Validating F1):\n")
        f.write("Class\t\tPrecision\tRecall\t\tF1 Score\tSupport\t\tTP\tFP\tFN\n")
        f.write("-----\t\t---------\t------\t\t--------\t-------\t\t---\t---\t---\n")
        for i, cls in enumerate(classes):
            precision_val, recall_val, f1_val, support_val = metrics_per_class[0][i], metrics_per_class[1][i], metrics_per_class[2][i], metrics_per_class[3][i]
            tp, fp, fn = TP_counts_ovr[i], FP_counts_ovr[i], FN_counts_ovr[i]
            f.write(f"{cls:<10}\t{precision_val:.4f}\t{recall_val:.4f}\t{f1_val:.4f}\t\t{support_val}\t\t{tp}\t{fp}\t{fn}\n")
            
        f.write("\nConfusion Matrix (OvR-Aligned Misclassification/Miss Counts):\n")
        f.write("Diagonal is TRUE POSITIVES. Off-diagonals are where a GT label (row) was missed and predicted as a column label.\n")
        f.write("\t" + "\t".join(classes) + "\n")
        for i, gt_cls in enumerate(classes):
            f.write(f"{gt_cls}\t" + "\t".join(map(str, cm_miss[i])) + "\n")
            
    print(f"\n✅ Metrics report saved to: {metrics_file}")
    print("\nFINAL SNIPPET-LEVEL CLASSIFICATION RESULTS:")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # --- 7. Plot Confusion Matrices ---
    
    # Plot absolute confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_miss, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Misclassification/Miss CM (Absolute Counts)")
    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_confusion_matrix_absolute.png")
    plt.close()

    # Plot percentage confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Misclassification/Miss CM (Row-Normalized Percentages)")
    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_confusion_matrix_percent.png")
    plt.close()

    print(f"✅ Confusion matrices saved to {output_dir}")
    return metrics_file


# --- ADJUSTED MAIN FUNCTION ---
def main():
    try:
        print("🚀 Starting Comprehensive Audio Classification Model Evaluation (Snippet-Level)")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = AudioClassification.TRAINED_WEIGHTS_PATH.parent
        folder_name = output_dir / f'evaluation_{timestamp}'

        model, mlb = load_model()
        thresholds = load_thresholds(mlb.classes_)
        # Save thresholds to evaluation folder
        thresholds_file = folder_name / "used_thresholds.json"
        folder_name.mkdir(parents=True, exist_ok=True)
        with open(thresholds_file, "w") as tf:
            json.dump(thresholds, tf, indent=2)
        print(f"✅ Thresholds saved to: {thresholds_file}")

        # --- ADJUSTMENT: Load the new GT-event snippet file ---
        test_cache_dir = AudioClassification.CACHE_DIR / "test" # Using "test" cache folder
        # get_tf_dataset is called without seconds_step=True, ensuring it loads the new GT-event segments
        test_dataset = get_tf_dataset(AudioClassification.TEST_SEGMENTS_FILE, mlb, test_cache_dir, batch_size=32, shuffle=False)

        if test_dataset is None or not any(True for _ in test_dataset):
            raise ValueError("Test dataset is empty. Check test data file and paths.")

        # --- ADJUSTMENT: Call the new evaluation function ---
        evaluate_and_report(
            model=model,
            test_dataset=test_dataset,
            mlb=mlb,
            thresholds=thresholds,
            output_dir=folder_name
        )

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        raise
    except ValueError as e:
        print(f"❌ Configuration or data error: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected evaluation error: {e}")
        raise


if __name__ == "__main__":
    main()