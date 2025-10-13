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

    # --- 3. Calculate Core Multi-Label Metrics (Per-Class TP/FP/FN/TN) ---
    # Calculate Macro F1 (average F1 score per class)
    macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
    
    # Per-class detailed metrics (Precision, Recall, F1, Support)
    metrics_per_class = precision_recall_fscore_support(
        test_true_labels, test_pred_binary, average=None, zero_division=0
    )
    
    # --- 4. Generate Multi-Label Confusion Matrix (Co-occurrence) ---    
    # Initialize CM as a C x C matrix
    cm_counts = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(min_samples):
        # Indices where the true label is 1
        true_indices = np.where(test_true_labels[i] == 1)[0]
        # Indices where the predicted label is 1
        pred_indices = np.where(test_pred_binary[i] == 1)[0]
        
        # Populate confusion matrix based on co-occurrence
        for gt_idx in true_indices:
            for pred_idx in pred_indices:
                cm_counts[gt_idx, pred_idx] += 1
    
    # Calculate percentages (row-normalized: percentage of the true class's total support)
    cm_percent = cm_counts.astype(float)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    cm_percent = np.divide(cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums!=0) * 100

    # --- 5. Save Metrics Summary to Text File ---
    metrics_file = output_dir / "classification_metrics_snippet_level.txt"
    with open(metrics_file, "w") as f:
        f.write("Snippet-Level Classification Metrics (Trained on GT Events):\n")
        f.write("="*60 + "\n")
        
        f.write(f"Total Snippets Evaluated: {min_samples}\n")
        f.write("\nOverall Metrics:\n")
        f.write(f"  Macro F1 Score (Average per class): {macro_f1:.4f}\n")
        
        f.write("\nPer-Class Metrics:\n")
        f.write("Class\t\tPrecision\tRecall\t\tF1 Score\tSupport\n")
        f.write("-----\t\t---------\t------\t\t--------\t-------\n")
        for i, cls in enumerate(classes):
            precision, recall, f1, support = metrics_per_class[0][i], metrics_per_class[1][i], metrics_per_class[2][i], metrics_per_class[3][i]
            f.write(f"{cls:<10}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t\t{support}\n")
            
        f.write("\nConfusion Matrix (Absolute Counts):\n")
        f.write("\t" + "\t".join(classes) + "\n")
        for i, gt_cls in enumerate(classes):
            f.write(f"{gt_cls}\t" + "\t".join(map(str, cm_counts[i])) + "\n")
            
    print(f"\n✅ Metrics report saved to: {metrics_file}")
    print("\nFINAL SNIPPET-LEVEL CLASSIFICATION RESULTS:")
    print(f"Macro F1 Score: {macro_f1:.4f}")

    # --- 6. Plot Confusion Matrices ---
    
    # Plot absolute confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Multi-Label Confusion Matrix (Absolute Counts)")
    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_confusion_matrix_absolute.png")
    plt.close()

    # Plot percentage confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Multi-Label Confusion Matrix (Row-Normalized Percentages)")
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

        test_cache_dir = AudioClassification.CACHE_DIR / "test" 
        test_dataset = get_tf_dataset(AudioClassification.TEST_SEGMENTS_FILE, mlb, test_cache_dir, batch_size=32, shuffle=False)

        if test_dataset is None or not any(True for _ in test_dataset):
            raise ValueError("Test dataset is empty. Check test data file and paths.")

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