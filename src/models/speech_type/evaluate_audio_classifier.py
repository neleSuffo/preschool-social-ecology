import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from constants import AudioClassification
from utils import load_thresholds, load_model, create_data_generators, setup_gpu_config
from tqdm import tqdm
from config import AudioConfig

import matplotlib.pyplot as plt
import seaborn as sns

# Setup GPU configuration
gpu_available = setup_gpu_config()
if gpu_available:
    print("✅ GPU configuration completed successfully")
else:
    print("⚠️ GPU configuration failed, will use CPU")


def evaluate_and_save_predictions(model, test_generator, mlb, thresholds, output_dir):
    """
    Generate model predictions for all test samples, apply thresholds,
    and save the results to a JSONL file for per-second analysis.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / "second_level_predictions.jsonl"
    print("📊 Generating predictions and saving to file...")

    test_true_labels = []
    test_metadata = []

    for i in tqdm(range(len(test_generator)), desc="Processing batches"):
        _, labels = test_generator[i]
        if len(labels) > 0:
            test_true_labels.extend(labels)
            batch_segments = test_generator.segments_data[
                i * test_generator.batch_size:(i + 1) * test_generator.batch_size
            ]
            for segment in batch_segments[:len(labels)]:
                test_metadata.append({
                    'start_time': segment.get('second', 0.0),
                    'audio_path': segment.get('audio_path', 'unknown')
                })

    test_predictions = model.predict(test_generator, verbose=1)

    min_samples = min(test_predictions.shape[0], len(test_true_labels))
    test_predictions = test_predictions[:min_samples]
    test_true_labels = np.array(test_true_labels[:min_samples])
    test_metadata = test_metadata[:min_samples]

    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[mlb.classes_[i]]).astype(int)
        for i in range(len(mlb.classes_))
    ]).T

    per_second_data = defaultdict(lambda: {'true': set(), 'pred': set(), 'audio_path': None})

    for i in range(min_samples):
        metadata = test_metadata[i]
        start_time = metadata['start_time']
        audio_path = metadata['audio_path']
        second = round(start_time)

        true_labels_indices = np.where(test_true_labels[i] == 1)[0]
        for idx in true_labels_indices:
            per_second_data[audio_path, second]['true'].add(mlb.classes_[idx])

        pred_labels_indices = np.where(test_pred_binary[i] == 1)[0]
        for idx in pred_labels_indices:
            per_second_data[audio_path, second]['pred'].add(mlb.classes_[idx])

        per_second_data[audio_path, second]['audio_path'] = audio_path

    with open(predictions_file, 'w') as f:
        for (audio_path, second), data in sorted(per_second_data.items()):
            record = {
                "audio_path": data['audio_path'],
                "second": second,
                "true_labels": sorted(list(data['true'])),
                "predicted_labels": sorted(list(data['pred'])),
            }
            f.write(json.dumps(record) + '\n')

    print(f"✅ Predictions saved to: {predictions_file}")
    return predictions_file


def compute_metrics_from_predictions(predictions_file, output_dir):
    """
    Evaluate multilabel per-second predictions per class and generate confusion matrix.
    Empty GT/prediction = 'silence'.
    """
    classes = AudioConfig.VALID_RTTM_CLASSES
    all_classes = classes + ["silence"]

    # Initialize per-class counts
    counts = {cls: dict(TP=0, FP=0, FN=0, TN=0) for cls in classes}

    # Initialize confusion matrix
    cm = {gt: {pred: 0 for pred in all_classes} for gt in all_classes}

    with open(predictions_file, "r") as f:
        for line in f:
            record = json.loads(line)
            gt_labels = record["true_labels"] or []
            pred_labels = record["predicted_labels"] or []

            # If empty, consider as silence
            if not gt_labels:
                gt_labels = ["silence"]
            if not pred_labels:
                pred_labels = ["silence"]

            # Update per-class TP/FP/FN/TN
            for cls in classes:
                gt_val = 1 if cls in gt_labels else 0
                pred_val = 1 if cls in pred_labels else 0

                if gt_val == 1 and pred_val == 1:
                    counts[cls]["TP"] += 1
                elif gt_val == 1 and pred_val == 0:
                    counts[cls]["FN"] += 1
                elif gt_val == 0 and pred_val == 1:
                    counts[cls]["FP"] += 1
                else:
                    counts[cls]["TN"] += 1

            # Update confusion matrix
            for gt in gt_labels:
                for pred in pred_labels:
                    if gt not in all_classes:
                        gt = "silence"
                    if pred not in all_classes:
                        pred = "silence"
                    cm[gt][pred] += 1

    # Save metrics and counts to text file
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics_summary.txt"
    with open(metrics_file, "w") as f:
        f.write("Per-class metrics:\n")
        f.write("="*40 + "\n")
        for cls in classes:
            TP, FP, FN, TN = counts[cls]["TP"], counts[cls]["FP"], counts[cls]["FN"], counts[cls]["TN"]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            f.write(f"{cls}: TP={TP}, FP={FP}, FN={FN}, TN={TN}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}\n")

        f.write("\nConfusion matrix (GT rows, Pred cols):\n")
        f.write("\t" + "\t".join(all_classes) + "\n")
        for gt in all_classes:
            f.write(f"{gt}\t" + "\t".join(str(cm[gt][pred]) for pred in all_classes) + "\n")

    # Plot confusion matrix (absolute)
    cm_array = np.array([[cm[gt][pred] for pred in all_classes] for gt in all_classes])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Multilabel Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_confusion_matrix.png")
    plt.close()

    # Plot confusion matrix (percentages, row-normalized)
    cm_percent = cm_array.astype(float)
    row_sums = cm_percent.sum(axis=1, keepdims=True)
    cm_percent = np.divide(cm_percent, row_sums, out=np.zeros_like(cm_percent), where=row_sums!=0)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot=True, fmt=".2%", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Multilabel Confusion Matrix (Percentages)")
    plt.tight_layout()
    plt.savefig(output_dir / "multilabel_confusion_matrix_percent.png")
    plt.close()

    # Custom confusion matrix logic to match per-class metrics
    cm_custom = {gt: {pred: 0 for pred in all_classes} for gt in all_classes}
    silence_label = "silence"
    with open(predictions_file, "r") as f:
        for line in f:
            record = json.loads(line)
            gt_labels = record["true_labels"] or []
            pred_labels = record["predicted_labels"] or []
            if not gt_labels:
                gt_labels = [silence_label]
            if not pred_labels:
                pred_labels = [silence_label]
            # Silence-silence
            if gt_labels == [silence_label] and pred_labels == [silence_label]:
                cm_custom[silence_label][silence_label] += 1
                continue
            # For each class
            for cls in classes:
                gt_has = cls in gt_labels
                pred_has = cls in pred_labels
                # TP
                if gt_has and pred_has:
                    cm_custom[cls][cls] += 1
                # FP
                if not gt_has and pred_has:
                    for gt in gt_labels:
                        cm_custom[gt][cls] += 1
                # FN
                if gt_has and not pred_has:
                    for pred in pred_labels:
                        cm_custom[cls][pred] += 1
    # Save custom confusion matrix to metrics file
    with open(metrics_file, "a") as f:
        f.write("\nCustom confusion matrix (GT rows, Pred cols):\n")
        f.write("\t" + "\t".join(all_classes) + "\n")
        for gt in all_classes:
            f.write(f"{gt}\t" + "\t".join(str(cm_custom[gt][pred]) for pred in all_classes) + "\n")
        # Add macro F1 for custom confusion matrix
        f.write("\nMacro F1 (per-class metrics): ")
        f1s = []
        for cls in classes:
            TP, FP, FN = counts[cls]["TP"], counts[cls]["FP"], counts[cls]["FN"]
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            f1s.append(f1)
        macro_f1 = np.mean(f1s)
        f.write(f"{macro_f1:.3f}\n")
    print(f"✅ Metrics saved to {metrics_file}")
    print(f"✅ Confusion matrix saved to {output_dir/'multilabel_confusion_matrix.png'}")
    print(f"✅ Percentage confusion matrix saved to {output_dir/'multilabel_confusion_matrix_percent.png'}")
    return counts, cm


def main():
    try:
        print("🚀 Starting Comprehensive Audio Classification Model Evaluation")
        print("=" * 70)

        test_segments_file = AudioClassification.TEST_SECONDS_FILE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = AudioClassification.TRAINED_WEIGHTS_PATH.parent
        folder_name = output_dir / f'evaluation_{timestamp}'

        model, mlb = load_model()
        thresholds = load_thresholds(mlb.classes_)

        segment_files = {'train': None, 'val': None, 'test': test_segments_file}
        _, _, test_generator = create_data_generators(segment_files, mlb)

        if test_generator is None or len(test_generator) == 0:
            raise ValueError("Test generator is empty. Check test data file and paths.")

        predictions_file = evaluate_and_save_predictions(
            model=model,
            test_generator=test_generator,
            mlb=mlb,
            thresholds=thresholds,
            output_dir=folder_name
        )

        compute_metrics_from_predictions(predictions_file, output_dir=folder_name)

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