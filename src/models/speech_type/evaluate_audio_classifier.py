import json
import numpy as np
from collections import defaultdict
from datetime import datetime
from constants import AudioClassification
from utils import load_thresholds, load_model, create_data_generators, setup_gpu_config
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score

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
    
    Parameters
    ----------
    model (tf.keras.Model):
        Trained multi-label audio classification model
    test_generator (EvaluationDataGenerator):
        Test data generator with deterministic ordering
    mlb (MultiLabelBinarizer):
        Fitted label encoder from training pipeline
    thresholds (dict):
        Dictionary mapping class names to decision thresholds
    output_dir (Path):
        Directory to save the prediction results file
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
    
    # Ensure shapes match before processing
    min_samples = min(test_predictions.shape[0], len(test_true_labels))
    test_predictions = test_predictions[:min_samples]
    test_true_labels = np.array(test_true_labels[:min_samples])
    test_metadata = test_metadata[:min_samples]
    
    # Apply class-specific thresholds
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[mlb.classes_[i]]).astype(int) 
        for i in range(len(mlb.classes_))
    ]).T
    
    # Aggregate predictions to per-second level
    per_second_data = defaultdict(lambda: {'true': set(), 'pred': set(), 'audio_path': None})
    
    # Aggregate true labels and binary predictions
    for i in range(min_samples):
        metadata = test_metadata[i]
        start_time = metadata['start_time']
        audio_path = metadata['audio_path']
        
        # Round the start time to the nearest second to create a key
        second = round(start_time)
        
        # Add true labels
        true_labels_indices = np.where(test_true_labels[i] == 1)[0]
        for idx in true_labels_indices:
            per_second_data[audio_path, second]['true'].add(mlb.classes_[idx])
        
        # Add predicted labels
        pred_labels_indices = np.where(test_pred_binary[i] == 1)[0]
        for idx in pred_labels_indices:
            per_second_data[audio_path, second]['pred'].add(mlb.classes_[idx])
            
        per_second_data[audio_path, second]['audio_path'] = audio_path

    # Write predictions to JSONL file
    with open(predictions_file, 'w') as f:
        for (audio_path, second), data in sorted(per_second_data.items()):
            # Infer 'no speech' label
            if not data['true']:
                data['true'].add('no speech')
            if not data['pred']:
                data['pred'].add('no speech')

            record = {
                "audio_path": data['audio_path'],
                "second": second,
                "true_labels": sorted(list(data['true'])),
                "predicted_labels": sorted(list(data['pred'])),
            }
            f.write(json.dumps(record) + '\n')
            
    print(f"✅ Predictions saved to: {predictions_file}")
    return predictions_file


def compute_metrics_from_predictions(predictions_file, mlb, output_dir):
    """
    Loads per-second predictions from a JSONL file and computes precision, recall, and F1 for each speech class independently.

    Parameters
    ----------
    predictions_file (Path):
        Path to the JSONL file containing per-second predictions
    mlb (MultiLabelBinarizer):
        Fitted label encoder
    output_dir (Path):
        Directory to save the final evaluation report
    """
    print("\n📈 Computing metrics from saved predictions...")
    
    # Load predictions from file
    true_labels_list = []
    pred_labels_list = []
    
    with open(predictions_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            true_labels_list.append(record['true_labels'])
            pred_labels_list.append(record['predicted_labels'])
    
    # Only use speech classes
    speech_classes = list(mlb.classes_)
    mlb_speech = MultiLabelBinarizer(classes=speech_classes)
    mlb_speech.fit(None)

    true_bin = mlb_speech.transform(true_labels_list)
    pred_bin = mlb_speech.transform(pred_labels_list)

    print("📊 Speech Class Metrics:")
    print("=" * 50)
    for i, class_name in enumerate(speech_classes):
        y_true = true_bin[:, i]
        y_pred = pred_bin[:, i]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        support = y_true.sum()
        print(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
    print("=" * 50)
    print(f"✅ Evaluation completed. Metrics saved to {output_dir}")

def main():
    """
    Execute comprehensive audio classification model evaluation pipeline.
    """   
    try:
        print("🚀 Starting Comprehensive Audio Classification Model Evaluation")
        print("=" * 70)
        
        test_segments_file = AudioClassification.TEST_SECONDS_FILE
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = AudioClassification.TRAINED_WEIGHTS_PATH.parent
        folder_name = output_dir / f'evaluation_{timestamp}'

        model, mlb = load_model()
        thresholds = load_thresholds(mlb.classes_)
        
        segment_files = {
            'train': None,
            'val': None,
            'test': test_segments_file
        }
        _, _, test_generator = create_data_generators(segment_files, mlb)
        
        if test_generator is None or len(test_generator) == 0:
            raise ValueError("Test generator is empty. Check test data file and paths.")
            
        # Refactored: First generate predictions and save to file
        predictions_file = evaluate_and_save_predictions(
            model=model,
            test_generator=test_generator,
            mlb=mlb,
            thresholds=thresholds,
            output_dir=folder_name
        )
        
        # Then, compute metrics from the saved predictions
        compute_metrics_from_predictions(
            predictions_file=predictions_file,
            mlb=mlb,
            output_dir=folder_name
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