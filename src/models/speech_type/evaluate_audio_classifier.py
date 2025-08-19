# filepath: /home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/models/speech_type/evaluate_audio_classifier.py
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall
from tqdm import tqdm
from constants import AudioClassification
from config import AudioConfig
from audio_classifier import MacroF1Score, FocalLoss

# --- Feature Extraction ---
def extract_enhanced_features(audio_path, start_time, duration, sr=16000, n_mels=256, hop_length=512, fixed_time_steps=None):
    """
    Extract enhanced audio features (mel-spectrogram + MFCC) from audio segment.
    Same function as used in training to ensure consistency.
    """
    if fixed_time_steps is None:
        fixed_time_steps = int(np.ceil(duration * sr / hop_length))
    
    try:
        y, sr_loaded = librosa.load(audio_path, sr=sr, offset=start_time, duration=duration)
        if sr_loaded != sr:
            y = librosa.resample(y, orig_sr=sr_loaded, target_sr=sr)
        
        expected_samples = int(duration * sr)
        if len(y) < expected_samples:
            y = np.pad(y, (0, expected_samples - len(y)), 'constant')
        elif len(y) > expected_samples:
            y = y[:expected_samples]
        
        if len(y) == 0:
            return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)
        
        y = y / (np.max(np.abs(y)) + 1e-6)
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])
        
        # Mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=2048, fmin=20, fmax=10000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
        mel_spectrogram_db = 2 * (mel_spectrogram_db - mel_spectrogram_db.min()) / (mel_spectrogram_db.max() - mel_spectrogram_db.min() + 1e-6) - 1
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        mfcc = 2 * (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-6) - 1
        
        # Concatenate features
        combined = np.concatenate([mel_spectrogram_db, mfcc], axis=0)
        
        if combined.shape[1] < fixed_time_steps:
            pad_width = fixed_time_steps - combined.shape[1]
            combined = np.pad(combined, ((0, 0), (0, pad_width)), 'constant', constant_values=-1)
        elif combined.shape[1] > fixed_time_steps:
            combined = combined[:, :fixed_time_steps]
        
        return combined
    
    except Exception as e:
        print(f"Error processing segment from {audio_path} [{start_time}s, duration {duration}s]: {e}")
        return np.zeros((n_mels + 13, fixed_time_steps), dtype=np.float32)

# --- Data Generator for Evaluation ---
class EvaluationDataGenerator(tf.keras.utils.Sequence):
    """
    Simplified data generator for evaluation (no augmentation, deterministic order).
    """
    def __init__(self, segments_file_path, mlb, n_mels, hop_length, sr, window_duration, fixed_time_steps, batch_size=32):
        self.segments_file_path = segments_file_path
        self.mlb = mlb
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.sr = sr
        self.window_duration = window_duration
        self.fixed_time_steps = fixed_time_steps
        self.batch_size = batch_size
        self.segments_data = self._load_segments_metadata()

    def __len__(self):
        return (len(self.segments_data) + self.batch_size - 1) // self.batch_size

    def _load_segments_metadata(self):
        segments = []
        with open(self.segments_file_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line.strip()))
        return segments

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.segments_data))
        batch_segments = self.segments_data[start_idx:end_idx]
        
        X_batch = []
        y_batch = []
        
        for segment in batch_segments:
            mel = extract_enhanced_features(
                segment['audio_path'], segment['start'], segment['duration'],
                sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length,
                fixed_time_steps=self.fixed_time_steps
            )
            
            if mel.ndim != 2:
                if mel.ndim == 3 and mel.shape[-1] == 1:
                    mel = mel.squeeze(axis=-1)
                else:
                    continue
                    
            X_batch.append(mel)
            multi_hot_labels = self.mlb.transform([segment['labels']])[0]
            y_batch.append(multi_hot_labels)
        
        if not X_batch:
            return np.array([]), np.array([])
            
        X_batch_np = np.array(X_batch)
        X_batch_final = np.expand_dims(X_batch_np, -1)
        return X_batch_final, np.array(y_batch)

# --- Evaluation Functions ---
def load_model_and_setup(model_path):
    """
    Load trained model and setup multi-label binarizer from training run.
    
    Parameters:
    ----------
    model_path (str or Path): Path to saved model (.h5 file)
    
    Returns:
    -------
    tuple: (model, mlb)
    """    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    # Setup multi-label binarizer
    mlb = MultiLabelBinarizer(classes=AudioConfig.VALID_RTTM_CLASSES)
    mlb.fit([[]])  # Initialize
    
    # Load model with custom objects
    custom_objects = {
        'MacroF1Score': MacroF1Score,
        'FocalLoss': FocalLoss
    }
    
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects=custom_objects)
    
    return model, mlb

def load_thresholds(run_dir, mlb_classes):
    """
    Load optimized thresholds from training run.
    
    Parameters:
    ----------
    run_dir (str or Path): Directory containing training run files
    mlb_classes (list): List of class names
    
    Returns:
    -------
    list: Optimized thresholds for each class
    """
    run_dir = Path(run_dir)
    thresholds_file = run_dir / 'thresholds.json'
    
    if thresholds_file.exists():
        with open(thresholds_file, 'r') as f:
            thresholds_dict = json.load(f)
        thresholds = [thresholds_dict.get(class_name, 0.5) for class_name in mlb_classes]
        print(f"Loaded optimized thresholds: {dict(zip(mlb_classes, thresholds))}")
    else:
        thresholds = [0.5] * len(mlb_classes)
        print("Using default thresholds (0.5) - optimized thresholds not found.")
    
    return thresholds

def create_evaluation_generator(test_segments_file, mlb):
    """
    Create data generator for evaluation.
    
    Parameters:
    ----------
    test_segments_file (str or Path): Path to test segments file
    mlb: Fitted MultiLabelBinarizer
    
    Returns:
    -------
    EvaluationDataGenerator: Test data generator
    """
    fixed_time_steps = int(np.ceil(AudioConfig.WINDOW_DURATION * AudioConfig.SR / AudioConfig.HOP_LENGTH))
    
    test_generator = EvaluationDataGenerator(
        test_segments_file, mlb,
        AudioConfig.N_MELS, AudioConfig.HOP_LENGTH, AudioConfig.SR,
        AudioConfig.WINDOW_DURATION, fixed_time_steps,
        batch_size=32
    )
    
    print(f"Created test generator with {len(test_generator)} batches")
    return test_generator

def evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir):
    """
    Perform comprehensive evaluation of the model on test set.
    
    Parameters:
    ----------
    model: Loaded Keras model
    test_generator: Test data generator
    mlb: MultiLabelBinarizer
    thresholds (list): Optimized thresholds for each class
    output_dir (str or Path): Directory to save evaluation results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Running comprehensive model evaluation...")
    print("=" * 60)
    
    # Get predictions and true labels
    print("📊 Generating predictions...")
    test_predictions = model.predict(test_generator, verbose=1)
    
    print("📋 Collecting true labels...")
    test_true_labels = []
    for i in tqdm(range(len(test_generator)), desc="Collecting labels"):
        _, labels = test_generator[i]
        test_true_labels.extend(labels)
    test_true_labels = np.array(test_true_labels)
    
    # Handle shape mismatches
    if test_predictions.shape[0] != test_true_labels.shape[0]:
        print(f"⚠️ Shape mismatch: predictions {test_predictions.shape[0]}, labels {test_true_labels.shape[0]}")
        min_samples = min(test_predictions.shape[0], test_true_labels.shape[0])
        test_predictions = test_predictions[:min_samples]
        test_true_labels = test_true_labels[:min_samples]
        print(f"✂️ Adjusted to {min_samples} samples")
    
    # Apply thresholds
    print("🎯 Applying optimized thresholds...")
    test_pred_binary = np.array([
        (test_predictions[:, i] > thresholds[i]).astype(int) 
        for i in range(len(mlb.classes_))
    ]).T
    
    # Calculate comprehensive metrics
    print("📈 Calculating metrics...")
    
    if test_true_labels.sum() > 0:
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            test_true_labels, test_pred_binary, average=None, zero_division=0
        )
        
        # Macro metrics
        macro_precision = precision_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_recall = recall_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        macro_f1 = f1_score(test_true_labels, test_pred_binary, average='macro', zero_division=0)
        
        # Micro metrics
        micro_precision = precision_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_recall = recall_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        micro_f1 = f1_score(test_true_labels, test_pred_binary, average='micro', zero_division=0)
        
        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(test_true_labels, test_pred_binary)
        
        # Print results
        print_evaluation_results(
            mlb.classes_, thresholds, 
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1,
            subset_accuracy,
            precision_per_class, recall_per_class, f1_per_class, support_per_class
        )
        
        # Save detailed results
        save_evaluation_results(
            output_dir, mlb.classes_, thresholds,
            test_true_labels, test_pred_binary, test_predictions,
            precision_per_class, recall_per_class, f1_per_class, support_per_class,
            macro_precision, macro_recall, macro_f1,
            micro_precision, micro_recall, micro_f1, subset_accuracy
        )
        
        # Generate plots
        generate_evaluation_plots(
            output_dir, mlb.classes_,
            test_true_labels, test_pred_binary, test_predictions
        )
        
    else:
        print("⚠️ Warning: No positive instances in test set for metrics calculation.")
    
    print(f"\n✅ Evaluation completed! Results saved to: {output_dir}")

def print_evaluation_results(class_names, thresholds, 
                           macro_precision, macro_recall, macro_f1,
                           micro_precision, micro_recall, micro_f1,
                           subset_accuracy,
                           precision_per_class, recall_per_class, f1_per_class, support_per_class):
    """Print formatted evaluation results to console."""
    
    print(f"\n📊 TEST SET EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"🎯 Optimized thresholds: {[f'{t:.3f}' for t in thresholds]}")
    
    print(f"\n📈 Overall Metrics:")
    print(f"  Subset Accuracy (exact match): {subset_accuracy:.4f}")
    print(f"  Macro Precision: {macro_precision:.4f}")
    print(f"  Macro Recall: {macro_recall:.4f}")
    print(f"  Macro F1-score: {macro_f1:.4f}")
    print(f"  Micro Precision: {micro_precision:.4f}")
    print(f"  Micro Recall: {micro_recall:.4f}")
    print(f"  Micro F1-score: {micro_f1:.4f}")
    
    print(f"\n📋 Per-Class Results:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} "
              f"{f1_per_class[i]:<10.4f} {support_per_class[i]:<10}")
    print("-" * 65)
    print(f"{'MACRO AVG':<15} {macro_precision:<10.4f} {macro_recall:<10.4f} {macro_f1:<10.4f}")
    print(f"{'MICRO AVG':<15} {micro_precision:<10.4f} {micro_recall:<10.4f} {micro_f1:<10.4f}")

def save_evaluation_results(output_dir, class_names, thresholds,
                          test_true_labels, test_pred_binary, test_predictions,
                          precision_per_class, recall_per_class, f1_per_class, support_per_class,
                          macro_precision, macro_recall, macro_f1,
                          micro_precision, micro_recall, micro_f1, subset_accuracy):
    """Save detailed evaluation results to files."""
    
    # Save summary metrics
    summary = {
        'overall_metrics': {
            'subset_accuracy': float(subset_accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'micro_precision': float(micro_precision),
            'micro_recall': float(micro_recall),
            'micro_f1': float(micro_f1)
        },
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(support_per_class[i]),
                'threshold': float(thresholds[i])
            } for i in range(len(class_names))
        },
        'test_set_size': len(test_true_labels),
        'thresholds': {class_names[i]: float(thresholds[i]) for i in range(len(class_names))}
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Save detailed predictions
    predictions_df = pd.DataFrame(test_predictions, columns=[f'{name}_prob' for name in class_names])
    predictions_binary_df = pd.DataFrame(test_pred_binary, columns=[f'{name}_pred' for name in class_names])
    true_labels_df = pd.DataFrame(test_true_labels, columns=[f'{name}_true' for name in class_names])
    
    detailed_results = pd.concat([predictions_df, predictions_binary_df, true_labels_df], axis=1)
    detailed_results.to_csv(output_dir / 'detailed_predictions.csv', index=False)
    
    print(f"💾 Saved evaluation summary to: {output_dir / 'evaluation_summary.json'}")
    print(f"💾 Saved detailed predictions to: {output_dir / 'detailed_predictions.csv'}")

def generate_evaluation_plots(output_dir, class_names, test_true_labels, test_pred_binary, test_predictions):
    """Generate evaluation plots and save them."""
    
    # 1. Per-class F1 scores
    f1_scores = [f1_score(test_true_labels[:, i], test_pred_binary[:, i], zero_division=0) 
                 for i in range(len(class_names))]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, f1_scores)
    plt.title('Per-Class F1 Scores')
    plt.ylabel('F1 Score')
    plt.xlabel('Classes')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_f1_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Prediction probability distributions
    fig, axes = plt.subplots(1, len(class_names), figsize=(4*len(class_names), 4))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        axes[i].hist(test_predictions[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{class_name} Prediction Probabilities')
        axes[i].set_xlabel('Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class support (number of positive samples per class)
    support_counts = test_true_labels.sum(axis=0)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, support_counts)
    plt.title('Class Support (Number of Positive Samples)')
    plt.ylabel('Number of Samples')
    plt.xlabel('Classes')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, support_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{int(count)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_support.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Generated evaluation plots in: {output_dir}")

def main():
    """
    Main function for model evaluation.
    """   
    try:
        print("🚀 Starting Audio Classification Model Evaluation")
        print("=" * 60)
        
        #Setup paths
        results_dir = Path(AudioClassification.RESULTS_DIR)
        model_path = results_dir / "best_model.h5"
        test_segments_file = AudioClassification.TEST_SEGMENTS_FILE
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = model_path.parent / 'resnet_gru_evaluation_' / timestamp
                
        # Load model and setup
        print("🧠 Loading model and setup...")
        model, mlb = load_model_and_setup(model_path)

        # Load optimized thresholds
        print("🎯 Loading optimized thresholds...")
        thresholds = load_thresholds(results_dir, mlb.classes_)
        
        # Create test data generator
        print("🔄 Creating test data generator...")
        test_generator = create_evaluation_generator(test_segments_file, mlb)
        
        # Run comprehensive evaluation
        print("🔍 Running evaluation...")
        evaluate_model_comprehensive(model, test_generator, mlb, thresholds, output_dir)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        raise

if __name__ == "__main__":
    main()