# filepath: /home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/models/person/evaluate_rnn_cnn_v2.py
"""
Evaluation script for CNN-RNN person classification model on test set.
Based on the improved training script architecture.

Usage:
  python evaluate_rnn_cnn_v2.py
"""

import os
import argparse
import json
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from constants import PersonClassification, DataPaths
from config import PersonConfig

# Import classes from training script
from train_rnn_cnn import collate_fn, sequence_features_from_cnn, calculate_metrics
from person_classifier import VideoFrameDataset, CNNEncoder, FrameRNNClassifier, load_model

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrices for each class.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n_samples, n_classes).
    y_pred : np.ndarray
        Predicted labels of shape (n_samples, n_classes).
    class_names : List[str]
        Names of the classes.
    output_dir : str
        Directory to save plots.
    """
    fig, axes = plt.subplots(1, len(class_names), figsize=(12, 5))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].figure.colorbar(im, ax=axes[i])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, format(cm[j, k], 'd'),
                           ha="center", va="center",
                           color="white" if cm[j, k] > thresh else "black")
        
        axes[i].set_title(f'{class_name.capitalize()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(['No', 'Yes'])
        axes[i].set_yticklabels(['No', 'Yes'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(metrics, output_dir):
    """Plot comparison of metrics across classes.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing calculated metrics.
    output_dir : str
        Directory to save plots.
    """
    # Extract metrics for each class
    metric_types = ['precision', 'recall', 'f1']
    
    data = []
    for class_name in PersonConfig.TARGET_LABELS:
        for metric_type in metric_types:
            key = f'{class_name}_{metric_type}'
            if key in metrics:
                data.append({
                    'Class': class_name.capitalize(),
                    'Metric': metric_type.capitalize(),
                    'Value': metrics[key]
                })
    
    # Add macro averages
    for metric_type in metric_types:
        key = f'macro_{metric_type}'
        if key in metrics:
            data.append({
                'Class': 'Macro Average',
                'Metric': metric_type.capitalize(),
                'Value': metrics[key]
            })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Metric', columns='Class', values='Value')
    ax = pivot_df.plot(kind='bar', width=0.8)
    plt.title('Performance Metrics by Class')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Class')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_distributions(all_probs, all_labels, output_dir):
    """Plot probability distributions for each class.
    
    Parameters
    ----------
    all_probs : np.ndarray
        Predicted probabilities of shape (n_samples, n_classes).
    all_labels : np.ndarray
        Ground truth labels of shape (n_samples, n_classes).
    output_dir : str
        Directory to save plots.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    class_names = PersonConfig.TARGET_LABELS
    
    for i, class_name in enumerate(class_names):
        # Separate probabilities by true label
        pos_probs = all_probs[all_labels[:, i] == 1, i]
        neg_probs = all_probs[all_labels[:, i] == 0, i]
        
        axes[i].hist(neg_probs, bins=50, alpha=0.7, label=f'True Negative (n={len(neg_probs)})', color='red')
        axes[i].hist(pos_probs, bins=50, alpha=0.7, label=f'True Positive (n={len(pos_probs)})', color='blue')
        axes[i].axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
        
        axes[i].set_title(f'{class_name.capitalize()} Probability Distribution')
        axes[i].set_xlabel('Predicted Probability')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(models, dataloader, device, output_dir):
    """Evaluate the model on test data and generate comprehensive results.
    
    Parameters
    ----------
    models : Tuple[nn.Module, nn.Module]
        Tuple containing (CNN encoder, RNN classifier) models.
    dataloader : DataLoader
        Test data loader.
    device : torch.device
        Device to run evaluation on.
    output_dir : str
        Directory to save results.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all evaluation metrics.
    """
    cnn, rnn = models
    cnn.eval()
    rnn.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    video_results = []
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    print("Running evaluation on test set...")
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for images_padded, labels_padded, lengths, video_ids in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)
            
            # Forward pass
            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            # Calculate loss
            mask_flat = mask.view(-1, 2)
            logits_flat = logits.view(-1, 2)[mask_flat[:, 0] != -100]
            labels_flat = labels_padded.view(-1, 2)[mask_flat[:, 0] != -100]
            
            if len(logits_flat) > 0:  # Only calculate loss if we have valid samples
                loss = criterion(logits_flat, labels_flat)
                total_loss += loss.item() * images_padded.size(0)
            
            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Collect results per video
            for i, video_id in enumerate(video_ids):
                seq_len = lengths[i].item()
                video_preds = preds[i, :seq_len].cpu().numpy()
                video_labels = labels_padded[i, :seq_len].cpu().numpy()
                video_probs = probs[i, :seq_len].cpu().numpy()
                
                # Calculate video-level statistics
                video_results.append({
                    'video_id': video_id,
                    'num_frames': seq_len,
                    'adult_frames': int(video_labels[:, 1].sum()),
                    'child_frames': int(video_labels[:, 0].sum()),
                    'predicted_adult_frames': int(video_preds[:, 1].sum()),
                    'predicted_child_frames': int(video_preds[:, 0].sum()),
                    'adult_precision': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[0],
                    'adult_recall': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[1],
                    'adult_f1': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[2],
                    'child_precision': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[0],
                    'child_recall': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[1],
                    'child_f1': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[2],
                    'avg_adult_prob': float(video_probs[:, 1].mean()),
                    'avg_child_prob': float(video_probs[:, 0].mean()),
                    'max_adult_prob': float(video_probs[:, 1].max()),
                    'max_child_prob': float(video_probs[:, 0].max()),
                    'min_adult_prob': float(video_probs[:, 1].min()),
                    'min_child_prob': float(video_probs[:, 0].min())
                })
            
            # Only keep valid predictions (not masked)
            valid_mask = mask.view(-1, 2)[:, 0]
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()
            valid_probs = probs.view(-1, 2)[valid_mask].cpu()
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
            all_probs.append(valid_probs)
            
            # Update progress bar
            progress_bar.set_postfix({
                'samples_processed': len(all_preds) * dataloader.batch_size
            })
    
    # Concatenate all results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_labels, all_preds, class_names=PersonConfig.TARGET_LABELS)
    metrics['test_loss'] = avg_loss
    
    # Add additional metrics
    metrics['total_samples'] = len(all_labels)
    metrics['adult_samples'] = int(all_labels[:, 1].sum())
    metrics['child_samples'] = int(all_labels[:, 0].sum())
    metrics['adult_prevalence'] = float(all_labels[:, 1].mean())
    metrics['child_prevalence'] = float(all_labels[:, 0].mean())
    
    # Save detailed results
    print(f"\nSaving results to {output_dir}")
    
    # Save video-level results
    video_df = pd.DataFrame(video_results)
    video_df.to_csv(os.path.join(output_dir, 'video_level_results.csv'), index=False)
    
    # Save frame-level predictions
    frame_results = pd.DataFrame({
        'child_true': all_labels[:, 0],
        'adult_true': all_labels[:, 1],
        'child_pred': all_preds[:, 0],
        'adult_pred': all_preds[:, 1],
        'child_prob': all_probs[:, 0],
        'adult_prob': all_probs[:, 1]
    })
    frame_results.to_csv(os.path.join(output_dir, 'frame_level_results.csv'), index=False)
    
    # Generate plots
    plot_confusion_matrices(all_labels, all_preds, PersonConfig.TARGET_LABELS, output_dir)
    plot_metrics_comparison(metrics, output_dir)
    plot_probability_distributions(all_probs, all_labels, output_dir)
    
    # Save metrics as JSON
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Generate and save classification report
    report_dict = {}
    for i, class_name in enumerate(PersonConfig.TARGET_LABELS):
        report_dict[class_name] = classification_report(
            all_labels[:, i], all_preds[:, i], output_dict=True, zero_division=0
        )
    
    with open(os.path.join(output_dir, 'classification_report.json'), 'w') as f:
        json.dump(report_dict, f, indent=4)
    
    # Save summary statistics
    summary_stats = {
        'model_info': {
            'backbone': PersonConfig.BACKBONE,
            'feat_dim': PersonConfig.FEAT_DIM,
            'rnn_hidden': PersonConfig.RNN_HIDDEN,
            'rnn_layers': PersonConfig.RNN_LAYERS,
            'bidirectional': PersonConfig.BIDIRECTIONAL,
            'sequence_length': PersonConfig.MAX_SEQ_LEN
        },
        'dataset_stats': {
            'total_videos': len(video_df),
            'total_frames': metrics['total_samples'],
            'adult_frames': metrics['adult_samples'],
            'child_frames': metrics['child_samples'],
            'adult_prevalence': metrics['adult_prevalence'],
            'child_prevalence': metrics['child_prevalence']
        },
        'performance_summary': {
            'macro_f1': metrics['macro_f1'],
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'adult_f1': metrics['adult_f1'],
            'child_f1': metrics['child_f1'],
            'test_loss': metrics['test_loss']
        }
    }
    
    with open(os.path.join(output_dir, 'summary_stats.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    return metrics

def setup_evaluation(args):
    """Setup output directory, device, and data loader."""   
    # Create timestamped evaluation directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"{PersonConfig.MODEL_NAME}_evaluation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Results will be saved to: {eval_dir}")
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test dataset with same transforms as training (but without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading test dataset...")
    test_ds = VideoFrameDataset(
        PersonClassification.TEST_CSV_PATH, 
        transform=transform,
        sequence_length=PersonConfig.MAX_SEQ_LEN,
        log_dir=eval_dir
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    print(f"Test dataset loaded: {len(test_ds)} sequences")
    return device, test_loader, test_ds, eval_dir

def print_results(metrics):
    """Print formatted evaluation results."""
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Total Samples: {metrics['total_samples']:,}")
    print(f"Adult Samples: {metrics['adult_samples']:,} ({metrics['adult_prevalence']:.1%})")
    print(f"Child Samples: {metrics['child_samples']:,} ({metrics['child_prevalence']:.1%})")
    
    print(f"\nPer-Class Results:")
    print(f"  Adult    - P: {metrics['adult_precision']:.3f}, R: {metrics['adult_recall']:.3f}, F1: {metrics['adult_f1']:.3f}")
    print(f"  Child    - P: {metrics['child_precision']:.3f}, R: {metrics['child_recall']:.3f}, F1: {metrics['child_f1']:.3f}")
    
    print(f"\nMacro Averages:")
    print(f"  Precision: {metrics['macro_precision']:.3f}")
    print(f"  Recall:    {metrics['macro_recall']:.3f}")
    print(f"  F1:        {metrics['macro_f1']:.3f}")
    print("="*70)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CNN-RNN person classification model on test set")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output_dir', type=str, default=PersonClassification.OUTPUT_DIR, help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, default=PersonClassification.TRAINED_WEIGHTS_PATH, help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=PersonConfig.BATCH_SIZE, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup evaluation components
    device, test_loader, test_ds, eval_dir = setup_evaluation(args)
    
    # Load model
    cnn, rnn_model = load_model(device, args.model_path)
    
    # Run evaluation
    metrics = evaluate_model((cnn, rnn_model), test_loader, device, eval_dir)
    
    # Print and log results
    print_results(metrics)
    test_ds.log_skipped_files()
    
    print(f"\nEvaluation complete! Results saved to: {eval_dir}")
    print(f"Key files generated:")
    print(f"  • test_metrics.json - Overall performance metrics")
    print(f"  • video_level_results.csv - Per-video detailed results")
    print(f"  • frame_level_results.csv - Per-frame predictions and probabilities")
    print(f"  • confusion_matrices.png - Confusion matrix visualization")
    print(f"  • metrics_comparison.png - Performance comparison chart")
    print(f"  • probability_distributions.png - Prediction probability distributions")
    print(f"  • summary_stats.json - Complete summary statistics")

if __name__ == '__main__':
    main()