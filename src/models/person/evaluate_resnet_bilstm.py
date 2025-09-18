# filepath: /home/nele_pauline_suffo/projects/naturalistic-social-analysis/src/models/person/evaluate_rnn_cnn_v2.py
"""
Evaluation script for CNN-RNN person classification model on test set.
Based on the improved training script architecture.
"""
import json
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from config import PersonConfig
from utils import load_model, setup_evaluation, calculate_metrics, sequence_features_from_cnn

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
    output_dir : Path
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
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
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
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
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
    output_dir : Path
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
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['test_loss'] = avg_loss
    
    # Add additional metrics
    metrics['total_samples'] = len(all_labels)
    metrics['adult_samples'] = int(all_labels[:, 1].sum())
    metrics['child_samples'] = int(all_labels[:, 0].sum())
    metrics['adult_prevalence'] = float(all_labels[:, 1].mean())
    metrics['child_prevalence'] = float(all_labels[:, 0].mean())
    
    # Save detailed results
    print(f"\nSaving results to {output_dir}")
    
    # Save frame-level predictions
    frame_results = pd.DataFrame({
        'child_true': all_labels[:, 0],
        'adult_true': all_labels[:, 1],
        'child_pred': all_preds[:, 0],
        'adult_pred': all_preds[:, 1],
        'child_prob': all_probs[:, 0],
        'adult_prob': all_probs[:, 1]
    })
    frame_results.to_csv(output_dir / 'frame_level_results.csv', index=False)

    # Generate plots
    plot_confusion_matrices(all_labels, all_preds, PersonConfig.TARGET_LABELS, output_dir)
    plot_metrics_comparison(metrics, output_dir)
    
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

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    return metrics

def main():
    """Main evaluation function."""
    # Setup evaluation components
    device, test_loader, test_ds, eval_dir = setup_evaluation()
    
    # Load model
    cnn, rnn_model = load_model(device)
    
    # Run evaluation
    metrics = evaluate_model((cnn, rnn_model), test_loader, device, eval_dir)
    
    # Print and log results
    test_ds.log_skipped_files()
    
    print(f"\nEvaluation complete! Results saved to: {eval_dir}")

if __name__ == '__main__':
    main()