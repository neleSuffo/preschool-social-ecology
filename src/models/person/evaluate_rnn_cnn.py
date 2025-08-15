"""
Evaluation script for CNN-RNN person classification model on test set.

Usage:
  python evaluate_rnn_cnn.py
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
import seaborn as sns
from constants import PersonClassification, DataPaths
from config import PersonConfig

# Import classes from training script
from train_rnn_cnn import (
    VideoFrameDataset, 
    CNNEncoder, 
    FrameRNNClassifier, 
    collate_fn,
    sequence_features_from_cnn,
    calculate_metrics
)

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
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], 
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[i].set_title(f'{class_name.capitalize()} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
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
                    'adult_frames': int(video_labels[:, 0].sum()),
                    'child_frames': int(video_labels[:, 1].sum()),
                    'predicted_adult_frames': int(video_preds[:, 0].sum()),
                    'predicted_child_frames': int(video_preds[:, 1].sum()),
                    'adult_precision': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[0],
                    'adult_recall': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[1],
                    'adult_f1': precision_recall_fscore_support(video_labels[:, 0], video_preds[:, 0], average='binary', zero_division=0)[2],
                    'child_precision': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[0],
                    'child_recall': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[1],
                    'child_f1': precision_recall_fscore_support(video_labels[:, 1], video_preds[:, 1], average='binary', zero_division=0)[2],
                    'avg_adult_prob': float(video_probs[:, 0].mean()),
                    'avg_child_prob': float(video_probs[:, 1].mean())
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
    
    # Save detailed results
    print(f"\nSaving results to {output_dir}")
    
    # Save video-level results
    video_df = pd.DataFrame(video_results)
    video_df.to_csv(os.path.join(output_dir, 'video_level_results.csv'), index=False)
    
    # Save frame-level predictions
    frame_results = pd.DataFrame({
        'adult_true': all_labels[:, 0],
        'child_true': all_labels[:, 1],
        'adult_pred': all_preds[:, 0],
        'child_pred': all_preds[:, 1],
        'adult_prob': all_probs[:, 0],
        'child_prob': all_probs[:, 1]
    })
    frame_results.to_csv(os.path.join(output_dir, 'frame_level_results.csv'), index=False)
    
    # Generate plots
    plot_confusion_matrices(all_labels, all_preds, PersonConfig.TARGET_LABELS, output_dir)
    plot_metrics_comparison(metrics, output_dir)
    
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
    
    return metrics

def load_model(device: str, model_path: str) -> tuple:
    """Load trained model from checkpoint.
    
    Parameters
    ----------
    model_path : str
        Path to the model checkpoint.
    device : torch.device
        Device to load the model on.
        
    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Loaded CNN and RNN models.
    """
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize models with same architecture as training
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)
    
    # Load state dicts
    cnn.load_state_dict(checkpoint['cnn_state'])
    rnn_model.load_state_dict(checkpoint['rnn_state'])
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    
    return cnn, rnn_model

def setup_evaluation(args):
    """Setup output directory, device, and data loader."""   
    # Create timestamped evaluation directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f"{PersonConfig.MODEL_NAME}_validation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Update args to use the new directory
    args.output_dir = eval_dir
    print(f"Results will be saved to: {eval_dir}")
    
    # Set up device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load test dataset
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
    return device, test_loader, test_ds

def print_results(metrics):
    """Print formatted evaluation results."""
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"\nPer-Class Results:")
    print(f"  Adult    - P: {metrics['adult_precision']:.3f}, R: {metrics['adult_recall']:.3f}, F1: {metrics['adult_f1']:.3f}")
    print(f"  Child    - P: {metrics['child_precision']:.3f}, R: {metrics['child_recall']:.3f}, F1: {metrics['child_f1']:.3f}")
    print(f"\nMacro Averages:")
    print(f"  Precision: {metrics['macro_precision']:.3f}")
    print(f"  Recall:    {metrics['macro_recall']:.3f}")
    print(f"  F1:        {metrics['macro_f1']:.3f}")
    print("="*50)

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate CNN-RNN person classification model on test set")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output_dir', type=str, default=PersonClassification.OUTPUT_DIR, help='Directory to save evaluation results')
    parser.add_argument('--model_path', type=str, default=PersonClassification.TRAINED_WEIGHTS_PATH, help='Path to the trained model weights')
    parser.add_argument('--batch_size', type=int, default=PersonConfig.BATCH_SIZE, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup evaluation components
    device, test_loader, test_ds = setup_evaluation(args)
    
    # Load model and run evaluation
    cnn, rnn_model = load_model(args.device, args.model_path)
    metrics = evaluate_model((cnn, rnn_model), test_loader, device, args.output_dir)
    
    # Print and log results
    print_results(metrics)
    test_ds.log_skipped_files()
    
    print(f"\nEvaluation complete! Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()