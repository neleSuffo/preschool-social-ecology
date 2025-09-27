"""
Environment, model, and data setup utilities.
"""

import os
import sys
import shutil
import json
import datetime
import numpy as np
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from config import PersonConfig
from constants import PersonClassification
from person_classifier import VideoFrameDataset, CNNEncoder, FrameRNNClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def setup_environment(is_training=True):
    """Sets up the output directory, device, and mixed-precision scaler.
    
    Parameters:
    -----------
    is_training: bool
        If True, creates a timestamped directory for training logs.
        If False, prepares an evaluation directory.
        
    Returns: Tuple[Path, torch.device, torch.cuda.amp.GradScaler|None]
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_training:
        out_dir = Path(PersonClassification.OUTPUT_DIR) / f"{PersonConfig.MODEL_NAME}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_script_and_hparams(out_dir)
    else:
        out_dir = PersonClassification.TRAINED_WEIGHTS_PATH.parent / f"{PersonConfig.MODEL_NAME}_evaluation_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None

    if device.type == 'cuda':
        print(f"Using device: {device}")
        
    return out_dir, device, scaler

def save_script_and_hparams(out_dir):
    """Save a copy of training script and hyperparameters to out_dir."""
    try:
        script_path = os.path.abspath(sys.argv[0])
        script_copy_path = os.path.join(out_dir, os.path.basename(script_path))
        shutil.copyfile(script_path, script_copy_path)
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")

    hparams = {attr: getattr(PersonConfig, attr) for attr in dir(PersonConfig) if not attr.startswith('_')}
    hparams_path = os.path.join(out_dir, "hyperparameters.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=4)

def setup_data_loaders(csv_path, batch_size, is_training, log_dir, is_feature_extraction=False, split_name=None):
    """Setup a single data loader based on parameters."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VideoFrameDataset(csv_path, transform=transform, log_dir=log_dir, is_feature_extraction=is_feature_extraction, split_name=split_name)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=os.cpu_count(), # Use all cores for data loading
        collate_fn=collate_fn, # Re-add the collate_fn to handle padding
        pin_memory=True,
        persistent_workers=True
    )
    return loader, dataset

def setup_models_and_optimizers(device: torch.device):
    """Create models, initialize non-pretrained weights, compile and return optimizers."""
    rnn_model = FrameRNNClassifier().to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

    rnn_model.apply(init_weights)

    try:
        rnn_model = torch.compile(rnn_model)
        print("Models compiled successfully!")
    except Exception as e:
        print(f"Model compilation skipped: {e}")

    opt_rnn = torch.optim.AdamW(
        rnn_model.parameters(),
        lr=PersonConfig.LR * 2.0,
        weight_decay=PersonConfig.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    criterion = nn.BCEWithLogitsLoss()

    return rnn_model, opt_rnn, criterion

def load_model(device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)
    
    def clean_state_dict(state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    cnn.load_state_dict(clean_state_dict(checkpoint['cnn_state']))
    rnn_model.load_state_dict(clean_state_dict(checkpoint['rnn_state']))
    
    return cnn, rnn_model

def collate_fn(batch):
    """Pads sequences to the max length in the batch. Handles both image and feature tensors."""
    batch_sizes = [item[2] for item in batch]
    max_len = max(batch_sizes)
    bs = len(batch)
    first_shape = batch[0][0].shape

    if len(first_shape) == 2:
        # Features: (seq_len, feat_dim)
        feat_dim = first_shape[1]
        images_padded = torch.zeros((bs, max_len, feat_dim))
    elif len(first_shape) == 3:
        # Images: (seq_len, C, H, W)
        C, H, W = first_shape[1:]
        images_padded = torch.zeros((bs, max_len, C, H, W))
    else:
        raise ValueError(f"Unexpected input shape: {first_shape}")

    labels_padded = torch.full((bs, max_len, 2), -100.0, dtype=torch.float32)
    lengths, video_ids = [], []

    for i, (imgs, labs, l, vid) in enumerate(batch):
        images_padded[i, :l] = imgs
        labels_padded[i, :l] = labs
        lengths.append(l)
        video_ids.append(vid)

    return images_padded, labels_padded, torch.tensor(lengths, dtype=torch.long), video_ids

def calculate_metrics(y_true, y_pred, class_names=PersonConfig.TARGET_LABELS):
    """Calculate precision, recall, and F1 for each class and macro averages."""
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()

    metrics = {}
    all_precisions, all_recalls, all_f1s, all_accuracies = [], [], [], []

    for i, class_name in enumerate(class_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        acc = accuracy_score(y_true[:, i], y_pred[:, i])
        metrics[f'{class_name.lower()}_precision'] = precision
        metrics[f'{class_name.lower()}_recall'] = recall
        metrics[f'{class_name.lower()}_f1'] = f1
        metrics[f'{class_name.lower()}_accuracy'] = acc
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(acc)

    metrics['macro_precision'] = np.mean(all_precisions)
    metrics['macro_recall'] = np.mean(all_recalls)
    metrics['macro_f1'] = np.mean(all_f1s)
    metrics['macro_accuracy'] = np.mean(all_accuracies)

    return metrics

def initialize_training_logging(out_dir: Path):
    """Initializes the training metrics CSV log file."""
    csv_log_path = out_dir / "training_metrics.csv"
    csv_headers = [
        'epoch', 'train_loss', 'val_loss',
        'train_adult_precision', 'train_adult_recall', 'train_adult_f1',
        'train_child_precision', 'train_child_recall', 'train_child_f1',
        'train_macro_precision', 'train_macro_recall', 'train_macro_f1',
        'val_adult_precision', 'val_adult_recall', 'val_adult_f1',
        'val_child_precision', 'val_child_recall', 'val_child_f1',
        'val_macro_precision', 'val_macro_recall', 'val_macro_f1'
    ]
    with open(csv_log_path, 'w') as f:
        f.write(','.join(csv_headers) + '\n')
    return csv_log_path

def log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics):
    """Appends a new row of metrics to the CSV log file."""
    row = [
        epoch, train_loss, val_loss,
        train_metrics['adult_precision'], train_metrics['adult_recall'], train_metrics['adult_f1'],
        train_metrics['child_precision'], train_metrics['child_recall'], train_metrics['child_f1'],
        train_metrics['macro_precision'], train_metrics['macro_recall'], train_metrics['macro_f1'],
        val_metrics['adult_precision'], val_metrics['adult_recall'], val_metrics['adult_f1'],
        val_metrics['child_precision'], val_metrics['child_recall'], val_metrics['child_f1'],
        val_metrics['macro_precision'], val_metrics['macro_recall'], val_metrics['macro_f1'],
    ]
    with open(csv_log_path, 'a') as f:
        f.write(','.join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]) + '\n')

def print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics):
    """Prints formatted training and validation results to the console."""
    print(f"\nEpoch {epoch}/{PersonConfig.NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"    Macro    - P: {train_metrics['macro_precision']:.3f}, R: {train_metrics['macro_recall']:.3f}, F1: {train_metrics['macro_f1']:.3f}")
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"    Macro    - P: {val_metrics['macro_precision']:.3f}, R: {val_metrics['macro_recall']:.3f}, F1: {val_metrics['macro_f1']:.3f}")

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrices for each class and create a multi-class confusion matrix."""
    fig, axes = plt.subplots(1, len(class_names), figsize=(12, 5))
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].figure.colorbar(im, ax=axes[i])
        
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
    plt.savefig(output_dir / 'confusion_matrices_binary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    y_true_multiclass, y_pred_multiclass = [], []
    for i in range(len(y_true)):
        true_class = 1 if y_true[i, 0] == 1 else (2 if y_true[i, 1] == 1 else 0)
        pred_class = 1 if y_pred[i, 0] == 1 else (2 if y_pred[i, 1] == 1 else 0)
        y_true_multiclass.append(true_class)
        y_pred_multiclass.append(pred_class)
    
    y_true_multiclass = np.array(y_true_multiclass)
    y_pred_multiclass = np.array(y_pred_multiclass)
    multiclass_labels = ['No Face', 'Child', 'Adult']
    cm_multiclass = confusion_matrix(y_true_multiclass, y_pred_multiclass, labels=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues', xticklabels=multiclass_labels, yticklabels=multiclass_labels, cbar_kws={'label': 'Count'})
    plt.title('Multi-class Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_multiclass_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    cm_multiclass_pct = cm_multiclass.astype('float') / cm_multiclass.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multiclass_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=multiclass_labels, yticklabels=multiclass_labels, cbar_kws={'label': 'Percentage (%)'})
    plt.title('Multi-class Confusion Matrix (Percentages)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_multiclass_percentages.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    cm_df_counts = pd.DataFrame(cm_multiclass, index=[f'True_{label}' for label in multiclass_labels], columns=[f'Pred_{label}' for label in multiclass_labels])
    cm_df_counts.to_csv(output_dir / 'confusion_matrix_multiclass_counts.csv')
    
    cm_df_pct = pd.DataFrame(cm_multiclass_pct, index=[f'True_{label}' for label in multiclass_labels], columns=[f'Pred_{label}' for label in multiclass_labels])
    cm_df_pct.to_csv(output_dir / 'confusion_matrix_multiclass_percentages.csv')

def plot_metrics_comparison(metrics, output_dir):
    """Plot comparison of metrics across classes."""
    metric_types = ['precision', 'recall', 'f1']
    data = []
    for class_name in PersonConfig.TARGET_LABELS:
        for metric_type in metric_types:
            key = f'{class_name}_{metric_type}'
            if key in metrics:
                data.append({'Class': class_name.capitalize(), 'Metric': metric_type.capitalize(), 'Value': metrics[key]})
    
    for metric_type in metric_types:
        key = f'macro_{metric_type}'
        if key in metrics:
            data.append({'Class': 'Macro Average', 'Metric': metric_type.capitalize(), 'Value': metrics[key]})
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Metric', columns='Class', values='Value')
    ax = pivot_df.plot(kind='bar', width=0.8)
    plt.title('Performance Metrics by Class')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Class')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', rotation=90, padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
