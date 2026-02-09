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
from typing import List
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from config import PersonConfig
from constants import PersonClassification
from models.person.person_classifier import VideoFrameDataset, FrameRNNClassifier, PersonDetectionClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def sequence_features_from_cnn(cnn, images_padded, lengths, device):
    """
    Extracts features from padded image sequences using the CNN encoder.
    
    Parameters:
    ----------
    cnn : nn.Module
        The CNN encoder model.
    images_padded : torch.Tensor
        Padded image sequences of shape (batch_size, max_seq_len, C, H, W).
    lengths : torch.Tensor
        Actual lengths of each sequence in the batch.
    device : torch.device
        The device to perform computations on.
        
    Returns:
    -------
    feats : torch.Tensor
        Extracted features of shape (batch_size, max_seq_len, feat_dim).   
    """
    bs, max_seq, C, H, W = images_padded.shape
    images_flat = images_padded.view(bs * max_seq, C, H, W).to(device)
    feats_flat = cnn(images_flat)
    feat_dim = feats_flat.shape[-1]
    feats = feats_flat.view(bs, max_seq, feat_dim)
    return feats

def setup_environment(is_training=True, num_outputs=2):
    """Sets up the output directory, device, and mixed-precision scaler.
    
    Parameters:
    ----------
    is_training : bool
        Whether the setup is for training or evaluation.
    num_outputs : int
        Number of output labels for the model (1 or 2).
        
    Returns:
    -------
    out_dir : Path
        The output directory for saving models and logs.
    device : torch.device
        The device to perform computations on.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "age-binary" if num_outputs == 2 else "person-only"
    
    if is_training:
        out_dir = Path(PersonClassification.OUTPUT_DIR) / f"YOLOFeature_BiLSTM_{model_name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_script_and_hparams(out_dir, num_outputs)
    else:
        # For evaluation, the path is typically derived from the checkpoint path
        # Using a temporary name here, the caller handles final eval directory
        out_dir = Path("./tmp_eval_dir") 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None

    if device.type == 'cuda':
        print(f"Using device: {device}")
        
    return out_dir, device, scaler

def save_script_and_hparams(out_dir, num_outputs):
    """Save a copy of training script and hyperparameters to out_dir.
    
    Parameters:
    ----------
    out_dir : Path
        The output directory to save files.
    num_outputs : int
        Number of output labels for the model (1 or 2).
    """
    try:
        script_path = os.path.abspath(sys.argv[0])
        script_copy_path = os.path.join(out_dir, os.path.basename(script_path))
        shutil.copyfile(script_path, script_copy_path)
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")

    hparams = {attr: getattr(PersonConfig, attr) for attr in dir(PersonConfig) if not attr.startswith('_')}
    hparams['RUNTIME_NUM_OUTPUTS'] = num_outputs
    hparams_path = os.path.join(out_dir, "hyperparameters.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=4)

def setup_data_loaders(csv_path, batch_size, is_training, log_dir, is_feature_extraction=True, split_name=None):
    """Setup a single data loader based on parameters.
    
    Parameters:
    ----------
    csv_path : str or Path
        Path to the CSV file containing data information.
    batch_size : int
        Batch size for the data loader.
    is_training : bool
        Whether the data loader is for training or evaluation.
    log_dir : str or Path
        Directory for logging skipped files.
    is_feature_extraction : bool
        Whether to load images for feature extraction or precomputed features.
    split_name : str
        Name of the dataset split (e.g., 'train', 'val', 'test').
        
    Returns:
    -------
    loader : DataLoader
        Configured data loader.
    dataset : VideoFrameDataset
        The underlying dataset used in the data loader.
    """
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
        num_workers=os.cpu_count(),
        collate_fn=collate_fn, 
        pin_memory=True,
        persistent_workers=True
    )
    return loader, dataset

def setup_models_and_optimizers(device: torch.device, num_outputs: int, target_labels: List[str]):
    """Create the combined model (YOLO Feature + BiLSTM) and one optimizer.   
     
    Parameters:
    ----------
    device : torch.device
        The device to perform computations on.
    num_outputs : int
        Number of output labels for the model (1 or 2).
    target_labels : List[str]
        List of target label names.
        
    Returns:
    -------
    model : nn.Module 
        The combined CNN + RNN model.
    optimizer : torch.optim.Optimizer
        The optimizer for training the model.
    criterion : nn.Module
        The loss function used for training.
    """
    
# 1. Initialize the combined E2E model
    model = PersonDetectionClassifier( # Now PersonDetectionClassifier in person_classifier.py
        rnn_feat_dim=PersonConfig.FEAT_DIM, 
        num_outputs=num_outputs
    ).to(device)

    def init_weights(m):
        """Initializes weights only for the classification/RNN layers."""
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

    model.rnn.apply(init_weights)
    
    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"Model compilation skipped: {e}")

    # 2. Create a single optimizer for ALL parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PersonConfig.LR,
        weight_decay=PersonConfig.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )
    
    # 3. Implement Weighted Loss
    try:
        if num_outputs == 2:
            pos_weight = torch.tensor(PersonConfig.POS_WEIGHTS_AGE_BINARY, device=device, dtype=torch.float32)
        elif num_outputs == 1:
            pos_weight = torch.tensor(PersonConfig.POS_WEIGHTS_PERSON_ONLY, device=device, dtype=torch.float32)
        else:
             raise ValueError("Unsupported number of outputs for pos_weight")
             
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted BCE loss. Pos weights ({num_outputs} classes): {pos_weight.tolist()}")
    except Exception:
        print("Warning: Using unweighted BCE loss. Check POS_WEIGHTS configuration.")
        criterion = nn.BCEWithLogitsLoss()
        
    return model, optimizer, criterion

def load_model(device, num_outputs):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim, num_outputs=num_outputs).to(device)
    
    def clean_state_dict(state_dict):
        return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    cnn.load_state_dict(clean_state_dict(checkpoint['cnn_state']))
    rnn_model.load_state_dict(clean_state_dict(checkpoint['rnn_state']))
    
    return cnn, rnn_model

def collate_fn(batch):
    """Pads sequences to the max length in the batch. Handles dynamic label size."""
    
    batch_sizes = [item[2] for item in batch] 
    max_len = max(batch_sizes)
    bs = len(batch)
    first_shape = batch[0][0].shape
    
    # Get the number of output labels from the first item
    num_outputs = batch[0][1].shape[1] 

    if len(first_shape) == 2:
        # Features: (seq_len, feat_dim)
        feat_dim = first_shape[1]
        images_padded = torch.zeros((bs, max_len, feat_dim))
    elif len(first_shape) == 4:
        # Images (E2E Mode): (seq_len, C, H, W)
        C, H, W = first_shape[1:]
        images_padded = torch.zeros((bs, max_len, C, H, W))
    else:
        raise ValueError(f"Unexpected input shape: {first_shape}")

    labels_padded = torch.full((bs, max_len, num_outputs), -100.0, dtype=torch.float32)
    lengths, video_ids = [], []

    for i, (imgs, labs, l, vid) in enumerate(batch):
        images_padded[i, :l] = imgs
        labels_padded[i, :l] = labs
        lengths.append(l)
        video_ids.append(vid)

    return images_padded, labels_padded, torch.tensor(lengths, dtype=torch.long), video_ids

def calculate_metrics(y_true, y_pred, class_names=None):
    """Calculate metrics for 1 or 2 classes."""
    if class_names is None:
        if y_true.shape[1] == 1:
            class_names = PersonConfig.TARGET_LABELS_PERSON_ONLY
        else:
            class_names = PersonConfig.TARGET_LABELS_AGE_BINARY
            
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()

    metrics = {}
    all_precisions, all_recalls, all_f1s, all_accuracies = [], [], [], []

    # Handle single class case (N, 1) to (N,) for binary metrics
    if len(class_names) == 1:
        y_true_i = y_true.squeeze()
        y_pred_i = y_pred.squeeze()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_i, y_pred_i, average='binary', pos_label=1, zero_division=0
        )
        acc = accuracy_score(y_true_i, y_pred_i)
        
        metrics[f'{class_names[0].lower()}_precision'] = precision
        metrics[f'{class_names[0].lower()}_recall'] = recall
        metrics[f'{class_names[0].lower()}_f1'] = f1
        metrics[f'{class_names[0].lower()}_accuracy'] = acc
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(acc)

    # Handle multi-class case (N, C)
    else:
        for i, class_name in enumerate(class_names):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred[:, i], average='binary', pos_label=1, zero_division=0
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

def initialize_training_logging(out_dir: Path, class_names: List[str]):
    """Initializes the training metrics CSV log file."""
    csv_log_path = out_dir / "training_metrics.csv"
    
    headers = ['epoch', 'train_loss', 'val_loss']
    for prefix in ['train', 'val']:
        for class_name in class_names:
            for metric in ['precision', 'recall', 'f1']:
                headers.append(f'{prefix}_{class_name.lower()}_{metric}')
        for metric in ['precision', 'recall', 'f1']:
            headers.append(f'{prefix}_macro_{metric}')

    with open(csv_log_path, 'w') as f:
        f.write(','.join(headers) + '\n')
    return csv_log_path

def log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics, class_names):
    """Appends a new row of metrics to the CSV log file."""
    row = [epoch, train_loss, val_loss]
    
    for metrics_dict in [train_metrics, val_metrics]:
        for class_name in class_names:
            row.extend([metrics_dict[f'{class_name.lower()}_precision'],
                        metrics_dict[f'{class_name.lower()}_recall'],
                        metrics_dict[f'{class_name.lower()}_f1']])
        row.extend([metrics_dict['macro_precision'],
                    metrics_dict['macro_recall'],
                    metrics_dict['macro_f1']])

    with open(csv_log_path, 'a') as f:
        f.write(','.join([f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]) + '\n')

def print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics, class_names):
    """Prints formatted training and validation results to the console."""
    print(f"\nEpoch {epoch}/{PersonConfig.NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    for class_name in class_names:
        print(f"    {class_name.capitalize():<8} - P: {train_metrics[f'{class_name.lower()}_precision']:.3f}, R: {train_metrics[f'{class_name.lower()}_recall']:.3f}, F1: {train_metrics[f'{class_name.lower()}_f1']:.3f}")
    print(f"    Macro    - P: {train_metrics['macro_precision']:.3f}, R: {train_metrics['macro_recall']:.3f}, F1: {train_metrics['macro_f1']:.3f}")
    
    print(f"  Val Loss: {val_loss:.4f}")
    for class_name in class_names:
        print(f"    {class_name.capitalize():<8} - P: {val_metrics[f'{class_name.lower()}_precision']:.3f}, R: {val_metrics[f'{class_name.lower()}_recall']:.3f}, F1: {val_metrics[f'{class_name.lower()}_f1']:.3f}")
    print(f"    Macro    - P: {val_metrics['macro_precision']:.3f}, R: {val_metrics['macro_recall']:.3f}, F1: {val_metrics['macro_f1']:.3f}")

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrices for 1 or 2 classes."""
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)

    if len(class_names) == 1:
        # Single class 'person'
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_true.squeeze(), y_pred.squeeze(), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['No Person', 'Person'], yticklabels=['No Person', 'Person'])
        ax.set_title('Person Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix_binary.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # Two classes 'child', 'adult'
        fig, axes = plt.subplots(1, len(class_names), figsize=(12, 5))
        for i, class_name in enumerate(class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].figure.colorbar(im, ax=axes[i])
            thresh = cm.max() / 2.
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    axes[i].text(k, j, format(cm[j, k], 'd'), ha="center", va="center", color="white" if cm[j, k] > thresh else "black")
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

def plot_metrics_comparison(metrics, output_dir, class_names):
    """Plot comparison of metrics across classes."""
    metric_types = ['precision', 'recall', 'f1']
    data = []
    
    all_classes = class_names + ['Macro Average']
    
    for class_name in class_names:
        for metric_type in metric_types:
            key = f'{class_name.lower()}_{metric_type}'
            if key in metrics:
                data.append({'Class': class_name.capitalize(), 'Metric': metric_type.capitalize(), 'Value': metrics[key]})
    
    for metric_type in metric_types:
        key = f'macro_{metric_type}'
        if key in metrics:
            data.append({'Class': 'Macro Average', 'Metric': metric_type.capitalize(), 'Value': metrics[key]})
    
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 6))
    pivot_df = df.pivot(index='Metric', columns='Class', values='Value')
    ax = pivot_df.plot(kind='bar', width=0.8, figsize=(10, 6))
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