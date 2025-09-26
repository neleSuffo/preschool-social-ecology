"""
Utility functions for CNN-RNN training pipeline.
"""

import os
import sys
import shutil
import json
import datetime
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from config import PersonConfig
from constants import PersonClassification
from person_classifier import VideoFrameDataset, CNNEncoder, FrameRNNClassifier
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Collate Function
# ---------------------------
def collate_fn(batch):
    """Collate function for DataLoader to handle variable sequence lengths.
    
    Pads sequences to the maximum length in the batch and creates masks for valid data.
    
    Parameters:
    -----------
    batch: List of tuples (images, labels, seq_len, video_id)
        images: Tensor of shape (seq_len, C, H, W)
        labels: Tensor of shape (seq_len, num_classes)
        seq_len: int, actual length of the sequence
        video_id: str, identifier for the video
    
    Returns:
    --------
    images_padded: 
        Tensor of shape (batch_size, sequence_length, C, H, W)
    labels_padded: 
        Tensor of shape (batch_size, sequence_length, num_classes) with padding masked as
        -100 for loss computation
    lengths: 
        Tensor of shape (batch_size,) with actual lengths of each sequence
    video_ids: 
        List of video IDs corresponding to each sequence in the batch
    """
    batch_sizes = [item[2] for item in batch]
    max_len = max(batch_sizes)
    bs = len(batch)
    C, H, W = batch[0][0].shape[1:]

    images_padded = torch.zeros((bs, max_len, C, H, W))
    labels_padded = torch.full((bs, max_len, 2), -100.0, dtype=torch.float32)  # mask with -100
    lengths, video_ids = [], []

    for i, (imgs, labs, l, vid) in enumerate(batch):
        images_padded[i, :l] = imgs
        labels_padded[i, :l] = labs
        lengths.append(l)
        video_ids.append(vid)

    return images_padded, labels_padded, torch.tensor(lengths, dtype=torch.long), video_ids


# ---------------------------
# Metrics
# ---------------------------
def calculate_metrics(y_true, y_pred, class_names=PersonConfig.TARGET_LABELS):
    """Calculate precision, recall, F1 for each class and macro averages.
    
    Parameters:
    -----------
    y_true: np.ndarray or torch.Tensor of shape (num_samples, num_classes)
        Ground truth binary labels.
    y_pred: np.ndarray or torch.Tensor of shape (num_samples, num_classes)
        Predicted binary labels.
    class_names: List of str
        Names of the classes corresponding to each column.

    Returns:
    --------
    metrics: dict
        Dictionary containing precision, recall, F1, and accuracy for each class and macro averages.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Handle binary case with 2 labels
    if y_true.shape[1] == 1 and len(class_names) == 2:
        y_true = np.column_stack([y_true, y_true])
        y_pred = np.column_stack([y_pred, y_pred])

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

    # Macro averages
    metrics['macro_precision'] = np.mean(all_precisions)
    metrics['macro_recall'] = np.mean(all_recalls)
    metrics['macro_f1'] = np.mean(all_f1s)
    metrics['macro_accuracy'] = np.mean(all_accuracies)

    return metrics

# ---------------------------
# Feature Extraction
# ---------------------------
def sequence_features_from_cnn(cnn, images_padded, lengths, device):
    """Extract CNN features from padded image sequences.
    
    Parameters:
    -----------
    cnn: torch.nn.Module
        Pretrained CNN model for feature extraction.
    images_padded: torch.Tensor of shape (batch_size, sequence_length, C, H, W)
        Padded image sequences.
    lengths: torch.Tensor of shape (batch_size,)
        Actual lengths of each sequence.    
    device: torch.device
        Device to perform computations on.
        
    Returns:
    --------
    feats: torch.Tensor of shape (batch_size, sequence_length, feat_dim)
        Extracted features for each image in the sequences.
    """
    bs, max_seq, C, H, W = images_padded.shape
    imgs = images_padded.view(bs * max_seq, C, H, W).to(device)
    feats = cnn(imgs)  # (bs * max_seq, feat_dim)
    feats = feats.view(bs, max_seq, -1)
    return feats


# ---------------------------
# Environment Setup
# ---------------------------
def setup_training_environment():
    """Setup output directory, device, and logging.
        
    Returns:
    --------
    out_dir: str
        Output directory for saving models and logs.
    device: torch.device
        Device to perform computations on.
    scaler: torch.cuda.amp.GradScaler or None
        GradScaler for mixed precision training if using CUDA, else None.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PersonClassification.OUTPUT_DIR / f"{PersonConfig.MODEL_NAME}_{timestamp}"
    # ensure out_dir exists
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_script_and_hparams(out_dir)

    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    return out_dir, device, scaler


# ---------------------------
# Data Loaders
# ---------------------------
def setup_data_loaders(out_dir):
    """Setup data loaders.
    
    Parameters:
    -----------
    out_dir: str
        Directory for logging dataset info.
        
    Returns:
    --------
    train_loader: DataLoader
        DataLoader for training dataset.
    val_loader: DataLoader
        DataLoader for validation dataset.
    test_loader: DataLoader
        DataLoader for test dataset.
    train_ds: VideoFrameDataset
        Training dataset instance.
    val_ds: VideoFrameDataset
        Validation dataset instance.
    test_ds: VideoFrameDataset
        Test dataset instance.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = VideoFrameDataset(PersonClassification.TRAIN_CSV_PATH, transform=transform, log_dir=out_dir)
    val_ds = VideoFrameDataset(PersonClassification.VAL_CSV_PATH, transform=transform, log_dir=out_dir)
    test_ds = VideoFrameDataset(PersonClassification.TEST_CSV_PATH, transform=transform, log_dir=out_dir)

    train_loader = DataLoader(train_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=PersonConfig.BATCH_SIZE_INFERENCE, shuffle=False, num_workers=4,
                             pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, train_ds, val_ds, test_ds


# ---------------------------
# Save script + hyperparams
# ---------------------------
def save_script_and_hparams(out_dir):
    """Save a copy of training script and hyperparameters to out_dir.
    
    Parameters:
    -----------
    out_dir: str
        Directory to save the script and hyperparameters.
    """
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
      
def setup_evaluation():
    """Setup output directory, device, and data loader."""   
    # Create timestamped evaluation directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = PersonClassification.TRAINED_WEIGHTS_PATH.parent / f"{PersonConfig.MODEL_NAME}_evaluation_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Load test dataset with same transforms as training (but without augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = VideoFrameDataset(
        PersonClassification.TEST_CSV_PATH, 
        transform=transform,
        sequence_length=PersonConfig.SEQUENCE_LENGTH,
        log_dir=eval_dir
    )
    
    test_loader = DataLoader(
        test_ds, 
        batch_size=PersonConfig.BATCH_SIZE_INFERENCE,  # Use larger batch size for inference
        shuffle=False, 
        num_workers=8,  # Increase workers for faster data loading
        collate_fn=collate_fn, 
        pin_memory=True,
        prefetch_factor=2,  # Prefetch more batches
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    return device, test_loader, test_ds, eval_dir

def load_model(device):
    """Load trained model from checkpoint.
    
    Parameters
    ----------
    device : torch.device
        Device to load the model on.
        
    Returns
    -------
    Tuple[nn.Module, nn.Module]
        Loaded CNN and RNN models.
    """
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    # Initialize models with same architecture as training
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)
    
    # Handle compiled models (strip _orig_mod prefix if present)
    def clean_state_dict(state_dict):
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned[k[10:]] = v  # Remove '_orig_mod.' prefix  fv
            else:
                cleaned[k] = v
        return cleaned
    
    # Load state dicts
    cnn.load_state_dict(clean_state_dict(checkpoint['cnn_state']))
    rnn_model.load_state_dict(clean_state_dict(checkpoint['rnn_state']))
    
    return cnn, rnn_model

# ---------------------------
# Visualization utilities  
# ---------------------------

def plot_confusion_matrices(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrices for each class and create multi-class confusion matrix.
    
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
    import seaborn as sns
    
    # Original binary confusion matrices for each class
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
    plt.savefig(output_dir / 'confusion_matrices_binary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convert multi-label to multi-class for comprehensive confusion matrix
    # Categories: 0=no_face, 1=child, 2=adult
    y_true_multiclass = []
    y_pred_multiclass = []
    
    for i in range(len(y_true)):
        # Ground truth conversion
        if y_true[i, 0] == 1:  # child
            true_class = 1
        elif y_true[i, 1] == 1:  # adult
            true_class = 2
        else:  # no face
            true_class = 0
        y_true_multiclass.append(true_class)
        
        # Prediction conversion
        if y_pred[i, 0] == 1:  # predicted child
            pred_class = 1
        elif y_pred[i, 1] == 1:  # predicted adult
            pred_class = 2
        else:  # predicted no face
            pred_class = 0
        y_pred_multiclass.append(pred_class)
    
    y_true_multiclass = np.array(y_true_multiclass)
    y_pred_multiclass = np.array(y_pred_multiclass)
    
    # Multi-class confusion matrix with counts
    multiclass_labels = ['No Face', 'Child', 'Adult']
    cm_multiclass = confusion_matrix(y_true_multiclass, y_pred_multiclass, labels=[0, 1, 2])
    
    # Plot count matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='Blues', 
                xticklabels=multiclass_labels, yticklabels=multiclass_labels,
                cbar_kws={'label': 'Count'})
    plt.title('Multi-class Confusion Matrix (Counts)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_multiclass_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot percentage matrix
    cm_multiclass_pct = cm_multiclass.astype('float') / cm_multiclass.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_multiclass_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=multiclass_labels, yticklabels=multiclass_labels,
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('Multi-class Confusion Matrix (Percentages)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_multiclass_percentages.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save confusion matrix data as CSV for reference
    import pandas as pd
    cm_df_counts = pd.DataFrame(cm_multiclass, 
                               index=[f'True_{label}' for label in multiclass_labels],
                               columns=[f'Pred_{label}' for label in multiclass_labels])
    cm_df_counts.to_csv(output_dir / 'confusion_matrix_multiclass_counts.csv')
    
    cm_df_pct = pd.DataFrame(cm_multiclass_pct,
                            index=[f'True_{label}' for label in multiclass_labels], 
                            columns=[f'Pred_{label}' for label in multiclass_labels])
    cm_df_pct.to_csv(output_dir / 'confusion_matrix_multiclass_percentages.csv')

def plot_metrics_comparison(metrics, output_dir):
    """Plot comparison of metrics across classes.
    
    Parameters
    ----------
    metrics : Dict[str, float]
        Dictionary containing calculated metrics.
    output_dir : str
        Directory to save plots.
    """
    import pandas as pd
    from config import PersonConfig
    
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