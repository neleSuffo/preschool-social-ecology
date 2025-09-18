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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from config import PersonConfig
from constants import PersonClassification
from person_classifier import VideoFrameDataset


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
    out_dir = os.path.join(PersonClassification.OUTPUT_DIR, f"{PersonConfig.MODEL_NAME}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    save_script_and_hparams(out_dir)

    # use cuda if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    return out_dir, device, scaler


# ---------------------------
# Data Loaders
# ---------------------------
def setup_data_loaders(out_dir):
    """Setup train and validation data loaders.
    
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
    train_ds: VideoFrameDataset
        Training dataset instance.
    val_ds: VideoFrameDataset
        Validation dataset instance.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_ds = VideoFrameDataset(PersonClassification.TRAIN_CSV_PATH, transform=transform, log_dir=out_dir)
    val_ds = VideoFrameDataset(PersonClassification.VAL_CSV_PATH, transform=transform, log_dir=out_dir)

    train_loader = DataLoader(train_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=True, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    return train_loader, val_loader, train_ds, val_ds


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
    eval_dir = os.path.join(PersonClassification.OUTPUT_DIR, f"{PersonConfig.MODEL_NAME}_evaluation_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)
    
    print(f"Results will be saved to: {eval_dir}")
    
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        batch_size=PersonConfig.BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    print(f"Test dataset loaded: {len(test_ds)} sequences")
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
    print(f"Loading model from {PersonClassification.TRAINED_WEIGHTS_PATH}")
    checkpoint = torch.load(PersonClassification.TRAINED_WEIGHTS_PATH, map_location=device)
    
    # Initialize models with same architecture as training
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=False, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)
    
    # Handle compiled models (strip _orig_mod prefix if present)
    def clean_state_dict(state_dict):
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                cleaned[k[10:]] = v  # Remove '_orig_mod.' prefix
            else:
                cleaned[k] = v
        return cleaned
    
    # Load state dicts
    cnn.load_state_dict(clean_state_dict(checkpoint['cnn_state']))
    rnn_model.load_state_dict(clean_state_dict(checkpoint['rnn_state']))
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    
    # Print loaded model metrics if available
    if 'val_metrics' in checkpoint:
        val_metrics = checkpoint['val_metrics']
        print(f"Loaded model validation performance:")
        print(f"  Macro F1: {val_metrics.get('macro_f1', 'N/A'):.3f}")
        print(f"  Adult F1: {val_metrics.get('adult_f1', 'N/A'):.3f}")
        print(f"  Child F1: {val_metrics.get('child_f1', 'N/A'):.3f}")
    
    return cnn, rnn_model