"""
End-to-end pipeline:
  ResNet (frame-level feature extractor) -> BiLSTM -> per-frame classifier

Usage:
  python train_cnn_rnn.py
"""

import os
import argparse
import shutil
import datetime
import json
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from constants import PersonClassification, DataPaths
from config import PersonConfig
from person_classifier import VideoFrameDataset, CNNEncoder, FrameRNNClassifier

# ---------------------------
# Dataset
# ---------------------------
# Transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Adds a 50% chance of flipping the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def collate_fn(batch):
    """Collate function for DataLoader to handle variable sequence lengths.
    
    Pads sequences to the maximum length in the batch and creates masks for valid data.
    
    Parameters
    ----------
    batch : List[Tuple]
        List of samples from the dataset, each containing frames, labels, length, and video_id.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]
        A tuple containing:
        - images_padded: torch.Tensor of shape (batch_size, max_seq_len, C, H, W)
        - labels_padded: torch.Tensor of shape (batch_size, max_seq_len, 2) with -100 for padding
        - lengths: torch.Tensor of actual sequence lengths
        - video_ids: List of video identifiers
    """
    batch_sizes = [item[2] for item in batch]
    max_len = max(batch_sizes)
    bs = len(batch)
    C, H, W = batch[0][0].shape[1:]

    images_padded = torch.zeros((bs, max_len, C, H, W))
    labels_padded = torch.full((bs, max_len, 2), -100.0, dtype=torch.float32)  # mask with -100
    lengths = []
    video_ids = []
    for i, (imgs, labs, l, vid) in enumerate(batch):
        images_padded[i, :l] = imgs
        labels_padded[i, :l] = labs
        lengths.append(l)
        video_ids.append(vid)
    return images_padded, labels_padded, torch.tensor(lengths, dtype=torch.long), video_ids

# ---------------------------
# Utilities: training + eval
# ---------------------------
def calculate_metrics(y_true, y_pred, class_names=PersonConfig.TARGET_LABELS):
    """Calculate precision, recall, F1 for each class and macro averages.
    
    Computes per-class and macro-averaged metrics for binary classification tasks.
    
    Parameters
    ----------
    y_true : torch.Tensor or np.ndarray
        Ground truth labels of shape (n_samples, n_classes).
    y_pred : torch.Tensor or np.ndarray
        Predicted labels of shape (n_samples, n_classes).
    class_names : List[str], default=['child', 'adult']
        Names of the classes for metric labeling.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing precision, recall, F1, and accuracy metrics for each class
        and their macro averages.
    """
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Ensure 2D shape
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # If we only have one column but expect two classes, duplicate it
    if y_true.shape[1] == 1 and len(class_names) == 2:
        y_true = np.column_stack([y_true, y_true])
        y_pred = np.column_stack([y_pred, y_pred])
    
    # Debug: Check class distribution
    print(f"Debug: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    for i, class_name in enumerate(class_names):
        true_pos = y_true[:, i].sum()
        pred_pos = y_pred[:, i].sum()
        total = len(y_true)
        print(f"  {class_name}: True positives: {true_pos}/{total} ({true_pos/total:.3f}), Predicted positives: {pred_pos}/{total} ({pred_pos/total:.3f})")
    
    metrics = {}
    
    # Calculate metrics for each class separately
    for i, class_name in enumerate(class_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        metrics[f'{class_name.lower()}_precision'] = precision
        metrics[f'{class_name.lower()}_recall'] = recall
        metrics[f'{class_name.lower()}_f1'] = f1
        metrics[f'{class_name.lower()}_accuracy'] = accuracy_score(y_true[:, i], y_pred[:, i])
    
    # Calculate macro averages (equal weight to both classes)
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_accuracies = []
    
    for i in range(len(class_names)):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average='binary', zero_division=0
        )
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)
        all_accuracies.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    
    metrics['macro_precision'] = np.mean(all_precisions)
    metrics['macro_recall'] = np.mean(all_recalls)
    metrics['macro_f1'] = np.mean(all_f1s)
    metrics['macro_accuracy'] = np.mean(all_accuracies)
    
    return metrics

def sequence_features_from_cnn(cnn, images_padded, lengths, device):
    """Extract CNN features from padded image sequences.
    
    Processes a batch of padded image sequences through the CNN to extract features
    for subsequent RNN processing.
    
    Parameters
    ----------
    cnn : CNNEncoder
        The CNN model for feature extraction.
    images_padded : torch.Tensor
        Padded image sequences of shape (batch_size, max_seq_len, C, H, W).
    lengths : torch.Tensor
        Actual sequence lengths for each sample.
    device : torch.device
        Device to run computations on.
        
    Returns
    -------
    torch.Tensor
        Extracted features of shape (batch_size, max_seq_len, feat_dim).
    """
    bs, max_seq, C, H, W = images_padded.shape
    imgs = images_padded.view(bs * max_seq, C, H, W).to(device)
    feats = cnn(imgs)  # (bs * max_seq, feat_dim)
    feats = feats.view(bs, max_seq, -1)
    return feats

def train_one_epoch(models, optimizers, criterion, dataloader, device, freeze_cnn=False, scaler=None, accumulation_steps=4):
    """Train the model for one epoch with gradient accumulation and mixed precision.
    
    Performs one epoch of training with support for CNN freezing, gradient accumulation,
    and automatic mixed precision training for improved memory efficiency and speed.
    
    Parameters
    ----------
    models : Tuple[nn.Module, nn.Module]
        Tuple containing (CNN encoder, RNN classifier) models.
    optimizers : Tuple[torch.optim.Optimizer, torch.optim.Optimizer]
        Tuple containing (CNN optimizer, RNN optimizer). CNN optimizer can be None if frozen.
    criterion : nn.Module
        Loss function for training.
    dataloader : DataLoader
        Training data loader.
    device : torch.device
        Device to run training on.
    freeze_cnn : bool, default=False
        Whether to freeze CNN parameters during training.
    scaler : torch.cuda.amp.GradScaler, optional
        Gradient scaler for mixed precision training.
    accumulation_steps : int, default=4
        Number of steps to accumulate gradients before updating.
        
    Returns
    -------
    Tuple[float, Dict[str, float]]
        A tuple containing:
        - avg_loss: Average loss for the epoch
        - metrics: Dictionary with training metrics (precision, recall, F1, accuracy)
    """
    cnn, rnn = models
    opt_cnn, opt_rnn = optimizers
    cnn.train(not freeze_cnn)
    rnn.train()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []

    # Add progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (images_padded, labels_padded, lengths, _) in enumerate(progress_bar):
        mask = (labels_padded != -100)  # (bs, max_seq, 2)
        images_padded = images_padded.to(device, non_blocking=True)
        labels_padded = labels_padded.to(device, non_blocking=True)

        # Mixed precision training
        if scaler and device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                if freeze_cnn:
                    with torch.no_grad():
                        feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
                else:
                    feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)

                logits = rnn(feats, lengths)  # (bs, max_seq, 2)

                # flatten and mask
                mask_flat = mask.view(-1, 2)
                logits_flat = logits.view(-1, 2)[mask_flat[:,0] != -100]
                labels_flat = labels_padded.view(-1, 2)[mask_flat[:,0] != -100]

                loss = criterion(logits_flat, labels_flat) / accumulation_steps  # Scale loss

            scaler.scale(loss).backward()
            
            # Only step optimizers every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(opt_rnn)
                if opt_cnn:
                    scaler.unscale_(opt_cnn)
                
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
                if opt_cnn:
                    torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)
                
                scaler.step(opt_rnn)
                if opt_cnn:
                    scaler.step(opt_cnn)
                scaler.update()
                
                opt_rnn.zero_grad()
                if opt_cnn:
                    opt_cnn.zero_grad()
        else:
            # Regular training with gradient accumulation
            if freeze_cnn:
                with torch.no_grad():
                    feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            else:
                feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)

            logits = rnn(feats, lengths)  # (bs, max_seq, 2)

            # flatten and mask
            mask_flat = mask.view(-1, 2)
            logits_flat = logits.view(-1, 2)[mask_flat[:,0] != -100]
            labels_flat = labels_padded.view(-1, 2)[mask_flat[:,0] != -100]

            loss = criterion(logits_flat, labels_flat) / accumulation_steps  # Scale loss
            loss.backward()
            
            # Only step optimizers every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
                if opt_cnn:
                    torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)
                if opt_cnn:
                    opt_cnn.step()
                opt_rnn.step()
                
                opt_rnn.zero_grad()
                if opt_cnn:
                    opt_cnn.zero_grad()

        total_loss += loss.item() * images_padded.size(0) * accumulation_steps  # Unscale for logging

        # Collect predictions and labels for metrics
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Only keep valid predictions (not masked) - keep original shape
            valid_mask = mask.view(-1, 2)[:, 0]  # Use first column as mask (both columns should be same)
            valid_preds = preds.view(-1, 2)[valid_mask]  # Keep 2D shape
            valid_labels = labels_padded.view(-1, 2)[valid_mask]  # Keep 2D shape
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

        # Update progress bar with current loss
        current_loss = total_loss / ((batch_idx + 1) * len(dataloader.dataset) / len(dataloader))
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}'
        })

    # Calculate final metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_labels, all_preds)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics

def eval_on_loader(models, criterion, dataloader, device):
    """Evaluate the model on a data loader.
    
    Performs evaluation on the given dataset and computes loss and classification metrics.
    
    Parameters
    ----------
    models : Tuple[nn.Module, nn.Module]
        Tuple containing (CNN encoder, RNN classifier) models.
    criterion : nn.Module
        Loss function for evaluation.
    dataloader : DataLoader
        Data loader for evaluation data.
    device : torch.device
        Device to run evaluation on.
        
    Returns
    -------
    Tuple[float, Dict[str, float]]
        A tuple containing:
        - avg_loss: Average loss for the evaluation
        - metrics: Dictionary with evaluation metrics (precision, recall, F1, accuracy)
    """
    cnn, rnn = models
    cnn.eval(); rnn.eval()
    total_loss = 0.0
    
    all_preds = []
    all_labels = []

    # Add progress bar for validation
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images_padded, labels_padded, lengths, _ in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)

            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            # Calculate loss
            mask_flat = mask.view(-1, 2)
            logits_flat = logits.view(-1, 2)[mask_flat[:,0] != -100]
            labels_flat = labels_padded.view(-1, 2)[mask_flat[:,0] != -100]
            
            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item() * images_padded.size(0)

            # Collect predictions and labels for metrics
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # Only keep valid predictions (not masked) - keep original shape
            valid_mask = mask.view(-1, 2)[:, 0]  # Use first column as mask (both columns should be same)
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()  # Keep 2D shape
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()  # Keep 2D shape
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

    # Calculate final metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = calculate_metrics(all_labels, all_preds)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, metrics

def setup_training_environment(args):
    """Setup output directory, device, and logging.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
        
    Returns
    -------
    Tuple[str, torch.device, torch.cuda.amp.GradScaler]
        Output directory, device, and gradient scaler.
    """
    # Create timestamped folder name with model abbreviation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PersonClassification.OUTPUT_DIR, f"{PersonConfig.MODEL_NAME}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save a copy of this script and hyperparams there
    save_script_and_hparams(out_dir, args)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Enable mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    return out_dir, device, scaler

def setup_data_loaders(out_dir):
    """Setup data loaders for training and validation.
    
    Parameters
    ----------
    out_dir : str
        Output directory for logging.
        
    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Training and validation data loaders.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Revert to standard ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = VideoFrameDataset(PersonClassification.TRAIN_CSV_PATH, transform=transform, 
                                 sequence_length=PersonConfig.MAX_SEQ_LEN, log_dir=out_dir)
    val_ds = VideoFrameDataset(PersonClassification.VAL_CSV_PATH, transform=transform, 
                               sequence_length=PersonConfig.MAX_SEQ_LEN, log_dir=out_dir)

    train_loader = DataLoader(train_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=True, num_workers=8,
                              collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, batch_size=PersonConfig.BATCH_SIZE, shuffle=False, num_workers=8,
                            collate_fn=collate_fn, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    return train_loader, val_loader, train_ds, val_ds

def setup_models_and_optimizers(device):
    """Setup models and optimizers.
    
    Parameters
    ----------
    device : torch.device
        Device to place models on.
        
    Returns
    -------
    Tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.Optimizer, nn.Module]
        CNN model, RNN model, CNN optimizer, RNN optimizer, and loss criterion.
    """
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=True, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)

    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    torch.nn.init.orthogonal_(param.data)
                else:
                    torch.nn.init.normal_(param.data)
    
    # Only initialize RNN (CNN is pretrained)
    rnn_model.apply(init_weights)

    try:
        cnn = torch.compile(cnn)
        rnn_model = torch.compile(rnn_model)
        print("Models compiled successfully!")
    except Exception as e:
        print(f"Could not compile models: {e}")

    # Improved learning rates and optimizers
    if not PersonConfig.FREEZE_CNN:
        opt_cnn = torch.optim.AdamW(cnn.parameters(), 
                                   lr=PersonConfig.LR * 0.01,  # Much lower for CNN
                                   weight_decay=PersonConfig.WEIGHT_DECAY * 0.1,
                                   betas=(0.9, 0.999))
    else:
        opt_cnn = None
    
    opt_rnn = torch.optim.AdamW(rnn_model.parameters(), 
                               lr=PersonConfig.LR * 2.0,  # Higher for RNN
                               weight_decay=PersonConfig.WEIGHT_DECAY,
                               betas=(0.9, 0.999))
    
    # Improved loss function with stronger class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0, 3.0]).to(device))
    
    print(f"Model setup complete:")
    print(f"  CNN learning rate: {PersonConfig.LR * 0.01 if not PersonConfig.FREEZE_CNN else 'Frozen'}")
    print(f"  RNN learning rate: {PersonConfig.LR * 2.0}")
    print(f"  Loss function: BCEWithLogitsLoss with pos_weight=[3.0, 3.0]")
    print(f"  Optimizer: AdamW with improved settings")
    
    return cnn, rnn_model, opt_cnn, opt_rnn, criterion

def initialize_training_logging(out_dir):
    """Initialize CSV logging for training metrics.
    
    Parameters
    ----------
    out_dir : str
        Output directory for log files.
        
    Returns
    -------
    str
        Path to the CSV log file.
    """
    csv_log_path = os.path.join(out_dir, "training_metrics.csv")
    csv_headers = [
        'epoch', 'train_loss', 'val_loss',
        'train_adult_precision', 'train_adult_recall', 'train_adult_f1',
        'train_child_precision', 'train_child_recall', 'train_child_f1',
        'train_macro_precision', 'train_macro_recall', 'train_macro_f1',
        'val_adult_precision', 'val_adult_recall', 'val_adult_f1',
        'val_child_precision', 'val_child_recall', 'val_child_f1',
        'val_macro_precision', 'val_macro_recall', 'val_macro_f1'
    ]
    
    # Create CSV file with headers
    with open(csv_log_path, 'w') as f:
        f.write(','.join(csv_headers) + '\n')
    
    return csv_log_path

def log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics):
    """Log metrics for one epoch to CSV file.
    
    Parameters
    ----------
    csv_log_path : str
        Path to the CSV log file.
    epoch : int
        Current epoch number.
    train_loss : float
        Training loss.
    val_loss : float
        Validation loss.
    train_metrics : Dict[str, float]
        Training metrics.
    val_metrics : Dict[str, float]
        Validation metrics.
    """
    csv_row = [
        epoch, train_loss, val_loss,
        train_metrics['adult_precision'], train_metrics['adult_recall'], train_metrics['adult_f1'],
        train_metrics['child_precision'], train_metrics['child_recall'], train_metrics['child_f1'],
        train_metrics['macro_precision'], train_metrics['macro_recall'], train_metrics['macro_f1'],
        val_metrics['adult_precision'], val_metrics['adult_recall'], val_metrics['adult_f1'],
        val_metrics['child_precision'], val_metrics['child_recall'], val_metrics['child_f1'],
        val_metrics['macro_precision'], val_metrics['macro_recall'], val_metrics['macro_f1']
    ]
    
    with open(csv_log_path, 'a') as f:
        f.write(','.join([f'{x:.6f}' if isinstance(x, float) else str(x) for x in csv_row]) + '\n')

def print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics):
    """Print training and validation results for one epoch.
    
    Parameters
    ----------
    epoch : int
        Current epoch number.
    train_loss : float
        Training loss.
    val_loss : float
        Validation loss.
    train_metrics : Dict[str, float]
        Training metrics.
    val_metrics : Dict[str, float]
        Validation metrics.
    """
    print(f"\nEpoch {epoch}/{PersonConfig.NUM_EPOCHS}")
    
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"    Adult    - P: {train_metrics['adult_precision']:.3f}, R: {train_metrics['adult_recall']:.3f}, F1: {train_metrics['adult_f1']:.3f}")
    print(f"    Child    - P: {train_metrics['child_precision']:.3f}, R: {train_metrics['child_recall']:.3f}, F1: {train_metrics['child_f1']:.3f}")
    print(f"    Macro    - P: {train_metrics['macro_precision']:.3f}, R: {train_metrics['macro_recall']:.3f}, F1: {train_metrics['macro_f1']:.3f}")
    
    print(f"  Val Loss: {val_loss:.4f}")
    print(f"    Adult    - P: {val_metrics['adult_precision']:.3f}, R: {val_metrics['adult_recall']:.3f}, F1: {val_metrics['adult_f1']:.3f}")
    print(f"    Child    - P: {val_metrics['child_precision']:.3f}, R: {val_metrics['child_recall']:.3f}, F1: {val_metrics['child_f1']:.3f}")
    print(f"    Macro    - P: {val_metrics['macro_precision']:.3f}, R: {val_metrics['macro_recall']:.3f}, F1: {val_metrics['macro_f1']:.3f}")

def save_script_and_hparams(out_dir, args):
    """Save a copy of the training script and hyperparameters to the output directory.
    
    Creates backup copies of the current script and saves all hyperparameters from PersonConfig
    as a JSON file for reproducibility.
    
    Parameters
    ----------
    out_dir : str
        Output directory path where files will be saved.
    args : argparse.Namespace
        Command line arguments passed to the script.
        
    Returns
    -------
    None
    """
    # Save a copy of the current script
    try:
        script_path = os.path.abspath(sys.argv[0])
        script_copy_path = os.path.join(out_dir, os.path.basename(script_path))
        shutil.copyfile(script_path, script_copy_path)
    except Exception as e:
        print(f"Warning: Could not copy script file: {e}")

    # Save hyperparameters as JSON
    hparams = {attr: getattr(PersonConfig, attr) for attr in dir(PersonConfig) if not attr.startswith('_')}
    hparams_path = os.path.join(out_dir, "hyperparameters.json")
    with open(hparams_path, "w") as f:
        json.dump(hparams, f, indent=4)

def handle_checkpointing_and_early_stopping(out_dir, epoch, cnn, rnn_model, opt_cnn, opt_rnn, 
                                           train_metrics, val_metrics, best_val_f1, patience_counter):
    """Handle model checkpointing and early stopping logic.
    
    Parameters
    ----------
    out_dir : str
        Output directory for saving checkpoints.
    epoch : int
        Current epoch number.
    cnn : nn.Module
        CNN model.
    rnn_model : nn.Module
        RNN model.
    opt_cnn : torch.optim.Optimizer
        CNN optimizer.
    opt_rnn : torch.optim.Optimizer
        RNN optimizer.
    train_metrics : Dict[str, float]
        Training metrics.
    val_metrics : Dict[str, float]
        Validation metrics.
    best_val_f1 : float
        Best validation F1 score so far.
    patience_counter : int
        Current patience counter.
        
    Returns
    -------
    Tuple[float, int, bool]
        Updated best_val_f1, patience_counter, and whether to stop early.
    """
    ckpt = {
        'epoch': epoch,
        'cnn_state': cnn.state_dict(),
        'rnn_state': rnn_model.state_dict(),
        'opt_rnn': opt_rnn.state_dict(),
        'opt_cnn': opt_cnn.state_dict() if opt_cnn else None,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics
    }
    
    # Save periodic checkpoints
    if epoch % 10 == 0 or epoch == PersonConfig.NUM_EPOCHS:
        ckpt_path = os.path.join(out_dir, f'ckpt_epoch{epoch:03d}.pth')
        torch.save(ckpt, ckpt_path)

    # Early stopping and best model tracking
    should_stop = False
    if val_metrics['macro_f1'] > best_val_f1:
        best_val_f1 = val_metrics['macro_f1']
        patience_counter = 0  # Reset patience counter
        best_path = os.path.join(out_dir, 'best.pth')
        torch.save(ckpt, best_path)
        print(f"  ⭐ New best macro F1: {best_val_f1:.3f}!")
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter}/{PersonConfig.PATIENCE} epochs")

        if patience_counter >= PersonConfig.PATIENCE:
            print(f"\n🛑 Early stopping triggered! No improvement for {PersonConfig.PATIENCE} epochs.")
            print(f"Best validation macro F1: {best_val_f1:.3f}")
            should_stop = True
    
    return best_val_f1, patience_counter, should_stop

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path):
    """Run the main training loop with early stopping.
    
    Parameters
    ----------
    out_dir : str
        Output directory for saving checkpoints.
    device : torch.device
        Device to run training on.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    cnn : nn.Module
        CNN model.
    rnn_model : nn.Module
        RNN model.
    opt_cnn : torch.optim.Optimizer
        CNN optimizer.
    opt_rnn : torch.optim.Optimizer
        RNN optimizer.
    criterion : nn.Module
        Loss function.
    csv_log_path : str
        Path to CSV log file.
        
    Returns
    -------
    float
        Best validation F1 score achieved.
    """
    best_val_f1 = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {PersonConfig.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {PersonConfig.PATIENCE} epochs")
    print("-" * 50)
    
    for epoch in range(1, PersonConfig.NUM_EPOCHS + 1):
        # Training and validation
        train_loss, train_metrics = train_one_epoch((cnn, rnn_model), (opt_cnn, opt_rnn), criterion, train_loader,
                                                    device, freeze_cnn=PersonConfig.FREEZE_CNN, scaler=scaler)
        val_loss, val_metrics = eval_on_loader((cnn, rnn_model), criterion, val_loader, device)
        
        # Log and print results
        print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics)
        
        # Handle checkpointing and early stopping
        best_val_f1, patience_counter, should_stop = handle_checkpointing_and_early_stopping(
            out_dir, epoch, cnn, rnn_model, opt_cnn, opt_rnn, train_metrics, val_metrics, best_val_f1, patience_counter
        )
        
        if should_stop:
            break
    
    if patience_counter < PersonConfig.PATIENCE:
        print(f"Training finished. Best macro F1: {best_val_f1:.3f}")
    else:
        print(f"Training stopped early. Best macro F1: {best_val_f1:.3f}")
    
    return best_val_f1
        
# ---------------------------
# Main: parse args, start training
# ---------------------------
def main():
    """Main training function for the CNN-RNN person classification model.
    
    Sets up the training pipeline including data loading, model initialization, 
    training loop with early stopping, and model checkpointing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--diagnose', action='store_true', help='Run diagnostics only')
    args = parser.parse_args()

    # Setup training environment
    out_dir, device, scaler = setup_training_environment(args)
    
    # Setup data loaders
    train_loader, val_loader, train_ds, val_ds = setup_data_loaders(out_dir)
    
    # Setup models and optimizers
    cnn, rnn_model, opt_cnn, opt_rnn, criterion = setup_models_and_optimizers(device)
    
    # Run diagnostics if requested
    if args.diagnose:
        print("Running diagnostics on training data...")
        diagnostics = diagnose_learning_issues((cnn, rnn_model), train_loader, device)
        suggestions = suggest_improvements(diagnostics)
        
        print("\n💡 Suggested Improvements:")
        for suggestion in suggestions:
            print(f"  • {suggestion}")
        return
    
    # Initialize logging
    csv_log_path = initialize_training_logging(out_dir)
    
    # Run training loop
    best_val_f1 = run_training_loop(out_dir, device, scaler, train_loader, val_loader, 
                                   cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path)
    
    # Log any skipped files
    train_ds.log_skipped_files()
    val_ds.log_skipped_files()
        
if __name__ == '__main__':
    main()

def diagnose_learning_issues(models, dataloader, device, num_batches=5):
    """Diagnose potential learning issues in the model.
    
    Parameters
    ----------
    models : Tuple[nn.Module, nn.Module]
        Tuple containing (CNN encoder, RNN classifier) models.
    dataloader : DataLoader
        Data loader to analyze.
    device : torch.device
        Device to run diagnostics on.
    num_batches : int, default=5
        Number of batches to analyze.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing diagnostic information.
    """
    cnn, rnn = models
    cnn.eval()
    rnn.eval()
    
    diagnostics = {
        'label_distribution': {'child': 0, 'adult': 0, 'total_samples': 0},
        'gradient_norms': [],
        'activations': [],
        'loss_values': [],
        'prediction_stats': {'all_zeros': 0, 'all_ones': 0, 'mixed': 0}
    }
    
    criterion = nn.BCEWithLogitsLoss()
    
    print("🔍 Running Learning Diagnostics...")
    print("=" * 50)
    
    with torch.no_grad():
        for batch_idx, (images_padded, labels_padded, lengths, _) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            mask = (labels_padded != -100)
            
            # Analyze label distribution
            valid_labels = labels_padded[mask]
            if len(valid_labels) > 0:
                adult_count = (valid_labels[:, 1] == 1).sum().item()
                child_count = (valid_labels[:, 0] == 1).sum().item()
                diagnostics['label_distribution']['adult'] += adult_count
                diagnostics['label_distribution']['child'] += child_count
                diagnostics['label_distribution']['total_samples'] += len(valid_labels)
            
            # Forward pass
            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            # Check activations
            diagnostics['activations'].append({
                'cnn_features_mean': feats.mean().item(),
                'cnn_features_std': feats.std().item(),
                'logits_mean': logits.mean().item(),
                'logits_std': logits.std().item(),
                'logits_min': logits.min().item(),
                'logits_max': logits.max().item()
            })
            
            # Check predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            valid_preds = preds[mask[:, :, 0]]
            
            if len(valid_preds) > 0:
                all_zeros = (valid_preds.sum(dim=1) == 0).sum().item()
                all_ones = (valid_preds.sum(dim=1) == 2).sum().item()
                mixed = len(valid_preds) - all_zeros - all_ones
                
                diagnostics['prediction_stats']['all_zeros'] += all_zeros
                diagnostics['prediction_stats']['all_ones'] += all_ones
                diagnostics['prediction_stats']['mixed'] += mixed
            
            # Calculate loss
            mask_flat = mask.view(-1, 2)
            logits_flat = logits.view(-1, 2)[mask_flat[:, 0] != -100]
            labels_flat = labels_padded.view(-1, 2)[mask_flat[:, 0] != -100]
            
            if len(logits_flat) > 0:
                loss = criterion(logits_flat, labels_flat)
                diagnostics['loss_values'].append(loss.item())
    
    # Print diagnostics
    print(f"📊 Label Distribution:")
    total = diagnostics['label_distribution']['total_samples']
    adult_pct = diagnostics['label_distribution']['adult'] / total * 100 if total > 0 else 0
    child_pct = diagnostics['label_distribution']['child'] / total * 100 if total > 0 else 0
    print(f"  Adult: {diagnostics['label_distribution']['adult']}/{total} ({adult_pct:.1f}%)")
    print(f"  Child: {diagnostics['label_distribution']['child']}/{total} ({child_pct:.1f}%)")
    
    if diagnostics['activations']:
        avg_activations = {
            key: np.mean([a[key] for a in diagnostics['activations']])
            for key in diagnostics['activations'][0].keys()
        }
        print(f"\n🧠 Activation Statistics:")
        print(f"  CNN Features - Mean: {avg_activations['cnn_features_mean']:.4f}, Std: {avg_activations['cnn_features_std']:.4f}")
        print(f"  Logits - Mean: {avg_activations['logits_mean']:.4f}, Std: {avg_activations['logits_std']:.4f}")
        print(f"  Logits Range: [{avg_activations['logits_min']:.4f}, {avg_activations['logits_max']:.4f}]")
    
    if diagnostics['loss_values']:
        avg_loss = np.mean(diagnostics['loss_values'])
        std_loss = np.std(diagnostics['loss_values'])
        print(f"\n💥 Loss Statistics:")
        print(f"  Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    
    pred_total = sum(diagnostics['prediction_stats'].values())
    if pred_total > 0:
        print(f"\n🎯 Prediction Patterns:")
        print(f"  All Zeros: {diagnostics['prediction_stats']['all_zeros']}/{pred_total} ({diagnostics['prediction_stats']['all_zeros']/pred_total*100:.1f}%)")
        print(f"  All Ones: {diagnostics['prediction_stats']['all_ones']}/{pred_total} ({diagnostics['prediction_stats']['all_ones']/pred_total*100:.1f}%)")
        print(f"  Mixed: {diagnostics['prediction_stats']['mixed']}/{pred_total} ({diagnostics['prediction_stats']['mixed']/pred_total*100:.1f}%)")
    
    # Identify potential issues
    print(f"\n⚠️  Potential Issues:")
    issues = []
    
    if adult_pct < 5 or child_pct < 5:
        issues.append(f"Severe class imbalance detected")
    
    if avg_activations and abs(avg_activations['logits_mean']) > 2:
        issues.append(f"Logits saturated (mean: {avg_activations['logits_mean']:.2f})")
    
    if avg_activations and avg_activations['logits_std'] < 0.1:
        issues.append(f"Low logit variance - model not learning")
    
    if diagnostics['prediction_stats']['all_zeros'] > pred_total * 0.9:
        issues.append(f"Model predicting mostly zeros")
    elif diagnostics['prediction_stats']['all_ones'] > pred_total * 0.9:
        issues.append(f"Model predicting mostly ones")
    
    if avg_loss and avg_loss > 1.5:
        issues.append(f"High loss indicates learning difficulties")
    
    if not issues:
        issues.append("No obvious issues detected")
    
    for issue in issues:
        print(f"  • {issue}")
    
    print("=" * 50)
    return diagnostics

def suggest_improvements(diagnostics):
    """Suggest improvements based on diagnostic results.
    
    Parameters
    ----------
    diagnostics : Dict[str, Any]
        Results from diagnose_learning_issues function.
        
    Returns
    -------
    List[str]
        List of suggested improvements.
    """
    suggestions = []
    
    # Check class imbalance
    total = diagnostics['label_distribution']['total_samples']
    if total > 0:
        adult_pct = diagnostics['label_distribution']['adult'] / total * 100
        child_pct = diagnostics['label_distribution']['child'] / total * 100
        
        if adult_pct < 10 or child_pct < 10:
            suggestions.append("Use stronger class weights in loss function")
            suggestions.append("Consider focal loss for extreme imbalance")
            suggestions.append("Try oversampling minority class")
    
    # Check activations
    if diagnostics['activations']:
        avg_activations = {
            key: np.mean([a[key] for a in diagnostics['activations']])
            for key in diagnostics['activations'][0].keys()
        }
        
        if avg_activations['logits_std'] < 0.1:
            suggestions.append("Increase learning rate")
            suggestions.append("Reduce regularization (weight decay, dropout)")
            suggestions.append("Check if CNN is frozen when it shouldn't be")
        
        if abs(avg_activations['logits_mean']) > 2:
            suggestions.append("Add gradient clipping")
            suggestions.append("Reduce learning rate")
            suggestions.append("Check for label encoding issues")
    
    # Check prediction patterns
    pred_total = sum(diagnostics['prediction_stats'].values())
    if pred_total > 0:
        zero_pct = diagnostics['prediction_stats']['all_zeros'] / pred_total
        one_pct = diagnostics['prediction_stats']['all_ones'] / pred_total
        
        if zero_pct > 0.8:
            suggestions.append("Model stuck predicting zeros - increase positive class weight")
            suggestions.append("Try different initialization")
        elif one_pct > 0.8:
            suggestions.append("Model stuck predicting ones - decrease positive class weight")
    
    # Check loss
    if diagnostics['loss_values']:
        avg_loss = np.mean(diagnostics['loss_values'])
        if avg_loss > 1.5:
            suggestions.append("High loss - check label format and data quality")
            suggestions.append("Consider simpler model architecture")
            suggestions.append("Increase training time")
    
    return suggestions
