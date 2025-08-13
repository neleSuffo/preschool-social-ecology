"""
End-to-end pipeline:
  ResNet (frame-level feature extractor) -> BiLSTM -> per-frame classifier

Usage:
  python train_cnn_rnn.py --train_csv frames_train.csv --val_csv frames_val.csv --out_dir checkpoints
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

# ---------------------------
# Dataset
# ---------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, csv_file, sequence_length=10, transform=None, log_dir=None):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.transform = transform
        self.skipped_files = []
        self.log_dir = log_dir

        self.grouped = []
        for video_id, group in self.data.groupby('video_id'):
            sorted_group = group.sort_values('frame_id')
            self.grouped.append(sorted_group.reset_index(drop=True))

    def log_skipped_files(self):
        """Write skipped files to a log file"""
        if self.log_dir and self.skipped_files:
            log_path = os.path.join(self.log_dir, "skipped_frames.txt")
            with open(log_path, 'w') as f:
                f.write(f"Total skipped frames: {len(self.skipped_files)}\n\n")
                f.write("Skipped files:\n")
                for file_path, reason in self.skipped_files:
                    f.write(f"{file_path} - {reason}\n")
            print(f"Logged {len(self.skipped_files)} skipped frames to {log_path}")

    def __len__(self):
        return sum(max(0, len(g) - self.sequence_length + 1) for g in self.grouped)

    def __getitem__(self, idx):
        total = 0
        for group_idx, group in enumerate(self.grouped):
            length = max(0, len(group) - self.sequence_length + 1)
            if idx < total + length:
                start_idx = idx - total
                frames = []
                labels = []
                video_id = None
                
                # Try to load frames, skip broken ones
                current_idx = start_idx
                frames_loaded = 0
                max_attempts = len(group) - start_idx  # Don't go beyond group bounds
                
                while frames_loaded < self.sequence_length and current_idx < len(group):
                    row = group.iloc[current_idx]
                    file_path = row['file_path']
                    
                    try:
                        # Check if file exists
                        if not os.path.exists(file_path):
                            self.skipped_files.append((file_path, "File not found"))
                            current_idx += 1
                            continue
                        
                        img = Image.open(file_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        frames.append(img)
                        labels.append([row['adult'], row['child']])
                        if video_id is None:
                            video_id = row['video_id']
                        frames_loaded += 1
                        
                    except Exception as e:
                        self.skipped_files.append((file_path, f"Error loading: {str(e)}"))
                        
                    current_idx += 1
                
                # If we couldn't load enough frames, pad with the last valid frame or skip this sequence
                if frames_loaded == 0:
                    # No valid frames found, try next sequence
                    total += length
                    continue
                elif frames_loaded < self.sequence_length:
                    # Pad with last frame
                    while len(frames) < self.sequence_length:
                        frames.append(frames[-1].clone())
                        labels.append(labels[-1].copy())
                
                frames = torch.stack(frames)  # shape: (seq_len, C, H, W)
                labels = torch.tensor(labels).float()  # shape: (seq_len, 2)
                return frames, labels, len(frames), video_id
            total += length
        raise IndexError("Index out of range")

# Transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # optionally normalize with ImageNet mean/std if using pretrained CNN
])

def collate_fn(batch):
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
# Model: CNN backbone + RNN
# ---------------------------
class CNNEncoder(nn.Module):
    """
    Uses a pretrained ResNet (without final FC) to extract per-frame features.
    Outputs feature vector for each frame.
    """
    def __init__(self, backbone='resnet18', pretrained=True, feat_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            res = models.resnet18(pretrained=pretrained)
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]  # remove FC and avgpool kept? last is avgpool
            self.encoder = nn.Sequential(*modules)  # outputs (B, feat_in, 1, 1)
            self.feat_dim = feat_in
        else:
            raise NotImplementedError("Only resnet18 implemented; swap if you want resnet50")
        # optional projection to reduce dim
        if self.feat_dim != feat_dim:
            self.project = nn.Linear(self.feat_dim, feat_dim)
            self.feat_dim = feat_dim
        else:
            self.project = None

    def forward(self, x):
        # x: (batch*seq_len, C, H, W)
        f = self.encoder(x)  # (N, feat_in, 1,1)
        f = f.view(f.size(0), -1)  # (N, feat_in)
        if self.project is not None:
            f = self.project(f)
        return f  # (N, feat_dim)

class FrameRNNClassifier(nn.Module):
    def __init__(self, feat_dim=512, rnn_hidden=256, rnn_layers=1, bidirectional=True, num_outputs=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feat_dim, hidden_size=rnn_hidden,
                           num_layers=rnn_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if rnn_layers>1 else 0.0)
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim//2, num_outputs)  # 2 outputs: adult, child
        )

    def forward(self, feats, lengths):
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(rnn_out)  # (batch, max_seq, 2)
        return logits

# ---------------------------
# Utilities: training + eval
# ---------------------------
def calculate_metrics(y_true, y_pred, class_names=['Adult', 'Child']):
    """Calculate precision, recall, F1 for each class and macro averages"""
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
    
    print(f"Debug: y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    
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
    """
    images_padded: (batch, max_seq, C, H, W)
    returns: feats_padded (batch, max_seq, feat_dim)
    """
    bs, max_seq, C, H, W = images_padded.shape
    imgs = images_padded.view(bs * max_seq, C, H, W).to(device)
    feats = cnn(imgs)  # (bs * max_seq, feat_dim)
    feats = feats.view(bs, max_seq, -1)
    return feats

def train_one_epoch(models, optimizers, criterion, dataloader, device, freeze_cnn=False, scaler=None, accumulation_steps=4):
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

def save_script_and_hparams(out_dir, args):
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
        
# ---------------------------
# Main: parse args, start training
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Create timestamped folder name with model abbreviation
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(PersonClassification.OUTPUT_DIR, f"{PersonConfig.MODEL_NAME}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save a copy of this script and hyperparams there
    save_script_and_hparams(out_dir, args)

    transform = transforms.Compose([
        transforms.Resize((192, 192)),  # Reduced from 224x224 for ~30% speedup
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

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Enable mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    cnn = CNNEncoder(backbone='resnet18', pretrained=True, feat_dim=256).to(device)  # Reduced from 512
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim, rnn_hidden=128,  # Reduced from 256
                                  rnn_layers=1, bidirectional=True,  # Reduced from 2 layers
                                  num_outputs=2, dropout=0.3).to(device)

    # Compile models for faster execution (PyTorch 2.0+)
    try:
        cnn = torch.compile(cnn)
        rnn_model = torch.compile(rnn_model)
        print("Models compiled successfully!")
    except Exception as e:
        print(f"Could not compile models: {e}")

    opt_cnn = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=PersonConfig.LR) if not PersonConfig.FREEZE_CNN else None
    opt_rnn = torch.optim.Adam(rnn_model.parameters(), lr=PersonConfig.LR)

    criterion = nn.BCEWithLogitsLoss()

    # Initialize CSV logging
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
    
    best_val_f1 = 0.0
    patience_counter = 0
    print(f"\nStarting training for {PersonConfig.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {PersonConfig.PATIENCE} epochs")
    print("-" * 50)
    
    for epoch in range(1, PersonConfig.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{PersonConfig.NUM_EPOCHS}")
        
        # Training
        train_loss, train_metrics = train_one_epoch((cnn, rnn_model), (opt_cnn, opt_rnn), criterion, train_loader,
                                                    device, freeze_cnn=PersonConfig.FREEZE_CNN, scaler=scaler)
        
        # Validation
        val_loss, val_metrics = eval_on_loader((cnn, rnn_model), criterion, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Adult    - P: {train_metrics['adult_precision']:.3f}, R: {train_metrics['adult_recall']:.3f}, F1: {train_metrics['adult_f1']:.3f}")
        print(f"    Child    - P: {train_metrics['child_precision']:.3f}, R: {train_metrics['child_recall']:.3f}, F1: {train_metrics['child_f1']:.3f}")
        print(f"    Macro    - P: {train_metrics['macro_precision']:.3f}, R: {train_metrics['macro_recall']:.3f}, F1: {train_metrics['macro_f1']:.3f}")
        
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    Adult    - P: {val_metrics['adult_precision']:.3f}, R: {val_metrics['adult_recall']:.3f}, F1: {val_metrics['adult_f1']:.3f}")
        print(f"    Child    - P: {val_metrics['child_precision']:.3f}, R: {val_metrics['child_recall']:.3f}, F1: {val_metrics['child_f1']:.3f}")
        print(f"    Macro    - P: {val_metrics['macro_precision']:.3f}, R: {val_metrics['macro_recall']:.3f}, F1: {val_metrics['macro_f1']:.3f}")

        # Log metrics to CSV
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

        ckpt = {
            'epoch': epoch,
            'cnn_state': cnn.state_dict(),
            'rnn_state': rnn_model.state_dict(),
            'opt_rnn': opt_rnn.state_dict(),
            'opt_cnn': opt_cnn.state_dict() if opt_cnn else None,
            'val_metrics': val_metrics,
            'train_metrics': train_metrics
        }
        ckpt_path = os.path.join(out_dir, f'ckpt_epoch{epoch:03d}.pth')
        if epoch % 10 == 0 or epoch == PersonConfig.NUM_EPOCHS:
            torch.save(ckpt, ckpt_path)

        # Early stopping and best model tracking
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
                break

    if patience_counter < PersonConfig.PATIENCE:
        print(f"Training finished. Best macro F1: {best_val_f1:.3f}")
    else:
        print(f"Training stopped early at epoch {epoch}. Best macro F1: {best_val_f1:.3f}")
    
    # Log any skipped files
    train_ds.log_skipped_files()
    val_ds.log_skipped_files()
        
if __name__ == '__main__':
    main()
