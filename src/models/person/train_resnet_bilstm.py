"""
End-to-end pipeline (refactored):
  ResNet (frame-level feature extractor) -> BiLSTM -> per-frame classifier

This file contains the high-level training loop and model orchestration.
"""

import os
import warnings
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from config import PersonConfig
from constants import PersonClassification
from utils import (
    setup_environment, 
    setup_data_loaders,
    setup_models_and_optimizers,
    calculate_metrics,
    initialize_training_logging,
    log_epoch_metrics,
    print_epoch_results,
)

# Set conservative thread limits to avoid DataLoader hangs
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings("ignore", message="Can't initialize NVML")
torch.set_num_threads(4)

# ---------------------------
# Training/Evaluation Functions (Core to this script)
# ---------------------------

def sequence_features_from_cnn(cnn, images_padded, lengths, device):
    """
    Extracts features from padded image sequences using the CNN encoder.
    """
    bs, max_seq, C, H, W = images_padded.shape
    images_flat = images_padded.view(bs * max_seq, C, H, W).to(device)
    feats_flat = cnn(images_flat)
    feat_dim = feats_flat.shape[-1]
    feats = feats_flat.view(bs, max_seq, feat_dim)
    return feats

def handle_checkpointing_and_early_stopping(out_dir, epoch, cnn, rnn_model, opt_cnn, opt_rnn,
                                           train_metrics, val_metrics, best_val_f1, patience_counter):
    """Saves checkpoints, manages early stopping, and updates patience counter."""
    ckpt = {
        'epoch': epoch,
        'cnn_state': cnn.state_dict(),
        'rnn_state': rnn_model.state_dict(),
        'opt_rnn': opt_rnn.state_dict() if opt_rnn else None,
        'opt_cnn': opt_cnn.state_dict() if opt_cnn else None,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
    }

    if epoch == PersonConfig.NUM_EPOCHS:
        last_path = Path(out_dir) / 'last.pth'
        torch.save(ckpt, last_path)

    should_stop = False
    if val_metrics['macro_f1'] > best_val_f1:
        best_val_f1 = val_metrics['macro_f1']
        patience_counter = 0
        best_path = Path(out_dir) / 'best.pth'
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

def eval_on_loader(models, criterion, dataloader, device):
    """Evaluate models on dataloader and return loss + metrics."""
    cnn, rnn = models
    cnn.eval()
    rnn.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images_padded, labels_padded, lengths, _ in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)

            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            mask_flat = mask.view(-1, 2)[:, 0]
            logits_flat = logits.view(-1, 2)[mask_flat]
            labels_flat = labels_padded.view(-1, 2)[mask_flat]

            loss = criterion(logits_flat, labels_flat)

            batch_size = images_padded.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            valid_preds = preds.view(-1, 2)[mask_flat].cpu()
            valid_labels = labels_padded.view(-1, 2)[mask_flat].cpu()

            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
    
    if not all_preds:
        raise RuntimeError("No valid predictions collected during evaluation")
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics

def eval_on_loader(models, criterion, dataloader, device):
    """Evaluate models on dataloader and return loss + metrics."""
    cnn, rnn = models
    cnn.eval()
    rnn.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images_padded, labels_padded, lengths, _ in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)

            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            mask_flat = mask.view(-1, 2)[:, 0]
            logits_flat = logits.view(-1, 2)[mask_flat]
            labels_flat = labels_padded.view(-1, 2)[mask_flat]

            loss = criterion(logits_flat, labels_flat)

            batch_size = images_padded.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            valid_preds = preds.view(-1, 2)[mask_flat].cpu()
            valid_labels = labels_padded.view(-1, 2)[mask_flat].cpu()

            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
    
    if not all_preds:
        raise RuntimeError("No valid predictions collected during evaluation")
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics

def train_one_epoch(models, optimizers, criterion, dataloader, device, freeze_cnn=False, scaler=None, accumulation_steps=4):
    """
    Train models for one epoch with gradient accumulation and optional AMP.
    """
    cnn, rnn = models
    opt_cnn, opt_rnn = optimizers

    cnn.train(not freeze_cnn)
    rnn.train()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (images_padded, labels_padded, lengths, _) in enumerate(progress_bar):
        images_padded = images_padded.to(device, non_blocking=True)
        labels_padded = labels_padded.to(device, non_blocking=True)
        mask = (labels_padded != -100)

        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            if freeze_cnn:
                with torch.no_grad():
                    feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            else:
                feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)

            logits = rnn(feats, lengths)
            mask_flat = mask.view(-1, 2)[:, 0]
            logits_flat = logits.view(-1, 2)[mask_flat]
            labels_flat = labels_padded.view(-1, 2)[mask_flat]
            loss = criterion(logits_flat, labels_flat) / accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                if opt_rnn:
                    scaler.unscale_(opt_rnn)
                if opt_cnn:
                    scaler.unscale_(opt_cnn)
                
            if opt_rnn:
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
                if opt_cnn:
                    torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)

            if scaler:
                if opt_rnn:
                    scaler.step(opt_rnn)
                if opt_cnn:
                    scaler.step(opt_cnn)
                scaler.update()
            else:
                if opt_rnn:
                    opt_rnn.step()
                if opt_cnn:
                    opt_cnn.step()

            if opt_rnn:
                opt_rnn.zero_grad()
            if opt_cnn:
                opt_cnn.zero_grad()
                
        # ... (rest of the metric collection logic remains the same)
        # Collect predictions and update progress bar
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            valid_mask = mask.view(-1, 2)[:, 0]
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

        batch_size = images_padded.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size
        avg_loss_so_far = total_loss / total_samples if total_samples else 0.0
        progress_bar.set_postfix({'loss': f"{avg_loss_so_far:.4f}"})

    if not all_preds:
        raise RuntimeError("No valid predictions collected during training epoch")

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path):
    """
    Orchestrates the full training process across multiple epochs.
    
    Parameters
    ----------
    out_dir : Path
        Directory to save outputs and checkpoints.
    device : torch.device
        Device to run training on.
    scaler : torch.cuda.amp.GradScaler
        Gradient scaler for mixed precision training.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data
    cnn : nn.Module
        CNN feature extractor model.
    rnn_model : nn.Module
        RNN classifier model.
    opt_cnn : torch.optim.Optimizer
        Optimizer for CNN model.
    opt_rnn : torch.optim.Optimizer
        Optimizer for RNN model.
    criterion : nn.Module
        Loss function.
    csv_log_path : Path
        Path to CSV file for logging training metrics.
    """
    best_val_f1 = 0.0
    patience_counter = 0

    print(f"\nStarting training for {PersonConfig.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {PersonConfig.PATIENCE} epochs")
    print("-" * 50)

    for epoch in range(1, PersonConfig.NUM_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            (cnn, rnn_model), (opt_cnn, opt_rnn), criterion, train_loader,
            device, freeze_cnn=PersonConfig.FREEZE_CNN, scaler=scaler,
        )

        val_loss, val_metrics = eval_on_loader((cnn, rnn_model), criterion, val_loader, device)

        print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics)

        best_val_f1, patience_counter, should_stop = handle_checkpointing_and_early_stopping(
            out_dir, epoch, cnn, rnn_model, opt_cnn, opt_rnn, train_metrics, val_metrics, best_val_f1, patience_counter
        )

        if should_stop:
            break

    if patience_counter < PersonConfig.PATIENCE:
        print(f"Training finished. Best macro F1: {best_val_f1:.3f}")
    else:
        print(f"Training stopped early. Best macro F1: {best_val_f1:.3f}")

# ---------------------------
# Entry point
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir, device, scaler = setup_environment(is_training=True)

    train_loader, train_ds = setup_data_loaders(PersonClassification.TRAIN_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=True, log_dir=out_dir)
    val_loader, val_ds = setup_data_loaders(PersonClassification.VAL_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=False, log_dir=out_dir)

    cnn, rnn_model, opt_cnn, opt_rnn, criterion = setup_models_and_optimizers(device)

    csv_log_path = initialize_training_logging(out_dir)

    best_val_f1 = run_training_loop(
        out_dir, device, scaler, train_loader, val_loader,
        cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path,
    )

    try:
        train_ds.log_skipped_files()
        val_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")

if __name__ == '__main__':
    main()