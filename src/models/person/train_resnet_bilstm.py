"""
End-to-end pipeline (refactored):
  ResNet-BiLSTM -> per-frame classifier (E2E fine-tuning)

This file contains the high-level training loop and model orchestration.
"""

import os
import warnings
import torch
from tqdm import tqdm
import argparse
import torch.nn as nn
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
import numpy as np

# Set conservative thread limits to avoid DataLoader hangs
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings("ignore", message="Can't initialize NVML")
torch.set_num_threads(4)

# ---------------------------
# Training/Evaluation Functions
# ---------------------------

def handle_checkpointing_and_early_stopping(out_dir, epoch, model, optimizer,
                                           train_metrics, val_metrics, best_val_f1, patience_counter):
    """Saves checkpoints, manages early stopping, and updates patience counter."""
    
    # 💥 CRITICAL FIX: Save the combined model and unified optimizer states
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict() if optimizer else None, # Use unified optimizer
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

def eval_on_loader(model, criterion, dataloader, device):
    """Evaluate models on dataloader and return loss + metrics."""
    
    # 💥 CRITICAL FIX: Use 'model' for eval
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        # Input is now 'images_padded'
        for images_padded, labels_padded, lengths, _ in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)

            # Pass images directly to the combined model
            logits = model(images_padded, lengths)
            
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

def train_one_epoch(model, optimizer, criterion, dataloader, device, scaler=None, accumulation_steps=4):
    """
    Train models for one epoch with gradient accumulation and optional AMP.
    """
    
    # 💥 CRITICAL FIX: Use 'model' for train
    model.train()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    # Input is now 'images_padded'
    for batch_idx, (images_padded, labels_padded, lengths, _) in enumerate(progress_bar):
        images_padded = images_padded.to(device, non_blocking=True)
        labels_padded = labels_padded.to(device, non_blocking=True)
        mask = (labels_padded != -100)

        # Diagnostic print for first batch of each epoch
        if batch_idx == 0:
            print("\n[DIAGNOSTIC] Image stats:")
            print(f"  mean: {images_padded.mean().item():.4f}, std: {images_padded.std().item():.4f}, min: {images_padded.min().item():.4f}, max: {images_padded.max().item():.4f}")
            print("[DIAGNOSTIC] Label distribution:")
            mask_flat = mask.view(-1, 2)[:, 0]
            valid_labels = labels_padded.view(-1, 2)[mask_flat].cpu().numpy()
            if valid_labels.ndim == 1:
                valid_labels = valid_labels.reshape(-1, 2)
            if valid_labels.shape[1] == 2:
                child_count = int((valid_labels[:,0] == 1).sum())
                adult_count = int((valid_labels[:,1] == 1).sum())
                print(f"  child frames: {child_count}, adult frames: {adult_count}, total valid: {valid_labels.shape[0]}")
            else:
                print(f"  Unexpected label shape: {valid_labels.shape}")

        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            # Pass images directly to the combined model
            logits = model(images_padded, lengths)
            mask_flat = mask.view(-1, 2)[:, 0]
            logits_flat = logits.view(-1, 2)[mask_flat]
            labels_flat = labels_padded.view(-1, 2)[mask_flat]
            loss = criterion(logits_flat, labels_flat) / accumulation_steps

        # Diagnostic print for loss value
        if batch_idx == 0:
            print(f"[DIAGNOSTIC] First batch loss: {loss.item():.6f}")

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                # 💥 CRITICAL FIX: Use 'optimizer' and 'model'
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # 💥 CRITICAL FIX: Use 'optimizer' and 'model'
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad()
                
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

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, model, optimizer, criterion, csv_log_path):
    """
    Orchestrates the full training process across multiple epochs.
    """
    best_val_f1 = 0.0
    patience_counter = 0

    print(f"\nStarting training for {PersonConfig.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {PersonConfig.PATIENCE} epochs")
    print("-" * 50)

    for epoch in range(1, PersonConfig.NUM_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model, optimizer, criterion, train_loader, device, scaler=scaler,
        )

        val_loss, val_metrics = eval_on_loader(model, criterion, val_loader, device)

        print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics)

        # 💥 CRITICAL FIX: Pass the combined model and optimizer
        best_val_f1, patience_counter, should_stop = handle_checkpointing_and_early_stopping(
            out_dir, epoch, model, optimizer, train_metrics, val_metrics, best_val_f1, patience_counter
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

    # 💥 NOTE: is_feature_extraction=True ensures the DataLoader loads raw images
    train_loader, train_ds = setup_data_loaders(
        PersonClassification.TRAIN_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=True, log_dir=out_dir, 
        is_feature_extraction=True, split_name='train'
    )
    val_loader, val_ds = setup_data_loaders(
        PersonClassification.VAL_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=False, log_dir=out_dir, 
        is_feature_extraction=True, split_name='val'
    )

    # Load combined model and optimizer
    model, optimizer, criterion = setup_models_and_optimizers(device) 

    csv_log_path = initialize_training_logging(out_dir)

    run_training_loop(
        out_dir, device, scaler, train_loader, val_loader,
        model, optimizer, criterion, csv_log_path, # Pass the correct variables
    )

    try:
        train_ds.log_skipped_files()
        val_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")

if __name__ == '__main__':
    main()