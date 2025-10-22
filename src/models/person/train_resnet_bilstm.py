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
    
    ckpt = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'opt_state': optimizer.state_dict() if optimizer else None,
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

def eval_on_loader(model, criterion, dataloader, device, class_names):
    """Evaluate models on dataloader and return loss + metrics."""
    
    model.eval()
    num_outputs = len(class_names)

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for images_padded, labels_padded, lengths, _ in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            
            # Mask: only include labels that are not padding (-100)
            mask = (labels_padded != -100) # shape (B, S, NUM_OUTPUTS)

            logits = model(images_padded, lengths) # shape (B, S, NUM_OUTPUTS)
            
            # Flatten to (B*S*NUM_OUTPUTS), then select valid entries based on the first label's mask
            mask_flat = mask.view(-1, num_outputs)[:, 0]
            logits_flat = logits.view(-1, num_outputs)[mask_flat]
            labels_flat = labels_padded.view(-1, num_outputs)[mask_flat]

            loss = criterion(logits_flat, labels_flat)

            batch_size = images_padded.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probs = torch.sigmoid(logits)
            preds = (probs > PersonConfig.CONFIDENCE_THRESHOLD).float()
            
            valid_preds = preds.view(-1, num_outputs)[mask_flat].cpu()
            valid_labels = labels_padded.view(-1, num_outputs)[mask_flat].cpu()

            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
    
    if not all_preds:
        # Fallback for empty dataloaders/splits
        return 0.0, calculate_metrics(torch.zeros(0, num_outputs), torch.zeros(0, num_outputs), class_names)
        
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds, class_names)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics

def train_one_epoch(model, optimizer, criterion, dataloader, device, class_names, scaler=None, accumulation_steps=4):
    """
    Train models for one epoch with gradient accumulation and optional AMP.
    """
    
    model.train()
    num_outputs = len(class_names)

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (images_padded, labels_padded, lengths, _) in enumerate(progress_bar):
        images_padded = images_padded.to(device, non_blocking=True)
        labels_padded = labels_padded.to(device, non_blocking=True)
        mask = (labels_padded != -100) # shape (B, S, NUM_OUTPUTS)

        # Diagnostic print for label distribution
        if batch_idx == 0:
            print("[DIAGNOSTIC] Label distribution:")
            mask_flat = mask.view(-1, num_outputs)[:, 0]
            valid_labels = labels_padded.view(-1, num_outputs)[mask_flat].cpu().numpy()
            
            if num_outputs == 1:
                pos_count = int((valid_labels[:,0] == 1).sum())
                print(f"  {class_names[0]} frames: {pos_count}, total valid: {valid_labels.shape[0]}")
            else: # age-binary
                child_count = int((valid_labels[:,0] == 1).sum())
                adult_count = int((valid_labels[:,1] == 1).sum())
                print(f"  child frames: {child_count}, adult frames: {adult_count}, total valid: {valid_labels.shape[0]}")

        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(images_padded, lengths) # shape (B, S, NUM_OUTPUTS)
            
            # Masking: select valid entries based on the first label's mask
            mask_flat = mask.view(-1, num_outputs)[:, 0]
            logits_flat = logits.view(-1, num_outputs)[mask_flat]
            labels_flat = labels_padded.view(-1, num_outputs)[mask_flat]
            
            loss = criterion(logits_flat, labels_flat) / accumulation_steps

        if batch_idx == 0:
            print(f"[DIAGNOSTIC] First batch loss: {loss.item():.6f}")

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            optimizer.zero_grad()
                
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > PersonConfig.CONFIDENCE_THRESHOLD).float()
            
            valid_preds = preds.view(-1, num_outputs)[mask_flat].cpu()
            valid_labels = labels_padded.view(-1, num_outputs)[mask_flat].cpu()
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

        batch_size = images_padded.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size
        avg_loss_so_far = total_loss / total_samples if total_samples else 0.0
        progress_bar.set_postfix({'loss': f"{avg_loss_so_far:.4f}"})

    if not all_preds:
        return 0.0, calculate_metrics(torch.zeros(0, num_outputs), torch.zeros(0, num_outputs), class_names)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds, class_names)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, model, optimizer, criterion, csv_log_path, class_names):
    """Orchestrates the full training process across multiple epochs."""
    best_val_f1 = 0.0
    patience_counter = 0

    print(f"\nStarting training for {PersonConfig.NUM_EPOCHS} epochs...")
    print(f"Early stopping patience: {PersonConfig.PATIENCE} epochs")
    print("-" * 50)

    for epoch in range(1, PersonConfig.NUM_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model, optimizer, criterion, train_loader, device, class_names, scaler=scaler,
        )

        val_loss, val_metrics = eval_on_loader(model, criterion, val_loader, device, class_names)

        print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics, class_names)
        log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics, class_names)

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
    parser.add_argument('--mode', choices=["person-only", "age-binary"], default="age-binary",
                       help='Select the classification mode to load the correct number of outputs.')
    args = parser.parse_args()

    # Determine runtime parameters
    if args.mode == "age-binary":
        class_names = PersonConfig.TARGET_LABELS_AGE_BINARY
        num_outputs = 2
    else:
        class_names = PersonConfig.TARGET_LABELS_PERSON_ONLY
        num_outputs = 1
        
    # Setup environment (output directory depends on mode)
    out_dir, device, scaler = setup_environment(is_training=True, num_outputs=num_outputs)

    # Load data loaders
    train_loader, train_ds = setup_data_loaders(
        PersonClassification.TRAIN_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=True, log_dir=out_dir, 
        is_feature_extraction=True, split_name='train'
    )
    val_loader, val_ds = setup_data_loaders(
        PersonClassification.VAL_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=False, log_dir=out_dir, 
        is_feature_extraction=True, split_name='val'
    )
    
    # Check for consistent number of output classes in data
    if train_ds.num_outputs != num_outputs or val_ds.num_outputs != num_outputs:
        raise ValueError(f"Data CSVs generated with {train_ds.num_outputs} outputs, but trying to train in {num_outputs}-output mode. Re-run create_input_csv_files.py with --classification-mode={args.mode}")

    # Load combined model and optimizer
    model, optimizer, criterion = setup_models_and_optimizers(device, num_outputs, class_names) 

    csv_log_path = initialize_training_logging(out_dir, class_names)

    run_training_loop(
        out_dir, device, scaler, train_loader, val_loader,
        model, optimizer, criterion, csv_log_path, class_names
    )

    try:
        train_ds.log_skipped_files()
        val_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")

if __name__ == '__main__':
    main()