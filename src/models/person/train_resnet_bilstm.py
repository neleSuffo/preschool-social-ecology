"""
End-to-end pipeline (refactored):
  BiLSTM -> per-frame classifier

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
from person_classifier import FrameRNNClassifier

# Set conservative thread limits to avoid DataLoader hangs
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
warnings.filterwarnings("ignore", message="Can't initialize NVML")
torch.set_num_threads(4)

# ---------------------------
# Training/Evaluation Functions
# ---------------------------

def handle_checkpointing_and_early_stopping(out_dir, epoch, rnn_model, opt_rnn,
                                           train_metrics, val_metrics, best_val_f1, patience_counter):
    """Saves checkpoints, manages early stopping, and updates patience counter."""
    ckpt = {
        'epoch': epoch,
        'rnn_state': rnn_model.state_dict(),
        'opt_rnn': opt_rnn.state_dict() if opt_rnn else None,
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

def eval_on_loader(rnn_model, criterion, dataloader, device):
    """Evaluate models on dataloader and return loss + metrics."""
    rnn_model.eval()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for features_padded, labels_padded, lengths, _ in progress_bar:
            features_padded = features_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)

            logits = rnn_model(features_padded, lengths)
            
            mask_flat = mask.view(-1, 2)[:, 0]
            logits_flat = logits.view(-1, 2)[mask_flat]
            labels_flat = labels_padded.view(-1, 2)[mask_flat]

            loss = criterion(logits_flat, labels_flat)

            batch_size = features_padded.size(0)
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

def train_one_epoch(rnn_model, opt_rnn, criterion, dataloader, device, scaler=None, accumulation_steps=4):
    """
    Train models for one epoch with gradient accumulation and optional AMP.
    """
    rnn_model.train()

    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, (features_padded, labels_padded, lengths, _) in enumerate(progress_bar):
        features_padded = features_padded.to(device, non_blocking=True)
        labels_padded = labels_padded.to(device, non_blocking=True)
        mask = (labels_padded != -100)

        with torch.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = rnn_model(features_padded, lengths)
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
                scaler.unscale_(opt_rnn)
                torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 5.0)
                scaler.step(opt_rnn)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 5.0)
                opt_rnn.step()
            opt_rnn.zero_grad()
                
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            valid_mask = mask.view(-1, 2)[:, 0]
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

        batch_size = features_padded.size(0)
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

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, rnn_model, opt_rnn, criterion, csv_log_path):
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
            rnn_model, opt_rnn, criterion, train_loader, device, scaler=scaler,
        )

        val_loss, val_metrics = eval_on_loader(rnn_model, criterion, val_loader, device)

        print_epoch_results(epoch, train_loss, val_loss, train_metrics, val_metrics)
        log_epoch_metrics(csv_log_path, epoch, train_loss, val_loss, train_metrics, val_metrics)

        best_val_f1, patience_counter, should_stop = handle_checkpointing_and_early_stopping(
            out_dir, epoch, rnn_model, opt_rnn, train_metrics, val_metrics, best_val_f1, patience_counter
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

    # feature extraction is done once seperately with extract_features.py
    train_loader, train_ds = setup_data_loaders(
        PersonClassification.TRAIN_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=True, log_dir=out_dir, is_feature_extraction=False, split_name='train'
    )
    val_loader, val_ds = setup_data_loaders(
        PersonClassification.VAL_CSV_PATH, PersonConfig.BATCH_SIZE, is_training=False, log_dir=out_dir, is_feature_extraction=False, split_name='val'
    )

    rnn_model, opt_rnn, criterion = setup_models_and_optimizers(device)

    csv_log_path = initialize_training_logging(out_dir)

    run_training_loop(
        out_dir, device, scaler, train_loader, val_loader,
        rnn_model, opt_rnn, criterion, csv_log_path,
    )

    try:
        train_ds.log_skipped_files()
        val_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")

if __name__ == '__main__':
    main()