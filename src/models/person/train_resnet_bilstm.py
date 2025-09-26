"""
End-to-end pipeline (refactored):
  ResNet (frame-level feature extractor) -> BiLSTM -> per-frame classifier

This file contains the high-level training loop and model orchestration.
Utility functions (data collate, metrics, feature extraction, data loaders,
and environment setup) were moved to `utils.py` to improve readability.

Usage:
  python train_cnn_rnn.py
"""
import os
# Set conservative thread limits to avoid DataLoader hangs
os.environ['OMP_NUM_THREADS'] = '4'  
os.environ['MKL_NUM_THREADS'] = '4'

import warnings
# Suppress NVML warnings due to driver/library version mismatch
warnings.filterwarnings("ignore", message="Can't initialize NVML")

import torch
torch.set_num_threads(4)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

import argparse
import torch.nn as nn
from tqdm import tqdm
from config import PersonConfig
from person_classifier import CNNEncoder, FrameRNNClassifier
from utils import (
    calculate_metrics,
    sequence_features_from_cnn,
    setup_training_environment,
    setup_data_loaders,
)

# ---------------------------
# Model + optimizer setup
# ---------------------------

def setup_models_and_optimizers(device: torch.device):
    """Create models, initialize non-pretrained weights, compile and return optimizers.

    - CNN is loaded from a pretrained backbone (optionally frozen via PersonConfig).
    - RNN (BiLSTM/linear head) is initialized and orthogonally initialized for recurrent weights.
    - Uses AdamW with different learning-rate schedules for CNN vs RNN.
    - Returns (cnn, rnn_model, opt_cnn, opt_rnn, criterion).
    """
    cnn = CNNEncoder(backbone=PersonConfig.BACKBONE, pretrained=True, feat_dim=PersonConfig.FEAT_DIM).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim).to(device)

    # Initialize linear and recurrent weights for the RNN model only
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if param.data.ndim >= 2:
                        nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)

    rnn_model.apply(init_weights)

    # Try compiling models (may fail on some environments) - fall back gracefully
    try:
        cnn = torch.compile(cnn)
        rnn_model = torch.compile(rnn_model)
        print("Models compiled successfully!")
    except Exception as e:
        print(f"Model compilation skipped: {e}")

    # Optimizers - lower LR for pretrained CNN, higher LR for RNN
    opt_cnn = None
    if not PersonConfig.FREEZE_CNN:
        opt_cnn = torch.optim.AdamW(
            cnn.parameters(),
            lr=PersonConfig.LR * 0.01,
            weight_decay=PersonConfig.WEIGHT_DECAY * 0.1,
            betas=(0.9, 0.999),
        )

    opt_rnn = torch.optim.AdamW(
        rnn_model.parameters(),
        lr=PersonConfig.LR * 2.0,
        weight_decay=PersonConfig.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    # Standard BCE loss without class weighting (classes are balanced)
    criterion = nn.BCEWithLogitsLoss()

    return cnn, rnn_model, opt_cnn, opt_rnn, criterion


# ---------------------------
# Training / Evaluation loops
# ---------------------------

def train_one_epoch(models, optimizers, criterion, dataloader, device, freeze_cnn=False, scaler=None, accumulation_steps=4):
    """Train models for one epoch with gradient accumulation and optional AMP.

    Returns avg_loss (per-sample) and metrics dict (macro and per-class).
    """
    cnn, rnn = models
    opt_cnn, opt_rnn = optimizers

    # Set training/eval modes
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

        # Feature extraction (with or without gradients depending on freeze_cnn)
        if scaler and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                if freeze_cnn:
                    with torch.no_grad():
                        feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
                else:
                    feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)

                logits = rnn(feats, lengths)

                # Mask and flatten only valid entries
                mask_flat = mask.view(-1, 2)[:, 0]
                logits_flat = logits.view(-1, 2)[mask_flat]
                labels_flat = labels_padded.view(-1, 2)[mask_flat]

                loss = criterion(logits_flat, labels_flat) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients and clip
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
            # No AMP path
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
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
                if opt_cnn:
                    torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)
                    opt_cnn.step()
                opt_rnn.step()

                opt_rnn.zero_grad()
                if opt_cnn:
                    opt_cnn.zero_grad()

        # Accumulate totals for logging
        batch_size = images_padded.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps  # unscale
        total_samples += batch_size

        # Collect predictions for metrics (do on CPU / detached)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            valid_mask = mask.view(-1, 2)[:, 0]
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()

            all_preds.append(valid_preds)
            all_labels.append(valid_labels)

        # Update progress bar
        avg_loss_so_far = total_loss / total_samples if total_samples else 0.0
        progress_bar.set_postfix({'loss': f"{avg_loss_so_far:.4f}"})

    # Finalize metrics
    if len(all_preds) == 0:
        raise RuntimeError("No valid predictions collected during training epoch")

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

    if len(all_preds) == 0:
        raise RuntimeError("No valid predictions collected during evaluation")

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss / total_samples

    return avg_loss, metrics


# ---------------------------
# Logging, checkpointing, printing
# ---------------------------

def initialize_training_logging(out_dir: str):
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
    print(f"\nEpoch {epoch}/{PersonConfig.NUM_EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"    Adult    - P: {train_metrics['adult_precision']:.3f}, R: {train_metrics['adult_recall']:.3f}, F1: {train_metrics['adult_f1']:.3f}")
    print(f"    Child    - P: {train_metrics['child_precision']:.3f}, R: {train_metrics['child_recall']:.3f}, F1: {train_metrics['child_f1']:.3f}")
    print(f"    Macro    - P: {train_metrics['macro_precision']:.3f}, R: {train_metrics['macro_recall']:.3f}, F1: {train_metrics['macro_f1']:.3f}")

    print(f"  Val Loss: {val_loss:.4f}")
    print(f"    Adult    - P: {val_metrics['adult_precision']:.3f}, R: {val_metrics['adult_recall']:.3f}, F1: {val_metrics['adult_f1']:.3f}")
    print(f"    Child    - P: {val_metrics['child_precision']:.3f}, R: {val_metrics['child_recall']:.3f}, F1: {val_metrics['child_f1']:.3f}")
    print(f"    Macro    - P: {val_metrics['macro_precision']:.3f}, R: {val_metrics['macro_recall']:.3f}, F1: {val_metrics['macro_f1']:.3f}")


def handle_checkpointing_and_early_stopping(out_dir, epoch, cnn, rnn_model, opt_cnn, opt_rnn,
                                           train_metrics, val_metrics, best_val_f1, patience_counter):
    ckpt = {
        'epoch': epoch,
        'cnn_state': cnn.state_dict(),
        'rnn_state': rnn_model.state_dict(),
        'opt_rnn': opt_rnn.state_dict() if opt_rnn else None,
        'opt_cnn': opt_cnn.state_dict() if opt_cnn else None,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
    }

    # Save last model weights
    if epoch == PersonConfig.NUM_EPOCHS:
        last_path = out_dir / 'last.pth'
        torch.save(ckpt, last_path)

    should_stop = False
    if val_metrics['macro_f1'] > best_val_f1:
        best_val_f1 = val_metrics['macro_f1']
        patience_counter = 0
        best_path = out_dir / 'best.pth'
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


# ---------------------------
# Main training loop orchestration
# ---------------------------

def run_training_loop(out_dir, device, scaler, train_loader, val_loader, cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path):
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

    return best_val_f1


# ---------------------------
# Entry point
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir, device, scaler = setup_training_environment()

    train_loader, val_loader, _, train_ds, val_ds, _ = setup_data_loaders(out_dir)

    cnn, rnn_model, opt_cnn, opt_rnn, criterion = setup_models_and_optimizers(device)

    csv_log_path = initialize_training_logging(out_dir)

    best_val_f1 = run_training_loop(
        out_dir, device, scaler, train_loader, val_loader,
        cnn, rnn_model, opt_cnn, opt_rnn, criterion, csv_log_path,
    )

    # Log skipped files and final housekeeping
    try:
        train_ds.log_skipped_files()
        val_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")


if __name__ == '__main__':
    main()