"""
Evaluation script for CNN-RNN person classification model on test set.
Based on the improved training script architecture with utility functions moved to utils.py.
"""
import json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from config import PersonConfig
from utils import (
    load_model, 
    setup_evaluation, 
    calculate_metrics, 
    sequence_features_from_cnn,
    plot_confusion_matrices,
    plot_metrics_comparison
)

def evaluate_model(models, dataloader, device, output_dir):
    """Evaluate the model on test data and generate comprehensive results.
    
    Parameters
    ----------
    models : Tuple[nn.Module, nn.Module]
        Tuple containing (CNN encoder, RNN classifier) models.
    dataloader : DataLoader
        Test data loader.
    device : torch.device
        Device to run evaluation on.
    output_dir : Path
        Directory to save results.
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all evaluation metrics.
    """
    cnn, rnn = models
    cnn.eval()
    rnn.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    video_results = []
    
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    
    print("Running evaluation on test set...")
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for images_padded, labels_padded, lengths, video_ids in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)
            
            # Forward pass
            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            # Calculate loss
            mask_flat = mask.view(-1, 2)
            logits_flat = logits.view(-1, 2)[mask_flat[:, 0] != -100]
            labels_flat = labels_padded.view(-1, 2)[mask_flat[:, 0] != -100]
            
            if len(logits_flat) > 0:  # Only calculate loss if we have valid samples
                loss = criterion(logits_flat, labels_flat)
                total_loss += loss.item() * images_padded.size(0)
            
            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Collect results per video
            for i, video_id in enumerate(video_ids):
                seq_len = lengths[i].item()
                video_preds = preds[i, :seq_len].cpu().numpy()
                video_labels = labels_padded[i, :seq_len].cpu().numpy()
                video_probs = probs[i, :seq_len].cpu().numpy()
                
                # Store basic video information for reference
                video_results.append({
                    'video_id': video_id,
                    'num_frames': seq_len,
                    'adult_frames': int(video_labels[:, 1].sum()),
                    'child_frames': int(video_labels[:, 0].sum()),
                    'no_face_frames': seq_len - int(video_labels[:, 0].sum()) - int(video_labels[:, 1].sum())
                })
            
            # Only keep valid predictions (not masked)
            valid_mask = mask.view(-1, 2)[:, 0]
            valid_preds = preds.view(-1, 2)[valid_mask].cpu()
            valid_labels = labels_padded.view(-1, 2)[valid_mask].cpu()
            valid_probs = probs.view(-1, 2)[valid_mask].cpu()
            
            all_preds.append(valid_preds)
            all_labels.append(valid_labels)
            all_probs.append(valid_probs)
            
            # Update progress bar
            progress_bar.set_postfix({
                'samples_processed': len(all_preds) * dataloader.batch_size
            })
    
    # Concatenate all results
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    avg_loss = total_loss / len(dataloader.dataset)
    
    # Calculate comprehensive metrics
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['test_loss'] = avg_loss
    
    # Add additional metrics
    metrics['total_samples'] = len(all_labels)
    metrics['adult_samples'] = int(all_labels[:, 1].sum())
    metrics['child_samples'] = int(all_labels[:, 0].sum())
    metrics['adult_prevalence'] = float(all_labels[:, 1].mean())
    metrics['child_prevalence'] = float(all_labels[:, 0].mean())
    
    # Save detailed results
    print(f"\nSaving results to {output_dir}")
    
    # Save frame-level predictions
    frame_results = pd.DataFrame({
        'child_true': all_labels[:, 0],
        'adult_true': all_labels[:, 1],
        'child_pred': all_preds[:, 0],
        'adult_pred': all_preds[:, 1],
        'child_prob': all_probs[:, 0],
        'adult_prob': all_probs[:, 1]
    })
    frame_results.to_csv(output_dir / 'frame_level_results.csv', index=False)

    # Generate plots
    plot_confusion_matrices(all_labels, all_preds, PersonConfig.TARGET_LABELS, output_dir)
    plot_metrics_comparison(metrics, output_dir)
    
    # Save essential metrics only
    summary_stats = {
        'performance_metrics': {
            # Per-class metrics
            'child_precision': metrics['child_precision'],
            'child_recall': metrics['child_recall'], 
            'child_f1': metrics['child_f1'],
            'adult_precision': metrics['adult_precision'],
            'adult_recall': metrics['adult_recall'],
            'adult_f1': metrics['adult_f1'],
            # Overall metrics
            'macro_precision': metrics['macro_precision'],
            'macro_recall': metrics['macro_recall'],
            'macro_f1': metrics['macro_f1']
        },
        'dataset_stats': {
            'total_videos': len(video_results),
            'total_frames': metrics['total_samples'],
            'adult_frames': metrics['adult_samples'],
            'child_frames': metrics['child_samples'],
            'no_face_frames': metrics['total_samples'] - metrics['adult_samples'] - metrics['child_samples']
        }
    }

    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    return metrics

def main():
    """Main evaluation function."""
    # Setup evaluation components
    device, test_loader, test_ds, eval_dir = setup_evaluation()
    
    # Load model
    cnn, rnn_model = load_model(device)
    
    # Run evaluation
    metrics = evaluate_model((cnn, rnn_model), test_loader, device, eval_dir)
    
    # Print and log results
    test_ds.log_skipped_files()
    
    print(f"\nEvaluation complete! Results saved to: {eval_dir}")

if __name__ == '__main__':
    main()