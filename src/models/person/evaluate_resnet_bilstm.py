"""
Evaluation script for CNN-RNN person classification model on test set.
"""
import json
import torch
import pandas as pd
from tqdm import tqdm
from config import PersonConfig
from constants import PersonClassification
from utils import setup_environment, setup_data_loaders, load_model, calculate_metrics, plot_confusion_matrices, plot_metrics_comparison

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


def evaluate_model(models, dataloader, device, output_dir):
    """Evaluate the model on test data and generate comprehensive results."""
    cnn, rnn = models
    cnn.eval()
    rnn.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    video_results = []
    
    print("Running evaluation on test set...")
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for images_padded, labels_padded, lengths, video_ids in progress_bar:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)
            mask = (labels_padded != -100)
            
            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            for i, video_id in enumerate(video_ids):
                seq_len = lengths[i].item()
                video_labels = labels_padded[i, :seq_len].cpu().numpy()
                
                video_results.append({
                    'video_id': video_id,
                    'num_frames': seq_len,
                    'adult_frames': int(video_labels[:, 1].sum()),
                    'child_frames': int(video_labels[:, 0].sum()),
                    'no_face_frames': seq_len - int(video_labels[:, 0].sum()) - int(video_labels[:, 1].sum())
                })
            
            valid_mask = mask.view(-1, 2)[:, 0]
            if valid_mask.sum() > 0:
                valid_preds = preds.view(-1, 2)[valid_mask]
                valid_labels = labels_padded.view(-1, 2)[valid_mask]
                valid_probs = probs.view(-1, 2)[valid_mask]
                
                all_preds.append(valid_preds.cpu())
                all_labels.append(valid_labels.cpu())
                all_probs.append(valid_probs.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    metrics = calculate_metrics(all_labels, all_preds)
    
    metrics['total_samples'] = len(all_labels)
    metrics['adult_samples'] = int(all_labels[:, 1].sum())
    metrics['child_samples'] = int(all_labels[:, 0].sum())
    metrics['adult_prevalence'] = float(all_labels[:, 1].mean())
    metrics['child_prevalence'] = float(all_labels[:, 0].mean())
    
    print(f"\nSaving results to {output_dir}")
    
    frame_results = pd.DataFrame({
        'child_true': all_labels[:, 0], 'adult_true': all_labels[:, 1],
        'child_pred': all_preds[:, 0], 'adult_pred': all_preds[:, 1],
        'child_prob': all_probs[:, 0], 'adult_prob': all_probs[:, 1]
    })
    frame_results.to_csv(output_dir / 'frame_level_results.csv', index=False)

    plot_confusion_matrices(all_labels, all_preds, PersonConfig.TARGET_LABELS, output_dir)
    plot_metrics_comparison(metrics, output_dir)
    
    summary_stats = {
        'performance_metrics': {
            'child_precision': metrics['child_precision'], 'child_recall': metrics['child_recall'], 'child_f1': metrics['child_f1'],
            'adult_precision': metrics['adult_precision'], 'adult_recall': metrics['adult_recall'], 'adult_f1': metrics['adult_f1'],
            'macro_precision': metrics['macro_precision'], 'macro_recall': metrics['macro_recall'], 'macro_f1': metrics['macro_f1']
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
    _, device, _ = setup_environment(is_training=False)
    test_loader, test_ds = setup_data_loaders(
        PersonClassification.TEST_CSV_PATH, 
        PersonConfig.BATCH_SIZE_INFERENCE, 
        is_training=False,
        log_dir=PersonClassification.TRAINED_WEIGHTS_PATH.parent
    )
    
    cnn, rnn_model = load_model(device)
    
    eval_dir = PersonClassification.TRAINED_WEIGHTS_PATH.parent / f"{PersonConfig.MODEL_NAME}_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_model((cnn, rnn_model), test_loader, device, eval_dir)
    
    try:
        test_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")
    
    print(f"\nEvaluation complete! Results saved to: {eval_dir}")

if __name__ == '__main__':
    main()