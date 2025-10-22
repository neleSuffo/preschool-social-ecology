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

def evaluate_model(models, dataloader, device, output_dir, class_names):
    """Evaluate the model on test data and generate comprehensive results."""
    cnn, rnn = models
    cnn.eval()
    rnn.eval()
    
    num_outputs = len(class_names)
    
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
            
            # Per-video stats
            for i, video_id in enumerate(video_ids):
                seq_len = lengths[i].item()
                video_labels = labels_padded[i, :seq_len].cpu().numpy()
                
                stats = {'video_id': video_id, 'num_frames': seq_len}
                
                if num_outputs == 2:
                    stats['adult_frames'] = int(video_labels[:, 1].sum())
                    stats['child_frames'] = int(video_labels[:, 0].sum())
                    stats['no_person_frames'] = seq_len - stats['adult_frames'] - stats['child_frames']
                else:
                    stats['person_frames'] = int(video_labels[:, 0].sum())
                    stats['no_person_frames'] = seq_len - stats['person_frames']
                    
                video_results.append(stats)
            
            # Per-frame stats for metrics
            valid_mask = mask.view(-1, num_outputs)[:, 0]
            if valid_mask.sum() > 0:
                valid_preds = preds.view(-1, num_outputs)[valid_mask]
                valid_labels = labels_padded.view(-1, num_outputs)[valid_mask]
                valid_probs = probs.view(-1, num_outputs)[valid_mask]
                
                all_preds.append(valid_preds.cpu())
                all_labels.append(valid_labels.cpu())
                all_probs.append(valid_probs.cpu())
    
    if not all_preds:
        print("No valid predictions collected. Evaluation skipped.")
        return {'macro_f1': 0.0}

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    
    metrics = calculate_metrics(all_labels, all_preds, class_names)
    
    metrics['total_samples'] = len(all_labels)
    if num_outputs == 2:
        metrics['adult_samples'] = int(all_labels[:, 1].sum())
        metrics['child_samples'] = int(all_labels[:, 0].sum())
    elif num_outputs == 1:
        metrics['person_samples'] = int(all_labels[:, 0].sum())
    
    print(f"\nSaving results to {output_dir}")
    
    frame_results_data = {}
    for i, label in enumerate(class_names):
        frame_results_data[f'{label}_true'] = all_labels[:, i]
        frame_results_data[f'{label}_pred'] = all_preds[:, i]
        frame_results_data[f'{label}_prob'] = all_probs[:, i]
        
    frame_results = pd.DataFrame(frame_results_data)
    frame_results.to_csv(output_dir / 'frame_level_results.csv', index=False)

    plot_confusion_matrices(all_labels, all_preds, class_names, output_dir)
    plot_metrics_comparison(metrics, output_dir, class_names)
    
    summary_stats = {'performance_metrics': {}, 'dataset_stats': {}}
    for metric_type in ['precision', 'recall', 'f1']:
        for class_name in class_names:
            summary_stats['performance_metrics'][f'{class_name}_{metric_type}'] = metrics[f'{class_name}_{metric_type}']
        summary_stats['performance_metrics'][f'macro_{metric_type}'] = metrics[f'macro_{metric_type}']

    summary_stats['dataset_stats']['total_videos'] = len(video_results)
    summary_stats['dataset_stats']['total_frames'] = metrics['total_samples']
    
    if num_outputs == 2:
        summary_stats['dataset_stats']['adult_frames'] = metrics['adult_samples']
        summary_stats['dataset_stats']['child_frames'] = metrics['child_samples']
        summary_stats['dataset_stats']['no_person_frames'] = metrics['total_samples'] - metrics['adult_samples'] - metrics['child_samples']
    else:
        summary_stats['dataset_stats']['person_frames'] = metrics['person_samples']
        summary_stats['dataset_stats']['no_person_frames'] = metrics['total_samples'] - metrics['person_samples']


    with open(output_dir / 'summary_stats.json', 'w') as f:
        json.dump(summary_stats, f, indent=4)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=["person-only", "age-binary"], default="age-binary",
                       help='Select the classification mode to load the correct model and run evaluation.')
    args = parser.parse_args()

    # Determine runtime parameters
    if args.mode == "age-binary":
        class_names = PersonConfig.TARGET_LABELS_AGE_BINARY
        num_outputs = 2
    else:
        class_names = PersonConfig.TARGET_LABELS_PERSON_ONLY
        num_outputs = 1
        
    _, device, _ = setup_environment(is_training=False, num_outputs=num_outputs)
    
    test_loader, test_ds = setup_data_loaders(
        PersonClassification.TEST_CSV_PATH, 
        PersonConfig.BATCH_SIZE_INFERENCE, 
        is_training=False,
        log_dir=PersonClassification.TRAINED_WEIGHTS_PATH.parent
    )
    
    # Check for consistent number of output classes in data
    if test_ds.num_outputs != num_outputs:
        raise ValueError(f"Data CSVs generated with {test_ds.num_outputs} outputs, but trying to evaluate in {num_outputs}-output mode. Check TEST_CSV_PATH.")
    
    cnn, rnn_model = load_model(device, num_outputs)
    
    eval_dir = PersonClassification.TRAINED_WEIGHTS_PATH.parent / f"{PersonConfig.MODEL_NAME}_{args.mode}_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    evaluate_model((cnn, rnn_model), test_loader, device, eval_dir, class_names)
    
    try:
        test_ds.log_skipped_files()
    except Exception as e:
        print(f"Warning: could not log skipped files: {e}")
    
    print(f"\nEvaluation complete! Results saved to: {eval_dir}")

if __name__ == '__main__':
    main()