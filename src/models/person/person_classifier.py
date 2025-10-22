import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torchvision import models
from ultralytics import YOLO
from config import PersonConfig
from constants import PersonClassification

class YOLOFeatureExtractor(nn.Module):
    """
    YOLOv12 (Ultralytics) Feature Extractor Backbone.
    Loads the YOLO model and strips the detection head to use the backbone
    for high-quality egocentric features.
    """
    def __init__(self, yolo_model_path=PersonConfig.YOLO_BACKBONE_PATH, feat_dim=PersonConfig.FEAT_DIM):
        super().__init__()
        try:
            # 1. Load the pre-trained YOLOv12l model
            yolo = YOLO("yolov12l.pt")  
            
            # 2. Extract the feature backbone
            # Conceptually extracts the backbone layers, skipping the final detection head.
            # We use a custom slicing index if provided by the model config, otherwise a safe fallback.
            backbone_end_index = yolo.model.model.yaml.get('backbone_end_index', -2)
            self.backbone = yolo.model.model[:backbone_end_index] 
            
            # Determine the final feature dimension and add a projection layer if needed
            self.gap = nn.AdaptiveAvgPool2d(1)
            
            # Conceptual feature dimension after the backbone and before projection/GAP
            # For YOLOv12l, the final layer output is typically 1024 (before fusion layers)
            feat_in = 1024 
            
            if feat_in != feat_dim:
                 self.project = nn.Linear(feat_in, feat_dim)
            else:
                 self.project = None

        except Exception as e:
            print(f"Error loading YOLO backbone: {e}. Falling back to a standard ResNet.")
            # Fallback implementation (retained for robustness)
            from torchvision import models
            # Use ResNet18 as a lightweight fallback
            res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.gap = None
            self.project = nn.Linear(feat_in, feat_dim) if feat_in != feat_dim else None

    def forward(self, x):
        """x shape: (batch_size * seq_len, C, H, W)"""
        x = self.backbone(x)
        
        if isinstance(x, list):
            # Take the final feature map from the list
            x = x[-1] 

        if x.dim() == 4: # (N, C, H, W)
            x = self.gap(x)
            x = x.view(x.size(0), -1)
        
        if self.project:
            x = self.project(x)
        return x
    
class FrameRNNClassifier(nn.Module):
    """Bidirectional LSTM classifier generalized for 1 or 2 outputs."""
    def __init__(self, 
                feat_dim=PersonConfig.FEAT_DIM, 
                rnn_hidden=PersonConfig.RNN_HIDDEN, 
                rnn_layers=PersonConfig.RNN_LAYERS, 
                bidirectional=PersonConfig.BIDIRECTIONAL, 
                num_outputs=PersonConfig.NUM_OUTPUTS,
                dropout=PersonConfig.DROPOUT):
        super().__init__()
        self.rnn = nn.LSTM(input_size=feat_dim, hidden_size=rnn_hidden,
                        num_layers=rnn_layers, batch_first=True,
                        bidirectional=bidirectional, dropout=dropout if rnn_layers>1 else 0.0)
        out_dim = rnn_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim//2, num_outputs) 
        )
        self.num_outputs = num_outputs

    def forward(self, feats, lengths):
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(rnn_out)
        return logits
    
class PersonDetectionClassifier(nn.Module):
    """
    End-to-end YOLO Feature + BiLSTM classifier.
    """
    def __init__(self, num_outputs=PersonConfig.NUM_OUTPUTS, **kwargs):
        super().__init__()

        self.cnn = YOLOFeatureExtractor(feat_dim=PersonConfig.FEAT_DIM)
        self.rnn = FrameRNNClassifier(feat_dim=self.cnn.feat_dim, num_outputs=num_outputs)

    def forward(self, images_padded, lengths):
        # FIX: Extract dimensions from the padded input tensor
        bs, max_seq, C, H, W = images_padded.shape
        
        images_flat = images_padded.view(bs * max_seq, C, H, W)
        feats_flat = self.cnn(images_flat)
        
        feat_dim = feats_flat.shape[-1]
        feats = feats_flat.view(bs, max_seq, feat_dim)
        
        logits = self.rnn(feats, lengths)
        return logits
    
class VideoFrameDataset(Dataset):
    """
    Dataset for loading sequences of video frames (generalized for 1 or 2 labels).
    """
    def __init__(self, csv_file, transform=None, log_dir=None, is_feature_extraction=False, split_name=None):
        self.csv_file = Path(csv_file)
        self.df = pd.read_csv(self.csv_file)
        self.grouped = self.df.groupby('video_id')
        self.transform = transform
        self.log_dir = Path(log_dir) if log_dir else None
        self.is_feature_extraction = is_feature_extraction
        self.split_name = split_name
        self.skipped_files = []
        self._create_sequence_indices()
        
        # Infer target labels from CSV columns
        all_labels = PersonConfig.TARGET_LABELS_AGE_BINARY + PersonConfig.TARGET_LABELS_PERSON_ONLY
        self.target_labels = [col for col in self.df.columns if col in all_labels]
        self.num_outputs = len(self.target_labels)


    def _create_sequence_indices(self):
        self.sequences = []
        for group_idx, (video_id, group) in enumerate(self.grouped):
            self.sequences.append({
                'video_id': video_id,
                'group_idx': group_idx,
                'sequence_length': len(group)
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_info = self.sequences[idx]
        video_id = sequence_info['video_id']
        sequence_length = sequence_info['sequence_length']
        
        frames_or_features = []
        labels = []

        for _, row in self.grouped.get_group(video_id).iterrows():
            file_path = row['file_path']
            current_labels = [row[label] for label in self.target_labels]
            
            if self.is_feature_extraction:
                try:
                    if not Path(file_path).exists():
                        self.skipped_files.append((file_path, "File not found"))
                        continue
                        
                    img = Image.open(file_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames_or_features.append(img)
                    labels.append(current_labels)
                except Exception as e:
                    self.skipped_files.append((file_path, f"Error loading: {str(e)}"))
                    continue
            else:
                # Logic for pre-extracted features
                video_name = Path(file_path).parent.name
                frame_id = row['frame_id']
                feature_dir = PersonClassification.TRAIN_CSV_PATH.parent / "extracted_features" / self.split_name / video_name
                feature_file = feature_dir / f"{frame_id:06d}.pt"
                
                try:
                    if not feature_file.exists():
                        self.skipped_files.append((str(feature_file), "Feature file not found"))
                        continue
                    
                    features = torch.load(feature_file)
                    frames_or_features.append(features)
                    labels.append(current_labels)
                except Exception as e:
                    self.skipped_files.append((str(feature_file), f"Error loading feature: {str(e)}"))
                    continue

        if not frames_or_features:
            return torch.empty(0), torch.empty(0), 0, video_id

        frames_or_features = torch.stack(frames_or_features)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return frames_or_features, labels, len(frames_or_features), video_id

    def log_skipped_files(self):
        if self.log_dir and self.skipped_files:
            log_path = self.log_dir / f"{self.csv_file.stem}_skipped_files.log"
            with open(log_path, 'w') as f:
                f.write(f"Logged at: {pd.Timestamp.now()}\n\n")
                for file_path, reason in self.skipped_files:
                    f.write(f"{file_path} - Reason: {reason}\n")
            print(f"Logged {len(self.skipped_files)} skipped files to {log_path}")