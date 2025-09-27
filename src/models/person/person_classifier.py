import os
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from torchvision import models
from config import PersonConfig

class FrameRNNClassifier(nn.Module):
    """Bidirectional LSTM classifier for sequence-level person classification.
    
    Takes CNN features from a sequence of frames and processes them through a bidirectional LSTM
    to output per-frame classifications for adult and child presence.
    
    Parameters
    ----------
    feat_dim : int, default=512
        Dimension of input CNN features.
    rnn_hidden : int, default=256
        Hidden dimension of the LSTM.
    rnn_layers : int
        Number of LSTM layers.
    bidirectional : bool, default=True
        Whether to use bidirectional LSTM.
    num_outputs : int, default=2
        Number of output classes (adult, child).
    dropout : float, default=0.3
        Dropout probability for regularization.
    """
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
            nn.Linear(out_dim//2, num_outputs)  # 2 outputs: adult, child
        )

    def forward(self, feats, lengths):
        """Forward pass through the RNN classifier.
        
        Processes CNN features through bidirectional LSTM and classification layers.
        
        Parameters
        ----------
        feats : torch.Tensor
            CNN features of shape (batch_size, sequence_length, feat_dim).
        lengths : torch.Tensor
            Actual sequence lengths for each sample in the batch.
            
        Returns
        -------
        torch.Tensor
            Classification logits of shape (batch_size, sequence_length, num_outputs).
        """
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(rnn_out)  # (batch, max_seq, 2)
        return logits

class CNNEncoder(nn.Module):
    """CNN feature extractor using pretrained ResNet backbone
    
    Extracts feature vectors from individual frames using a pretrained ResNet model.
    The final fully connected layer is removed to get feature representations.
    
    Parameters
    ----------
    backbone : str, default='resnet18'
        The ResNet architecture to use ('resnet18' or 'resnet50').
    pretrained : bool, default=True
        Whether to use ImageNet pretrained weights.
    feat_dim : int, default=512
        Desired feature dimension. If different from backbone output, adds projection layer.
    """
    def __init__(self, backbone='resnet18', pretrained=True, feat_dim=512):
        super().__init__()
        
        # Load the appropriate model and weights
        if backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            res = models.resnet18(weights=weights)
            # ResNet's last layer is 'fc'
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]
            self.encoder = nn.Sequential(*modules)
        
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            res = models.efficientnet_b0(weights=weights)
            # EfficientNet's last layer is 'classifier'
            feat_in = res.classifier[1].in_features
            # The 'features' module already gives us what we need
            self.encoder = res.features
        
        else:
            raise NotImplementedError(f"Backbone '{backbone}' not implemented; supported: resnet18, efficientnet_b0")

        self.feat_dim = feat_in

        # Optional projection layer
        if self.feat_dim != feat_dim:
            self.project = nn.Linear(self.feat_dim, feat_dim)
            self.feat_dim = feat_dim
        else:
            self.project = None

    def forward(self, x):
        """Forward pass through the CNN encoder.
        
        Processes a batch of images through the backbone to extract features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch_size * seq_len, C, H, W).
            
        Returns
        -------
        torch.Tensor
            Feature vectors of shape (batch_size * seq_len, feat_dim).
        """
        x = self.encoder(x)
        
        # Global average pooling for different architectures
        if len(x.shape) == 4:  # If we have spatial dimensions (batch, channels, height, width)
            x = x.mean([2, 3])  # Global average pooling: (batch, channels)
        
        # Apply projection if needed
        if self.project:
            x = self.project(x)
        return x

class VideoFrameDataset(Dataset):
    """
    Dataset for loading sequences of video frames.
    
    This class can operate in two modes:
    1. **Frame-level loading (is_feature_extraction=True or default):** Loads raw image files and applies transformations.
    2. **Feature-level loading (is_feature_extraction=False and features are pre-extracted):** Loads pre-extracted feature tensors from disk.
    
    This is designed to work with the custom collate function to handle sequences of varying lengths.
    """
    def __init__(self, csv_file, transform=None, log_dir=None, is_feature_extraction=False):
        """
        Initializes the dataset.

        Args:
            csv_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            log_dir (string, optional): Directory to save log files for skipped items.
            is_feature_extraction (bool, optional): If True, loads raw image files. If False,
                                                    assumes pre-extracted features will be loaded.
        """
        self.csv_file = Path(csv_file)
        self.grouped = pd.read_csv(self.csv_file).groupby('video_id')
        self.transform = transform
        self.log_dir = Path(log_dir) if log_dir else None
        self.is_feature_extraction = is_feature_extraction
        self.skipped_files = []
        self._create_sequence_indices()

    def _create_sequence_indices(self):
        """
        Creates a list of tuples (group_index, start_index, end_index)
        to represent each video sequence.
        """
        self.sequences = []
        for group_idx, (video_id, group) in enumerate(self.grouped):
            sequence_length = len(group)
            self.sequences.append({
                'video_id': video_id,
                'group_idx': group_idx,
                'start_idx': 0,
                'end_idx': sequence_length - 1,
                'sequence_length': sequence_length
            })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a single sequence of frames or features.
        """
        sequence_info = self.sequences[idx]
        group_idx = sequence_info['group_idx']
        video_id = sequence_info['video_id']
        sequence_length = sequence_info['sequence_length']
        
        frames_or_features = []
        labels = []
        frames_loaded = 0

        # Load each item in the sequence
        for i in range(sequence_length):
            row = self.grouped.get_group(video_id).iloc[i]
            
            # Use appropriate file path based on mode
            if self.is_feature_extraction:
                file_path = row['file_path']
                try:
                    if not Path(file_path).exists():
                        self.skipped_files.append((file_path, "File not found"))
                        continue
                    
                    img = Image.open(file_path).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames_or_features.append(img)
                    labels.append([row['child'], row['adult']])
                    frames_loaded += 1
                except Exception as e:
                    self.skipped_files.append((file_path, f"Error loading: {str(e)}"))
                    continue
            else:
                # Assuming features are saved in a subfolder named by the video ID
                feature_dir = Path(f"/path/to/extracted_features/{video_id}")
                feature_file = feature_dir / f"{row['frame_id']:06d}.pt"
                
                try:
                    if not feature_file.exists():
                        self.skipped_files.append((str(feature_file), "Feature file not found"))
                        continue
                    
                    features = torch.load(feature_file)
                    frames_or_features.append(features)
                    labels.append([row['child'], row['adult']])
                    frames_loaded += 1
                except Exception as e:
                    self.skipped_files.append((str(feature_file), f"Error loading feature: {str(e)}"))
                    continue

        if not frames_or_features:
            return torch.empty(0), torch.empty(0), 0, video_id

        # Stack the tensors
        frames_or_features = torch.stack(frames_or_features)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return frames_or_features, labels, frames_loaded, video_id

    def log_skipped_files(self):
        """
        Logs skipped files to a file in the log directory.
        """
        if self.log_dir and self.skipped_files:
            log_path = self.log_dir / f"{self.csv_file.stem}_skipped_files.log"
            with open(log_path, 'w') as f:
                f.write(f"Logged at: {pd.Timestamp.now()}\n\n")
                for file_path, reason in self.skipped_files:
                    f.write(f"{file_path} - Reason: {reason}\n")
            print(f"Logged {len(self.skipped_files)} skipped files to {log_path}")