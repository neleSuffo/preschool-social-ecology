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
        
        # Handle new torchvision weights API (deprecated 'pretrained' parameter)
        if backbone == 'resnet18':
            if pretrained:
                res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                res = models.resnet18(weights=None)
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]  # remove FC and avgpool kept? last is avgpool
            self.encoder = nn.Sequential(*modules)  # outputs (B, feat_in, 1, 1)
            self.feat_dim = feat_in
        elif backbone == 'resnet50':
            if pretrained:
                res = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                res = models.resnet50(weights=None)
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]  # remove FC and avgpool kept? last is avgpool
            self.encoder = nn.Sequential(*modules)  # outputs (B, feat_in, 1, 1)
            self.feat_dim = feat_in
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented; supported: resnet18, resnet50")
        # optional projection to reduce dim
        if self.feat_dim != feat_dim:
            self.project = nn.Linear(self.feat_dim, feat_dim)
            self.feat_dim = feat_dim
        else:
            self.project = None

    def forward(self, x):
        """Forward pass through the CNN encoder.
        
        Processes a batch of images through the ResNet backbone to extract features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (batch_size * seq_len, C, H, W).
            
        Returns
        -------
        torch.Tensor
            Feature vectors of shape (batch_size * seq_len, feat_dim).
        """
        # x: (batch*seq_len, C, H, W)
        f = self.encoder(x)  # (N, feat_in, 1,1)
        f = f.view(f.size(0), -1)  # (N, feat_in)
        if self.project is not None:
            f = self.project(f)
        return f  # (N, feat_dim)

class VideoFrameDataset(Dataset):
    def __init__(self, csv_file, sequence_length=PersonConfig.SEQUENCE_LENGTH, transform=None, log_dir=None):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.transform = transform
        self.skipped_files = []
        self.log_dir = log_dir

        self.grouped = []
        for video_id, group in self.data.groupby('video_id'):
            sorted_group = group.sort_values('frame_id')
            self.grouped.append(sorted_group.reset_index(drop=True))

        # Build a list of valid sequence start indices
        self.sequence_indices = []
        for group_idx, group in enumerate(self.grouped):
            if len(group) >= self.sequence_length:
                for start_idx in range(len(group) - self.sequence_length + 1):
                    self.sequence_indices.append((group_idx, start_idx))

    def __len__(self):
        """Get the total number of sequences in the dataset."""
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        """Get a sequence of frames and labels for training."""
        
        # Get the group index and start index from the pre-built list
        group_idx, start_idx = self.sequence_indices[idx]
        group = self.grouped[group_idx]
        
        frames = []
        labels = []
        video_id = None
        frames_loaded = 0
        
        # Determine the video_id and frame_ids for logging purposes
        row_start = group.iloc[start_idx]
        video_id = row_start['video_id']
        start_frame_id = row_start['frame_id']
        
        frame_ids_in_sequence = []
        for i in range(self.sequence_length):
            current_idx = start_idx + i
            row = group.iloc[current_idx]
            file_path = row['file_path']
            frame_ids_in_sequence.append(row['frame_id'])

            try:
                # Check if file exists
                if not Path(file_path).exists():
                    self.skipped_files.append((file_path, "File not found"))
                    print(f"File not found: {file_path}")
                    continue
                
                img = Image.open(file_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                frames.append(img)
                labels.append([row['child'], row['adult']])
                frames_loaded += 1
            except Exception as e:
                # This is the key change: print the error message
                print(f"Error loading file {file_path}: {str(e)}")
                self.skipped_files.append((file_path, f"Error loading: {str(e)}"))
        
        if frames_loaded == 0:
            # Log the specific sequence details before raising the error
            error_message = (
                f"Could not load any frames for sequence {idx}. "
                f"Video ID: {video_id}. "
                f"Frames attempted: {frame_ids_in_sequence}"
            )
            # You can also write this to a log file if you prefer
            print(f"ERROR: {error_message}")
            raise ValueError(error_message)
        
        if frames_loaded < self.sequence_length:
            # Pad with the last successfully loaded frame
            while len(frames) < self.sequence_length:
                frames.append(frames[-1].clone())
                labels.append(labels[-1].copy())
        
        frames = torch.stack(frames)
        labels = torch.tensor(labels).float()
        
        return frames, labels, frames_loaded, video_id

    def log_skipped_files(self):
        """Write skipped files to a log file for debugging purposes."""
        # ... (This method remains the same)
        if self.log_dir and self.skipped_files:
            log_path = Path(self.log_dir) / "skipped_frames.txt"
            with open(log_path, 'w') as f:
                f.write(f"Total skipped frames: {len(self.skipped_files)}\n\n")
                f.write("Skipped files:\n")
                for file_path, reason in self.skipped_files:
                    f.write(f"{file_path} - {reason}\n")
            print(f"Logged {len(self.skipped_files)} skipped frames to {log_path}")