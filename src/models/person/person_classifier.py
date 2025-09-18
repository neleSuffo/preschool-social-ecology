import os
import pandas as pd
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
    """CNN feature extractor using pretrained ResNet backbone.
    
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
        self.sequence_length = sequence_length # number of frames per sequence
        self.transform = transform
        self.skipped_files = []
        self.log_dir = log_dir

        self.grouped = []
        for video_id, group in self.data.groupby('video_id'):
            sorted_group = group.sort_values('frame_id')
            self.grouped.append(sorted_group.reset_index(drop=True))

    def log_skipped_files(self):
        """Write skipped files to a log file for debugging purposes.
        
        Creates a text file containing all frames that were skipped during dataset loading,
        along with the reason for skipping (file not found, loading error, etc.).
        """
        if self.log_dir and self.skipped_files:
            log_path = os.path.join(self.log_dir, "skipped_frames.txt")
            with open(log_path, 'w') as f:
                f.write(f"Total skipped frames: {len(self.skipped_files)}\n\n")
                f.write("Skipped files:\n")
                for file_path, reason in self.skipped_files:
                    f.write(f"{file_path} - {reason}\n")
            print(f"Logged {len(self.skipped_files)} skipped frames to {log_path}")

    def __len__(self):
        """Get the total number of sequences in the dataset.
        
        Calculates the number of possible sequences that can be created from all video groups,
        considering the sequence length parameter.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Total number of sequences available in the dataset.
        """
        return sum(max(0, len(g) - self.sequence_length + 1) for g in self.grouped)

    def __getitem__(self, idx):
        """Get a sequence of frames and labels for training.
        
        Extracts a sequence of consecutive frames from a video along with their corresponding labels.
        Handles missing or corrupted files by skipping them and logging the issues.
        
        Parameters
        ----------
        idx : int
            Index of the sequence to retrieve.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, int, str]
            A tuple containing:
            - frames: torch.Tensor of shape (seq_len, C, H, W) with frame images
            - labels: torch.Tensor of shape (seq_len, 2) with adult/child labels  
            - length: int indicating actual sequence length
            - video_id: str identifier for the source video
        """
        total = 0
        for group_idx, group in enumerate(self.grouped):
            length = max(0, len(group) - self.sequence_length + 1)
            if idx < total + length:
                start_idx = idx - total
                frames = []
                labels = []
                video_id = None
                
                # Try to load frames, skip broken ones
                current_idx = start_idx
                frames_loaded = 0
                max_attempts = len(group) - start_idx  # Don't go beyond group bounds
                
                while frames_loaded < self.sequence_length and current_idx < len(group):
                    row = group.iloc[current_idx]
                    file_path = row['file_path']
                    
                    try:
                        # Check if file exists
                        if not os.path.exists(file_path):
                            self.skipped_files.append((file_path, "File not found"))
                            current_idx += 1
                            continue
                        
                        img = Image.open(file_path).convert('RGB')
                        if self.transform:
                            img = self.transform(img)
                        frames.append(img)
                        labels.append([row['child'], row['adult']])
                        if video_id is None:
                            video_id = row['video_id']
                        frames_loaded += 1
                        
                    except Exception as e:
                        self.skipped_files.append((file_path, f"Error loading: {str(e)}"))
                        
                    current_idx += 1
                
                # If we couldn't load enough frames, pad with the last valid frame or skip this sequence
                if frames_loaded == 0:
                    # No valid frames found, try next sequence
                    total += length
                    continue
                elif frames_loaded < self.sequence_length:
                    # Pad with last frame
                    while len(frames) < self.sequence_length:
                        frames.append(frames[-1].clone())
                        labels.append(labels[-1].copy())
                
                frames = torch.stack(frames)  # shape: (seq_len, C, H, W)
                labels = torch.tensor(labels).float()  # shape: (seq_len, 2)
                return frames, labels, len(frames), video_id
            total += length
        raise IndexError("Index out of range")