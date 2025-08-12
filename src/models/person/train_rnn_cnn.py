"""
train_cnn_rnn.py

End-to-end pipeline:
  ResNet (frame-level feature extractor) -> BiLSTM -> per-frame classifier

Usage:
  python train_cnn_rnn.py --train_csv frames_train.csv --val_csv frames_val.csv --out_dir checkpoints
"""

import os
import argparse
from collections import defaultdict
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# ---------------------------
# Dataset
# ---------------------------
class VideoFrameDataset(Dataset):
    def __init__(self, csv_file, sequence_length=10, transform=None):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.transform = transform

        # Group frames by video_id, then sort by frame_id to create sequences
        self.grouped = []
        for video_id, group in self.data.groupby('video_id'):
            sorted_group = group.sort_values('frame_id')
            self.grouped.append(sorted_group.reset_index(drop=True))

    def __len__(self):
        return sum(max(0, len(g) - self.sequence_length + 1) for g in self.grouped)

    def __getitem__(self, idx):
        # Find which video and which start frame corresponds to idx
        total = 0
        for group in self.grouped:
            length = max(0, len(group) - self.sequence_length + 1)
            if idx < total + length:
                start_idx = idx - total
                frames = []
                labels = []
                for i in range(start_idx, start_idx + self.sequence_length):
                    row = group.iloc[i]
                    img = Image.open(row['file_path']).convert('RGB')
                    if self.transform:
                        img = self.transform(img)
                    frames.append(img)
                    labels.append([row['adult'], row['child']])
                frames = torch.stack(frames)  # shape: (seq_len, C, H, W)
                labels = torch.tensor(labels).float()  # shape: (seq_len, 2)
                return frames, labels
            total += length
        raise IndexError("Index out of range")

# Transforms for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # optionally normalize with ImageNet mean/std if using pretrained CNN
])

def collate_fn(batch):
    batch_sizes = [item[2] for item in batch]
    max_len = max(batch_sizes)
    bs = len(batch)
    C, H, W = batch[0][0].shape[1:]

    images_padded = torch.zeros((bs, max_len, C, H, W))
    labels_padded = torch.full((bs, max_len, 2), -100.0, dtype=torch.float32)  # mask with -100
    lengths = []
    video_ids = []
    for i, (imgs, labs, l, vid) in enumerate(batch):
        images_padded[i, :l] = imgs
        labels_padded[i, :l] = labs
        lengths.append(l)
        video_ids.append(vid)
    return images_padded, labels_padded, torch.tensor(lengths, dtype=torch.long), video_ids

# ---------------------------
# Model: CNN backbone + RNN
# ---------------------------
class CNNEncoder(nn.Module):
    """
    Uses a pretrained ResNet (without final FC) to extract per-frame features.
    Outputs feature vector for each frame.
    """
    def __init__(self, backbone='resnet18', pretrained=True, feat_dim=512):
        super().__init__()
        if backbone == 'resnet18':
            res = models.resnet18(pretrained=pretrained)
            feat_in = res.fc.in_features
            modules = list(res.children())[:-1]  # remove FC and avgpool kept? last is avgpool
            self.encoder = nn.Sequential(*modules)  # outputs (B, feat_in, 1, 1)
            self.feat_dim = feat_in
        else:
            raise NotImplementedError("Only resnet18 implemented; swap if you want resnet50")
        # optional projection to reduce dim
        if self.feat_dim != feat_dim:
            self.project = nn.Linear(self.feat_dim, feat_dim)
            self.feat_dim = feat_dim
        else:
            self.project = None

    def forward(self, x):
        # x: (batch*seq_len, C, H, W)
        f = self.encoder(x)  # (N, feat_in, 1,1)
        f = f.view(f.size(0), -1)  # (N, feat_in)
        if self.project is not None:
            f = self.project(f)
        return f  # (N, feat_dim)

class FrameRNNClassifier(nn.Module):
    def __init__(self, feat_dim=512, rnn_hidden=256, rnn_layers=1, bidirectional=True, num_outputs=2, dropout=0.3):
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
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        logits = self.classifier(rnn_out)  # (batch, max_seq, 2)
        return logits

# ---------------------------
# Utilities: training + eval
# ---------------------------
def sequence_features_from_cnn(cnn, images_padded, lengths, device):
    """
    images_padded: (batch, max_seq, C, H, W)
    returns: feats_padded (batch, max_seq, feat_dim)
    """
    bs, max_seq, C, H, W = images_padded.shape
    imgs = images_padded.view(bs * max_seq, C, H, W).to(device)
    with torch.no_grad():
        # we might want to set cnn.eval() or trainable depending on fine-tuning
        pass
    feats = cnn(imgs)  # (bs * max_seq, feat_dim)
    feats = feats.view(bs, max_seq, -1)
    return feats

def train_one_epoch(models, optimizers, criterion, dataloader, device, freeze_cnn=False):
    cnn, rnn = models
    opt_cnn, opt_rnn = optimizers
    cnn.train(not freeze_cnn)
    rnn.train()
    total_loss = 0.0
    total_correct = 0
    total_labels = 0

    for images_padded, labels_padded, lengths, _ in dataloader:
        mask = (labels_padded != -100)  # (bs, max_seq, 2)
        images_padded = images_padded.to(device)
        labels_padded = labels_padded.to(device)

        if freeze_cnn:
            with torch.no_grad():
                feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
        else:
            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)

        logits = rnn(feats, lengths)  # (bs, max_seq, 2)

        # flatten and mask
        mask_flat = mask.view(-1, 2)
        logits_flat = logits.view(-1, 2)[mask_flat[:,0] != -100]  # adult
        labels_flat = labels_padded.view(-1, 2)[mask_flat[:,0] != -100]

        loss = criterion(logits_flat, labels_flat)

        opt_rnn.zero_grad()
        if opt_cnn:
            opt_cnn.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 5.0)
        if opt_cnn:
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 5.0)
        if opt_cnn:
            opt_cnn.step()
        opt_rnn.step()

        total_loss += loss.item() * images_padded.size(0)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct = (preds == labels_padded) & (labels_padded != -100)
            total_correct += correct.sum().item()
            total_labels += (labels_padded != -100).sum().item()

    return total_loss / len(dataloader.dataset), total_correct / max(1, total_labels)

def eval_on_loader(models, criterion, dataloader, device):
    cnn, rnn = models
    cnn.eval(); rnn.eval()
    total_loss = 0.0
    total_correct = 0
    total_frames = 0

    with torch.no_grad():
        for images_padded, labels_padded, lengths, _ in dataloader:
            images_padded = images_padded.to(device)
            labels_padded = labels_padded.to(device)
            lengths = lengths.to(device)

            feats = sequence_features_from_cnn(cnn, images_padded, lengths, device)
            logits = rnn(feats, lengths)
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels_padded.view(-1)

            loss = criterion(logits_flat, labels_flat)
            total_loss += loss.item() * images_padded.size(0)

            preds = logits.argmax(dim=-1)
            mask = (labels_padded != -100)
            total_correct += ((preds == labels_padded) & mask).sum().item()
            total_frames += mask.sum().item()

    return total_loss / len(dataloader.dataset), total_correct / max(1, total_frames)

# ---------------------------
# Main: parse args, start training
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--out_dir', default='checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)  # number of videos per batch
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freeze_cnn', action='store_true')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # transforms — tune to your dataset resolution
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_ds = VideoFrameDataset(args.train_csv, transform=transform, max_seq_len=args.max_seq_len)
    val_ds = VideoFrameDataset(args.val_csv, transform=transform, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_fn, pin_memory=True)

    device = torch.device(args.device)

    # models
    cnn = CNNEncoder(backbone='resnet18', pretrained=True, feat_dim=512).to(device)
    rnn_model = FrameRNNClassifier(feat_dim=cnn.feat_dim, rnn_hidden=256,
                                rnn_layers=2, bidirectional=True,
                                num_outputs=2, dropout=0.3).to(device)

    # optimizers
    opt_cnn = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=args.lr) if not args.freeze_cnn else None
    opt_rnn = torch.optim.Adam(rnn_model.parameters(), lr=args.lr)
    
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch((cnn, rnn_model), (opt_cnn, opt_rnn), criterion, train_loader,
                                                device, freeze_cnn=args.freeze_cnn)
        val_loss, val_acc = eval_on_loader((cnn, rnn_model), criterion, val_loader, device)
        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.4f}")

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'cnn_state': cnn.state_dict(),
            'rnn_state': rnn_model.state_dict(),
            'opt_rnn': opt_rnn.state_dict(),
            'opt_cnn': opt_cnn.state_dict() if opt_cnn else None,
            'val_acc': val_acc
        }
        ckpt_path = os.path.join(args.out_dir, f'ckpt_epoch{epoch:03d}.pth')
        torch.save(ckpt, ckpt_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.out_dir, 'best.pth')
            torch.save(ckpt, best_path)
            print("  New best saved.")

    print("Training finished. Best val acc:", best_val_acc)

if __name__ == '__main__':
    main()
