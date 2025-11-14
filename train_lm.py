#!/usr/bin/env python3
"""
train_lm.py
Train a small word-level LSTM language model from scratch in PyTorch.
Usage examples (see bottom of file for recommended parameter sets):
  python train_lm.py --data_path dataset.txt --epochs 5 --batch_size 64 --seq_len 30 --embed 128 --hidden 256 --nlayers 2 --lr 1e-3 --save_dir runs/exp1
"""

import argparse
import os
import math
import random
from collections import Counter
import time
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------
# Reproducibility
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------
# Simple tokenizer (word-level)
# ------------------------
class Vocab:
    def __init__(self, tokens, min_freq=1, unk_token="<unk>", pad_token="<pad>"):
        self.unk_token = unk_token
        self.pad_token = pad_token
        ctr = Counter(tokens)
        # Keep tokens with freq >= min_freq
        items = [tok for tok, f in ctr.items() if f >= min_freq]
        # sort by frequency (desc) then alpha to be deterministic
        items.sort(key=lambda x: (-ctr[x], x))
        self.itos = [pad_token, unk_token] + items
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def __len__(self):
        return len(self.itos)
    def encode(self, tokens):
        return [self.stoi.get(t, self.stoi[self.unk_token]) for t in tokens]
    def decode(self, indices):
        return [self.itos[i] for i in indices]

def tokenize_text(text):
    # simple split on whitespace. You can replace with better tokenizer if allowed.
    tokens = text.replace("\n", " <nl> ").split()
    return tokens

# ------------------------
# Dataset: sliding windows of tokens -> next token prediction
# ------------------------
class LMDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = token_ids
        # create windows: for i from 0..len-data - seq_len - 1: input = tokens[i:i+seq_len], target = tokens[i+1:i+seq_len+1]
        self.n_samples = max(0, len(self.data) - seq_len)
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y

# ------------------------
# Model: Embedding -> LSTM -> Linear
# ------------------------
class LMLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, nlayers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, nlayers, batch_first=True, dropout=dropout if nlayers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.nlayers = nlayers
    def forward(self, x, hidden=None):
        # x: [B, seq_len]
        emb = self.embed(x)  # [B, seq_len, E]
        out, hidden = self.lstm(emb, hidden)  # out: [B, seq_len, H]
        logits = self.fc(out)  # [B, seq_len, V]
        return logits, hidden

# ------------------------
# Utilities: train / eval / perplexity / plot
# ------------------------
def perplexity(loss):
    return math.exp(loss) if loss < 100 else float('inf')

def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=None):
    model.train()
    total_loss = 0.0
    n_tokens = 0
    start = time.time()
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits, _ = model(xb)
        # reshape for loss: (B*seq_len, V)
        B, S, V = logits.shape
        loss = criterion(logits.view(B*S, V), yb.view(B*S))
        loss_val = loss.item()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss_val * (B*S)
        n_tokens += (B*S)
    avg_loss = total_loss / n_tokens
    elapsed = time.time() - start
    return avg_loss, elapsed

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            B,S,V = logits.shape
            loss = criterion(logits.view(B*S, V), yb.view(B*S))
            total_loss += loss.item() * (B*S)
            n_tokens += (B*S)
    avg_loss = total_loss / n_tokens
    return avg_loss

def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ------------------------
# Main training flow
# ------------------------
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    with open(args.data_path, "r", encoding="utf-8") as f:
        raw = f.read()
    tokens = tokenize_text(raw)
    vocab = Vocab(tokens, min_freq=args.min_freq)
    token_ids = vocab.encode(tokens)

    # split
    n = len(token_ids)
    train_cut = int(n * args.train_frac)
    val_cut = int(n * (args.train_frac + args.val_frac))
    train_ids = token_ids[:train_cut]
    val_ids = token_ids[train_cut:val_cut]
    test_ids = token_ids[val_cut:]

    train_ds = LMDataset(train_ids, args.seq_len)
    val_ds = LMDataset(val_ids, args.seq_len)
    test_ds = LMDataset(test_ids, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model
    model = LMLSTM(len(vocab), embed_size=args.embed, hidden_size=args.hidden, nlayers=args.nlayers, dropout=args.dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi[vocab.pad_token])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    history = {"train_loss": [], "val_loss": []}

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t_loss, t_time = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad=args.clip_grad)
        v_loss = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        print(f"Epoch {epoch:03d} | train_loss {t_loss:.4f} | val_loss {v_loss:.4f} | train_ppl {perplexity(t_loss):.3f} | val_ppl {perplexity(v_loss):.3f} | time {t_time:.1f}s")
        # save best model
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocab': vocab.itos,
                'args': vars(args)
            }, os.path.join(args.save_dir, "best_model.pt"))

        # save checkpoint each epoch
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'history': history,
            'vocab': vocab.itos
        }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pt"))

        # early stopping optional
        if args.early_stop is not None and epoch - np.argmin(val_losses) >= args.early_stop:
            print("Early stopping triggered.")
            break

    # final evaluation on test
    best_ckpt = torch.load(os.path.join(args.save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt['model_state'])
    test_loss = eval_epoch(model, test_loader, criterion, device)
    test_ppl = perplexity(test_loss)
    print(f"Test loss: {test_loss:.4f} | Test perplexity: {test_ppl:.3f}")

    # Save results and plots
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({
            "train_loss": train_losses,
            "val_loss": val_losses,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "vocab_size": len(vocab),
            "n_train_tokens": len(train_ids),
            "n_val_tokens": len(val_ids),
            "n_test_tokens": len(test_ids)
        }, f, indent=2)

    plot_losses(train_losses, val_losses, os.path.join(args.save_dir, "loss_plot.png"))
    print(f"Saved plots and model to {args.save_dir}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to text dataset (single file).")
    parser.add_argument("--save_dir", type=str, default="runs/exp", help="Directory to save models/plots.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--embed", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--early_stop", type=int, default=None, help="Epochs patience to early stop")
    args = parser.parse_args()
    main(args)
