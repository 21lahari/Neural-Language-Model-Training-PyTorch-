## Neural Language Model Training (PyTorch)
This repository contains code and experiments for Assignment 2: Neural Language Model Training with PyTorch. The project demonstrates training and evaluation of a word-level LSTM-based language model on a provided dataset, including underfitting, overfitting, and best-fit scenarios.

## Dataset
Single text file: dataset.txt (only this dataset is used).

## Implementation
Framework: PyTorch, implemented from scratch (no pre-trained models, no high-level language model libraries).

Architecture: Word-level LSTM language model (Embedding → LSTM → Linear).

Loss: CrossEntropyLoss.

Optimizer: Adam.

Reproducibility: Fixed random seed (--seed 42).

Checkpoints and Results saved in runs/<scenario>/ subfolders.

## Requirements
Python >= 3.8
torch
numpy
matplotlib

## Quick commands
Underfit:
python train_lm.py --data_path dataset.txt --save_dir runs/underfit --epochs 5 --batch_size 64 --seq_len 20 --embed 32 --hidden 32 --nlayers 1 --lr 5e-3 --seed 42

Overfit (run briefly and stop after 3–6 epochs):
python train_lm.py --data_path dataset.txt --save_dir runs/overfit --epochs 20 --batch_size 32 --seq_len 30 --embed 512 --hidden 1024 --nlayers 3 --dropout 0.0 --lr 1e-3 --seed 42

Best-fit:
python train_lm.py --data_path dataset.txt --save_dir runs/bestfit_quick --epochs 10 --batch_size 64 --seq_len 20 --embed 96 --hidden 192 --nlayers 1 --dropout 0.4 --lr 1e-3 --early_stop 3 --min_freq 2 --clip_grad 0.5 --seed 42
