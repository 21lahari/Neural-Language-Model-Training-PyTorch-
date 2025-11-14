# Neural Language Model — Assignment

## Files
- train_lm.py : Single-file PyTorch LM (Embedding + LSTM + Linear)
- dataset.txt : Provided dataset (use only this file)
- runs/ : Contains output per experiment (loss plot, results.json, best_model.pt)

## Quick commands
Underfit:
python train_lm.py --data_path dataset.txt --save_dir runs/underfit --epochs 5 --batch_size 64 --seq_len 20 --embed 32 --hidden 32 --nlayers 1 --lr 5e-3 --seed 42

Overfit (run briefly and stop after 3–6 epochs):
python train_lm.py --data_path dataset.txt --save_dir runs/overfit --epochs 20 --batch_size 32 --seq_len 30 --embed 512 --hidden 1024 --nlayers 3 --dropout 0.0 --lr 1e-3 --seed 42

Best-fit:
python train_lm.py --data_path dataset.txt --save_dir runs/bestfit_quick --epochs 10 --batch_size 64 --seq_len 20 --embed 96 --hidden 192 --nlayers 1 --dropout 0.4 --lr 1e-3 --early_stop 3 --min_freq 2 --clip_grad 0.5 --seed 42
