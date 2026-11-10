"""
Loading a dataset and splitting it into training / validation / test sets.

  training set    -> the network learns from it (weights are updated)
  validation set  -> we watch it during training to detect overfitting and choose hyper-parameters
  test set        -> used ONCE at the very end: an honest estimate of the error on new data
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robot_learning_course.paths import find_file


def load_dataset(path):
    """Return (q, x) as float32 tensors: q (N, 7) joint angles [rad], x (N, 3) flange position [m]."""
    d = np.load(find_file(path))
    q = torch.tensor(d["q"], dtype=torch.float32)
    x = torch.tensor(d["x"], dtype=torch.float32)
    print(f"[data] loaded {len(q)} samples from {path}")
    return q, x


def split_indices(n, val_fraction=0.15, test_fraction=0.15, seed=0):
    """Shuffle the indices 0..n-1 and cut them into three groups."""
    # TODO (optional exercise 3): consecutive samples (10 ms apart) are almost identical, so a random
    #   split puts "twins" in train and test -> the test error is too optimistic. Split by TIME instead
    #   (e.g. last 15% of the recording = test) and compare the test errors.
    perm = np.random.default_rng(seed).permutation(n)
    n_val, n_test = int(val_fraction * n), int(test_fraction * n)
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]  # train, val, test


def make_loader(inputs, targets, batch_size=256, shuffle=True):
    """A DataLoader serves the data in small random "mini-batches"."""
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle)
