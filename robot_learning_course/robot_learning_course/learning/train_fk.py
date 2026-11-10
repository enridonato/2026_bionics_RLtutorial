"""
Learn the FORWARD kinematics of the FR3:   x = f_theta(q)     q: 7 joint angles -> x: 3D flange position

This is supervised learning (regression): the dataset gives us the right answer x for every input q.

Run:  train_fk
      train_fk --epochs 50 --hidden 128 128
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")  # draw plots into files (no window needed)
import matplotlib.pyplot as plt
import torch

from robot_learning_course.learning.data import load_dataset, make_loader, split_indices
from robot_learning_course.learning.models import MLP, save_model
from robot_learning_course.paths import DATA_DIR, MODELS_DIR


def mean_error_mm(model, q, x):
    """Average distance between predicted and true position, in millimetres."""
    with torch.no_grad():  # no gradients needed: we are only measuring
        return 1000 * (model(q) - x).norm(dim=1).mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train a forward-kinematics MLP")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "fk_dataset.npz"))
    parser.add_argument("--out", default=os.path.join(MODELS_DIR, "fk_model.pt"))
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    torch.manual_seed(0)

    # ---- 1. DATA: load and split
    q, x = load_dataset(args.data)
    train, val, test = split_indices(len(q))
    loader = make_loader(q[train], x[train], args.batch)

    # ---- 2. MODEL: 7 inputs -> 3 outputs
    model = MLP(in_dim=7, out_dim=3, hidden=args.hidden, angle_input=True)
    model.set_normalization(q[train], x[train])
    print(f"[fk] network with {sum(p.numel() for p in model.parameters())} trainable parameters")

    # ---- 3. LOSS and OPTIMIZER
    # Loss on NORMALIZED outputs, so that x, y and z count the same
    def loss_fn(x_pred, x_true):
        return (((x_pred - x_true) / model.out_std) ** 2).mean()   # Mean Squared Error

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Slowly reduce the learning rate: big steps at the beginning, fine steps at the end
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- 4. TRAINING LOOP
    history = {"train": [], "val": []}
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        for q_batch, x_batch in loader:           # one mini-batch at a time
            loss = loss_fn(model(q_batch), x_batch)  # a) forward pass: predict and measure the error
            optimizer.zero_grad()                  # b) forget the gradients of the previous step
            loss.backward()                        # c) backpropagation: d(loss)/d(every weight)
            optimizer.step()                       # d) move every weight a little bit downhill
        scheduler.step()

        model.eval()
        history["train"].append(mean_error_mm(model, q[train], x[train]))
        history["val"].append(mean_error_mm(model, q[val], x[val]))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d} | train error {history['train'][-1]:7.2f} mm "
                  f"| val error {history['val'][-1]:7.2f} mm | {time.time() - start:5.1f} s")

    # ---- 5. FINAL TEST (data never seen during training) and save
    print(f"[fk] TEST error: {mean_error_mm(model, q[test], x[test]):.2f} mm")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_model(model, args.out)

    plt.figure(figsize=(6, 4))
    plt.semilogy(history["train"], label="train")
    plt.semilogy(history["val"], label="validation")
    plt.xlabel("epoch"), plt.ylabel("mean position error [mm]"), plt.legend(), plt.grid(True, which="both")
    plt.title("Forward kinematics learning curve")
    plt.tight_layout()
    plt.savefig(args.out.replace(".pt", "_curve.png"), dpi=150)
    print(f"[fk] saved model to {args.out} (and the learning curve next to it)")


if __name__ == "__main__":
    main()
