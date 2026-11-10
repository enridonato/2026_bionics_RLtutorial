"""
Learn the INVERSE kinematics of the FR3:   q = g_phi(x)     x: 3D position -> q: 7 joint angles

Two ways, chosen with --mode:

  naive   Plain supervised learning on the same dataset, with inputs and labels swapped:
          minimize || g(x) - q ||^2.
          PROBLEM: the FR3 has 7 joints for 3 coordinates -> infinitely many q reach the same x.
          The dataset contains many different "right answers" for (almost) the same x, and the
          network learns their AVERAGE... which in general does NOT reach x.

  distal  "Distal teacher" (Jordan & Rumelhart, 1992). We do not ask WHICH q to choose: we only
          ask that the chosen q REACHES x, checked through the learned forward model f (frozen):
          minimize || f(g(x)) - x ||^2  +  beta * || g(x) - q_home ||^2
          The small second term breaks the tie: among all solutions, prefer one close to home.

The result is checked with the TRUE kinematics: is f_true(g(x)) close to x?

Run:  train_ik --mode naive
      train_ik --mode distal
"""
import argparse
import os
import time

import torch

from robot_learning_course import fr3
from robot_learning_course.kinematics import TrueKinematics
from robot_learning_course.learning.data import load_dataset, make_loader, split_indices
from robot_learning_course.learning.models import MLP, load_model, save_model
from robot_learning_course.paths import DATA_DIR, MODELS_DIR, find_file


def reaching_error_mm(ik_model, x, true_kin):
    """Send the predicted q to the TRUE robot kinematics: how far from the target do we end up?"""
    with torch.no_grad():
        q_pred = ik_model(x).numpy()
    x_reached = torch.tensor(true_kin.fk(q_pred), dtype=torch.float32)
    return 1000 * (x_reached - x).norm(dim=1)  # one error per target, in mm


def naive_loss(ik, x_batch, q_batch):
    """Plain supervised learning: the output must equal the recorded joint angles."""
    return (((ik(x_batch) - q_batch) / ik.out_std) ** 2).mean()


def distal_loss(ik, fk, x_batch, home, q_range, beta):
    """Distal teacher: no joint labels needed, the output must REACH x through the learned forward model."""
    q_pred = ik(x_batch)
    reach = (((fk(q_pred) - x_batch) / fk.out_std) ** 2).mean()   # does q_pred reach x?
    posture = (((q_pred - home) / q_range) ** 2).mean()           # is it close to home?
    return reach + beta * posture


def main():
    parser = argparse.ArgumentParser(description="Train an inverse-kinematics MLP")
    parser.add_argument("--mode", choices=["naive", "distal"], default="distal")
    parser.add_argument("--data", default=os.path.join(DATA_DIR, "fk_dataset.npz"))
    parser.add_argument("--fk", default=os.path.join(MODELS_DIR, "fk_model.pt"), help="learned forward model")
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3, help="weight of the 'stay close to home' term")
    args = parser.parse_args()
    out = args.out or os.path.join(MODELS_DIR, f"ik_{args.mode}.pt")
    torch.manual_seed(0)

    # ---- 1. DATA (note: inputs and labels are swapped with respect to train_fk)
    q, x = load_dataset(args.data)
    train, val, test = split_indices(len(q))
    loader = make_loader(x[train], q[train])
    true_kin = TrueKinematics()

    # ---- 2. MODEL: 3 inputs -> 7 outputs
    if args.mode == "naive":
        ik = MLP(in_dim=3, out_dim=7)
        ik.set_normalization(x[train], q[train])
    else:
        fk = load_model(find_file(args.fk))
        for p in fk.parameters():
            p.requires_grad_(False)          # the forward model is FROZEN: we only train ik
        ik = MLP(in_dim=3, out_dim=7, bounded=True)
        ik.set_normalization(x[train], q[train])
        ik.set_bounds(fr3.JOINT_LOWER, fr3.JOINT_UPPER)   # outputs always inside the joint limits
        home = torch.tensor(fr3.HOME, dtype=torch.float32)
        q_range = torch.tensor(fr3.JOINT_UPPER - fr3.JOINT_LOWER, dtype=torch.float32)

    # ---- 3. TRAINING LOOP (identical to train_fk: only the loss changed)
    optimizer = torch.optim.Adam(ik.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start = time.time()
    for epoch in range(args.epochs):
        ik.train()
        for x_batch, q_batch in loader:
            if args.mode == "naive":
                loss = naive_loss(ik, x_batch, q_batch)
            else:
                loss = distal_loss(ik, fk, x_batch, home, q_range, args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            ik.eval()
            err = reaching_error_mm(ik, x[val][:2000], true_kin).mean().item()
            print(f"epoch {epoch:4d} | loss {loss.item():.5f} | val reaching error {err:7.1f} mm "
                  f"| {time.time() - start:5.1f} s")

    # ---- 4. TEST with the true kinematics
    errors = reaching_error_mm(ik.eval(), x[test], true_kin)
    print(f"[ik-{args.mode}] TEST reaching error: mean {errors.mean():.1f} mm, "
          f"median {errors.median():.1f} mm, 90% of targets below {errors.quantile(0.9):.1f} mm")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_model(ik, out)
    print(f"[ik-{args.mode}] saved model to {out}")


if __name__ == "__main__":
    main()
