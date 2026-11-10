"""Quick checks that everything works:  python -m pytest test/"""
import numpy as np
import torch

from robot_learning_course import fr3
from robot_learning_course.kinematics import TrueKinematics
from robot_learning_course.learning.models import MLP
from robot_learning_course.scene import build_scene, joint_indices


def test_multi_robot_scene():
    model, data, bases = build_scene(num_envs=4)
    assert model.nu == 4 * 7
    qpos_ids, _, _ = joint_indices(model, 3)
    assert np.allclose(data.qpos[qpos_ids], fr3.HOME)


def test_true_kinematics_shapes():
    tk = TrueKinematics()
    assert tk.fk(fr3.HOME).shape == (3,)
    assert tk.jacobian(fr3.HOME).shape == (3, 7)


def test_mlp_jacobian_shape():
    net = MLP(7, 3, hidden=(16, 16), angle_input=True)
    J = torch.autograd.functional.jacobian(net, torch.zeros(7))
    assert J.shape == (3, 7)


def test_pretrained_models():
    """The plan-B models must stay accurate (re-run after regenerating pretrained/)."""
    import os
    from robot_learning_course.learning.models import load_model
    folder = os.path.join(os.path.dirname(__file__), "..", "pretrained")
    fk = load_model(os.path.join(folder, "fk_model.pt"))
    ik = load_model(os.path.join(folder, "ik_distal.pt"))
    tk = TrueKinematics()
    q = fr3.clip_to_limits(fr3.HOME + np.random.default_rng(0).normal(0, 0.3, (100, 7)))
    with torch.no_grad():
        x_fk = fk(torch.tensor(q, dtype=torch.float32)).numpy()
    assert 1000 * np.linalg.norm(x_fk - tk.fk(q), axis=1).mean() < 5.0           # FK error [mm]
    J = torch.autograd.functional.jacobian(fk, torch.tensor(fr3.HOME, dtype=torch.float32)).numpy()
    J_true = tk.jacobian(fr3.HOME)
    assert np.linalg.norm(J - J_true) / np.linalg.norm(J_true) < 0.05             # Jacobian error
    x = tk.fk(q)
    with torch.no_grad():
        q_ik = ik(torch.tensor(x, dtype=torch.float32)).numpy()
    assert 1000 * np.linalg.norm(tk.fk(q_ik) - x, axis=1).mean() < 10.0           # IK reaching error [mm]


def test_collected_labels_are_exact(tmp_path, monkeypatch):
    """Every recorded pair (q, x) must agree with the true kinematics: no 2 ms label error."""
    from robot_learning_course import run_collect
    output = str(tmp_path / "data.npz")
    monkeypatch.setattr("sys.argv", ["run_collect", "--num-envs", "2", "--num-samples", "200", "--headless", "--output", output])
    run_collect.main()
    d = np.load(output)
    assert d["q"].shape == (200, 7)
    assert np.linalg.norm(TrueKinematics().fk(d["q"]) - d["x"], axis=1).max() < 1e-6


def test_control_follows_the_circle(tmp_path, monkeypatch, capsys):
    """Proportional control lags a moving target by V/K, about 30 mm on the circle (see the slides)."""
    import re
    from robot_learning_course import run_control
    monkeypatch.setattr(run_control, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["run_control", "--shape", "circle", "--seconds", "12", "--headless"])
    run_control.main()
    error = float(re.search(r"last 8 s: ([0-9.]+) mm", capsys.readouterr().out).group(1))
    assert 25 < error < 36
    assert (tmp_path / "tracking_error.png").exists()
