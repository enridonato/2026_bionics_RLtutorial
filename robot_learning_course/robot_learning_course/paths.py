"""
Where files are stored.

  ~/robot_learning_files/data/     datasets you collect   (*.npz)
  ~/robot_learning_files/models/   networks you train     (*.pt) and training plots (*.png)

If you did not manage to collect data or to train a model, ready-made ones ("pretrained")
ship with the package: find_file() falls back to them automatically.
"""
import os

WORK_DIR = os.path.expanduser("~/robot_learning_files")
DATA_DIR = os.path.join(WORK_DIR, "data")
MODELS_DIR = os.path.join(WORK_DIR, "models")


def package_dir(folder):
    """A folder shipped with the package ("mujoco_models", "pretrained")."""
    return os.path.join(os.path.dirname(__file__), "..", folder)


def find_file(path):
    """Return `path` if it exists, otherwise the file with the same name in the pretrained folder."""
    if os.path.exists(path):
        return path
    fallback = os.path.join(package_dir("pretrained"), os.path.basename(path))
    if os.path.exists(fallback):
        print(f"[paths] '{path}' not found -> using the pretrained file '{fallback}'")
        return fallback
    raise FileNotFoundError(f"Neither '{path}' nor '{fallback}' exist")
