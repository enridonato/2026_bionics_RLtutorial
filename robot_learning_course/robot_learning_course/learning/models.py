"""
The neural network used everywhere in the course: a Multi-Layer Perceptron (MLP).

    input --(sin, cos)--> normalize --> [Linear -> SiLU] x L --> Linear --> de-normalize --> output

Normalization is stored INSIDE the model (as "buffers"), so the saved file is self-contained:
you give it raw joint angles in rad and you get raw positions in m. Nothing else to remember.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256, 256), angle_input=False, bounded=False):
        super().__init__()
        self.config = dict(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden),
                           angle_input=angle_input, bounded=bounded)
        # If the inputs are ANGLES, we feed the network sin(q) and cos(q) instead of q:
        # angles are periodic, and the kinematics is made of sines and cosines anyway.
        self.angle_input = angle_input
        feat_dim = 2 * in_dim if angle_input else in_dim

        layers, d = [], feat_dim
        for h in hidden:
            # SiLU(z) = z * sigmoid(z): a SMOOTH version of ReLU. We will differentiate the network
            # to get the Jacobian, so smoothness matters (ReLU gives a "staircase" Jacobian).
            # TODO (optional exercise 2): replace nn.SiLU() with nn.ReLU() or nn.Tanh(), retrain, compare.
            layers += [nn.Linear(d, h), nn.SiLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

        # Statistics of the data. register_buffer = saved with the model, but NOT trained.
        self.register_buffer("in_mean", torch.zeros(feat_dim))
        self.register_buffer("in_std", torch.ones(feat_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std", torch.ones(out_dim))
        # Only for bounded=True: the output is squashed inside [out_low, out_high] (e.g. joint limits)
        self.bounded = bounded
        self.register_buffer("out_low", torch.zeros(out_dim))
        self.register_buffer("out_high", torch.ones(out_dim))

    def features(self, inp):
        if self.angle_input:
            return torch.cat([torch.sin(inp), torch.cos(inp)], dim=-1)
        return inp

    def set_normalization(self, inputs, outputs):
        """Compute mean and standard deviation of the TRAINING data (tensors of shape (N, dim))."""
        feats = self.features(inputs)
        self.in_mean.copy_(feats.mean(0))
        self.in_std.copy_(feats.std(0) + 1e-8)
        self.out_mean.copy_(outputs.mean(0))
        self.out_std.copy_(outputs.std(0) + 1e-8)

    def set_bounds(self, low, high):
        self.out_low.copy_(torch.as_tensor(low, dtype=torch.float32))
        self.out_high.copy_(torch.as_tensor(high, dtype=torch.float32))

    def forward(self, inp):
        z = (self.features(inp) - self.in_mean) / self.in_std   # 1. normalize: inputs ~ mean 0, std 1
        y = self.net(z)                                  # 2. the neural network itself
        if self.bounded:                                 # 3a. squash inside [low, high]
            return self.out_low + (self.out_high - self.out_low) * torch.sigmoid(y)
        return y * self.out_std + self.out_mean          # 3b. back to physical units


def save_model(model, path):
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_model(path):
    """Rebuild a saved MLP (architecture + weights + normalization), ready to use."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # data only: never runs code from the file
    model = MLP(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # we only use it, we do not train it anymore
    return model
