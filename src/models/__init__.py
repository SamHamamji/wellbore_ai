import torch

from .cnn2d import WaveCnn2d
from .cnn3d import WaveCnn3d
from .mlp import WaveMlp

models: dict[str, type[torch.nn.Module]] = {
    "cnn2d": WaveCnn2d,
    "cnn3d": WaveCnn3d,
    "mlp": WaveMlp,
}

__all__ = ["WaveCnn2d", "WaveCnn3d", "WaveMlp", "models"]
