import torch

from .cnn2d import WaveCnn2d
from .cnn3d import WaveCnn3d
from .wave_mlp import WaveMlp
from .fft_mlp import FftMlp

models: dict[str, type[torch.nn.Module]] = {
    "cnn2d": WaveCnn2d,
    "cnn3d": WaveCnn3d,
    "wave_mlp": WaveMlp,
    "fft_mlp": FftMlp,
}

__all__ = ["WaveCnn2d", "WaveCnn3d", "WaveMlp", "models"]
