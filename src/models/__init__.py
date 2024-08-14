import torch

from .wave_cnn2d import WaveCnn2d
from .wave_mlp import WaveMlp
from .fft_mlp import FftMlp

models: dict[str, type[torch.nn.Module]] = {
    "wave_cnn2d": WaveCnn2d,
    "wave_mlp": WaveMlp,
    "fft_mlp": FftMlp,
}

__all__ = ["WaveCnn2d", "FftMlp", "WaveMlp", "models"]
