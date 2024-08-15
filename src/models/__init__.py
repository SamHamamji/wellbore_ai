import torch

from .fft_cnn2d import FftCnn2d
from .wave_mlp import WaveMlp
from .fft_mlp import FftMlp

models: dict[str, type[torch.nn.Module]] = {
    "fft_cnn2d": FftCnn2d,
    "wave_mlp": WaveMlp,
    "fft_mlp": FftMlp,
}

__all__ = ["FftCnn2d", "FftMlp", "WaveMlp", "models"]
