import torch

from .fft_cnn2d import FftCnn2d
from .fft_mlp import FftMlp
from .polar_fft_cnn2d import PolarFftCnn2d
from .wave_mlp import WaveMlp

models: dict[str, type[torch.nn.Module]] = {
    "fft_cnn2d": FftCnn2d,
    "fft_mlp": FftMlp,
    "polar_fft_cnn2d": PolarFftCnn2d,
    "wave_mlp": WaveMlp,
}

__all__ = ["FftCnn2d", "FftMlp", "PolarFftCnn2d", "WaveMlp", "models"]
