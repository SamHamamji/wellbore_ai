import torch

from .amplitude_fft_cnn2d import AmplitudeFftCnn2d
from .fft_cnn2d import FftCnn2d
from .fft_cnn2d_inv import FftCnn2dInv
from .fft_mlp import FftMlp
from .polar_fft_cnn2d import PolarFftCnn2d
from .wave_mlp import WaveMlp

models: dict[str, type[torch.nn.Module]] = {
    "amplitude_fft_cnn2d": AmplitudeFftCnn2d,
    "fft_cnn2d": FftCnn2d,
    "fft_cnn2d_inv": FftCnn2dInv,
    "fft_mlp": FftMlp,
    "polar_fft_cnn2d": PolarFftCnn2d,
    "wave_mlp": WaveMlp,
}

__all__ = [
    "AmplitudeFftCnn2d",
    "FftCnn2d",
    "FftCnn2dInv",
    "FftMlp",
    "PolarFftCnn2d",
    "WaveMlp",
    "models",
]
