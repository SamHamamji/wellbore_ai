import torch

from .amplitude_fft_cnn2d import AmplitudeFftCnn2d
from .dispersion_curve_mlp import DispersionCurveMlp
from .fft_cnn2d import FftCnn2d
from .fft_cnn2d_inv import FftCnn2dInv
from .fft_mlp import FftMlp
from .polar_fft_cnn2d import PolarFftCnn2d
from .wave_cnn2d import WaveCnn2d
from .wave_mlp import WaveMlp

models: dict[str, type[torch.nn.Module]] = {
    "amplitude_fft_cnn2d": AmplitudeFftCnn2d,
    "dispersion_curve_mlp": DispersionCurveMlp,
    "fft_cnn2d": FftCnn2d,
    "fft_cnn2d_inv": FftCnn2dInv,
    "fft_mlp": FftMlp,
    "polar_fft_cnn2d": PolarFftCnn2d,
    "wave_cnn2d": WaveCnn2d,
    "wave_mlp": WaveMlp,
}

__all__ = [
    "AmplitudeFftCnn2d",
    "FftCnn2d",
    "FftCnn2dInv",
    "FftMlp",
    "PolarFftCnn2d",
    "WaveCnn2d",
    "WaveMlp",
    "models",
]
