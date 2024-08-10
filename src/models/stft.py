import torch


class Stft(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wave: torch.Tensor):
        fourier = torch.stft(
            wave,
            n_fft=64,
            window=torch.hann_window(64),
            return_complex=True,
            normalized=False,
        )
        return torch.view_as_real(fourier)
