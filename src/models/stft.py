import torch


class Stft(torch.nn.Module):
    def __init__(self, n_fft: int):
        self.n_fft = n_fft
        super().__init__()
        self.eval()

    def forward(self, wave: torch.Tensor):
        fourier = torch.stft(
            wave,
            n_fft=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
            normalized=False,
        )
        return torch.view_as_real(fourier)
