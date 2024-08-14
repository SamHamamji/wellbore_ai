import torch


class FftLayer(torch.nn.Module):
    def __init__(self, time_dim=-1, complex_dim=-1):
        super().__init__()
        self.time_dim = time_dim
        self.complex_dim = complex_dim

    def forward(self, wave: torch.Tensor):
        ft = torch.fft.rfft(wave, dim=self.time_dim)  # pylint: disable=not-callable
        return torch.view_as_real(ft).movedim(-1, self.complex_dim)
