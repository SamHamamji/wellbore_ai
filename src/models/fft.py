import torch


class FftLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, wave: torch.Tensor):
        ft = torch.fft.rfft(wave, dim=-1)  # pylint: disable=not-callable
        return torch.view_as_real(ft)
