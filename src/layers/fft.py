import torch


class FftLayer(torch.nn.Module):
    def __init__(self, time_dim=-1, complex_dim=-1, polar_decomposition=False):
        super().__init__()
        self.time_dim = time_dim
        self.complex_dim = complex_dim
        self.polar_decomposition = polar_decomposition

    def forward(self, wave: torch.Tensor):
        # pylint: disable=not-callable
        ft: torch.Tensor = torch.fft.rfft(wave, dim=self.time_dim)
        if self.polar_decomposition:
            ft = torch.stack((ft.abs(), ft.angle()), dim=-1)
        else:
            ft = torch.view_as_real(ft)
        return ft.movedim(-1, self.complex_dim)
