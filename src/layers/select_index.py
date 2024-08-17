import torch


class SelectIndexLayer(torch.nn.Module):
    def __init__(self, dim: int, slices: tuple[slice, ...]):
        super().__init__()
        self.dim = dim
        self.slices = slices

    def forward(self, x: torch.Tensor):
        return x[tuple(self.slices)]
