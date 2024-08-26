import torch


class SelectIndexLayer(torch.nn.Module):
    def __init__(self, slices: tuple[slice | int, ...]):
        super().__init__()
        self.slices = slices

    def forward(self, x: torch.Tensor):
        return x[tuple(self.slices)]
