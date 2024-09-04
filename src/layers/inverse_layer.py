import torch


class InverseLayer(torch.nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        return x.div(self.scale).reciprocal()
