import typing

import torch


class AbsoluteRelativeError(torch.nn.Module):
    def __init__(self, reduction: typing.Literal["mean", "sum"] | None = None) -> None:
        self.reduction = reduction
        super().__init__()

    def forward(self, pred: torch.Tensor, y_true: torch.Tensor):
        mare = torch.abs(pred - y_true) / y_true

        if self.reduction == "mean":
            return torch.mean(mare)
        elif self.reduction == "sum":
            return torch.sum(mare)

        return mare
