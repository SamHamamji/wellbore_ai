import typing

import torch
import torch.types

Metric = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


absolute_error: Metric = lambda y, pred: (pred - y)
squared_error: Metric = lambda y, pred: (pred - y).square_()
relative_error: Metric = lambda y, pred: (pred - y).div_(y)
binary_cross_entropy: Metric = (
    lambda y, pred: pred.log().mul_(y).add_((1 - pred).log_().mul_(1 - y)).negative_()
)

error_metrics: dict[str, Metric] = {
    "absolute_error": absolute_error,
    "squared_error": squared_error,
    "relative_error": relative_error,
    "binary_cross_entropy": binary_cross_entropy,
}
