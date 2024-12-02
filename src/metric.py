import typing

import torch

Metric = typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


absolute_error: Metric = lambda y, pred: (pred - y)
squared_error: Metric = lambda y, pred: (pred - y).square_()
relative_error: Metric = lambda y, pred: (pred - y).div_(y)
binary_cross_entropy: Metric = (
    lambda y, pred: (pred.clamp(min=1e-10).log_().mul_(y))
    .add_((1 - pred).clamp(min=1e-10).log_().mul_(1 - y))
    .negative_()
)
accuracy: Metric = lambda y, pred: (pred.round() == y).float()

error_metrics: dict[str, Metric] = {
    "absolute_error": absolute_error,
    "squared_error": squared_error,
    "relative_error": relative_error,
    "binary_cross_entropy": binary_cross_entropy,
}
