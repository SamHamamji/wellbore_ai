import typing

import torch
import torch.types

Metric = typing.Callable[
    [torch.Tensor, torch.Tensor], torch.Tensor | torch.types.Number
]
