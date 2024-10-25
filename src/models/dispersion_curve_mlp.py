import torch


class DispersionCurveMlp(torch.nn.Sequential):
    # pylint: disable=W0613
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        linear_layers = [
            torch.nn.Linear(input_shape.numel(), 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(128),
            torch.nn.Linear(128, output_shape.numel()),
        ]

        super().__init__(
            torch.nn.Flatten(-len(input_shape), -1),
            *linear_layers,
        )
