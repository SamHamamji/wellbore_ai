import torch

from src.layers import FftLayer, SelectIndexLayer


class WaveCnn2d(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -3, polar_decomposition=True),
        SelectIndexLayer((slice(0, 1), slice(None), slice(None))),
    )

    # pylint: disable=W0613
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 9)
        conv_padding = (1, 0)

        conv_layers = [
            torch.nn.Conv2d(1, 2, kernel_size, 1, conv_padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
            torch.nn.Conv2d(2, 4, kernel_size, 1, conv_padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2), (2, 2)),
        ]

        linear_layers = [
            torch.nn.Linear(2232, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, output_shape.numel()),
        ]

        super().__init__(
            # torch.nn.Unflatten(-2, (1, input_shape[0])),
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )