import torch

from src.layers.fft import FftLayer


class AmplitudeFftCnn2d(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -3, polar_decomposition=True),
    )

    # pylint: disable=W0613
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 9)
        conv_padding = (1, 0)

        conv_layers = [
            torch.nn.Conv2d(2, 4, kernel_size, 1, conv_padding),
            torch.nn.BatchNorm2d(4),
            torch.nn.MaxPool2d((3, 3), (3, 3)),
            torch.nn.Conv2d(4, 8, kernel_size, 1, conv_padding),
            torch.nn.BatchNorm2d(8),
            torch.nn.AvgPool2d((2, 2), (2, 2)),
        ]

        linear_layers = [
            torch.nn.Linear(1968, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, output_shape.numel()),
        ]

        super().__init__(
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )