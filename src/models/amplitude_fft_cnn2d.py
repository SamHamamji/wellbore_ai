import torch

from src.layers import FftLayer, SelectIndexLayer


class AmplitudeFftCnn2d(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -3, polar_decomposition=True),
        SelectIndexLayer((slice(0, 1), slice(None), slice(None))),
    )

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 9)
        conv_padding = (1, 0)

        conv_layers = [
            torch.nn.Conv2d(1, 4, kernel_size, 1, conv_padding),
            torch.nn.MaxPool2d((3, 3), (3, 3)),
            torch.nn.Conv2d(4, 8, kernel_size, 1, conv_padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 3), (3, 3)),
        ]

        linear_layers = [
            torch.nn.Linear(656, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape.numel()),
        ]

        super().__init__(
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )
