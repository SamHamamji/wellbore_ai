import torch


class WaveCnn2d(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
    )

    # pylint: disable=W0613
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 15)
        conv_padding = (1, 0)

        conv_layers = [
            torch.nn.Conv2d(1, 4, kernel_size, 1, conv_padding),
            torch.nn.MaxPool2d((3, 3), (3, 3)),
            torch.nn.Conv2d(4, 8, kernel_size, 1, conv_padding),
            torch.nn.MaxPool2d((3, 3), (3, 3)),
        ]

        linear_layers = [
            torch.nn.Linear(1320, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_shape.numel()),
        ]

        super().__init__(
            torch.nn.Unflatten(-2, (1, input_shape[0])),
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )
