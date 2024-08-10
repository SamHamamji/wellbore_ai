import torch


class Wave2dCnn(torch.nn.Sequential):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_sizes = [(3, 9), (3, 9), (3, 9)]
        padding = [(2, 0), (2, 0), (2, 0)]
        strides = [1, 1, 1, 1]
        channels = [1, 8, 16, 32, 64]

        conv_layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    (kernel_sizes[i]),
                    strides[i],
                    padding[i],
                ),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((kernel_sizes[i]), 2),
            )
            for i in range(len(kernel_sizes))
        ]

        linear_layers = [
            torch.nn.Linear(11456, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_shape.numel()),
        ]

        super().__init__(
            torch.nn.Unflatten(-2, (1, input_shape[0])),
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )
