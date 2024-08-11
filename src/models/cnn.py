import torch


class Wave2dCnn(torch.nn.Sequential):
    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 9)
        padding = (1, 0)

        conv_layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2), 2),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(4, 8, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2), 2),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(8, 16, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d((2, 2), 2),
            ),
        ]

        linear_layers = [
            torch.nn.Linear(2912, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_shape.numel()),
        ]

        super().__init__(
            torch.nn.Unflatten(-2, (1, input_shape[0])),
            *conv_layers,
            torch.nn.Flatten(-3, -1),
            *linear_layers,
        )
