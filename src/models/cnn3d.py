import torch


class WaveCnn3d(torch.nn.Sequential):
    class MoveDim(torch.nn.Module):
        def __init__(self, source: int, destination: int):
            super().__init__()
            self.source_dim = source
            self.destination_dim = destination

        def forward(self, x: torch.Tensor):
            return x.movedim(self.source_dim, self.destination_dim)

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        kernel_size = (3, 3, 9)
        padding = (1, 1, 0)

        conv_layers = [
            torch.nn.Sequential(
                torch.nn.Conv3d(2, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.Conv3d(4, 4, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d((3, 3, 3), 3),
            ),
            torch.nn.Sequential(
                torch.nn.Conv3d(4, 8, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.MaxPool3d((3, 3, 3), 3),
            ),
        ]

        linear_layers = [
            torch.nn.Linear(144, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, output_shape.numel()),
        ]

        super().__init__(
            self.MoveDim(-1, -4),
            *conv_layers,
            torch.nn.Flatten(-4, -1),
            *linear_layers,
        )
