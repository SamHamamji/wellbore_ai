import torch

from src.layers import FftLayer


class FftMlp(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(time_dim=-1, complex_dim=-1),
    )

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        layers = [
            torch.nn.Flatten(-len(input_shape), -1),
            torch.nn.Linear(input_shape.numel(), 64, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(2, output_shape.numel(), dtype=torch.float32),
        ]

        super().__init__(*layers)
