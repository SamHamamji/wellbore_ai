import torch


class WaveMlp(torch.nn.Sequential):
    dataset_x_transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
    )

    def __init__(self, input_shape: torch.Size, output_shape: torch.Size):
        layers = [
            torch.nn.Flatten(-len(input_shape), -1),
            #
            torch.nn.Linear(input_shape.numel(), 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.2),
            #
            torch.nn.Linear(512, output_shape.numel()),
        ]

        super().__init__(*layers)
