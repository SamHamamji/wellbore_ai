import torch


class WaveMlp(torch.nn.Sequential):
    def __init__(self, input_size: int, output_size: int):
        layers = [
            torch.nn.Linear(input_size, 64, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2, dtype=torch.float32),
            torch.nn.ReLU(),
            torch.nn.Linear(2, output_size, dtype=torch.float32),
        ]

        super().__init__(*layers)
