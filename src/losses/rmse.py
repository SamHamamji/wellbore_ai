import torch


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, x, y):
        return torch.sqrt(self.mse_loss(x, y))
