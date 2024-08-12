import argparse

import numpy as np
import torch.utils.data
import lightning as L

from src.data.dataset import WaveDataset
from src.data.split import split_dataset
from src.models import WaveCnn3d
from src.models.stft import Stft
from src.train_test import train, test


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--splits", type=float, nargs="+")
parser.add_argument("--seed", type=int, default=0)


class LitModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return self.optimizer


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        transform=Stft(64),
    )
    train_dataloader, val_dataloader, test_dataloader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])

    model = WaveCnn3d(x_shape, y_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    if args.input_path is not None:
        checkpoint = torch.load(args.input_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    lit_model = LitModel(model, loss_fn, optimizer)

    trainer = L.Trainer(max_epochs=args.epochs, val_check_interval=0.1)
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    if args.output_path is not None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, args.output_path)
