import torch

from src.data.dataset import WaveDataset


def new_checkpoint(
    path: str,
    ds: WaveDataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    ds_kwargs = {
        "data_dir": ds.data_dir,
        "target_length": ds.target_length,
        "dtype": ds.dtype,
        "x_transform": ds.x_transform,
        "y_transform": ds.y_transform,
        "bounds": ds.bounds,
    }

    new_checkpoint = {
        "ds_kwargs": ds_kwargs,
        "model_state_dict": model.state_dict(),
        "model_type": type(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": type(optimizer),
        "epoch": epoch,
    }
    torch.save(new_checkpoint, path)
    print(f"\nSaved checkpoint to {path}\n")


def update_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
):
    checkpoint: dict = torch.load(path, weights_only=False)

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"] = epoch

    torch.save(checkpoint, path)
    print(f"\nUpdated checkpoint in {path}\n")


def load_checkpoint(
    checkpoint_path: str,
) -> tuple[WaveDataset, torch.nn.Module, torch.optim.Optimizer, int]:
    checkpoint: dict = torch.load(checkpoint_path, weights_only=False)

    ds = WaveDataset(**checkpoint["ds_kwargs"])
    x_shape, y_shape = map(lambda t: t.shape, ds[0])

    model_type = checkpoint["model_type"]
    model: torch.nn.Module = model_type(x_shape, y_shape)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_type = checkpoint["optimizer_type"]
    optimizer: torch.optim.Optimizer = optimizer_type(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    initial_epoch: int = checkpoint["epoch"]

    return ds, model, optimizer, initial_epoch
