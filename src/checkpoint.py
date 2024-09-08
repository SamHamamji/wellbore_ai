import torch

from src.data.dataset import WaveDataset
from src.pad_state_dicts import pad_model_state_dict, pad_optimizer_state_dict


def new_checkpoint(
    path: str,
    ds: WaveDataset,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    ds_kwargs = {
        "data_dir": ds.data_dir,
        "target_length": ds.target_length,
        "dtype": ds.dtype,
        "x_transform": ds.x_transform,
        "bounds": ds.bounds,
    }

    new_checkpoint = {
        "ds_kwargs": ds_kwargs,
        "model_state_dict": model.state_dict(),
        "model_type": type(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_type": type(optimizer),
        "scheduler_state_dict": scheduler.state_dict(),
        "scheduler_type": type(scheduler),
    }
    torch.save(new_checkpoint, path)
    print(f"\nSaved checkpoint to {path}")


def update_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    checkpoint: dict = torch.load(path, weights_only=False)

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)
    print(f"\nUpdated checkpoint in {path}")


def load_checkpoint(checkpoint_path: str):
    checkpoint: dict = torch.load(checkpoint_path, weights_only=False)

    ds = WaveDataset(**checkpoint["ds_kwargs"])
    x_shape, y_shape = map(lambda t: t.shape, ds[0])

    model_type: type[torch.nn.Module] = checkpoint["model_type"]
    model = model_type(x_shape, y_shape)
    model_state_dict = checkpoint["model_state_dict"]

    optimizer_type = checkpoint["optimizer_type"]
    optimizer: torch.optim.Optimizer = optimizer_type(model.parameters())
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    scheduler_type = checkpoint["scheduler_type"]
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = scheduler_type(optimizer)

    pad_model_state_dict(model_state_dict, model, 0.001)
    if optimizer_state_dict["state"]:
        pad_optimizer_state_dict(optimizer_state_dict, model, 0, 0)

    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model_type must be a subclass of Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer_type must be a subclass of Optimizer")
    if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        raise ValueError("scheduler_type must be a subclass of ReduceLROnPlateau")

    return ds, model, optimizer, scheduler
