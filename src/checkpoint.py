import json

import torch

from src.data.dataset import WaveDataset
from src.pad_state_dicts import pad_model_state_dict, pad_optimizer_state_dict
from src.history import History


class Checkpoint:
    def __init__(
        self,
        ds: WaveDataset,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        history: History,
    ):
        self.ds = ds
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = history

    def save(self, path: str):
        file_content = {
            "ds_kwargs": self.ds.get_kwargs(),
            "history_state_dict": self.history.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_type": type(self.model),
            "optimizer_type": type(self.optimizer),
            "scheduler_type": type(self.scheduler),
        }

        torch.save(file_content, path)
        print()
        print(f"Saved checkpoint to {path}")

    @staticmethod
    def load_from_path(path: str) -> "Checkpoint":
        file_content: dict = torch.load(path, weights_only=False)

        model_type = file_content["model_type"]
        optimizer_type = file_content["optimizer_type"]
        scheduler_type = file_content["scheduler_type"]

        ds_kwargs = file_content["ds_kwargs"]
        model_state_dict = file_content["model_state_dict"]
        optimizer_state_dict = file_content["optimizer_state_dict"]
        scheduler_state_dict = file_content["scheduler_state_dict"]
        history_state_dict = file_content.get("history_state_dict", {})

        ds = WaveDataset(**ds_kwargs)
        model: torch.nn.Module = model_type(*map(lambda t: t.shape, ds[0]))
        optimizer: torch.optim.Optimizer = optimizer_type(model.parameters())
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = scheduler_type(
            optimizer
        )
        history = History()

        pad_model_state_dict(model_state_dict, model, 0.001)
        if optimizer_state_dict["state"]:
            pad_optimizer_state_dict(optimizer_state_dict, model, 0, 0)

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)
        history.load_state_dict(history_state_dict)

        if not isinstance(model, torch.nn.Module):
            raise ValueError("model_type must be a subclass of Module")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError("optimizer_type must be a subclass of Optimizer")
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            raise ValueError("scheduler_type must be a subclass of ReduceLROnPlateau")

        return Checkpoint(ds, model, optimizer, scheduler, history)

    def print(self):
        print(f"Dataset kwargs: {self.ds.get_kwargs()}")
        print()

        X, y = next(iter(self.ds))
        print(f"Sample shapes: {X.shape} {y.shape}")
        print(f"Model: {self.model}")
        for _, layer in self.model.named_children():
            X = layer(X)
            print(f"{layer.__class__.__name__} output shape: {X.shape}")
        print()

        print("Parameters:")
        for name, param in self.model.named_parameters():
            print(f"  {name}, {param.shape}, {param.numel()}")
        print()

        print(
            f"Scheduler state dict: {json.dumps(self.scheduler.state_dict(), indent=2)}"
        )
        print()
