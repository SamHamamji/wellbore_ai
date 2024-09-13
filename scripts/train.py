import argparse

import numpy as np
import torch.utils.data

from src.checkpoint import update_checkpoint, load_checkpoint
from src.data.split import split_dataset
from src.metric import Metric
from src.train_test import train, test


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)

training_args = parser.add_argument_group("Training")
training_args.add_argument("--epochs", type=int, required=True)
training_args.add_argument("--lr", type=float)
training_args.add_argument("--lr_threshold", type=float)
training_args.add_argument("--lr_factor", type=float)
training_args.add_argument("--lr_cooldown", type=int)
training_args.add_argument("--lr_patience", type=int)

data_args = parser.add_argument_group("Data Processing")
data_args.add_argument("--batch_size", type=int, default=1)
data_args.add_argument("--dataloader_workers", type=int, default=0)
data_args.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))


def update_training_params(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
):
    if args.lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        scheduler.get_last_lr()[0] = args.lr

    for arg in ("lr_threshold", "lr_factor", "lr_cooldown", "lr_patience"):
        arg_value = getattr(args, arg)
        if arg_value is not None:
            setattr(scheduler, arg.replace("lr_", ""), arg_value)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds, model, optimizer, scheduler = load_checkpoint(args.checkpoint_path)
    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, args.splits)
    )

    metrics: dict[str, Metric] = {
        "rmse": lambda y, pred: (y - pred).square().mean(0).sqrt(),
    }
    print(f"Training metrics: {test(train_loader, model, metrics)}")
    print(f"Validation metrics: {test(val_loader, model, metrics)}")

    update_training_params(args, optimizer, scheduler)
    scheduler_state_dict = scheduler.state_dict()
    training_params = {
        "lr": scheduler.get_last_lr()[0],
        "threshold": scheduler_state_dict["threshold"],
        "factor": scheduler_state_dict["factor"],
        "cooldown": scheduler_state_dict["cooldown"],
        "patience": scheduler_state_dict["patience"],
    }
    print(f"Training parameters: {training_params}", end="\n\n")

    try:
        train(
            train_loader,
            val_loader,
            model,
            args.epochs,
            lambda y, pred: (y - pred).square(),
            optimizer,
            scheduler,
        )
    except KeyboardInterrupt:
        if input("\nInterrupted, save model? [y/N] ").lower() not in ["y", "yes"]:
            exit(0)

    update_checkpoint(args.checkpoint_path, ds, model, optimizer, scheduler)
