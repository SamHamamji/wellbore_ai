import argparse
import typing

import numpy as np
import torch.utils.data

from src.checkpoint import update_checkpoint, load_checkpoint
from src.data.split import split_dataset
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


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds, model, optimizer, scheduler = load_checkpoint(args.checkpoint_path)

    if args.lr:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr
        scheduler.get_last_lr()[0] = args.lr

    for arg in ("lr_threshold", "lr_factor", "lr_cooldown", "lr_patience"):
        if arg in args:
            setattr(scheduler, arg.replace("lr_", ""), getattr(args, arg))

    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    metrics: dict[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "rmse": lambda y, pred: (y - pred).square().mean(0).sqrt(),
        "mae": lambda y, pred: (y - pred).abs().mean(0),
        "mare": lambda y, pred: ((y - pred).abs() / y).mean(0),
    }

    val_metrics = test(val_loader, model, metrics)
    print("Validation metrics:", val_metrics, end="\n\n")

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

    update_checkpoint(args.checkpoint_path, model, optimizer, scheduler)
