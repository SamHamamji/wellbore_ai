import argparse

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
training_args.add_argument("--learning_rate", type=float, required=True)

data_args = parser.add_argument_group("Data Processing")
data_args.add_argument("--batch_size", type=int, default=1)
data_args.add_argument("--dataloader_workers", type=int, default=0)
data_args.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds, model, optimizer, initial_epoch = load_checkpoint(args.checkpoint_path)

    for param_group in optimizer.param_groups:
        param_group["lr"] = args.learning_rate

    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    loss_fn = torch.nn.MSELoss(reduction="sum")

    val_metrics = test(dataloader=val_loader, model=model, loss_fn=loss_fn)
    print("Validation metrics:", val_metrics, end="\n\n")

    epoch = train(
        train_loader,
        val_loader,
        model,
        (initial_epoch, initial_epoch + args.epochs),
        loss_fn,
        optimizer,
    )
    interrupted: bool = epoch < initial_epoch + args.epochs

    if not interrupted or input("\nInterrupted, save model? [y/N] ").lower() in [
        "y",
        "yes",
    ]:
        update_checkpoint(args.checkpoint_path, model, optimizer, epoch)
