import argparse

import numpy as np
import torch.utils.data

from src.data.dataset import WaveDataset
from src.data.split import split_dataset
from src.models import models
from src.train_test import train, test


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

path_args = parser.add_argument_group("Paths")
path_args.add_argument("--data_dir", type=str, required=True)
path_args.add_argument("--input_path", type=str, default=None)
path_args.add_argument("--output_path", type=str, default=None)

training_args = parser.add_argument_group("Training")
training_args.add_argument("--model_type", type=str, choices=models.keys())
training_args.add_argument("--epochs", type=int, required=True)
training_args.add_argument("--learning_rate", type=float, default=0.001)

data_args = parser.add_argument_group("Data Processing")
data_args.add_argument("--batch_size", type=int, default=1)
data_args.add_argument("--dataloader_workers", type=int, default=0)
data_args.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (args.input_path is None) == (args.model_type is None):
        parser.error("Either --input_path or --model_type must be specified")
    elif args.input_path is not None:
        checkpoint = torch.load(args.input_path, weights_only=False)
        model_type: type[torch.nn.Module] = checkpoint["model_type"]
    else:
        checkpoint = None
        model_type = models[args.model_type]

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        transform=(
            model_type.dataset_transform  # type: ignore
            if hasattr(model_type, "dataset_transform")
            else None
        ),
    )
    train_loader, test_loader, val_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])
    print(f"Sample shapes: {x_shape=} {y_shape=}", end="\n\n")

    model = model_type(x_shape, y_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_metrics = test(dataloader=test_loader, model=model, loss_fn=loss_fn)
    print("Testing metrics:", test_metrics, end="\n\n")

    train(train_loader, test_loader, model, args.epochs, loss_fn, optimizer)

    if args.output_path is not None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": model_type,
        }
        torch.save(checkpoint, args.output_path)
