"""Tests a model from a checkpoint. Each split is tested separately."""

import argparse

import numpy as np
import torch.utils.data

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.data.dataset import WellboreDataset
from src.train_test import test
import src.metric as metric

ds_splits = ("train", "validation", "test")

parser = argparse.ArgumentParser()

parser.add_argument("--paths", type=str, required=True, nargs="+")
parser.add_argument("--sum_aggregate", action="store_true")
for ds_split in ds_splits:
    parser.add_argument(f"--{ds_split}", action="store_true")

# Dataloader
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))

# Dataset
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument(
    "--noise_type",
    type=str,
    default=WellboreDataset.noise_types.__args__[0],
    choices=WellboreDataset.noise_types.__args__,
)
parser.add_argument("--noise_std", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    metrics: dict[str, metric.Metric] = {
        "rmse": lambda y, pred: metric.squared_error(y, pred).mean(0).sqrt_(),
        "mae": lambda y, pred: metric.absolute_error(y, pred).abs_().mean(0),
        "error std": lambda y, pred: metric.absolute_error(y, pred).std(0),
        "mare": lambda y, pred: metric.relative_error(y, pred).abs_().mean(0),
        "relative error std": lambda y, pred: metric.relative_error(y, pred).std(0),
    }

    if not any(getattr(args, split_name) for split_name in ds_splits):
        raise ValueError("No dataset split specified")

    for path in args.paths:
        extra_ds_kwargs = {
            "noise_type": args.noise_type,
            "noise_std": args.noise_std,
        }
        if args.data_dir is not None:
            extra_ds_kwargs["data_dir"] = args.data_dir

        checkpoint = Checkpoint.load_from_path(path, extra_ds_kwargs)
        param_num = sum(p.numel() for p in checkpoint.model.parameters())

        print(
            f"Model {path}, {param_num:.2e} parameters, epoch {checkpoint.scheduler.last_epoch}"
        )
        for split_name, ds_split in zip(
            ds_splits, split_dataset(checkpoint.ds, args.splits)
        ):
            if not getattr(args, split_name):
                continue

            loader = torch.utils.data.DataLoader(
                ds_split,
                batch_size=args.batch_size,
                num_workers=args.dataloader_workers,
            )

            results = test(loader, checkpoint.model, metrics)
            print(f"\t{split_name.capitalize()} metrics ({len(ds_split)} samples):")
            for metric_name, result in results.items():
                if isinstance(result, torch.Tensor) and args.sum_aggregate:
                    result = result.sum()
                print(f"\t\t{metric_name}: {result}")
            print()
