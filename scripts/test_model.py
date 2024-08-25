import argparse

import numpy as np
import torch.utils.data

from src.data.split import split_dataset
from src.train_test import test
from src.checkpoint import load_checkpoint


parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--test", action="store_true")
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds, model, _, epoch = load_checkpoint(args.model_path)

    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    loss_fn = torch.nn.MSELoss(reduction="sum")

    print(f"Epoch {epoch}")
    print(f"Dataset x transform: {ds.x_transform}")
    print(f"Dataset x bounds: {ds.bounds}")

    print("Train metrics: ", test(train_loader, model, loss_fn))
    print("Validation metrics: ", test(val_loader, model, loss_fn))
    if args.test:
        print("Test metrics: ", test(test_loader, model, loss_fn))
