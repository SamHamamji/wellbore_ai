import argparse
import os

import numpy as np
import torch.utils.data

from src.checkpoint import Checkpoint
from src.data.dataset import WellboreDataset
from src.history import History
from src.models import models


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--path", type=str, default=None)
parser.add_argument("--model_type", type=str, choices=models.keys())
parser.add_argument(
    "--signal_type",
    type=str,
    choices=WellboreDataset.signal_types.__args__,
    required=True,
)
parser.add_argument(
    "--label_type",
    type=str,
    choices=WellboreDataset.label_types.__args__,
    required=True,
)
parser.add_argument("--target_signal_length", type=int, required=False)
parser.add_argument("--max_vs", type=int, required=False)
parser.add_argument("--max_vp", type=int, required=False)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_type = models[args.model_type]

    ds = WellboreDataset(
        data_dir=args.data_dir,
        target_signal_length=args.target_signal_length,
        dtype=torch.float32,
        signal_type=args.signal_type,
        label_type=args.label_type,
        label_bounds=(
            range(args.max_vs) if args.max_vs else None,
            range(args.max_vp) if args.max_vp else None,
        ),
        x_transform=(
            model_type.dataset_x_transform  # type: ignore
            if hasattr(model_type, "dataset_x_transform")
            else None
        ),
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])
    model = model_type(x_shape, y_shape)
    output_shape: torch.Size = model(torch.ones(x_shape)).shape

    print(f"Sample shapes: {x_shape=} {y_shape=} {output_shape=}")
    if os.path.exists(args.path) and input(
        f"File {args.path} already exists, replace? [y/N] "
    ).lower() not in ["y", "yes"]:
        exit(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=1 - 1e-10,
        threshold=0,
    )
    history = History()

    checkpoint = Checkpoint(ds, model, optimizer, scheduler, history)
    checkpoint.save(args.path)
