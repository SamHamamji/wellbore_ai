import argparse

import numpy as np
import torch.utils.data

from src.checkpoint import new_checkpoint
from src.data.dataset import WaveDataset
from src.models import models


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--model_type", type=str, choices=models.keys())
parser.add_argument("--max_vs", type=int, required=False)
parser.add_argument("--max_vp", type=int, required=False)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_type = models[args.model_type]

    ds = WaveDataset(
        data_dir=args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        bounds=(
            range(args.max_vs) if args.max_vs else None,
            range(args.max_vp) if args.max_vp else None,
        ),
        x_transform=(
            model_type.dataset_x_transform  # type: ignore
            if hasattr(model_type, "dataset_x_transform")
            else None
        ),
        y_transform=(
            model_type.dataset_y_transform  # type: ignore
            if hasattr(model_type, "dataset_y_transform")
            else None
        ),
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])
    print(f"Sample shapes: {x_shape=} {y_shape=}")

    model = model_type(x_shape, y_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=1 - 1e-10,
        threshold=0,
    )

    new_checkpoint(args.checkpoint_path, ds, model, optimizer, scheduler)
