import argparse

import numpy as np
import torch.utils.data

from src.data.dataset import WaveDataset
from src.data.split import split_dataset
from src.data.file_filter_fn import get_filter_fn_by_vs_vp
from src.train_test import test


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.model_path, weights_only=False)
    model_type: type[torch.nn.Module] = checkpoint["model_type"]

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        filter_fn=get_filter_fn_by_vs_vp(max_vp=7000),
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
    train_loader, test_loader, val_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    x, y = ds[0]

    model = model_type(x.shape, y.shape)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_fn = torch.nn.MSELoss(reduction="sum")

    print("Train metrics: ", test(train_loader, model, loss_fn))
    print("Test metrics: ", test(test_loader, model, loss_fn))
