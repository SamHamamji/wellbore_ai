import argparse

import numpy as np
import torch.utils.data

from src.data.dataset import WaveDataset
from src.data.split import split_dataset
from src.models.mlp import WaveMlp
from src.models.cnn import Wave2dCnn
from src.train_test import train, test


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
    )
    train_dataloader, test_dataloader, val_dataloader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), (0.7, 0.2, 0.1))
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])

    model = Wave2dCnn(x_shape, y_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    if args.input_path is not None:
        checkpoint = torch.load(args.input_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    test_metrics = test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
    print("Testing metrics:", test_metrics, end="\n\n")

    train(train_dataloader, test_dataloader, model, args.epochs, loss_fn, optimizer)

    if args.output_path is not None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, args.output_path)
