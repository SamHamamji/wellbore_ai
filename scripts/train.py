import argparse

import ignite
import ignite.base
import ignite.contrib
import ignite.contrib.handlers
import ignite.engine
import ignite.handlers
import ignite.handlers.tqdm_logger
import ignite.metrics
import ignite.metrics.regression.mean_absolute_relative_error
import ignite.utils
import numpy as np
import torch.utils.data

from src.data.dataset import WaveDataset
from src.data.split import split_dataset
from src.models import WaveCnn3d, Stft


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--dataloader_workers", type=int, default=0)
parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        transform=Stft(64),
    )
    train_loader, val_loader, test_loader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=args.batch_size,
            num_workers=args.dataloader_workers,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), args.splits)
    )

    x_shape, y_shape = map(lambda t: t.shape, ds[0])

    model = WaveCnn3d(x_shape, y_shape)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    trainer = ignite.engine.create_supervised_trainer(
        model, optimizer, loss_fn, deterministic=True
    )

    val_metrics: dict[str, ignite.metrics.Metric] = {
        # "mare": ignite.metrics.regression.mean_absolute_relative_error.MeanAbsoluteRelativeError(),
        "mse": ignite.metrics.Loss(loss_fn),
    }

    train_evaluator = ignite.engine.create_supervised_evaluator(model, val_metrics)
    val_evaluator = ignite.engine.create_supervised_evaluator(model, val_metrics)

    def epoch_completed(trainer: ignite.engine.Engine):
        train_metrics = train_evaluator.run(train_loader).metrics
        val_metrics = val_evaluator.run(val_loader).metrics

        print(f"Training metrics: {train_metrics}")
        print(f"Validation metrics: {val_metrics}")
        print()

    model_checkpoint = ignite.handlers.ModelCheckpoint(
        "models",
        n_saved=2,
        filename_prefix="best",
        score_function=(lambda engine: -engine.state.metrics["mse"]),
        score_name="mse",
        global_step_transform=ignite.contrib.handlers.global_step_from_engine(trainer),
        require_empty=False,
    )

    trainer.add_event_handler(ignite.engine.Events.EPOCH_COMPLETED, epoch_completed)

    train_pbar = ignite.handlers.tqdm_logger.ProgressBar()
    train_pbar.attach(trainer, output_transform=lambda x: {"loss": x})

    # val_pbar = ignite.handlers.tqdm_logger.ProgressBar()
    # val_pbar.attach(val_evaluator, output_transform=lambda x: {"loss": x})

    val_evaluator.add_event_handler(
        ignite.engine.Events.COMPLETED,
        model_checkpoint,
        {"model": model, "optimizer": optimizer},
    )

    trainer.run(train_loader, max_epochs=args.epochs)
