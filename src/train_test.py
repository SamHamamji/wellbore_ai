import time
import typing

import torch
import torch.utils.data

from src.metric import Metric

def train_single_epoch(
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
):
    b_num = len(loader)
    b_len = len(str(b_num)) + 1
    total_loss = torch.Tensor()

    model.train()
    start_time = time.time()

    for i, (x_batch, y_batch) in enumerate(loader):
        optimizer.zero_grad()
        pred_batch = model(x_batch)

        loss = loss_fn(pred_batch, y_batch)
        total_loss = torch.concat((total_loss, loss.detach()))

        loss.sum().backward()
        optimizer.step()

        delay = time.time() - start_time

        print(
            f"\033[KBatch loss: {loss.mean(0).sum(0):<.2f} [{i+1:{b_len}d} / {b_num:<{b_len}d}] | {delay:.2f}s",
            end="\r",
        )

    return dict(
        loss=total_loss.mean(0).sum(0).item(),
    )


def train(
    train_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    epochs: int,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
):
    [lr] = scheduler.get_last_lr()

    for _ in range(epochs):
        state_dict = scheduler.state_dict()
        if state_dict["cooldown_counter"] != 0:
            epoch_state = f"{state_dict["cooldown_counter"]}/{state_dict["cooldown"]} cooldown"
        else:
            epoch_state = f"{state_dict["num_bad_epochs"]}/{state_dict["patience"]} bad epochs"

        print(f"Epoch {scheduler.last_epoch}:")
        print(f"Best loss: {state_dict["best"]} | {epoch_state} | {lr:.2e} learning rate")

        initial_time = time.time()
        train_metrics = train_single_epoch(train_loader, model, loss_fn, optimizer)
        training_time = time.time() - initial_time

        scheduler_metric = train_metrics["loss"]
        scheduler.step(scheduler_metric)  # type: ignore\

        print(f"\033[KTraining metrics ({training_time:.1f}s): {train_metrics}")
        print("Validating...", end="\r")

        val_metrics = test(
            val_loader,
            model,
            {"loss": lambda y, pred: loss_fn(pred, y).mean(0).sum(0).item()},
        )
        testing_time = time.time() - initial_time - training_time

        print(f"Validation metrics ({testing_time:.1f}s): {val_metrics}")
        print()

        if lr != scheduler.get_last_lr()[0]:
            lr = scheduler.get_last_lr()[0]
            scheduler.best = scheduler_metric


def test(
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    metrics: dict[str, Metric],
):
    model.eval()
    total_preds = torch.Tensor()
    total_y = torch.Tensor()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_batch: torch.Tensor
            pred_batch: torch.Tensor = model(x_batch)

            total_preds = torch.concat((total_preds, pred_batch))
            total_y = torch.concat((total_y, y_batch))

    return {
        metric_name: metric_fn(total_y, total_preds)
        for metric_name, metric_fn in metrics.items()
    }
