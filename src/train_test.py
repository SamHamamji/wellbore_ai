import time
import typing

import torch
import torch.utils.data


def train_single_epoch(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
):
    b_num = len(dataloader)
    b_len = len(str(b_num)) + 1
    total_loss = torch.Tensor()

    model.train()
    start_time = time.time()

    for i, (x_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        pred_batch = model(x_batch)

        loss = loss_fn(pred_batch, y_batch)
        total_loss = torch.concat((total_loss, loss.detach()))

        loss.sum().backward()
        optimizer.step()

        delay = time.time() - start_time

        print(
            f"Batch loss: {loss.sum():<7f} [{i+1:{b_len}d} / {b_num:<{b_len}d}] | {delay:4f}s",
            end="\r",
        )

    return dict(
        loss=total_loss.mean(0),
    )


def train(
    train_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    epochs: tuple[int, int],
    loss_fn: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
):
    try:
        for epoch in range(epochs[0], epochs[1]):
            initial_time = time.time()
            print(f"Epoch {epoch}:")
            train_metrics = train_single_epoch(
                train_dataloader, model, loss_fn, optimizer
            )
            training_time = time.time() - initial_time
            print(f"Training metrics: {train_metrics} | {training_time:.2f}s")
            print("Testing...", end="\r")

            val_metrics = test(
                val_dataloader,
                model,
                {"loss": lambda y, pred: loss_fn(pred, y).mean(0)},
            )

            testing_time = time.time() - initial_time - training_time
            print(f"Validation metrics: {val_metrics} | {testing_time:.2f}s")
            print()
    except KeyboardInterrupt:
        return epoch

    return epoch + 1


def test(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    metrics: dict[str, typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
):
    model.eval()
    total_preds = torch.Tensor()
    total_y = torch.Tensor()

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_batch: torch.Tensor
            pred_batch: torch.Tensor = model(x_batch)

            total_preds = torch.concat((total_preds, pred_batch))
            total_y = torch.concat((total_y, y_batch))

    return {
        metric_name: metric_fn(total_y, total_preds)
        for metric_name, metric_fn in metrics.items()
    }
