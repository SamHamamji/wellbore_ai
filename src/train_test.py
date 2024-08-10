import time
import torch
import torch.utils.data
import torch.nn as nn


def train_single_epoch(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    batch_num = len(dataloader)
    batch_len = len(str(batch_num)) + 1
    model.train()
    start_time = time.time()

    for i, (x_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        pred_batch = model(x_batch)
        loss: torch.Tensor = loss_fn(pred_batch, y_batch)
        loss.backward()

        optimizer.step()

        delay = time.time() - start_time

        print(
            f"\rloss: {loss:<7f} [{i+1:{batch_len}d} / {batch_num:<{batch_len}d}] {delay:4f}s",
            end="",
        )


def train(
    train_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    test_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    epochs: int,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    def clear_lines(n: int):
        if n <= 0:
            return
        print("\r\x1b[2K" + "\033[1A\x1b[2K" * (n - 1), end="")

    def print_epoch_paragraph(lines_to_clear: int, subheader: str, *contents: str):
        clear_lines(lines_to_clear)
        print(f"Epoch {epoch}: {subheader}")
        for content in contents:
            print(content)

    for epoch in range(epochs):
        initial_time = time.time()
        print_epoch_paragraph(0, "Training...")
        train_single_epoch(train_dataloader, model, loss_fn, optimizer)

        training_time = time.time() - initial_time
        print_epoch_paragraph(2, f"{training_time:.2f}s", "Testing...")

        train_metrics = test(train_dataloader, model, loss_fn)
        test_metrics = test(test_dataloader, model, loss_fn)

        testing_time = time.time() - initial_time - training_time
        print_epoch_paragraph(
            3,
            f"{training_time:.2f}s | {testing_time:.2f}s",
            f"Training metrics: {train_metrics}",
            f"Testing metrics: {test_metrics}",
        )
        print()


def test(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: nn.Module,
    loss_fn: nn.Module,
):
    model.eval()
    sample_num: int = 0
    loss: float = 0.0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            pred_batch: torch.Tensor = model(x_batch)
            loss += loss_fn(pred_batch, y_batch).item()

            sample_num += y_batch.shape[0]

    return dict(
        avg_loss=loss / sample_num,
    )
