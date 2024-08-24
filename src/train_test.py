import time
import torch
import torch.utils.data


def train_single_epoch(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    batch_num = len(dataloader)
    batch_len = len(str(batch_num)) + 1
    total_loss = torch.Tensor([0.0])
    sample_num: int = 0

    model.train()
    start_time = time.time()

    for i, (x_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        pred_batch = model(x_batch)
        loss: torch.Tensor = loss_fn(pred_batch, y_batch)
        loss.backward()

        optimizer.step()

        delay = time.time() - start_time
        sample_num += x_batch.shape[0]
        total_loss += loss

        print(
            f"Batch loss: {loss:<7f} [{i+1:{batch_len}d} / {batch_num:<{batch_len}d}] {delay:4f}s",
            end="\r",
        )

    return dict(
        rmse_loss=torch.sqrt(total_loss / sample_num).item(),
    )


def train(
    train_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    epochs: tuple[int, int],
    loss_fn: torch.nn.Module,
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

            val_metrics = test(val_dataloader, model, loss_fn)

            testing_time = time.time() - initial_time - training_time
            print(f"Validation metrics: {val_metrics} | {testing_time:.2f}s")
            print()
    except KeyboardInterrupt:
        return epoch

    return epoch + 1


def test(
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
):
    model.eval()
    total_loss = torch.Tensor([0.0])
    sample_num: int = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            pred_batch: torch.Tensor = model(x_batch)

            total_loss += loss_fn(pred_batch, y_batch)
            sample_num += y_batch.shape[0]

    return dict(
        rmse_loss=torch.sqrt(total_loss / sample_num).item(),
    )
