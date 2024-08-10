import torch.utils.data

from src.data.dataset import WaveDataset
from src.data.split import split_dataset

from src.models.mlp import WaveMlp
from src.train_test import train, test


if __name__ == "__main__":
    ds = WaveDataset(
        "dataset/ISO Wr",
        dims_to_flatten=(-2, -1),
        target_length=1541,
        dtype=torch.float32,
    )
    dataloader = torch.utils.data.DataLoader(ds, batch_size=16)

    train_dataloader, test_dataloader, val_dataloader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=32,
            drop_last=False,
            num_workers=6,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), (0.7, 0.2, 0.1))
    )

    model = WaveMlp(13 * 1541, 2)

    loss_fn = torch.nn.MSELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    test_metrics = test(dataloader=test_dataloader, model=model, loss_fn=loss_fn)
    print("Testing metrics:", test_metrics, end="\n\n")

    train(
        train_dataloader,
        test_dataloader,
        model,
        10,
        loss_fn,
        optimizer,
    )
