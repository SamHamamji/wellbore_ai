import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.data.split import split_dataset

from src.models.mlp import WaveMlp
from src.models.stft import Stft
from src.train_test import train, test


def plot(wave: torch.Tensor):
    stft_module = Stft()
    spect = stft_module(wave)

    app = dash.Dash()
    app.layout = [
        dash.html.Div(
            [
                dash.html.H1(f"Receiver {index}"),
                dash.dcc.Graph(figure=px.imshow(spect[index, :, :, 0])),
                dash.dcc.Graph(figure=px.imshow(spect[index, :, :, 1])),
                dash.dcc.Graph(figure=px.line(y=wave[index])),
            ]
        )
        for index in [1, 5, 9, 13]
    ]
    app.run(debug=True)


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
    print("Testing metrics:", test_metrics)

    train(
        train_dataloader,
        test_dataloader,
        model,
        10,
        loss_fn,
        optimizer,
    )
