import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.data.split import split_dataset


def spectrogram(wave: torch.Tensor):
    fourier = torch.stft(
        wave,
        n_fft=64,
        window=torch.hann_window(64),
        return_complex=True,
        normalized=False,
    )
    return torch.view_as_real(fourier)


def plot(wave: torch.Tensor):
    spect = spectrogram(wave)

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
    ds = WaveDataset("dataset/ISO Wr", target_length=1541)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=16)

    train_dataloader, test_dataloader, val_dataloader = (
        torch.utils.data.DataLoader(
            ds_split,
            batch_size=16,
        )
        for ds_split in split_dataset(ds, torch.ones(len(ds)), (0.7, 0.2, 0.1))
    )
