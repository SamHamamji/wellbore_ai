import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset


def spectrogram(wave: torch.Tensor):
    fourier = torch.stft(
        wave,
        n_fft=128,
        window=torch.hann_window(128),
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
    ds = WaveDataset("dataset/ISO Wr")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

    x, y = ds[0]

    for x, y in ds:
        if x.shape[1] == 1541:
            continue
        print(x.shape)
        print(spectrogram(x).shape)
        plot(x)
