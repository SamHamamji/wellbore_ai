import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.models.stft import Stft


def plot(wave: torch.Tensor):
    stft_module = Stft(64)
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
        for index in [0, 4, 8, 12]
    ]
    app.run(debug=True)


if __name__ == "__main__":
    ds = WaveDataset("dataset/ISO Wr")

    x, y = ds[0]

    plot(x)
