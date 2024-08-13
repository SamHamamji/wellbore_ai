import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.layers import FftLayer


def plot(wave: torch.Tensor):
    fft = FftLayer()
    spect = fft(wave)

    # pylint: disable=not-callable
    frequencies = torch.fft.rfftfreq(wave.shape[-1], d=1.0)

    app = dash.Dash()
    app.layout = [
        dash.html.Div(
            [
                dash.html.H1(f"Receiver {index}"),
                dash.dcc.Graph(figure=px.line(x=frequencies, y=spect[index, :, 0])),
                dash.dcc.Graph(figure=px.line(x=frequencies, y=spect[index, :, 1])),
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
