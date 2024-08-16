import argparse

import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.layers import FftLayer


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)


def plot_fft(wave: torch.Tensor, transform: torch.nn.Module):
    spect = transform(wave)

    # pylint: disable=not-callable
    frequencies = torch.fft.rfftfreq(wave.shape[-1], d=1.0)

    app = dash.Dash()
    app.layout = [
        dash.html.Div(
            [
                dash.html.H1(f"Receiver {receiver}"),
                dash.html.H3("Raw signal"),
                dash.dcc.Graph(figure=px.line(y=wave[receiver])),
                dash.html.H3("Amplitude"),
                dash.dcc.Graph(figure=px.line(x=frequencies, y=spect[receiver, :, 0])),
                dash.html.H3("Phase"),
                dash.dcc.Graph(figure=px.line(x=frequencies, y=spect[receiver, :, 1])),
            ]
        )
        for receiver in [0, 4, 8, 12]
    ]
    app.run(debug=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        dtype=torch.float32,
        target_length=1541,
    )
    transform = FftLayer(time_dim=-1, complex_dim=-1, polar_decomposition=True)

    x, y = ds[args.sample_index]

    print(y)

    plot_fft(x, transform)
