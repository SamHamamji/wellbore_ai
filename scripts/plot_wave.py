import argparse

import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.layers import FftLayer, SelectIndexLayer


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--polar", action="store_true")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)


def plot_fft(wave: torch.Tensor, transform: torch.nn.Module):
    with torch.no_grad():
        spect = transform(wave)

    sample_rate = 6.63130e-06
    # pylint: disable=not-callable
    frequencies = torch.fft.rfftfreq(wave.shape[-1], d=sample_rate)
    time = torch.arange(wave.shape[-1]) * sample_rate

    app = dash.Dash()

    app.layout = []
    for receiver in [0, 4, 8, 12]:
        signal = px.line(
            x=time, y=wave[receiver], labels={"x": "Time", "y": "Amplitude"}
        )
        channel_1 = px.line(
            x=frequencies, y=spect[receiver, :, 0], labels={"x": "Frequency (Hz)"}
        )
        channel_2 = px.line(
            x=frequencies, y=spect[receiver, :, 1], labels={"x": "Frequency (Hz)"}
        )
        div = dash.html.Div(
            [
                dash.html.H1(f"Receiver {receiver}"),
                dash.html.H3("Raw signal"),
                dash.dcc.Graph(figure=signal),
                dash.html.H3("Channel 1 (Cosine / Amplitude)"),
                dash.dcc.Graph(figure=channel_1),
                dash.html.H3("Channel 2 (Sine / Phase)"),
                dash.dcc.Graph(figure=channel_2),
            ]
        )
        app.layout.append(div)

    app.run(debug=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        dtype=torch.float32,
        target_length=1541,
    )
    transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -1, args.polar),
        SelectIndexLayer((slice(None), slice(None), slice(None))),
    )
    x, y = ds[args.sample_index]

    plot_fft(x, transform)
