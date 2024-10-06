import argparse

import torch.utils.data
import dash
import plotly.graph_objects as go

from src.data.dataset import WaveDataset
from src.layers import FftLayer, SelectIndexLayer


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--polar", action="store_true")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--noise_type",
    type=str,
    default=WaveDataset.noise_types.__args__[0],
    choices=WaveDataset.noise_types.__args__,
)
parser.add_argument("--noise_std", type=float, default=None)


def get_offset(data: torch.Tensor, index: int):
    if index == 0:
        return torch.zeros(1, *data.shape[2:])
    return data[:index].max(dim=1, keepdim=True).values.sum(0)


def plot_fft(wave: torch.Tensor, transform: torch.nn.Module):
    with torch.no_grad():
        spect = transform(wave)

    sampling_interval = 6.63130e-06
    # pylint: disable=not-callable
    frequency = torch.fft.rfftfreq(wave.shape[-1], d=sampling_interval)
    time = torch.arange(wave.shape[-1]) * sampling_interval

    app = dash.Dash()

    signal_traces = []
    channel_1_traces = []
    channel_2_traces = []
    for i in range(wave.shape[0]):
        name = f"Receiver {i}"
        signal_offset = get_offset(wave, i)
        spect_offset = get_offset(spect, i)
        signal_traces.append(go.Scatter(x=time, y=wave[i] + signal_offset, name=name))
        channel_1_traces.append(
            go.Scatter(x=frequency, y=spect[i, :, 0] + spect_offset[..., 0], name=name)
        )
        channel_2_traces.append(
            go.Scatter(x=frequency, y=spect[i, :, 1] + spect_offset[..., 1], name=name)
        )

    signals_plot = go.Figure(
        signal_traces,
        layout={
            "title": "Raw signals",
            "xaxis_title": "Time",
            "yaxis_title": "Amplitude",
        },
    )
    channel_1_plot = go.Figure(
        channel_1_traces,
        layout={
            "title": "Channel 1 (Cosine / Amplitude)",
            "xaxis_title": "Frequency (Hz)",
            "yaxis_title": "Amplitude",
        },
    )
    channel_2_plot = go.Figure(
        channel_2_traces,
        layout={
            "title": "Channel 2 (Sine / Phase)",
            "xaxis_title": "Frequency (Hz)",
            "yaxis_title": "Amplitude",
        },
    )

    app.layout = [
        dash.html.H1("Raw signals"),
        dash.dcc.Graph(figure=signals_plot),
        dash.dcc.Graph(figure=channel_1_plot),
        dash.dcc.Graph(figure=channel_2_plot),
    ]

    app.run(debug=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        dtype=torch.float32,
        target_length=1541,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
    )
    transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -1, args.polar),
        SelectIndexLayer((slice(None), slice(None), slice(None))),
    )
    x, y = ds[args.sample_index]

    plot_fft(x, transform)
