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


def get_scaled_data(data: torch.Tensor, start: float, end: float):
    offset_per_receiver = data.max(dim=1).values.mean(0)
    offset = torch.arange(0, data.shape[0]) * offset_per_receiver
    data = data + offset.expand(data.T.shape).T

    return data * (end - start) / offset_per_receiver / (data.shape[0] - 1) + start


def get_offset(data: torch.Tensor, index: int):
    if index == 0:
        return torch.zeros(1, *data.shape[2:])
    return data.max(dim=1).values.mean(0) * index


def plot_fft(wave: torch.Tensor, transform: torch.nn.Module):
    with torch.no_grad():
        spect = transform(wave)

    sampling_interval = 6.63130e-06
    start_receiver = 3.275
    end_receiver = 3.275 + 1.828

    # pylint: disable=not-callable
    frequency = torch.fft.rfftfreq(wave.shape[-1], d=sampling_interval)
    time = torch.arange(wave.shape[-1]) * sampling_interval

    wave = get_scaled_data(wave, start_receiver, end_receiver)
    spect[..., 0] = get_scaled_data(spect[..., 0], start_receiver, end_receiver)
    spect[..., 1] = get_scaled_data(spect[..., 1], start_receiver, end_receiver)

    signals_plot = go.Figure(
        [
            go.Scatter(x=time, y=signal, name=f"Receiver {i}")
            for i, signal in enumerate(wave)
        ],
        layout={
            "title": "Raw signals",
            "xaxis_title": "Time",
            "yaxis_title": "Distance from source",
        },
    )
    channel_1_plot = go.Figure(
        [
            go.Scatter(x=frequency, y=signal, name=f"Receiver {i}")
            for i, signal in enumerate(spect[..., 0])
        ],
        layout={
            "title": "Channel 1 (Cosine / Amplitude)",
            "xaxis_title": "Frequency (Hz)",
            "yaxis_title": "Distance from source",
        },
    )
    channel_2_plot = go.Figure(
        [
            go.Scatter(x=frequency, y=signal, name=f"Receiver {i}")
            for i, signal in enumerate(spect[..., 0])
        ],
        layout={
            "title": "Channel 2 (Sine / Phase)",
            "xaxis_title": "Frequency (Hz)",
            "yaxis_title": "Distance from source",
        },
    )

    app = dash.Dash()
    app.layout = [
        dash.html.H1("Wave plots"),
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
