import argparse

import torch.utils.data
import dash
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.plotter_engines import plotter_engines
from src.data.dataset import WaveDataset
from src.layers import FftLayer


SAMPLING_INTERVAL = 6.63130e-06
START_RECEIVER = 3.275
END_RECEIVER = 3.275 + 1.828


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
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def get_scaled_data(data: torch.Tensor, start: float, end: float):
    offset_per_receiver = data.max(dim=1).values.mean(0)
    offset = torch.arange(0, data.shape[0]) * offset_per_receiver
    data = data + offset.expand(data.T.shape).T

    return data * (end - start) / offset_per_receiver / (data.shape[0] - 1) + start


def get_offset(data: torch.Tensor, index: int):
    if index == 0:
        return torch.zeros(1, *data.shape[2:])
    return data.max(dim=1).values.mean(0) * index


def plot_wave_plotly(
    wave: torch.Tensor,
    time_scale: torch.Tensor,
    spect: torch.Tensor,
    frequency_scale: torch.Tensor,
    polar: bool,
):
    figures = []
    for data, scale, title, xlabel in zip(
        [wave, spect[..., 0], spect[..., 1]],
        [time_scale, frequency_scale, frequency_scale],
        [
            "Raw signals",
            f"Channel 1 {'Amplitude' if polar else 'Cosine'}",
            f"Channel 2 {'Phase' if polar else 'Sine'}",
        ],
        ["Time", "Frequency (Hz)", "Frequency (Hz)"],
    ):
        figures.append(
            go.Figure(
                [
                    go.Scatter(x=scale, y=signal, name=f"Receiver {i}")
                    for i, signal in enumerate(data)
                ],
                layout={
                    "title": title,
                    "xaxis_title": xlabel,
                    "yaxis_title": "Distance from source (m)",
                },
            )
        )

    app = dash.Dash()
    app.layout = [
        dash.html.H1("Wave plots"),
        *[dash.dcc.Graph(figure=figure) for figure in figures],
    ]
    app.run(debug=True)


def plot_wave_matplotlib(
    wave: torch.Tensor,
    time_scale: torch.Tensor,
    spect: torch.Tensor,
    frequency_scale: torch.Tensor,
    polar: bool,
):
    for data, scale, title, xlabel in zip(
        [wave, spect[..., 0], spect[..., 1]],
        [time_scale, frequency_scale, frequency_scale],
        [
            "Raw signals",
            f"Channel 1 {'Amplitude' if polar else 'Cosine'}",
            f"Channel 2 {'Phase' if polar else 'Sine'}",
        ],
        ["Time (s)", "Frequency (Hz)", "Frequency (Hz)"],
    ):
        plt.figure()
        for i, signal in enumerate(data):
            plt.plot(scale, signal, label=f"Receiver {i}")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Distance from source (m)")

    plt.show()


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
    )
    wave, _ = ds[args.sample_index]

    with torch.no_grad():
        spect = transform(wave)

    wave = get_scaled_data(wave, START_RECEIVER, END_RECEIVER)
    spect[..., 0] = get_scaled_data(spect[..., 0], START_RECEIVER, END_RECEIVER)
    spect[..., 1] = get_scaled_data(spect[..., 1], START_RECEIVER, END_RECEIVER)

    # pylint: disable=not-callable
    frequency_scale = torch.fft.rfftfreq(wave.shape[-1], d=SAMPLING_INTERVAL)
    time_scale = torch.arange(wave.shape[-1]) * SAMPLING_INTERVAL

    if args.engine == "plotly":
        plot_wave_plotly(wave, time_scale, spect, frequency_scale, args.polar)
    else:
        plot_wave_matplotlib(wave, time_scale, spect, frequency_scale, args.polar)
