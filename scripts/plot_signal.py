import argparse

import torch.utils.data
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.plotter_engines import plotter_engines
from src.data.dataset import WellboreDataset
from src.layers import FftLayer


SAMPLING_INTERVAL = 6.63130e-06
START_RECEIVER = 3.275
END_RECEIVER = 3.275 + 1.828


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--polar", action="store_true")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--target_signal_length", type=int, required=False)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--signal_type",
    type=str,
    default=WellboreDataset.signal_types.__args__[0],
    choices=WellboreDataset.signal_types.__args__,
)
parser.add_argument(
    "--noise_type",
    type=str,
    default=WellboreDataset.noise_types.__args__[0],
    choices=WellboreDataset.noise_types.__args__,
)
parser.add_argument("--noise_std", type=float, default=None)
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def get_scaled_data(data: torch.Tensor, start: float, end: float):
    if data.shape[0] == 1:
        return data

    offset_per_receiver = data.max(dim=1).values.mean(0)
    offset = torch.arange(0, data.shape[0]) * offset_per_receiver
    data = data + offset.expand(data.T.shape).T

    return data * (end - start) / offset_per_receiver / (data.shape[0] - 1) + start


def plot_dispersion_curve(
    dispersion_curve: torch.Tensor, frequency_scale: torch.Tensor, engine: str
):
    dispersion_curve = get_scaled_data(
        dispersion_curve, START_RECEIVER, END_RECEIVER
    ).squeeze()
    frequency_scale = frequency_scale.squeeze()

    if engine == "plotly":
        plot_dispersion_curve_plotly(dispersion_curve, frequency_scale)
    else:
        plot_dispersion_curve_matplotlib(dispersion_curve, frequency_scale)


def plot_dispersion_curve_plotly(dispersion_curve: torch.Tensor, scale: torch.Tensor):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scale, y=dispersion_curve))
    fig.update_layout(
        title="Dispersion curve",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Slowness (μs/m)",
        yaxis_range=[0, 400],
    )
    fig.show()


def plot_dispersion_curve_matplotlib(
    dispersion_curve: torch.Tensor, scale: torch.Tensor
):
    plt.figure()
    plt.plot(scale, dispersion_curve)
    plt.title("Dispersion curve")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Slowness (μs/m)")
    plt.ylim(0, 400)

    plt.show()


def plot_wave(wave: torch.Tensor, polar: bool, engine: str):
    transform = FftLayer(-1, -1, polar)
    with torch.no_grad():
        spect = transform(wave)

    wave = get_scaled_data(wave, START_RECEIVER, END_RECEIVER)
    spect[..., 0] = get_scaled_data(spect[..., 0], START_RECEIVER, END_RECEIVER)
    spect[..., 1] = get_scaled_data(spect[..., 1], START_RECEIVER, END_RECEIVER)

    # pylint: disable=not-callable
    frequency_scale = torch.fft.rfftfreq(wave.shape[-1], d=SAMPLING_INTERVAL)
    time_scale = torch.arange(wave.shape[-1]) * SAMPLING_INTERVAL

    if engine == "plotly":
        plot_wave_plotly(wave, time_scale, spect, frequency_scale, polar)
    else:
        plot_wave_matplotlib(wave, time_scale, spect, frequency_scale, polar)


def plot_wave_plotly(
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
        ["Time", "Frequency (Hz)", "Frequency (Hz)"],
    ):
        figure = go.Figure(
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

        figure.show()


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

    ds = WellboreDataset(
        args.data_dir,
        dtype=torch.float32,
        signal_type=args.signal_type,
        target_signal_length=args.target_signal_length,
        noise_type=args.noise_type,
        noise_std=args.noise_std,
    )
    wave, _ = ds[args.sample_index]

    if args.signal_type == "waveform":
        plot_wave(wave, args.polar, args.engine)
    else:
        frequency_scale = ds.get_frequency_scale(args.sample_index)
        plot_dispersion_curve(wave, frequency_scale, args.engine)
