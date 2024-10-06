import argparse

import torch.utils.data
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.data.dataset import WaveDataset
from src.metric import Metric
from src.plotter_engines import plotter_engines


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument("--noise_std_min", type=float, default=0)
parser.add_argument("--noise_std_max", type=float, required=True)
parser.add_argument("--noise_std_step", type=float, required=True)
parser.add_argument(
    "--noise_type",
    type=str,
    default=WaveDataset.noise_types.__args__[0],
    choices=WaveDataset.noise_types.__args__,
)
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def get_figure_plotly(errors: torch.Tensor, noise_stds: torch.Tensor):
    fig = go.Figure(
        go.Scatter(x=noise_stds, y=errors, mode="lines", showlegend=False),
        layout={
            "xaxis_title": "Noise std",
            "yaxis_title": "MARE",
        },
    )
    return fig


def show_plot_matplotlib(errors: torch.Tensor, noise_stds: torch.Tensor):
    plt.subplots()

    plt.plot(noise_stds, errors)
    plt.xlabel(f"{args.noise_type} noise std".capitalize())
    plt.ylabel("MARE")
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = Checkpoint.load_from_path(args.path)
    checkpoint.model.eval()
    checkpoint.ds.noise_type = args.noise_type
    std_range = torch.arange(
        start=args.noise_std_min,
        end=args.noise_std_max,
        step=args.noise_std_step,
    )

    error_metric: Metric = lambda y, pred: ((pred - y) / y).abs().mean(0).sum()
    errors = []
    for std in std_range:
        checkpoint.ds.noise_std = std.item()
        test_subset = split_dataset(checkpoint.ds, args.splits)[2]
        loader = torch.utils.data.DataLoader(
            test_subset,
            num_workers=12,
            batch_size=int(len(test_subset)),
        )

        X, y = next(iter(loader))
        X: torch.Tensor
        y: torch.Tensor

        with torch.no_grad():
            pred = checkpoint.model(X)

        errors.append(error_metric(y, pred).item())

    errors = torch.tensor(errors)

    if args.engine == "plotly":
        get_figure_plotly(errors, std_range).show()
    elif args.engine == "matplotlib":
        show_plot_matplotlib(errors, std_range)
