import argparse

import torch.utils.data
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.data.dataset import WellboreDataset
from src.plotter_engines import plotter_engines
import src.metric as metric


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument("--noise_std_min", type=float, default=0)
parser.add_argument("--noise_std_max", type=float, required=True)
parser.add_argument("--noise_std_step", type=float, required=True)
parser.add_argument(
    "--noise_type",
    type=str,
    default=WellboreDataset.noise_types.__args__[0],
    choices=WellboreDataset.noise_types.__args__,
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


def show_plot_matplotlib(
    errors: torch.Tensor,
    noise_stds: torch.Tensor,
    noise_type: WellboreDataset.noise_types,
):
    plt.figure()

    xlabel = f"{noise_type.replace("_", " ").capitalize()} noise std"
    if noise_type == "additive_relative":
        xlabel = "Additive noise std (%)"
        noise_stds.mul_(100)

    plt.plot(noise_stds, errors, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("MARE")
    plt.tight_layout()
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

    error_metric: metric.Metric = (
        lambda y, pred: metric.relative_error(y, pred).abs_().mean(0).sum(0)
    )
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
        show_plot_matplotlib(errors, std_range, args.noise_type)
