"""Plots predictions of a model on its testing dataset."""

import argparse

import numpy as np
import torch.utils.data
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.plotter_engines import plotter_engines
import src.metric as metric


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--proportion", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument(
    "--layout", type=str, choices=("horizontal", "vertical"), default="horizontal"
)
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def get_predictions_figure_plotly(y: torch.Tensor, pred: torch.Tensor, label_name: str):
    boundaries = torch.stack([y.min(), y.max()])

    fig = go.Figure(
        [
            go.Scatter(x=y, y=pred, mode="markers", showlegend=False),
            go.Scatter(x=boundaries, y=boundaries, mode="lines", showlegend=False),
        ],
        layout={
            "xaxis_title": f"True {label_name}",
            "yaxis_title": f"Predicted {label_name}",
        },
    )
    return fig


def get_error_distribution_plotly(error: torch.Tensor, label_name: str):
    return go.Figure(
        go.Histogram(x=error, histnorm="probability", showlegend=False),
        layout={
            "xaxis_title": f"Relative {label_name} error",
            "yaxis_title": "Occurrences",
        },
    )


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = Checkpoint.load_from_path(args.path)

    test_subset = split_dataset(checkpoint.ds, args.splits)[2]
    loader = torch.utils.data.DataLoader(
        test_subset,
        num_workers=12,
        batch_size=int(len(test_subset) * args.proportion),
    )

    error_metric = metric.relative_error

    X, y = next(iter(loader))
    X: torch.Tensor
    y: torch.Tensor

    checkpoint.model.eval()
    with torch.no_grad():
        pred = checkpoint.model(X)

    label_names = checkpoint.ds.get_label_names()
    if args.layout == "horizontal":
        fig, axes = plt.subplots(
            2, (len(label_names) + 1) // 2, figsize=(4 * len(label_names), 8)
        )
    elif args.layout == "vertical":
        fig, axes = plt.subplots(
            (len(label_names) + 1) // 2, 2, figsize=(8, 4 * len(label_names))
        )
    else:
        raise ValueError(f"Invalid layout: {args.layout}")
    axes = axes.reshape(-1)

    for label_index, label_name in enumerate(label_names):
        label_y = y[..., label_index]
        label_pred = pred[..., label_index]

        error = error_metric(label_y, label_pred)
        assert isinstance(error, torch.Tensor)

        if args.engine == "plotly":
            get_predictions_figure_plotly(label_y, label_pred, label_name).show()
            get_error_distribution_plotly(error, label_name).show()
        elif args.engine == "matplotlib":
            ax: plt.Axes = axes[label_index]

            boundaries = torch.stack([label_y.min(), label_y.max()])

            ax.scatter(label_y, label_pred)
            ax.plot(boundaries, boundaries, "r--", label="y=x")
            ax.text(
                0.05,
                0.95,
                f"MARE: {error.abs().mean():.3f}",
                ha="left",
                va="top",
                transform=ax.transAxes,
            )

            ax.set_xlabel(f"True {label_name}", fontsize=32)
            ax.set_ylabel(f"Predicted {label_name}", fontsize=32)

    fig.tight_layout()
    plt.show()
