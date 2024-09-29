import argparse

import numpy as np
import torch.utils.data
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.metric import Metric
from src.plotter_engines import plotter_engines


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


def get_predictions_figure_plotly(
    y: torch.Tensor, pred: torch.Tensor, target_name: str
):
    boundaries = torch.stack([y.min(), y.max()])

    fig = go.Figure(
        [
            go.Scatter(x=y, y=pred, mode="markers", showlegend=False),
            go.Scatter(x=boundaries, y=boundaries, mode="lines", showlegend=False),
        ],
        layout={
            "xaxis_title": f"True {target_name}",
            "yaxis_title": f"Predicted {target_name}",
        },
    )
    return fig


def get_error_distribution_plotly(error: torch.Tensor, target_name: str):
    return go.Figure(
        go.Histogram(x=error, histnorm="probability", showlegend=False),
        layout={
            "xaxis_title": f"Relative {target_name} error",
            "yaxis_title": "Occurrences",
        },
    )


def set_predictions_ax_matplotlib(y: torch.Tensor, pred: torch.Tensor, ax: plt.Axes):
    boundaries = torch.stack([y.min(), y.max()])

    ax.scatter(y, pred)
    ax.plot(boundaries, boundaries, "r--", label="y=x")


def set_error_distribution_ax_matplotlib(error: torch.Tensor, ax: plt.Axes):
    ax.hist(error, bins=20, density=True)
    ax.legend(
        [f"Mean abs: {error.abs().mean():.3f}\nStandard dev: {error.std():.3f}"],
        handlelength=0,
        handletextpad=0,
        prop={"size": 12},
    )


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = Checkpoint.load_from_path(args.path)

    test_subset = split_dataset(checkpoint.ds, args.splits)[2]
    loader = torch.utils.data.DataLoader(
        test_subset,
        num_workers=8,
        batch_size=int(len(test_subset) * args.proportion),
    )

    error_metric: Metric = lambda y, pred: ((pred - y) / y)

    X, y = next(iter(loader))
    X: torch.Tensor
    y: torch.Tensor

    checkpoint.model.eval()
    with torch.no_grad():
        pred = checkpoint.model(X)

    label_names = checkpoint.ds.get_label_names()
    if args.layout == "horizontal":
        fig, axes = plt.subplots(2, len(label_names), figsize=(4 * len(label_names), 8))
    elif args.layout == "vertical":
        fig, axes = plt.subplots(len(label_names), 2, figsize=(8, 4 * len(label_names)))
        axes = axes.T
    else:
        raise ValueError(f"Invalid layout: {args.layout}")

    for target_index, target_name in enumerate(label_names):
        target_y = y[..., target_index]
        target_pred = pred[..., target_index]

        error = error_metric(target_y, target_pred)
        assert isinstance(error, torch.Tensor)

        if args.engine == "plotly":
            get_predictions_figure_plotly(target_y, target_pred, target_name).show()
            get_error_distribution_plotly(error, target_name).show()
        elif args.engine == "matplotlib":
            set_predictions_ax_matplotlib(target_y, target_pred, axes[0, target_index])
            axes[0, target_index].set_xlabel(f"True {target_name}")

            set_error_distribution_ax_matplotlib(error, axes[1, target_index])
            axes[1, target_index].set_xlabel(f"Relative {target_name} error")

            if args.layout == "vertical":
                axes[0, target_index].set_ylabel(f"Predicted {target_name}")
                axes[1, target_index].set_ylabel("Frequency")

    if args.layout == "horizontal":
        axes[0, 0].set_ylabel("Predicted values")
        axes[1, 0].set_ylabel("Frequency")

    fig.set_layout_engine("tight")
    plt.show()
