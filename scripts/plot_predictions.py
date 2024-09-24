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
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def plot_predictions_plotly(
    y: torch.Tensor, pred: torch.Tensor, error: torch.Tensor, target_name: str
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
    fig.show()

    fig = go.Figure(
        go.Histogram(x=error, histnorm="probability", showlegend=False),
        layout={
            "xaxis_title": "Relative error",
            "yaxis_title": "Frequency",
        },
    )
    fig.show()


def plot_predictions_matplotlib(
    y: torch.Tensor, pred: torch.Tensor, error: torch.Tensor, target_name: str
):
    boundaries = torch.stack([y.min(), y.max()])

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    ax1.scatter(y, pred)
    ax1.plot(boundaries, boundaries, "r--", label="y=x")
    ax1.set_xlabel(f"True {target_name}")
    ax1.set_ylabel(f"Predicted {target_name}")

    ax2.hist(error, bins=20, density=True)
    ax2.set_xlabel(f"Relative {target_name} error")
    ax2.set_ylabel(f"Occurrences")

    fig.subplots_adjust(bottom=0.1)
    plt.show()


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

    x, y = next(iter(loader))
    x: torch.Tensor
    y: torch.Tensor

    checkpoint.model.eval()
    with torch.no_grad():
        pred = checkpoint.model(x)

    label_names = checkpoint.ds.get_label_names()

    for target_index, target_name in enumerate(label_names):
        target_y = y[..., target_index]
        target_pred = pred[..., target_index]

        error = error_metric(target_y, target_pred)

        if args.engine == "plotly":
            plot_predictions_plotly(y, pred, error, target_name)
        elif args.engine == "matplotlib":
            plot_predictions_matplotlib(y, pred, error, target_name)
