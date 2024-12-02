import argparse

import torch
import matplotlib.pyplot as plt

from src.checkpoint import Checkpoint
from src.data.split import split_dataset
from src.plotter_engines import plotter_engines


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--splits", type=float, nargs="+", default=(0.7, 0.2, 0.1))
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)


def get_tpr_fpr(y_true: torch.Tensor, y_scores: torch.Tensor):
    thresholds = torch.sort(y_scores, descending=True).values
    thresholds = torch.cat([thresholds, torch.tensor([-float("inf")])])

    tpr = []
    fpr = []
    num_positives = (y_true == 1).sum().item()
    num_negatives = (y_true == 0).sum().item()

    for threshold in thresholds:
        predictions = y_scores >= threshold
        true_positives = ((predictions == 1) & (y_true == 1)).sum().item()
        false_positives = ((predictions == 1) & (y_true == 0)).sum().item()

        tpr.append(true_positives / num_positives if num_positives > 0 else 0)
        fpr.append(false_positives / num_negatives if num_negatives > 0 else 0)

    return torch.tensor(fpr), torch.tensor(tpr)


def plot_roc_curve_matplotlib(fpr: torch.Tensor, tpr: torch.Tensor, roc_auc: float):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.5f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_roc_curve_plotly(fpr: torch.Tensor, tpr: torch.Tensor, roc_auc: float):
    raise NotImplementedError()


if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = Checkpoint.load_from_path(args.path)
    checkpoint.model.eval()

    test_subset = split_dataset(checkpoint.ds, args.splits)[2]
    loader = torch.utils.data.DataLoader(test_subset, num_workers=12, batch_size=24)

    y_true = []
    y_scores = []

    for X, y in loader:
        with torch.no_grad():
            pred: torch.Tensor = checkpoint.model(X)

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true).flatten()
    y_scores = torch.cat(y_scores).flatten()

    fpr, tpr = get_tpr_fpr(y_true, y_scores)
    roc_auc = torch.trapz(tpr, fpr).item()

    if args.engine == "plotly":
        plot_roc_curve_plotly(fpr, tpr, roc_auc)
    elif args.engine == "matplotlib":
        plot_roc_curve_matplotlib(fpr, tpr, roc_auc)
