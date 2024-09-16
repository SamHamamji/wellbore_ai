import argparse

import numpy as np
import torch.utils.data
import plotly.graph_objects as go

from src.checkpoint import Checkpoint


parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, required=True)
parser.add_argument("--proportion", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = Checkpoint.load_from_path(args.path)

    loader = torch.utils.data.DataLoader(
        checkpoint.ds,
        num_workers=8,
        batch_size=int(len(checkpoint.ds) * args.proportion),
    )

    x, y = next(iter(loader))
    x: torch.Tensor
    y: torch.Tensor

    checkpoint.model.eval()
    with torch.no_grad():
        pred = checkpoint.model(x)

    for target_index, target_name in enumerate(checkpoint.ds.get_label_names()):
        target_y = y[..., target_index]
        target_pred = pred[..., target_index]

        boundaries = torch.stack([target_y.min(), target_y.max()])

        traces = [
            go.Scatter(x=target_y, y=target_pred, mode="markers", showlegend=False),
            go.Scatter(x=boundaries, y=boundaries, mode="lines", showlegend=False),
        ]
        fig = go.Figure(
            traces,
            layout={
                "xaxis_title": f"True {target_name}",
                "yaxis_title": f"Predicted {target_name}",
            },
        )
        fig.show()
