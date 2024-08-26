import argparse

import numpy as np
import torch.utils.data
import plotly.express as px

from src.checkpoint import load_checkpoint


parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--proportion", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds, model, _, _ = load_checkpoint(args.checkpoint_path)

    loader = torch.utils.data.DataLoader(
        ds, num_workers=8, batch_size=int(len(ds) * args.proportion)
    )

    x, y = next(iter(loader))

    model.eval()
    with torch.no_grad():
        pred = model(x)

    for target in range(y.shape[-1]):
        fig = px.scatter(
            x=y[..., target],
            y=pred[..., target],
            labels={"x": "y", "y": f"pred {target}"},
        )
        fig.add_scatter(x=y[..., target], y=y[..., target])
        fig.show()
