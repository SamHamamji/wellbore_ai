import argparse

import numpy as np
import torch.utils.data
import plotly.express as px

from src.data.dataset import WaveDataset
from src.models import WaveCnn3d, FftLayer


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        target_length=1541,
        dtype=torch.float32,
        transform=FftLayer(),
    )
    loader = torch.utils.data.DataLoader(ds, num_workers=8, batch_size=len(ds) // 8)

    x, y = next(iter(loader))

    model = WaveCnn3d(x.shape[1:], y.shape[1:])

    if args.model_path is not None:
        model_state_dict = torch.load(args.model_path, weights_only=True)[
            "model_state_dict"
        ]
        model.load_state_dict(model_state_dict)

    model.eval()

    with torch.no_grad():
        pred = model(x)

    for target in range(x.shape[-1]):
        fig = px.scatter(
            x=y[..., target],
            y=pred[..., target],
            labels={"x": "y", "y": f"pred {target}"},
        )
        fig.add_scatter(x=y[..., target], y=y[..., target])
        fig.show()
