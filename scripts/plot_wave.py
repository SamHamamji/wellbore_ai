import argparse

import torch.utils.data
import dash
import plotly.express as px

from src.data.dataset import WaveDataset
from src.data.file_filter_fn import get_filter_fn_by_vs_vp
from src.layers import FftLayer, SelectIndexLayer


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--polar", action="store_true")
parser.add_argument("--sample_index", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)


def plot_fft(wave: torch.Tensor, transform: torch.nn.Module):
    with torch.no_grad():
        spect = transform(wave)

    app = dash.Dash()
    app.layout = [
        dash.html.Div(
            [
                dash.html.H1(f"Receiver {receiver}"),
                dash.html.H3("Raw signal"),
                dash.dcc.Graph(figure=px.line(y=wave[receiver])),
                dash.html.H3("Amplitude"),
                dash.dcc.Graph(figure=px.line(y=spect[receiver, :, 0])),
                dash.html.H3("Phase"),
                dash.dcc.Graph(figure=px.line(y=spect[receiver, :, 1])),
            ]
        )
        for receiver in [0, 4, 8, 12]
    ]
    app.run(debug=True)


if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ds = WaveDataset(
        args.data_dir,
        dtype=torch.float32,
        filter_fn=get_filter_fn_by_vs_vp(),
        target_length=1541,
    )
    transform = torch.nn.Sequential(
        torch.nn.LazyBatchNorm1d(),
        FftLayer(-1, -1, args.polar),
        SelectIndexLayer(-1, (slice(None), slice(200), slice(None))),
    )
    x, y = ds[args.sample_index]

    plot_fft(x, transform)
