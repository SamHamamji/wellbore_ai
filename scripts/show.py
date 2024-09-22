import argparse

from src.checkpoint import Checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument(
    "--plotter", type=str, default="plotly", choices=("plotly", "matplotlib")
)

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = Checkpoint.load_from_path(args.path)

    checkpoint.print()
    checkpoint.history.plot(args.plotter)
