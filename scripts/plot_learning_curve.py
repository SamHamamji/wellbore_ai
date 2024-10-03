import argparse

from src.checkpoint import Checkpoint
from src.plotter_engines import plotter_engines

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, required=True)
parser.add_argument(
    "--engine",
    type=str,
    default=plotter_engines.__args__[0],
    choices=plotter_engines.__args__,
)

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = Checkpoint.load_from_path(args.path)
    checkpoint.history.plot(args.engine)
