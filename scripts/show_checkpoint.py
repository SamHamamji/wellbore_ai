import argparse
import json

from src.checkpoint import Checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint = Checkpoint.load_from_path(args.checkpoint_path)

    print(f"Dataset x transform: {checkpoint.ds.x_transform}")
    print(f"Dataset y bounds: {checkpoint.ds.bounds}")
    print()

    print(f"Model: {checkpoint.model}")
    print()

    print("Parameters:")
    for name, param in checkpoint.model.named_parameters():
        print(f"  {name}, {param.shape}, {param.numel()}")

    print(
        f"Scheduler state dict: {json.dumps(checkpoint.scheduler.state_dict(), indent=2)}"
    )
    print(f"Optimizer: {checkpoint.optimizer}")
