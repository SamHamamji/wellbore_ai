import argparse
import json

from src.checkpoint import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    ds, model, optimizer, scheduler = load_checkpoint(args.checkpoint_path)

    print(f"Dataset x transform: {ds.x_transform}")
    print(f"Dataset y bounds: {ds.bounds}")
    print()

    print(f"Model: {model}")
    print()

    print("Parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}, {param.shape}, {param.numel()}")

    print(f"Scheduler state dict: {json.dumps(scheduler.state_dict(), indent=2)}")
    print(f"Optimizer: {optimizer}")
