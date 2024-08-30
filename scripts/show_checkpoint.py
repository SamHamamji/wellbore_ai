import argparse

from src.checkpoint import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()

    ds, model, _, epoch = load_checkpoint(args.checkpoint_path)

    print(f"Model type: {type(model)}")

    print(f"Dataset x transform: {ds.x_transform}")
    print(f"Dataset y bounds: {ds.bounds}")

    print(f"Epoch {epoch}")
    print(model)
    for name, param in model.named_parameters():
        print(name, param.shape, param.numel())
