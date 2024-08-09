import torch.utils.data

from src.data.dataset import WaveDataset

if __name__ == "__main__":
    ds = WaveDataset("data/ISO Wr")
    dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

    for i in dataloader:
        print(i)
        break
