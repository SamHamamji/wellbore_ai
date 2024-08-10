import os

import scipy
import torch
import torch.utils.data


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        self.data_dir = data_dir

        self.files = list(
            map(lambda file: os.path.join(data_dir, file), os.listdir(data_dir))
        )
        self.files = list(filter(lambda s: s.endswith(".mat"), self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        file = self.files[index]
        data = scipy.io.loadmat(file)

        wave = torch.from_numpy(data["wavearray_param"]).T
        target = torch.Tensor((data["vs_r"].item(), data["vp_r"].item()))

        return (wave, target)
