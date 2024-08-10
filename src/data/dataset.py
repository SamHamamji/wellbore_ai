import os

import scipy
import torch
import torch.utils.data


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        dims_to_flatten: tuple[int, int] | None = None,
        target_length: int | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_length = target_length
        self.dims_to_flatten = dims_to_flatten
        self.dtype = dtype

        self.files = list(
            map(lambda file: os.path.join(data_dir, file), os.listdir(data_dir))
        )
        self.files = list(filter(lambda s: s.endswith(".mat"), self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        file = self.files[index]
        data = scipy.io.loadmat(file)

        wave = (
            torch.from_numpy(data["wavearray_param"]).T[1:].to(dtype=self.dtype)
        )  # Drop time row
        target = torch.Tensor((data["vs_r"].item(), data["vp_r"].item())).to(
            dtype=self.dtype
        )

        if self.target_length is not None:
            wave = wave[..., : self.target_length]

        if self.dims_to_flatten is not None:
            wave = wave.flatten(*self.dims_to_flatten)

        return (wave, target)
