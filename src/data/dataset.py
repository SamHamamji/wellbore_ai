import os
import typing

import scipy
import torch
import torch.utils.data


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        target_length: int | None = None,
        filter_fn: typing.Callable[[str], bool] | None = None,
        x_transform: torch.nn.Module | None = None,
        y_transform: torch.nn.Module | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_length = target_length
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.dtype = dtype

        self.files = list(
            map(lambda file: os.path.join(data_dir, file), os.listdir(data_dir))
        )
        if filter_fn is not None:
            self.files = list(filter(filter_fn, self.files))

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

        if self.x_transform is not None:
            with torch.no_grad():
                wave: torch.Tensor = self.x_transform(wave)

        if self.y_transform is not None:
            with torch.no_grad():
                target: torch.Tensor = self.y_transform(target)

        return (wave, target)
