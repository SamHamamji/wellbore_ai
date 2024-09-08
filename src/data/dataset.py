import os
import re

import scipy
import torch
import torch.utils.data

num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
target_variables = f"vs({num})vp({num})(?:eps({num})gam({num})del({num}))?"
wave_file_regex = f".*(?:ISO|VTI)_{target_variables}_MP_dipole.mat$"


def filter_files(file: str, bounds: tuple[range | None, ...]) -> bool:
    match = re.match(wave_file_regex, file)
    if match is None:
        return False

    groups = match.groups()

    return all(
        bound.start <= float(groups[i]) < bound.stop
        for i, bound in enumerate(bounds)
        if bound is not None
    )


class WaveDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        target_length: int | None = None,
        x_transform: torch.nn.Module | None = None,
        bounds: tuple[range | None, ...] = (),
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_length = target_length
        self.x_transform = x_transform
        self.dtype = dtype
        self.bounds = bounds

        self.files = list(
            map(lambda file: os.path.join(data_dir, file), os.listdir(data_dir))
        )

        self.files = list(filter(lambda file: filter_files(file, bounds), self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        file = self.files[index]
        data: dict = scipy.io.loadmat(file)

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

        return (wave, target)
