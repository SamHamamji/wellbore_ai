import os
import re

import numpy as np
import scipy
import torch
import torch.utils.data
import typing

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
    label_types = typing.Literal["isotropic", "stiffness", "thomsen", "velocities"]

    def __init__(
        self,
        data_dir: str,
        target_length: int | None = None,
        x_transform: torch.nn.Module | None = None,
        label_type: label_types = "isotropic",
        bounds: tuple[range | None, ...] = (),
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_length = target_length
        self.x_transform = x_transform
        self.dtype = dtype
        self.bounds = bounds
        self.label_type = label_type

        self.files = list(
            map(lambda file: os.path.join(data_dir, file), os.listdir(data_dir))
        )

        self.files = list(filter(lambda file: filter_files(file, bounds), self.files))

    def __len__(self):
        return len(self.files)

    def get_label_names(self):
        if self.label_type == "isotropic":
            return ("Vs (m/s)", "Vp (m/s)")
        if self.label_type == "stiffness":
            return ("c11", "c13", "c33", "c44", "c66")
        if self.label_type == "velocities":
            return ("Vs_0", "Vp_0", "Vs_90", "Vp_90", "Vs_45")
        raise NotImplementedError()

    def __getitem__(self, index: int):
        file = self.files[index]
        data: dict = scipy.io.loadmat(file)

        wave = (
            torch.from_numpy(data["wavearray_param"]).T[1:].to(dtype=self.dtype)
        )  # Drop time row

        if self.target_length is not None:
            wave = wave[..., : self.target_length]

        if self.x_transform is not None:
            with torch.no_grad():
                wave: torch.Tensor = self.x_transform(wave)

        if self.label_type == "isotropic":
            target = torch.Tensor((data["vs_r"].item(), data["vp_r"].item())).to(
                dtype=self.dtype
            )
        elif self.label_type == "stiffness":
            target = torch.Tensor(
                (
                    data["c11_r"].item(),
                    data["c13_r"].item(),
                    data["c33_r"].item(),
                    data["c44_r"].item(),
                    data["c66_r"].item(),
                )
            )

        elif self.label_type == "velocities":
            density = data["dens_r"].item()

            C11 = data["c11_r"].item() * 1e9
            C13 = data["c13_r"].item() * 1e9
            C33 = data["c33_r"].item() * 1e9
            C44 = data["c44_r"].item() * 1e9
            C66 = data["c66_r"].item() * 1e9

            M1 = 0.25 * (C11 - C33) ** 2 + (C13 + C44) ** 2
            M = 0.5 * (C11 + C33) + C44 + M1**0.5

            target = torch.Tensor(
                (
                    data["vs_r"].item(),  # VS_0
                    data["vp_r"].item(),  # VP_0
                    (C66 / density) ** 0.5,  # VS_90
                    (C11 / density) ** 0.5,  # VP_90
                    (M / (2 * density)) ** 0.5,  # VS_45
                )
            )

        else:
            raise NotImplementedError()

        return (wave, target)
