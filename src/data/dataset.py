import glob
import os
import re
import typing

import scipy
import torch
import torch.utils.data

num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
target_variables = f"vs({num})vp({num})(?:eps({num})gam({num})del({num}))?"
wave_file_regex = f".*(?:ISO|VTI)_{target_variables}_MP_dipole.mat$"
wave_file_regex = re.compile(wave_file_regex)


def filter_file(file: str, bounds: tuple[range | None, ...]) -> bool:
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
    noise_types = typing.Literal[
        "noiseless", "additive", "additive_relative", "multiplicative"
    ]

    def __init__(
        self,
        data_dir: str,
        target_signal_length: int | None = None,
        label_type: label_types = "isotropic",
        bounds: tuple[range | None, ...] = (),
        noise_type: noise_types = "noiseless",
        noise_std: float = 0.0,
        x_transform: torch.nn.Module | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_signal_length = target_signal_length
        self.label_type = label_type
        self.bounds = bounds
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.x_transform = x_transform
        self.dtype = dtype

        self.files = list(
            filter(
                lambda file: filter_file(file.split("/")[-1], bounds),
                glob.iglob(os.path.join(data_dir, "**"), recursive=True),
            )
        )
        assert len(self.files) > 0

    def get_kwargs(self):
        return {
            "data_dir": self.data_dir,
            "target_signal_length": self.target_signal_length,
            "x_transform": self.x_transform,
            "label_type": self.label_type,
            "bounds": self.bounds,
            "dtype": self.dtype,
        }

    def get_label_names(self):
        if self.label_type == "isotropic":
            return ("Vs (m/s)", "Vp (m/s)")
        if self.label_type == "stiffness":
            return ("c11", "c13", "c33", "c44", "c66")
        if self.label_type == "velocities":
            return (
                "Vs$_{0}$ (m/s)",
                "Vp$_{0}$ (m/s)",
                "Vs$_{90}$ (m/s)",
                "Vp$_{90}$ (m/s)",
                "Vp$_{45}$ (m/s)",
            )
        raise NotImplementedError()

    def get_thomsens_params(self, file_path: str):
        match = re.match(wave_file_regex, file_path)
        assert match is not None
        return match.groups()[2:]

    def get_stiffnesses(self, data: dict) -> tuple[float, float, float, float, float]:
        return (
            data["c11_r"].item() * 1e9,
            data["c13_r"].item() * 1e9,
            data["c33_r"].item() * 1e9,
            data["c44_r"].item() * 1e9,
            data["c66_r"].item() * 1e9,
        )

    def get_stiffnesses_iso(
        self, data: dict
    ) -> tuple[float, float, float, float, float]:
        density = data["dens_r"].item()
        Vs = data["vs_r"].item()
        Vp = data["vp_r"].item()
        c33 = density * Vp**2
        c44 = density * Vs**2
        return (c33, abs(c33 - c44) - c44, c33, c44, c44)

    def get_velocities(
        self, data: dict, file_path: str
    ) -> tuple[float, float, float, float, float]:
        thomsens_params = self.get_thomsens_params(file_path)
        density = data["dens_r"].item()
        c11, _, c33, c44, c66 = self.get_stiffnesses(data)
        M1 = (
            0.25 * (c11 - c33) ** 2
            + 2 * c33 * (c33 - c44) * float(thomsens_params[2])
            + (c33 - c44) ** 2
        )  # assumes strong anisotropy
        M = 0.5 * (c11 + c33) + c44 + M1**0.5

        return (
            data["vs_r"].item(),  # VS_0
            data["vp_r"].item(),  # VP_0
            (c66 / density) ** 0.5,  # VS_90
            (c11 / density) ** 0.5,  # VP_90
            (M / (2 * density)) ** 0.5,  # VP_45
        )

    def get_velocities_iso(
        self, data: dict
    ) -> tuple[float, float, float, float, float]:
        Vs = data["vs_r"].item()
        Vp = data["vp_r"].item()
        return (Vs, Vp, Vs, Vp, Vp)

    def get_labels(self, data: dict, file_path: str):
        if self.label_type == "isotropic":
            return torch.Tensor((data["vs_r"].item(), data["vp_r"].item()))
        if self.label_type == "stiffness":
            if "c11_r" not in data:
                return torch.Tensor(self.get_stiffnesses_iso(data))
            return torch.Tensor(self.get_stiffnesses(data))
        if self.label_type == "velocities":
            if "c11_r" not in data:
                return torch.Tensor(self.get_velocities_iso(data))
            return torch.Tensor(self.get_velocities(data, file_path))
        raise NotImplementedError()

    def __getitem__(self, index: int):
        file_path = self.files[index]
        data: dict = scipy.io.loadmat(file_path)

        wave = (
            torch.from_numpy(data["wavearray_param"]).T[1:].to(dtype=self.dtype)
        )  # Drop time row

        if self.target_signal_length is not None:
            wave = wave[..., : self.target_signal_length]

        if self.noise_type == "additive":
            wave.add_(torch.normal(0, self.noise_std, wave.shape))
        if self.noise_type == "additive_relative":
            peak = wave.abs().max().item()
            wave.add_(torch.normal(0, self.noise_std * peak, wave.shape))
        if self.noise_type == "multiplicative":
            wave.mul_(torch.normal(1, self.noise_std, wave.shape))

        if self.x_transform is not None:
            with torch.no_grad():
                wave: torch.Tensor = self.x_transform(wave)

        labels = self.get_labels(data, file_path).to(dtype=self.dtype)

        return (wave, labels)

    def __len__(self):
        return len(self.files)
