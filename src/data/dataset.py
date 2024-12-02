import glob
import os
import re
import typing

import scipy
import torch
import torch.utils.data

num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
target_variables = f"vs({num})vp({num})(?:eps({num})gam({num})del({num}))?"
matlab_file_regex = re.compile(f".*(?:ISO|VTI)_{target_variables}_MP_dipole.mat$")


def filter_file(file: str, label_bounds: tuple[range | None, ...]) -> bool:
    match = re.match(matlab_file_regex, file)
    if match is None:
        return False

    groups = match.groups()

    return all(
        bound.start <= float(groups[i]) < bound.stop
        for i, bound in enumerate(label_bounds)
        if bound is not None
    )


class WellboreDataset(torch.utils.data.Dataset):
    data_field_types = typing.Literal[
        "waveform",
        "dispersion_curve",
        "isotropic",
        "stiffness",
        "thomsen",
        "velocities",
        "anisotropy",
    ]
    noise_types = typing.Literal[
        "noiseless", "additive", "additive_relative", "multiplicative"
    ]

    def __init__(
        self,
        data_dir: str,
        target_signal_length: int | None = None,
        signal_type: data_field_types = "waveform",
        label_type: data_field_types = "isotropic",
        label_bounds: tuple[range | None, ...] = (),
        noise_type: noise_types = "noiseless",
        noise_std: float = 0.0,
        x_transform: torch.nn.Module | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.data_dir = data_dir
        self.target_signal_length = target_signal_length
        self.signal_type: WellboreDataset.data_field_types = signal_type
        self.label_type: WellboreDataset.data_field_types = label_type
        self.label_bounds = label_bounds
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.x_transform = x_transform
        self.dtype = dtype

        self.files = list(
            filter(
                lambda file: filter_file(file.split("/")[-1], label_bounds),
                glob.iglob(os.path.join(data_dir, "**"), recursive=True),
            )
        )
        assert len(self.files) > 0

    def get_thomsens_params(self, file_path: str):
        match = re.match(matlab_file_regex, file_path)
        assert match is not None
        return torch.Tensor(list(map(float, match.groups()[2:])))

    def get_stiffnesses(self, data: dict):
        if "c11_r" in data:
            stiffnesses = (
                data["c11_r"].item(),
                data["c13_r"].item(),
                data["c33_r"].item(),
                data["c44_r"].item(),
                data["c66_r"].item(),
            )
        else:
            density = data["dens_r"].item()
            Vs = data["vs_r"].item()
            Vp = data["vp_r"].item()
            c33 = density * Vp**2
            c44 = density * Vs**2
            stiffnesses = (c33, abs(c33 - c44) - c44, c33, c44, c44)

        return torch.Tensor(stiffnesses) * 1e9

    def get_velocities(self, data: dict, file_path: str):
        Vs = data["vs_r"].item()
        Vp = data["vp_r"].item()
        if "c11_r" in data:
            thomsens_params = self.get_thomsens_params(file_path)
            density = data["dens_r"].item()
            c11, _, c33, c44, c66 = self.get_stiffnesses(data).tolist()
            M1 = (
                0.25 * (c11 - c33) ** 2
                + 2 * c33 * (c33 - c44) * thomsens_params[2].item()
                + (c33 - c44) ** 2
            )  # assumes strong anisotropy
            M = 0.5 * (c11 + c33) + c44 + M1**0.5

            velocities = (
                Vs,  # VS_0
                Vp,  # VP_0
                (c66 / density) ** 0.5,  # VS_90
                (c11 / density) ** 0.5,  # VP_90
                (M / (2 * density)) ** 0.5,  # VP_45
            )
        else:
            velocities = (Vs, Vp, Vs, Vp, Vp)

        return torch.Tensor(velocities)

    def get_anisotropy(self, data: dict):
        return torch.tensor(("c11_r" not in data,))

    def get_isotropic_velocities(self, data: dict):
        return torch.Tensor((data["vs_r"].item(), data["vp_r"].item()))

    def get_waveform(self, data: dict):
        return torch.from_numpy(data["wavearray_param"]).T[1:]  # drop time row

    def get_dispersion_curve(self, data: dict):
        return torch.from_numpy(data["slowness_param"]).T

    def get_field_data(
        self, field: data_field_types, data: dict, file_path: str
    ) -> torch.Tensor:
        if field == "anisotropy":
            return self.get_anisotropy(data)
        if field == "isotropic":
            return self.get_isotropic_velocities(data)
        if field == "stiffness":
            return self.get_stiffnesses(data)
        if field == "velocities":
            return self.get_velocities(data, file_path)
        if field == "thomsen":
            return self.get_thomsens_params(file_path)
        if field == "waveform":
            return self.get_waveform(data)
        if field == "dispersion_curve":
            return self.get_dispersion_curve(data)
        raise NotImplementedError()

    def __getitem__(self, index: int):
        file_path = self.files[index]
        data: dict = scipy.io.loadmat(file_path)

        signal = self.get_field_data(self.signal_type, data, file_path).to(
            dtype=self.dtype
        )
        labels = self.get_field_data(self.label_type, data, file_path).to(
            dtype=self.dtype
        )

        if self.target_signal_length is not None:
            signal = signal[..., : self.target_signal_length]

        if self.noise_type == "additive":
            signal.add_(torch.normal(0, self.noise_std, signal.shape))
        if self.noise_type == "additive_relative":
            peak = signal.abs().max().item()
            signal.add_(torch.normal(0, self.noise_std * peak, signal.shape))
        if self.noise_type == "multiplicative":
            signal.mul_(torch.normal(1, self.noise_std, signal.shape))

        if self.x_transform is not None:
            with torch.no_grad():
                signal: torch.Tensor = self.x_transform(signal)

        return (signal, labels)

    def __len__(self):
        return len(self.files)

    def get_kwargs(self):
        return {
            "data_dir": self.data_dir,
            "target_signal_length": self.target_signal_length,
            "signal_type": self.signal_type,
            "label_type": self.label_type,
            "label_bounds": self.label_bounds,
            "noise_type": self.noise_type,
            "noise_std": self.noise_std,
            "x_transform": self.x_transform,
            "dtype": self.dtype,
        }

    def get_frequency_scale(self, index: int) -> torch.Tensor:
        file_path = self.files[index]
        data: dict = scipy.io.loadmat(file_path)
        return torch.from_numpy(data["frequency_param"].T)

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
        if self.label_type == "anisotropy":
            return ("Anisotropy",)
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"Dataset: {self.get_kwargs()}"
