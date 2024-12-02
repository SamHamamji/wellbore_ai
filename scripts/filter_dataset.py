"""Duplicates a dataset while removing all its unused fields"""

import argparse
import glob
import os

import scipy

from src.data.dataset import filter_file

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)


matlab_data_fields = {
    "c11_r",
    "c13_r",
    "c33_r",
    "c44_r",
    "c66_r",
    "dens_r",
    "vs_r",
    "vp_r",
    "wavearray_param",
    "slowness_param",
    "frequency_param",
}


if __name__ == "__main__":
    args = parser.parse_args()

    files = list(
        filter(
            lambda file: filter_file(file.split("/")[-1], ()),
            glob.iglob(os.path.join(args.input_path, "**"), recursive=True),
        )
    )

    for file_path in files:
        data = scipy.io.loadmat(file_path)

        for key in list(data.keys()):
            if key not in matlab_data_fields:
                del data[key]

        new_file_path = os.path.relpath(file_path, args.input_path)
        new_file_path = os.path.join(args.output_path, new_file_path)

        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        scipy.io.savemat(new_file_path, data)
