import typing

import matplotlib

matplotlib.use("gtk4cairo")
matplotlib.rcParams["font.family"] = "Liberation Serif"
matplotlib.rcParams["font.size"] = 20

plotter_engines = typing.Literal["plotly", "matplotlib"]