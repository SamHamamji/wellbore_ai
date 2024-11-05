import typing

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from src.plotter_engines import plotter_engines


class History:
    def __init__(self):
        self._state_dict = {}

    def append(self, **kwargs: typing.Any):
        if not self._state_dict:
            self._state_dict = {key: [] for key in kwargs}

        for key, value in kwargs.items():
            self._state_dict[key].append(value)

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict: dict[str, list]):
        self._state_dict = state_dict

    def plot(self, engine: plotter_engines):
        if engine == "plotly":
            self.plot_plotly()
        elif engine == "matplotlib":
            self.plot_matplotlib()

    def plot_plotly(self):
        traces = [
            go.Scatter(
                y=self._state_dict[metric],
                x=self._state_dict["epoch"],
                mode="lines",
                name=metric,
                showlegend=True,
                yaxis="y2" if metric == "learning_rate" else None,
            )
            for metric in self._state_dict
            if metric != "epoch"
        ]

        fig = go.Figure(
            traces,
            layout={
                "xaxis_title": "Epoch",
                "yaxis": {
                    "title": "Loss",
                    "type": "log",
                },
                "yaxis2": {
                    "title": "Learning rate",
                    "overlaying": "y",
                    "side": "right",
                    "position": 1,
                    "type": "log",
                    "showgrid": False,
                    "range": [1e-10, None],
                },
            },
        )
        fig.show()

    def plot_matplotlib(self):
        _, ax1 = plt.subplots()

        epochs = self._state_dict["epoch"]

        for metric in self._state_dict:
            if metric in ("epoch", "learning_rate"):
                continue
            ax1.plot(
                epochs,
                self._state_dict[metric],
                "-",
                label=metric.replace("_", " ").capitalize(),
            )

        ax2 = ax1.twinx()
        ax2.plot(
            epochs,
            self._state_dict["learning_rate"],
            "g-",
            label="Learning rate",
        )
        ax2.set_ylim(top=1e-1, bottom=7e-5)

        ax1.set_ylabel("Losses")
        ax1.set_yscale("log")
        ax1.set_xlabel("Epoch")
        ax1.legend(frameon=False)

        ax2.set_ylabel("Learning rate")
        ax2.set_yscale("log")
        ax2.legend(frameon=False, loc="center right")

        plt.tight_layout()
        plt.show()
