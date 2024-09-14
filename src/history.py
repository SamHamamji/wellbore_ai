import typing

import plotly.graph_objects as go


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

    def plot(self):
        traces = [
            go.Scatter(
                y=self._state_dict[metric],
                x=self._state_dict["epoch"],
                mode="lines",
                name=metric,
                showlegend=True,
                yaxis=f"y2" if metric == "learning_rate" else None,
            )
            for i, metric in enumerate(self._state_dict)
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
