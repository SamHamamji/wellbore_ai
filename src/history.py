import typing


class History:
    def __init__(self, state_dict: dict[str, list]):
        self._state_dict = state_dict

    def append(self, **kwargs: dict[str, typing.Any]):
        if not self._state_dict:
            self._state_dict = {key: [] for key in kwargs}

        for key, value in kwargs.items():
            self._state_dict[key].append(value)

    def state_dict(self):
        return self._state_dict
