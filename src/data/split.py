import numpy as np
import torch
import torch.utils.data
import sklearn.model_selection


def get_index_splits(labels: torch.Tensor, proportions: tuple[float, ...]):
    assert sum(proportions) == 1, f"{sum(proportions)=}"

    values = np.arange(len(labels))
    remaining_ratio = 1.0
    splits: list[torch.Tensor] = []

    for proportion in proportions[:-1]:
        values, split = sklearn.model_selection.train_test_split(
            values,
            stratify=labels[values],
            test_size=proportion / remaining_ratio,
            random_state=0,
        )
        remaining_ratio = remaining_ratio - proportion

        splits.append(torch.from_numpy(split))

    splits.append(torch.from_numpy(values))

    return tuple(splits)


def split_dataset(
    dataset: torch.utils.data.Dataset,
    labels: torch.Tensor,
    proportions: tuple[float, ...],
):
    index_splits = get_index_splits(labels, proportions)

    return tuple(torch.utils.data.Subset(dataset, indexes) for indexes in index_splits)  # type: ignore
