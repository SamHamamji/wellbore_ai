import torch
import torch.utils.data
import sklearn.model_selection


def get_index_splits(proportions: tuple[float, ...], labels: torch.Tensor):
    assert round(sum(proportions), 5) == 1, f"{sum(proportions)=}"

    values = torch.arange(len(labels))
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

        splits.append(split)

    splits.append(values)

    return tuple(splits)


def split_dataset(
    dataset: torch.utils.data.Dataset,
    proportions: tuple[float, ...],
    labels: torch.Tensor | None = None,
):
    index_splits = get_index_splits(
        proportions,
        torch.ones(len(dataset)) if labels is None else labels,  # type: ignore
    )

    return tuple(torch.utils.data.Subset(dataset, indexes) for indexes in index_splits)  # type: ignore
