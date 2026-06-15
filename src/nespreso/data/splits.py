"""Train/val/test dataset splitting helpers."""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset, Subset, random_split


class IndexedSubset(Subset):
    """Subset wrapper that also returns the original dataset index."""

    def __init__(self, subset: Subset) -> None:
        super().__init__(subset.dataset, subset.indices)

    def __getitem__(self, idx: int) -> tuple[Any, Any, int]:
        inputs, profiles = super().__getitem__(idx)
        original_idx = self.indices[idx]
        return inputs, profiles, original_idx

    def __getitems__(self, indices: list[int]) -> list[tuple[Any, Any, int]]:
        return [self.__getitem__(idx) for idx in indices]


def split_dataset(
    dataset: Dataset,
    train_size: float,
    val_size: float,
    test_size: float,
    batch_size: int = 32,
    use_batches: bool = True,
) -> tuple[IndexedSubset, IndexedSubset, IndexedSubset]:
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
    - dataset: The entire dataset to be split.
    - train_size, val_size, test_size: Proportions for splitting. They should sum to 1.

    Returns:
    - train_dataset, val_dataset, test_dataset: Split datasets.
    """
    total_size = len(dataset)
    train_len = int(total_size * train_size)
    val_len = int(total_size * val_size)
    test_len = total_size - train_len - val_len

    raw_train, raw_val, raw_test = random_split(dataset, [train_len, val_len, test_len])
    return IndexedSubset(raw_train), IndexedSubset(raw_val), IndexedSubset(raw_test)
