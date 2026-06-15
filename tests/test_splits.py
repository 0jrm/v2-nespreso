"""Unit tests for dataset splitting helpers."""

import torch
from torch.utils.data import Dataset, Subset

from nespreso.data.splits import IndexedSubset, split_dataset


class _TinyDataset(Dataset):
    def __init__(self, n=10):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.tensor([float(idx)], dtype=torch.float32)
        y = torch.tensor([float(idx) + 0.5], dtype=torch.float32)
        return x, y


def test_split_dataset_returns_indexed_subsets():
    ds = _TinyDataset(10)
    train, val, test = split_dataset(ds, 0.7, 0.15, 0.15)
    assert isinstance(train, IndexedSubset)
    assert isinstance(val, IndexedSubset)
    assert isinstance(test, IndexedSubset)
    assert len(train) + len(val) + len(test) == 10


def test_indexed_subset_getitems_delegates_to_getitem():
    ds = _TinyDataset(4)
    subset = IndexedSubset(Subset(ds, [1, 3]))
    batch = subset.__getitems__([0, 1])
    assert len(batch) == 2
    assert batch[0][2] == 1
    assert batch[1][2] == 3
