"""Training-run reporting helpers hoisted from the monolith __main__ block."""

from __future__ import annotations

from typing import Any, Mapping


def print_training_params(
    full_dataset: Any,
    *,
    input_params: Mapping[str, bool],
    n_components: int,
    batch_size: int,
    min_depth: float,
    max_depth: float,
    learning_rate: float,
    dropout_prob: float,
    train_size: float,
    val_size: float,
    test_size: float,
    layers_config: list[int],
) -> None:
    true_params = [param for param, value in input_params.items() if value]
    print(f"\nNumber of profiles: {len(full_dataset)}")
    print("Parameters used:", ", ".join(true_params))
    print(f"Min depth: {min_depth}, Max depth: {max_depth}")
    print(f"Number of components used: {n_components} x2")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout probability: {dropout_prob}")
    print(f"Train/test/validation split: {train_size}/{test_size}/{val_size}")
    print(f"Layer configuration: {layers_config}\n")
