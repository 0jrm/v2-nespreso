"""Train/val/test profiles-per-month stacked bar chart (hoisted from monolith __main__)."""

from __future__ import annotations

import calendar

import matplotlib.pyplot as plt
import pandas as pd

from nespreso.analysis.monthly import count_profiles_per_month
from nespreso.experiments.validation_context import ValidationContext


def run_monthly_distribution(ctx: ValidationContext) -> None:
    """Make a bar plot showing how many profiles are in train/val/test per month."""
    train_dataset = ctx.train_dataset
    val_dataset = ctx.val_dataset
    test_dataset = ctx.test_dataset
    train_indices = ctx.train_indices
    val_indices = ctx.val_indices
    test_indices = ctx.test_indices

    train_counts = count_profiles_per_month(train_dataset.dataset, train_indices)
    val_counts = count_profiles_per_month(val_dataset.dataset, val_indices)
    test_counts = count_profiles_per_month(test_dataset.dataset, test_indices)

    # Combine all dates and get unique months
    all_months = sorted(set(train_counts.index) | set(val_counts.index) | set(test_counts.index))

    # Combine all counts into a single DataFrame
    df = pd.DataFrame({"Train": train_counts, "Validation": val_counts, "Test": test_counts})

    # Calculate the total number of profiles for each month
    df_total = df.sum(axis=1)
    # Calculate the percentage for each dataset
    df_percentage = df.div(df_total, axis=0) * 100

    # Update the index to display month abbreviations
    df_percentage.index = [calendar.month_abbr[i] for i in df_percentage.index]

    # Plot
    ax = df_percentage.plot(kind="bar", stacked=True, figsize=(15, 6), width=0.8)
    plt.title("Profiles per Month")
    plt.xlabel("Month")
    plt.ylabel("%")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncols=3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")

    # Add total number labels on top of each bar
    for i, total in enumerate(df_total):
        ax.text(
            i,
            0,
            f"Total:\n{total:,.0f}",
            ha="center",
            va="bottom",
        )

    # Add percentage labels on each bar segment
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="center")

    # Set y-axis to show percentages from 0 to 100
    plt.ylim(0, 100)  # Increase to 105 to accommodate total labels?

    plt.tight_layout()
    plt.show()
