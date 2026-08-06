"""Completion verification for fel centroid production, used as a hard gate before analysis."""

import os
from typing import List, Tuple

import pandas as pd


def check_fel_completion(cwd: str) -> List[Tuple[int, int, int]]:
    """
    Verify that every centroid's production MD has reached its target
    cycle count.

    Reads ``fel/clustering_summary.csv`` (written by
    ``FreeEnergyCalculator._save_clustering_summary``), which already
    tracks ``production_cycles_done``/``production_cycles_target`` per
    centroid, so no cycle-count recomputation is needed here.

    Args:
        cwd (str): Working directory containing the ``fel/`` output
            folder (same directory a ``run``/``fel`` call was made
            from).

    Returns:
        list[tuple[int, int, int]]: One ``(centroid_frame,
            production_cycles_done, production_cycles_target)`` tuple per
            centroid that has **not** reached its target. An empty list
            means every centroid is complete.

    Raises:
        FileNotFoundError: If ``fel/clustering_summary.csv`` does
            not exist (no ``fel`` run has completed at all).
    """
    csv_path = f"{cwd}/fel/clustering_summary.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'pyadmd fel' first."
        )

    df = pd.read_csv(csv_path)
    incomplete = []
    for _, row in df.iterrows():
        done   = int(row['production_cycles_done'])
        target = int(row['production_cycles_target'])
        if done < target:
            incomplete.append((int(row['centroid_frame']), done, target))
    return incomplete
