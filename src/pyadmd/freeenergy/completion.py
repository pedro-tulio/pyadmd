"""Completion verification for freeenergy centroid production, used as a hard gate before analysis."""

import os
from typing import List, Tuple

import pandas as pd


def check_freeenergy_completion(cwd: str) -> List[Tuple[int, int, int]]:
    """
    Verify that every centroid's production MD has reached its target
    cycle count.

    Reads ``freeenergy/clustering_summary.csv`` (written by
    ``FreeEnergyCalculator._save_clustering_summary``), which already
    tracks ``production_cycles_done``/``production_cycles_target`` per
    centroid, so no cycle-count recomputation is needed here.

    Args:
        cwd (str): Working directory containing the ``freeenergy/`` output
            folder (same directory a ``run``/``freeenergy`` call was made
            from).

    Returns:
        list[tuple[int, int, int]]: One ``(centroid_frame,
            production_cycles_done, production_cycles_target)`` tuple per
            centroid that has **not** reached its target. An empty list
            means every centroid is complete.

    Raises:
        FileNotFoundError: If ``freeenergy/clustering_summary.csv`` does
            not exist (no ``freeenergy`` run has completed at all).
    """
    csv_path = f"{cwd}/freeenergy/clustering_summary.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'pyadmd freeenergy' first."
        )

    df = pd.read_csv(csv_path)
    incomplete = []
    for _, row in df.iterrows():
        done   = int(row['production_cycles_done'])
        target = int(row['production_cycles_target'])
        if done < target:
            incomplete.append((int(row['centroid_frame']), done, target))
    return incomplete
