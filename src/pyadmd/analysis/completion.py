"""Completion verification for pyadmd replica runs, used as a hard gate before analysis."""

import os
from typing import Any, Dict, List, Tuple

from pyadmd.io.dcd import find_last_completed_cycle


def check_pyadmd_completion(params: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """
    Verify that every pyAdMD replica has reached its target cycle count.

    Used by ``analyze -src pyadmd`` as a hard gate before any analysis
    runs: mirrors the assumption the time axis already makes (that
    ``sim_time``/``end_loop`` reflects reality) but makes it an explicit,
    verified precondition instead of a silent one.

    Args:
        params (dict): Parameters dict as loaded from ``pyAdMD_params.json``
            (raw JSON, i.e. ``params['args']`` is a plain dict, as returned
            by ``Analyzer._load_parameters``).

    Returns:
        list[tuple[int, int, int]]: One ``(replica, last_completed_cycle,
            target_cycle)`` tuple per replica that has **not** reached
            ``params['end_loop']``. An empty list means every replica is
            complete.

    Raises:
        ValueError: If ``end_loop`` is missing from ``params`` (e.g. a
            corrupted or pre-existing ``pyAdMD_params.json``), since
            completion cannot be verified without it.
    """
    end_loop = params.get('end_loop')
    if end_loop is None:
        raise ValueError(
            "'end_loop' missing from pyAdMD_params.json; cannot verify "
            "replica completion."
        )

    cwd      = params.get('cwd', os.getcwd())
    args     = params.get('args', {})
    replicas = args.get('replicas', 10)

    incomplete = []
    for rep in range(1, replicas + 1):
        rep_dir    = f"{cwd}/rep{rep}"
        last_cycle = find_last_completed_cycle(rep_dir)
        if last_cycle < end_loop:
            incomplete.append((rep, last_cycle, end_loop))
    return incomplete
