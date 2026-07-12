"""Pure geometry helpers with no I/O or simulation-engine dependencies."""

import numpy as np
from MDAnalysis.analysis.align import rotation_matrix as mda_rotation_matrix


def _kabsch_align(mob: np.ndarray, ref: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Perform mass-weighted Kabsch (rigid-body) alignment of mobile onto reference structure.

    Computes the optimal rotation matrix that minimizes the mass-weighted root-mean-square
    deviation (RMSD) between two sets of coordinates. Uses pure numpy and MDAnalysis
    rotation_matrix function; no Universe objects required.

    Args:
        mob (np.ndarray): Mobile structure coordinates, shape (n, 3), in Ångströms.
        ref (np.ndarray): Reference structure coordinates, shape (n, 3), in Ångströms.
        masses (np.ndarray): Per-atom masses for weighting, shape (n,), in atomic mass units.

    Returns:
        aligned_mob (np.ndarray): Rotated and translated mobile coordinates superposed
            onto the reference, shape (n, 3), in Ångströms.

    Note:
        Both structures are first translated to their respective mass-weighted centers
        of mass before rotation. The final result is then translated to the reference
        center of mass.
    """
    w = masses / masses.sum()
    ref_com = w @ ref
    mob_com = w @ mob
    R, _ = mda_rotation_matrix(mob - mob_com, ref - ref_com, weights=masses)
    return (mob - mob_com) @ R.T + ref_com
