"""Engine-agnostic system state: SystemState dataclass and reference-state I/O."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader


@dataclass
class SystemState:
    """Container for the system state converted from NAMD input files."""
    positions_nm:     np.ndarray        # (N, 3), nm
    velocities_nm_ps: np.ndarray        # (N, 3), nm/ps
    box_vectors_nm:   List[np.ndarray]  # [a, b, c], each (3,) nm


def make_reference_universe(psffile: str, positions_ang: np.ndarray) -> mda.Universe:
    """
    Build an MDAnalysis Universe from a PSF topology and a positions array.

    Replaces the pattern ``mda.Universe(psffile, coorfile, format='NAMDBIN')``
    used throughout the codebase solely to obtain topology + real coordinates.
    Works regardless of which input engine (NAMD or OpenMM) produced the run.

    Args:
        psffile (str): Path to the PSF topology file.
        positions_ang (np.ndarray): Atomic positions in Å, shape (N, 3).

    Returns:
        mda.Universe: Universe with ``atoms.positions`` set to ``positions_ang``.
    """
    # positions_ang must be shaped (1, N, 3) for MemoryReader (1 frame).
    # This creates a topology-only Universe with a synthetic single-frame
    # trajectory, which makes u.atoms.positions immediately accessible
    # without requiring any on-disk trajectory file.
    u = mda.Universe(psffile,
                     positions_ang[np.newaxis, :, :].astype(np.float32),
                     format=MemoryReader,
                     n_atoms=positions_ang.shape[0])
    return u


def save_reference_state(input_dir: str, positions_ang: np.ndarray, box_vectors_nm: List[np.ndarray], velocities_nm_ps: Optional[np.ndarray] = None) -> None:
    """
    Persist the initial reference positions, box vectors, and velocities to
    ``inputs/``.

    Written once at ``run`` time so that ``freeenergy`` and any future
    analysis subcommands can recover the reference state without re-reading
    engine-specific input files. Persisting velocities is required so that
    ``restart``/``append`` can correctly initialise replicas that had not yet
    started a single cycle (no OpenMM checkpoint to load velocities from).

    Files written:
        ``{input_dir}/init_reference_positions_ang.npy``
            Shape (N, 3), float64, Å.
        ``{input_dir}/init_reference_box_nm.npy``
            Shape (3, 3), float64, nm (rows = a, b, c vectors).
        ``{input_dir}/init_reference_velocities_nm_ps.npy``
            Shape (N, 3), float64, nm/ps. Only written if ``velocities_nm_ps``
            is provided (kept optional for backward compatibility).

    Args:
        input_dir (str): Path to the ``inputs/`` directory.
        positions_ang (np.ndarray): Atomic positions in Å, shape (N, 3).
        box_vectors_nm (list of np.ndarray): Three box vectors in nm.
        velocities_nm_ps (np.ndarray, optional): Atomic velocities in nm/ps,
            shape (N, 3).
    """
    np.save(os.path.join(input_dir, 'init_reference_positions_ang.npy'),
            positions_ang.astype(np.float64))
    np.save(os.path.join(input_dir, 'init_reference_box_nm.npy'),
            np.array(box_vectors_nm, dtype=np.float64))
    if velocities_nm_ps is not None:
        np.save(os.path.join(input_dir, 'init_reference_velocities_nm_ps.npy'),
                velocities_nm_ps.astype(np.float64))


def load_reference_state(input_dir: str) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """
    Load the saved reference positions, box vectors, and velocities from
    ``inputs/``.

    Args:
        input_dir (str): Path to the ``inputs/`` directory.

    Returns:
        positions_ang (np.ndarray): Shape (N, 3), float64, Å.
        box_vectors_nm (list of np.ndarray): Three (3,) arrays in nm.
        velocities_nm_ps (np.ndarray or None): Shape (N, 3), float64, nm/ps.
            ``None`` if the file predates this feature (older pyAdMD run),
            in which case callers must fall back to re-reading engine-specific
            input files before starting any not-yet-begun replica.

    Raises:
        FileNotFoundError: If the position or box .npy file is missing (run
            not yet completed or produced by a very old version of pyAdMD).
    """
    pos_path = os.path.join(input_dir, 'init_reference_positions_ang.npy')
    box_path = os.path.join(input_dir, 'init_reference_box_nm.npy')
    vel_path = os.path.join(input_dir, 'init_reference_velocities_nm_ps.npy')
    for p in (pos_path, box_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Reference state file not found: {p}\n"
                "Re-run 'pyadmd run' with the current version to generate it."
            )
    positions_ang  = np.load(pos_path)
    box_arr        = np.load(box_path)   # (3, 3)
    box_vectors_nm = [box_arr[0], box_arr[1], box_arr[2]]
    velocities_nm_ps = np.load(vel_path) if os.path.exists(vel_path) else None
    return positions_ang, box_vectors_nm, velocities_nm_ps
