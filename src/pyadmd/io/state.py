"""Engine-agnostic system state: SystemState dataclass and reference-state I/O."""

import functools
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader


# Parsed PSF topologies, keyed by absolute PSF path: {path: (mtime, Topology)}.
# Parsing a solvated PSF costs ~100 MB of transient and partly retained memory,
# and ``make_reference_universe`` is called at every ENM recalculation.
_TOPOLOGY_CACHE: Dict[str, Tuple[float, Any]] = {}


def _patch_pdbwriter_cache() -> bool:
    """
    Stop MDAnalysis' ``PDBWriter`` from keeping every written Universe alive.

    Recent MDAnalysis releases decorate ``PDBWriter._deduce_PDB_atom_name``
    with ``functools.cache``. On an instance method this makes the cache key
    include ``self``, so each writer -- and through it the written AtomGroup
    and its whole Universe -- is retained for the lifetime of the process.
    Every PDB write (``AtomGroup.write``) then leaks a Universe: about
    0.5 MB for a 2,300-atom selection and proportionally more for a full
    system, several times per ENM recalculation.

    The method only reads class-level constants through ``self``, so it is
    replaced by an equivalent one whose cache is keyed on the atom and
    residue names alone. The PDB output is unchanged.

    Returns:
        bool: True if the patch was applied; False if the installed
            MDAnalysis does not cache this method (nothing to fix) or its
            layout differs.

    Note:
        Applied once, when this module is imported. This module is imported by
        every component that writes PDB files (ENM, free energy and the
        simulation runner), so no call site needs to change.
    """
    try:
        from MDAnalysis.coordinates import PDB
        cached = PDB.PDBWriter.__dict__.get('_deduce_PDB_atom_name')
        orig = getattr(cached, '__wrapped__', None)
        if orig is None or not hasattr(cached, 'cache_clear'):
            return False

        @functools.lru_cache(maxsize=None)
        def _deduce(atomname, resname):
            return orig(PDB.PDBWriter, atomname, resname)

        PDB.PDBWriter._deduce_PDB_atom_name = (
            lambda self, atomname, resname: _deduce(atomname, resname)
        )
        return True
    except Exception:
        return False


_patch_pdbwriter_cache()


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

    Note:
        The PSF is parsed only on the first call for a given file; later calls
        build the Universe on the already-parsed topology, so no new topology
        is allocated. The Universes returned therefore share one topology
        object: they are meant for atom selection, masses, positions and
        writing, and topology attributes must not be added or modified on
        them. The cache is refreshed if the PSF file's modification time
        changes.
    """
    # positions_ang must be shaped (1, N, 3) for MemoryReader (1 frame).
    # This creates a topology-only Universe with a synthetic single-frame
    # trajectory, which makes u.atoms.positions immediately accessible
    # without requiring any on-disk trajectory file.
    coords = positions_ang[np.newaxis, :, :].astype(np.float32)
    n_atoms = positions_ang.shape[0]

    key = os.path.abspath(psffile)
    mtime = os.path.getmtime(psffile)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None and cached[0] == mtime:
        return mda.Universe(cached[1], coords, format=MemoryReader, n_atoms=n_atoms)

    u = mda.Universe(psffile, coords, format=MemoryReader, n_atoms=n_atoms)
    _TOPOLOGY_CACHE[key] = (mtime, u._topology)
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
