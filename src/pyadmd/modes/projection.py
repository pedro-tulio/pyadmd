"""Shared normal-mode projection utilities.

Extracted from ``pyadmd.fel.calculator.FreeEnergyCalculator`` so the same
mass-weighted Cα-projection math is used by both the free energy landscape
(FEL) protocol and the ``pyadmd analyze`` mode-projection analysis, instead
of two independent implementations drifting apart. Callers pass in the
handful of primitives (psf/coor/rst paths, engine type, mode type) rather
than an object, so either caller can use it without depending on the
other's internal state.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda

from pyadmd.io.state import make_reference_universe


def load_single_mode_vector(input_dir: str, psffile: str, coorfile: Optional[str],
                            rstfile: Optional[str], input_engine: str,
                            nm_type: str, mode_num: int) -> np.ndarray:
    """
    Load a single mode vector file and return per-atom Cartesian
    displacements.

    Resolves the on-disk path exactly as
    ``FreeEnergyCalculator._load_single_mode_vector`` did:
      - ``nm_type == 'charmm'``: ``{input_dir}/mode_nm{mode_num}.crd``
      - otherwise (ENM): ``{input_dir}/{base_name}_enm/{base_name}_{ca|heavy}_mode_{mode_num}.xyz``,
        where ``base_name`` is the coorfile (NAMD) or rstfile (OpenMM)
        basename used at ENM-generation time, falling back to the PSF
        filename stem if neither is available.

    Args:
        input_dir (str): Path to the ``inputs/`` directory.
        psffile (str): Path to the PSF topology file (used for the
            fallback ``base_name`` only).
        coorfile (str, optional): NAMD ``.coor`` path, if the original run
            used NAMD input.
        rstfile (str, optional): OpenMM ``.rst`` path, if the original run
            used OpenMM input.
        input_engine (str): ``'NAMD'`` or ``'OPENMM'`` (case-sensitive,
            matching ``args.source.upper()``).
        nm_type (str): ``'charmm'``, ``'ca'``, or ``'heavy'`` (lowercase).
        mode_num (int): Mode number to load.

    Returns:
        numpy.ndarray: (n_prot_atoms, 3) mode vector positions in Å.

    Raises:
        FileNotFoundError: If the mode vector file does not exist.
    """
    if nm_type == 'charmm':
        path        = f"{input_dir}/mode_nm{mode_num}.crd"
        file_format = "CRD"
    else:
        base_name = os.path.splitext(os.path.basename(psffile))[0]
        if input_engine == 'NAMD' and coorfile:
            base_name = os.path.splitext(os.path.basename(coorfile))[0]
        elif input_engine != 'NAMD' and rstfile:
            base_name = os.path.splitext(os.path.basename(rstfile))[0]
        prefix = "ca" if nm_type == 'ca' else "heavy"
        path   = (f"{input_dir}/{base_name}_enm/"
                  f"{base_name}_{prefix}_mode_{mode_num}.xyz")
        file_format = "XYZ"

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    u = mda.Universe(path, format=file_format)
    return u.atoms.positions.copy()


def get_projection_setup(psffile: str, input_dir: str,
                         ref_positions_ang: Optional[np.ndarray],
                         coorfile: Optional[str], rstfile: Optional[str],
                         input_engine: str, nm_type: str, modes: List[int],
                         console) -> Tuple[np.ndarray, np.ndarray, float,
                                           np.ndarray, Dict[int, np.ndarray]]:
    """
    Resolve Cα selection, reference positions, and normalised mode vectors
    for mode-projection analysis.

    Identical in behavior to
    ``FreeEnergyCalculator._get_projection_setup``, parametrized so callers
    outside ``FreeEnergyCalculator`` (e.g. ``Analyzer``) can use it too.

    Args:
        psffile (str): Path to the PSF topology file.
        input_dir (str): Path to the ``inputs/`` directory.
        ref_positions_ang (numpy.ndarray, optional): Saved reference
            positions in Å (from ``load_reference_state``), shape (N, 3).
            When ``None``, falls back to reading the NAMD coorfile
            directly (only valid when ``input_engine == 'NAMD'``).
        coorfile (str, optional): NAMD ``.coor`` path.
        rstfile (str, optional): OpenMM ``.rst`` path.
        input_engine (str): ``'NAMD'`` or ``'OPENMM'``.
        nm_type (str): ``'charmm'``, ``'ca'``, or ``'heavy'`` (lowercase).
        modes (list[int]): Mode numbers to load and normalise.
        console (ConsoleConfig): Console configuration object for
            formatted output.

    Returns:
        ca_ix_full (numpy.ndarray): (n_ca,) global Cα atom indices in the
            full system.
        ca_masses (numpy.ndarray): (n_ca,) Cα atomic masses in amu.
        M_ca (float): Total Cα mass.
        ref_pos_ca_ang (numpy.ndarray): (n_ca, 3) reference Cα positions
            in Å.
        mode_vectors_ca (dict): {mode_num: (n_ca, 3) Cartesian-normalised
            mode vector}.

    Raises:
        RuntimeError: If no reference positions and no NAMD coorfile are
            available to build the reference structure, or if no valid
            mode vectors could be loaded.
    """
    if ref_positions_ang is not None:
        u_ref = make_reference_universe(psffile, ref_positions_ang)
    elif input_engine == 'NAMD' and coorfile:
        u_ref = mda.Universe(psffile, coorfile, format='NAMDBIN')
    else:
        raise RuntimeError(
            "Cannot load reference positions for mode projection: "
            "saved reference state not found and no NAMD coorfile available. "
            "Re-run 'pyadmd run' with the current version to generate "
            "inputs/init_reference_positions_ang.npy."
        )

    prot_atoms = u_ref.select_atoms("protein")
    ca_atoms   = u_ref.select_atoms("protein and name CA")

    ca_ix_full     = ca_atoms.ix.copy()
    ca_masses      = ca_atoms.masses.copy()
    M_ca           = float(ca_masses.sum())
    ref_pos_ca_ang = ca_atoms.positions.copy()

    # Map global Cα index -> position within protein-only ordering
    prot_to_pos   = {int(gix): pos for pos, gix in enumerate(prot_atoms.ix)}
    ca_ix_in_prot = np.array([prot_to_pos[int(gix)] for gix in ca_ix_full])

    mode_vectors_ca: Dict[int, np.ndarray] = {}
    for mode_num in modes:
        try:
            vec_full = load_single_mode_vector(
                input_dir, psffile, coorfile, rstfile, input_engine,
                nm_type, mode_num,
            )   # (n_prot, 3) Å

            # Guard against an atom-count/order mismatch.
            if vec_full.shape[0] != len(prot_atoms):
                print(f"{console.PGM_WRN}Mode {console.WRN}{mode_num}{console.STD} "
                      f"vector has {console.WRN}{vec_full.shape[0]}{console.STD} atoms, "
                      f"expected {console.EXT}{len(prot_atoms)}{console.STD} protein "
                      "atoms; skipping.")
                continue

            q_ca = vec_full[ca_ix_in_prot]                  # (n_ca, 3)
            norm = np.linalg.norm(q_ca)
            if norm < 1e-10:
                print(f"{console.PGM_WRN}Mode {console.WRN}{mode_num}{console.STD} Cα vector is "
                      "near-zero after extraction; skipping.")
                continue
            mode_vectors_ca[mode_num] = q_ca / norm
        except FileNotFoundError as exc:
            print(f"{console.PGM_WRN}Mode file not found for mode "
                  f"{console.WRN}{mode_num}{console.STD}: {console.WRN}{exc}{console.STD}. Skipping.")

    if not mode_vectors_ca:
        raise RuntimeError("No valid mode vectors could be loaded for mode projection.")

    return ca_ix_full, ca_masses, M_ca, ref_pos_ca_ang, mode_vectors_ca


def compute_mode_projections_df(psffile: str,
                                unit_dcd_pairs: List[Tuple[int, Optional[str]]],
                                unit_col: str, cycle_ps: float,
                                ca_ix_full: np.ndarray, ca_masses: np.ndarray,
                                M_ca: float, ref_pos_ca_ang: np.ndarray,
                                mode_vectors_ca: Dict[int, np.ndarray]) -> pd.DataFrame:
    """
    Compute signed MRMS displacement of every trajectory frame along each
    individual mode vector, for one or more units (replicas or fel
    centroids), as a single combined DataFrame.

    Same formula as ``FreeEnergyCalculator.compute_mode_projections``:

        d_j = (1/√M) Σ_i √m_i · (r_i − r₀ᵢ) · q_{ij}

    Sign is preserved so downstream plots/histograms can distinguish both
    directions. Unlike the FEL-only predecessor (which returned flat
    per-mode arrays across all units concatenated), this returns one row
    per (unit, frame) so results can be traced back to their source unit
    and per-frame timestamp.

    Args:
        psffile (str): Path to the PSF topology file.
        unit_dcd_pairs (list[tuple[int, str or None]]): One
            ``(unit_id, dcd_path)`` pair per unit to project. ``dcd_path``
            entries that are ``None`` or point to a missing file are
            skipped (e.g. a failed FEL centroid).
        unit_col (str): Column name for the unit identifier
            (``'replica'`` or ``'centroid_frame'``).
        cycle_ps (float): Elapsed time per trajectory frame in
            picoseconds, used to compute each frame's ``time`` value as
            ``frame_index * cycle_ps`` (0-based within each unit's own
            trajectory).
        ca_ix_full (numpy.ndarray): (n_ca,) global Cα atom indices.
        ca_masses (numpy.ndarray): (n_ca,) Cα atomic masses in amu.
        M_ca (float): Total Cα mass.
        ref_pos_ca_ang (numpy.ndarray): (n_ca, 3) reference Cα positions
            in Å.
        mode_vectors_ca (dict): {mode_num: (n_ca, 3) normalised mode
            vector}, as returned by ``get_projection_setup``.

    Returns:
        pandas.DataFrame: Columns ``[unit_col, 'time', 'mode_{n1}',
            'mode_{n2}', ...]`` (mode columns sorted ascending by mode
            number), one row per analyzed frame across all units. Empty
            (with the correct columns) if no unit had a usable DCD.
    """
    sqrt_M_ca   = float(np.sqrt(M_ca))
    sqrt_masses = np.sqrt(ca_masses)          # (n_ca,) pre-computed
    mode_nums   = sorted(mode_vectors_ca.keys())
    columns     = [unit_col, 'time'] + [f'mode_{m}' for m in mode_nums]

    rows: List[Dict[str, float]] = []
    for unit_id, dcd_file in unit_dcd_pairs:
        if dcd_file is None or not os.path.exists(dcd_file):
            continue
        u = mda.Universe(psffile, dcd_file, format="DCD")
        for i, ts in enumerate(u.trajectory):
            curr_ca = u.atoms.positions[ca_ix_full]        # (n_ca, 3) Å
            disp    = curr_ca - ref_pos_ca_ang              # (n_ca, 3) Å
            mw_disp = (disp.T * sqrt_masses).T             # mass-weighted

            row: Dict[str, float] = {unit_col: unit_id, 'time': i * cycle_ps}
            for mode_num in mode_nums:
                q_ca = mode_vectors_ca[mode_num]
                mrms = float(np.sum(mw_disp * q_ca)) / sqrt_M_ca
                row[f'mode_{mode_num}'] = mrms
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    return df[columns]
