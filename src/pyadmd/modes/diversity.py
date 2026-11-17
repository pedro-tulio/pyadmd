"""Optional transverse diversification (beta) on top of alpha harmonization.

Used only when ``--subspace-beta`` is positive. With beta = 0 the runner calls
``pyadmd.modes.subspace.harmonize_and_transport`` directly and this module is
not touched.

Frame convention: all inputs are in the frame of the stored reference
(``old_coords``) and the current MD frame (``new_coords``). Every returned
vector (excitation direction and harmonized active basis) is rotated back into
the current MD frame, so the stored basis and its reference positions always
share one frame.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pyadmd.modes.subspace import (
    SubspaceResult,
    harmonize_and_transport,
    kabsch_rotation,
)


@dataclass
class DiversityResult:
    """
    Diagnostics of one beta update.

    Attributes:
        transport (SubspaceResult): Alpha-harmonized result, already rotated
            into the current MD frame (``q_new``, ``harmonized_modes``).
        reservoir_modes (np.ndarray): Transported full reservoir, shape (M, n, 3),
            current MD frame.
        excluded (np.ndarray): Reservoir indices excluded for overlap with the
            active subspace.
        selected (np.ndarray): Reservoir indices used for the supplementary mixture.
        coefficients (np.ndarray): Equal-magnitude coefficients b of the selected
            directions (sum of squares = 1).
        projections (np.ndarray): Cosines of each reservoir direction with q_c.
        transverse (np.ndarray): Transverse separations sqrt(1 - p^2).
        stiffness (np.ndarray): Effective stiffness of each transported direction.
        scores (np.ndarray): Selection scores.
        overlap_active (np.ndarray): Squared overlap of each direction with the
            active subspace.
        reservoir_coefficients (np.ndarray): Triangular-transform coefficients C
            with W = E C (raw new-mode coordinates).
        reservoir_singular_values (np.ndarray): Continuity singular values of the
            full reservoir.
        mixture_stiffness (float): Effective stiffness of the supplementary mixture.
        mixture_cross_term (float): Off-diagonal contribution to that stiffness.
        cosine_continuity (float): cos(Q_new, Q_c).
        deviation_deg (float): Angle between Q_new and Q_c, degrees.
        modal_coefficients (np.ndarray): Coefficients of Q_new in the new modal basis.
        modal_participation (float): Inverse participation ratio of the modal coefficients.
        collectivity (float): Atomic collectivity of the final direction.
    """
    transport: SubspaceResult
    reservoir_modes: np.ndarray
    excluded: np.ndarray
    selected: np.ndarray
    coefficients: np.ndarray
    projections: np.ndarray
    transverse: np.ndarray
    stiffness: np.ndarray
    scores: np.ndarray
    overlap_active: np.ndarray
    reservoir_coefficients: np.ndarray
    reservoir_singular_values: np.ndarray
    mixture_stiffness: float
    mixture_cross_term: float
    cosine_continuity: float
    deviation_deg: float
    modal_coefficients: np.ndarray
    modal_participation: float
    collectivity: float


def validate_diversity(alpha: float, beta: float, active: list[int],
                       search: list[int], beta_count: int = 3) -> None:
    """
    Validate beta-related options before any simulation starts.

    Args:
        alpha (float): Alpha mixing factor.
        beta (float): Beta amplitude; 0 disables diversification and skips the
            remaining checks.
        active (list[int]): Active mode labels (dimension m).
        search (list[int]): Search (reservoir) mode labels (dimension M).
        beta_count (int): Number of supplementary directions B.

    Raises:
        ValueError: On any invalid combination.
    """
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("--subspace-beta must be finite and non-negative")
    if (isinstance(beta_count, bool) or not isinstance(beta_count, Integral)
            or beta_count < 1):
        raise ValueError("--subspace-beta-count must be a positive integer")
    if beta == 0:
        return
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("--subspace-alpha must be finite and between 0 and 1")
    if not active or len(set(active)) != len(active):
        raise ValueError("diversification requires a nonempty list of distinct active modes")
    if len(set(search)) != len(search):
        raise ValueError("diversification requires distinct search modes")
    required = len(active) + beta_count
    if len(search) < required:
        raise ValueError(
            f"diversification requires at least {required} search modes "
            f"({len(active)} active + {beta_count} beta); got {len(search)}. "
            "Expand --subspace-search-modes or reduce -nm or --subspace-beta-count.")
    if not set(active).issubset(search) or min(search) < 7:
        raise ValueError("search modes must contain the active modes and exclude modes 1-6")


def _basis(modes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduced QR of flattened mode vectors.

    Args:
        modes (np.ndarray): Shape (k, n_atoms, 3), finite, full column rank.

    Returns:
        tuple: (Q, R) with Q of shape (3 n_atoms, k) orthonormal and R (k, k).

    Raises:
        ValueError: If the shape is wrong, values are non-finite, or the
            columns are linearly dependent.
    """
    a = np.asarray(modes, dtype=float)
    if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
        raise ValueError("basis must be finite with shape (k, n_atoms, 3)")
    a = a.reshape(a.shape[0], -1).T
    if a.shape[1] > a.shape[0] or np.linalg.matrix_rank(a) != a.shape[1]:
        raise ValueError("basis must have full column rank")
    return np.linalg.qr(a, mode="reduced")


def _rank(values: np.ndarray, candidates: np.ndarray, count: int) -> np.ndarray:
    """
    Indices of the ``count`` largest values among ``candidates``.

    Values are compared after rounding to 1e-12 relative to the largest
    magnitude; ties keep the persistent reservoir order (lowest index first).
    """
    scale = max(float(np.max(np.abs(values[candidates]))), 1e-300)
    rounded = np.round(values[candidates] / scale, 12)
    return candidates[np.lexsort((candidates, -rounded))[:count]]


def harmonize_with_diversity(old_modes: np.ndarray, new_modes: np.ndarray,
                             q_current: np.ndarray, old_coords: np.ndarray,
                             new_coords: np.ndarray, masses: np.ndarray,
                             alpha: float, beta: float,
                             old_reservoir: np.ndarray,
                             modal_stiffness: np.ndarray,
                             beta_count: int = 3) -> DiversityResult:
    """
    Add ``beta_count`` transverse, soft directions to the alpha-corrected direction.

    The active subspace is transported by ``harmonize_and_transport``. A full
    reservoir of M directions is transported separately by a square Procrustes
    alignment against ``old_reservoir``. The m directions with the largest
    overlap with the active subspace are excluded; the ``beta_count`` largest
    remaining scores are combined with equal absolute coefficients, signed to
    point toward the alpha direction.

    Args:
        old_modes (np.ndarray): Stored active basis U_m, shape (m, n, 3), in the
            frame of ``old_coords``.
        new_modes (np.ndarray): Newly computed search modes V_M, shape (M, n, 3),
            computed on ``new_coords``.
        q_current (np.ndarray): Corrected excitation direction Q_corr, shape (n, 3),
            in the current MD frame.
        old_coords (np.ndarray): Reference structure of the stored basis, (n, 3), Å.
        new_coords (np.ndarray): Current structure, (n, 3), Å.
        masses (np.ndarray): Per-atom masses, (n,), amu.
        alpha (float): Alpha mixing factor in [0, 1].
        beta (float): Beta amplitude, finite and > 0. Use
            ``harmonize_and_transport`` for beta = 0.
        old_reservoir (np.ndarray): Stored full reservoir, shape (M, n, 3).
        modal_stiffness (np.ndarray): Symmetric M x M matrix E^T D E for the new
            input basis E (not a diagonal approximation).
        beta_count (int): Number of supplementary directions B (>= 1).

    Returns:
        DiversityResult: Excitation direction, active basis, reservoir and
            diagnostics, all in the current MD frame.

    Raises:
        ValueError: On invalid inputs, insufficient reservoir size, unstable or
            near-singular stiffness, or too few meaningful candidates.
        RuntimeError: If the reservoir loses orthonormality or the update
            reverses the continuity direction.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("harmonize_with_diversity requires finite beta > 0; "
                         "use harmonize_and_transport for beta = 0")
    if (isinstance(beta_count, bool) or not isinstance(beta_count, Integral)
            or beta_count < 1):
        raise ValueError("beta_count must be a positive integer")

    old_modes = np.asarray(old_modes, dtype=float)
    new_modes = np.asarray(new_modes, dtype=float)
    old_reservoir = np.asarray(old_reservoir, dtype=float)
    m, n, _ = old_modes.shape
    M = new_modes.shape[0]

    if M < m + beta_count:
        raise ValueError(f"reservoir too small: need M >= m + B = {m + beta_count}, got {M}")
    if old_reservoir.shape != new_modes.shape:
        raise ValueError("old and new reservoirs must have identical dimensions")

    q = np.asarray(q_current, dtype=float)
    if q.shape != (n, 3) or not np.isfinite(q).all():
        raise ValueError("Q must be finite with shape (n_atoms, 3)")
    masses = np.asarray(masses, dtype=float)
    if masses.shape != (n,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("masses must be finite and positive, one per atom")
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-14:
        raise ValueError("Q has zero norm")

    # Rotation of the new structure onto the stored reference (same as the alpha step).
    r = kabsch_rotation(new_coords, old_coords, masses)

    # Alpha step. Q is passed in the reference frame; the returned direction is
    # rotated back to the MD frame below.
    active = harmonize_and_transport(
        old_modes, new_modes, (q / q_norm) @ r, old_coords, new_coords,
        masses, alpha,
    )
    # Still in the reference frame here (rotated back at the end of this function).
    qc = active.q_new.reshape(-1)

    # Full reservoir: new modes rotated into the reference frame, then QR.
    v, triangular = _basis(new_modes @ r)
    a, _ = _basis(old_reservoir)
    left, sigma, right_t = np.linalg.svd(a.T @ v, full_matrices=False)
    orientation = right_t.T @ left.T              # (M, M), orthogonal
    w = v @ orientation                           # (3n, M), reference frame
    c = np.linalg.solve(triangular, orientation)  # W = E C, raw new-mode coords

    K = np.asarray(modal_stiffness, dtype=float)
    if K.shape != (M, M) or not np.isfinite(K).all():
        raise ValueError("modal stiffness must be a finite M x M matrix")
    if not np.allclose(K, K.T, atol=1e-12, rtol=1e-10):
        raise ValueError("modal stiffness must be symmetric")
    eigenvalues = np.linalg.eigvalsh(K)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-8 * eigenvalues[-1]:
        raise ValueError("reservoir contains unstable or near-zero stiffness "
                         "(relative threshold 1e-8)")

    g = c.T @ K @ c
    kappa = np.diag(g).copy()
    reference_stiffness = float(eigenvalues[0])

    proj = w.T @ qc
    transverse = np.sqrt(np.maximum(0.0, 1.0 - np.clip(proj, -1.0, 1.0) ** 2))
    scores = transverse * np.sqrt(reference_stiffness / kappa)

    h = active.harmonized_modes.reshape(m, -1).T  # reference frame
    overlap = np.sum((h.T @ w) ** 2, axis=0)
    all_ix = np.arange(M)
    excluded = _rank(overlap, all_ix, m)
    candidates = np.setdiff1d(all_ix, excluded)
    if np.count_nonzero(scores[candidates] > 1e-12) < beta_count:
        raise ValueError(
            f"fewer than {beta_count} meaningful transverse candidates after excluding "
            f"{m} active-associated directions from {M} reservoir modes; use a larger "
            "reservoir or fewer beta directions in a fresh run.")
    selected = _rank(scores, candidates, beta_count)

    b = np.full(beta_count, 1.0 / np.sqrt(beta_count))
    b[proj[selected] < -1e-12] *= -1.0
    supplementary = w[:, selected] @ b            # whole vectors, unit norm
    if not np.isclose(np.linalg.norm(supplementary), 1.0, atol=1e-10):
        raise RuntimeError("reservoir lost orthonormality")

    raw = qc + beta * supplementary if beta <= 1 else qc / beta + supplementary
    q_ref = raw / np.linalg.norm(raw)
    cosine = float(np.clip(q_ref @ qc, -1.0, 1.0))
    if cosine <= 0.0:
        raise RuntimeError("diversification reversed the continuity direction")

    g_sel = g[np.ix_(selected, selected)]
    mixture_stiffness = float(b @ g_sel @ b)
    mixture_cross = mixture_stiffness - float(np.sum(b ** 2 * np.diag(g_sel)))

    modal = np.linalg.solve(triangular, v.T @ q_ref)
    power = modal ** 2
    participation = float(power.sum() ** 2 / np.sum(power ** 2)) if power.sum() > 0 else 0.0

    # Rotate everything back into the current MD frame (U_m frame correction).
    back = r.T
    q_new = q_ref.reshape(n, 3) @ back
    active.q_new = q_new
    active.harmonized_modes = active.harmonized_modes @ back
    reservoir_out = w.T.reshape(M, n, 3) @ back

    displacement = q_new / np.sqrt(masses)[:, None]
    atom_weight = np.sum(displacement ** 2, axis=1)
    atom_weight = atom_weight / atom_weight.sum()
    positive = atom_weight > 0
    collectivity = float(
        np.exp(-np.sum(atom_weight[positive] * np.log(atom_weight[positive]))) / n)

    return DiversityResult(
        transport=active,
        reservoir_modes=reservoir_out,
        excluded=excluded,
        selected=selected,
        coefficients=b,
        projections=proj,
        transverse=transverse,
        stiffness=kappa,
        scores=scores,
        overlap_active=overlap,
        reservoir_coefficients=c,
        reservoir_singular_values=np.clip(sigma, 0.0, 1.0),
        mixture_stiffness=mixture_stiffness,
        mixture_cross_term=mixture_cross,
        cosine_continuity=cosine,
        deviation_deg=float(np.degrees(np.arccos(cosine))),
        modal_coefficients=modal,
        modal_participation=participation,
        collectivity=collectivity,
    )


def write_diversity_spectrum(calc, positions_ang: np.ndarray, base_name: str,
                             nm_type: str, search_modes: list[int],
                             input_dir: str, psffile: str,
                             cutoff=None, spring_constant: float = 1.0,
                             max_modes=None, rtb: bool = False) -> str:
    """
    Compute the ENM spectrum for a beta reservoir and store it as an NPZ file.

    Performs the same pipeline as ``ENMCalculator.compute_enm`` (reference
    PDB, system, Hessian, mass weighting, diagonalization), but instead of
    writing XYZ/trajectory files it stores the reservoir modes (in the
    representative-atom basis), the modal stiffness matrix E^T D E, and the
    mapping of representative atoms to input PDB indices. The output path is
    ``{input_dir}/{base_name}_enm/{base_name}_{prefix}_diversity_spectrum.npz``,
    the path read by ``SimulationRunner._read_diversity_spectrum``.

    Args:
        calc (ENMCalculator): Calculator instance (same as the runner's).
        positions_ang (np.ndarray): Current positions in Å, shape (N, 3).
        base_name (str): Base name of the ENM run (e.g. ``step_120``).
        nm_type (str): ``'ca'`` or ``'heavy'``.
        search_modes (list[int]): Reservoir mode labels (column indices).
        input_dir (str): Directory holding ``{base_name}_enm/``.
        psffile (str): PSF topology path.
        cutoff (float, optional): Forwarded to ``create_system``.
        spring_constant (float): Forwarded to ``create_system``.
        max_modes (int, optional): Forwarded to the diagonalization.
        rtb (bool): Use the RTB reduction, as ``compute_enm`` does.

    Returns:
        str: Path of the written NPZ file.

    Raises:
        ValueError: If a label is below 7 or beyond the computed spectrum, or
            if the representative-atom mapping is inconsistent.
    """
    from openmm import app, unit
    from pyadmd.enm.rtb import build_blocks, project_and_diagonalize
    from pyadmd.io.state import make_reference_universe
    from pyadmd.simulation.system_builder import OpenMMSystemBuilder

    search_modes = [int(m) for m in search_modes]
    if min(search_modes) < 7:
        raise ValueError("reservoir labels must be >= 7 (rigid-body modes excluded)")

    output_folder = os.path.join(input_dir, f"{base_name}_enm")
    os.makedirs(output_folder, exist_ok=True)
    output_prefix = os.path.join(output_folder, base_name)
    prefix = "ca" if nm_type.lower() == "ca" else "heavy"

    prot = make_reference_universe(psffile, positions_ang).select_atoms("protein or nucleic")
    pdb_file = f"{output_prefix}.pdb"
    prot.write(pdb_file, file_format="PDB")

    system, topology, positions = calc.create_system(
        pdb_file, model_type=nm_type, cutoff=cutoff,
        output_prefix=output_prefix, spring_constant=spring_constant,
    )
    hessian = calc.hessian_enm(system, positions)
    mw_hessian = calc.mass_weight_hessian(hessian, system)

    n_particles = system.getNumParticles()
    if rtb:
        positions_nm = np.array(
            [[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)],
            dtype=np.float64)
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])
        masses[masses == 0] = 1.0
        _, enm, _ = project_and_diagonalize(
            mw_hessian, positions_nm, masses, build_blocks(topology),
            diagonalize_fn=calc.compute_normal_modes,
            n_modes=max_modes, use_gpu=True, console=calc.console)
    else:
        _, enm, _ = calc.compute_normal_modes(mw_hessian, n_modes=max_modes, use_gpu=True)

    columns = np.asarray(search_modes, dtype=int)
    if np.any(columns >= enm.shape[1]):
        raise ValueError("reservoir exceeds the computed internal spectrum; "
                         "increase max_modes")
    E = enm[:, columns]
    K = E.T @ (mw_hessian @ E)

    original = app.PDBFile(pdb_file)
    atoms = list(original.topology.atoms())
    if prefix == "heavy":
        atom_input_indices = np.array([a.index for a in atoms
                                       if a.element != app.element.hydrogen])
    else:
        atom_input_indices = np.array([
            a.index for a in atoms
            if (a.name in ("P", "C1'", "C2")
                if a.residue.name.upper() in OpenMMSystemBuilder.NUCLEIC_RESIDUES
                else a.name == "CA")])
    if len(atom_input_indices) != n_particles:
        raise ValueError("ENM representative atom mapping mismatch")

    masses_out = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])
    out_path = f"{output_prefix}_{prefix}_diversity_spectrum.npz"
    np.savez(out_path,
             schema=np.array(1),
             mode_numbers=columns,
             column_indices=columns,
             modes=E.T.reshape(len(columns), n_particles, 3),
             stiffness=(K + K.T) / 2.0,
             atom_input_indices=atom_input_indices,
             enm_masses=masses_out)
    return out_path
