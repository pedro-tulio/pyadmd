"""Rotations-Translations of Blocks (RTB) coarse-graining for ENM normal-mode diagonalization.

RTB is not a separate force field: it is an alternate diagonalization backend
for the same mass-weighted ENM Hessian already built by ``ENMCalculator``.
Atoms are grouped into rigid blocks (one residue per block), the mass-weighted
Hessian is projected onto the 6-DOF-per-block subspace, the reduced Hessian is
diagonalized, and the resulting eigenvectors are projected back to full
Cartesian space. The output convention (mass-weighted "tilde"-space
eigenvectors, raw + filtered eigenvalue arrays) exactly matches
``ENMCalculator.compute_normal_modes``'s own return contract, so everything
downstream works unmodified regardless of whether RTB was used.

Block size is fixed at one residue per block. Implications, stated
explicitly:

  - HEAVY model: each 1-residue block contains multiple heavy atoms
    (backbone + sidechain, protein or nucleic), so each block genuinely
    contributes up to 6 DOF (a reduction from 3*n_heavy_atoms down to
    ~6*n_residues).
  - Cα model, protein residues: each block is a single Cα point, which
    has no meaningful rotation about itself (``--rtb`` combined with a
    protein-only Cα system is mathematically equivalent to plain Cα).
  - Cα model, nucleic acid residues: ``ENMCalculator`` represents each
    nucleotide with a three-bead SBP block (P, C1', C2), which spans a
    genuine rigid-body subspace and contributes up to 6 DOF per residue.
    A protein-nucleic complex run under the Cα model therefore *does*
    see a real reduction on its nucleic portion, even though its protein
    portion does not.

References:
    Durand, Trinquier & Sanejouand, Biopolymers 1994, 34:759-771.
    Tama, Gadea, Marques & Sanejouand, Proteins 2000, 41:1-7.
"""

from typing import Any, Callable, List, Tuple

import numpy as np
from scipy.sparse import coo_array, issparse, csr_array


def build_blocks(topology: Any) -> List[List[int]]:
    """
    Partition an OpenMM topology's atoms into per-residue blocks.

    Blocks are formed by grouping consecutive atoms sharing the same
    residue, in the order ``topology.atoms()`` yields them. Fixed at one
    residue per block (see module docstring).

    Args:
        topology (openmm.app.Topology): Reduced-resolution ENM topology
            (Cα-only or heavy-atom-only), as built by
            ``ENMCalculator._create_ca_system``/``_create_heavy_system``.

    Returns:
        list[list[int]]: One list of (0-based, topology-local) atom
            indices per block, in residue order. Never empty for a
            non-empty topology.
    """
    blocks: List[List[int]] = []
    current_block: List[int] = []
    current_residue = None

    for atom in topology.atoms():
        if atom.residue is not current_residue:
            if current_block:
                blocks.append(current_block)
            current_block = []
            current_residue = atom.residue
        current_block.append(atom.index)

    if current_block:
        blocks.append(current_block)

    return blocks


def build_block_projection(positions_nm: np.ndarray, masses: np.ndarray,
                           blocks: List[List[int]],
                           drop_tol: float = 1e-8) -> Tuple[csr_array, List[int]]:
    """
    Build the orthonormal block-rigid-body projection matrix P.

    For each block, up to 6 generator vectors are built directly in
    mass-weighted ("tilde") Cartesian space:
      - 3 translations:  sqrt(m_i) * e_k,                restricted to the block's atoms.
      - 3 rotations:     sqrt(m_i) * (r_i - r_cm) x e_k,  restricted to the block's atoms,
                          r_cm = the block's mass-weighted center of mass.
    Each block's 6 candidate generators are Gram-Schmidt-orthonormalized
    independently; any generator whose residual norm falls below
    ``drop_tol`` after orthogonalization against earlier generators in
    the same block is dropped (handles single-atom blocks, which have no
    rotational DOF, and 2-atom blocks, which have no meaningful rotation
    about their own long axis). Blocks never share atoms, so generators
    from different blocks are automatically mutually orthogonal --  no
    global orthogonalization step is needed.

    Args:
        positions_nm (np.ndarray): Block-atom equilibrium positions,
            shape (N, 3), in nm (or any consistent length unit -- only
            relative positions within each block matter).
        masses (np.ndarray): Per-atom masses, shape (N,), in amu.
        blocks (list[list[int]]): Per-block atom index lists, as returned
            by ``build_blocks``. Indices must be valid row indices into
            ``positions_nm``/``masses``.
        drop_tol (float): Norm threshold below which a candidate
            generator (after Gram-Schmidt orthogonalization against
            earlier generators in the same block) is discarded as
            numerically degenerate. Default: 1e-8.

    Returns:
        P (scipy.sparse.csr_array): Shape (3*N, n_reduced), orthonormal
            columns spanning the block rigid-body subspace in
            mass-weighted Cartesian space.
        block_col_counts (list[int]): Number of valid (non-dropped)
            generator columns contributed by each block, in block order;
            sums to ``n_reduced``. Returned for diagnostics/logging only.

    Raises:
        ValueError: If no valid generator columns are found at all (e.g.
            empty ``blocks``).
    """
    n_atoms = positions_nm.shape[0]
    n_dof   = 3 * n_atoms
    sqrt_m  = np.sqrt(masses)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    col_idx = 0
    block_col_counts: List[int] = []

    eye3 = np.eye(3)

    for block in blocks:
        block   = np.asarray(block, dtype=np.int64)
        n_b     = len(block)
        pos_b   = positions_nm[block]          # (n_b, 3)
        m_b     = masses[block]                # (n_b,)
        sqrt_mb = sqrt_m[block]                # (n_b,)

        com = (m_b[:, np.newaxis] * pos_b).sum(axis=0) / m_b.sum()
        rel = pos_b - com                      # (n_b, 3)

        # 6 candidate generators, each flattened to length 3*n_b
        # (atom-major: [x0,y0,z0, x1,y1,z1, ...], matching the global DOF
        # ordering used below).
        candidates = np.empty((6, n_b, 3))
        for k in range(3):
            candidates[k]     = 0.0
            candidates[k, :, k] = sqrt_mb                       # translation k
            candidates[3 + k]  = sqrt_mb[:, np.newaxis] * np.cross(rel, eye3[k])  # rotation k

        candidates = candidates.reshape(6, 3 * n_b)

        # Global DOF indices for this block's atoms (atom-major x,y,z).
        global_dof = np.empty(3 * n_b, dtype=np.int64)
        global_dof[0::3] = 3 * block
        global_dof[1::3] = 3 * block + 1
        global_dof[2::3] = 3 * block + 2

        # Gram-Schmidt orthonormalization within the block.
        orthonormal: List[np.ndarray] = []
        for g in candidates:
            v = g.copy()
            for q in orthonormal:
                v -= np.dot(v, q) * q
            norm = np.linalg.norm(v)
            if norm > drop_tol:
                orthonormal.append(v / norm)

        block_col_counts.append(len(orthonormal))

        for q in orthonormal:
            nz = np.flatnonzero(q)
            rows.extend(global_dof[nz].tolist())
            cols.extend([col_idx] * len(nz))
            vals.extend(q[nz].tolist())
            col_idx += 1

    if col_idx == 0:
        raise ValueError("build_block_projection: no valid generator columns "
                         "were assembled (empty or degenerate blocks).")

    P = coo_array((vals, (rows, cols)), shape=(n_dof, col_idx)).tocsr()
    return P, block_col_counts


def project_and_diagonalize(mw_hessian: Any, positions_nm: np.ndarray,
                            masses: np.ndarray, blocks: List[List[int]],
                            diagonalize_fn: Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            n_modes: int = None, use_gpu: bool = False,
                            console: Any = None,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Diagonalize a mass-weighted ENM Hessian via the RTB block-projection
    reduction, returning output in the exact same shape/convention as
    ``ENMCalculator.compute_normal_modes``.

    Pipeline: build the block projection matrix P (``build_block_projection``),
    reduce the Hessian (``H_reduced = P.T @ mw_hessian @ P``, always small
    and dense by construction), diagonalize it via the caller-supplied
    ``diagonalize_fn`` (avoids duplicating/importing the diagonalization
    logic - pass ``ENMCalculator.compute_normal_modes`` bound to an
    instance), then project the resulting eigenvectors back to full
    Cartesian mass-weighted space (``modes_full = P @ modes_reduced``).

    Args:
        mw_hessian: Mass-weighted ENM Hessian (3N x 3N), sparse or dense,
            as returned by ``ENMCalculator.mass_weight_hessian``.
        positions_nm (np.ndarray): Equilibrium positions, shape (N, 3).
        masses (np.ndarray): Per-atom masses, shape (N,), amu.
        blocks (list[list[int]]): Per-block atom index lists, as returned
            by ``build_blocks``.
        diagonalize_fn: Callable with the same signature/contract as
            ``ENMCalculator.compute_normal_modes(hessian, n_modes=None,
            use_gpu=False) -> (frequencies, modes, eigenvalues)``, called
            here on the small reduced Hessian.
        n_modes (int, optional): Forwarded to ``diagonalize_fn``.
        use_gpu (bool): Forwarded to ``diagonalize_fn``. Since the
            reduced Hessian is always small (dense, at most a few
            thousand DOF), this mainly matters for very large full
            systems where even the reduced problem could benefit.
        console: Console configuration object for formatted output
            (optional; plain ``print`` is used if omitted).

    Returns:
        frequencies (np.ndarray): Same as ``compute_normal_modes``, shape (k,).
        modes_full (np.ndarray): Eigenvectors projected back to full
            Cartesian mass-weighted space, shape (3N, k) -- same
            convention as ``compute_normal_modes``'s own output, so
            every downstream consumer works unmodified.
        eigenvalues (np.ndarray): Raw (unfiltered) eigenvalues of the
            *reduced* problem, passed through unchanged from
            ``diagonalize_fn`` (same semantics as the non-RTB path).
    """
    def _log(msg: str) -> None:
        if console is not None:
            print(f"{console.PGM_NAM}{msg}")
        else:
            print(msg)

    n_dof_full = mw_hessian.shape[0]
    P, block_col_counts = build_block_projection(positions_nm, masses, blocks)
    n_reduced = P.shape[1]

    _log(f"RTB: {len(blocks)} blocks, reduced Hessian is "
         f"{n_reduced}x{n_reduced} (from {n_dof_full}x{n_dof_full} full DOF).")

    H_reduced = P.T @ mw_hessian @ P
    if issparse(H_reduced):
        H_reduced = H_reduced.toarray()
    H_reduced = np.asarray(H_reduced)
    H_reduced = 0.5 * (H_reduced + H_reduced.T)   # enforce exact symmetry

    frequencies, modes_reduced, eigenvalues = diagonalize_fn(
        H_reduced, n_modes=n_modes, use_gpu=use_gpu
    )

    modes_full = P @ modes_reduced   # (3N, k)
    norms = np.linalg.norm(modes_full, axis=0)
    norms[norms < 1e-12] = 1.0
    modes_full = modes_full / norms

    return frequencies, modes_full, eigenvalues
