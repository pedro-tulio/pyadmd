"""Normal-mode subspace harmonization and transport of the excitation direction."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SubspaceResult:
    """
    Container for the outputs of ``harmonize_and_transport``.

    Attributes:
        q_new (np.ndarray): Final excitation direction Q_final, shape
            (n_atoms, 3), unit norm.
        harmonized_modes (np.ndarray): Harmonized m-dimensional basis V_harm,
            shape (m, n_atoms, 3), orthonormal. Expressed in the frame of the
            old coordinates (``old_coords``), i.e. after the Kabsch rotation
            of the new modes.
        singular_values (np.ndarray): Singular values of the overlap matrix
            U^T V, shape (m,), clipped to [0, 1] and sorted in descending
            order. They are the cosines of the principal angles.
        principal_angles_deg (np.ndarray): Principal angles between the old
            active subspace and the matched subspace, shape (m,), in degrees.
        omega (float): Global overlap indicator, the mean of the squared
            singular values.
        rotation (np.ndarray): Kabsch rotation matrix R, shape (3, 3), that
            superposes the new coordinates onto the old ones (applied as
            ``new_modes @ R``).
        matching_coefficients (np.ndarray): Coefficients of the M search modes
            in each of the m matched directions (the block R_m of the right
            singular vectors), shape (M, m).
    """
    q_new: np.ndarray
    harmonized_modes: np.ndarray
    singular_values: np.ndarray
    principal_angles_deg: np.ndarray
    omega: float
    rotation: np.ndarray
    matching_coefficients: np.ndarray


def _orthonormal_columns(modes: np.ndarray) -> np.ndarray:
    """
    Return an orthonormal column basis spanning a set of mode vectors.

    Each mode is flattened (atom-major: x, y, z per atom) into a column and
    the columns are orthonormalized by reduced QR decomposition.

    Args:
        modes (np.ndarray): Mode vectors, shape (k, n_atoms, 3).

    Returns:
        np.ndarray: Orthonormal basis, shape (3 * n_atoms, k), spanning the
            same subspace as ``modes``.

    Raises:
        ValueError: If ``modes`` does not have shape (k, n_atoms, 3).
    """
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 3 or modes.shape[2] != 3:
        raise ValueError("modes must have shape (k, n_atoms, 3)")
    a = modes.reshape(modes.shape[0], -1).T
    q, _ = np.linalg.qr(a, mode="reduced")
    return q


def kabsch_rotation(mobile: np.ndarray,
                    reference: np.ndarray,
                    weights: np.ndarray | None = None) -> np.ndarray:
    """
    Compute the weighted Kabsch rotation superposing a structure onto a reference.

    Both structures are centered on their weighted centers; the optimal
    proper rotation is obtained from the SVD of their weighted covariance
    matrix. Rotation only: the translation is not returned.

    Args:
        mobile (np.ndarray): Mobile coordinates, shape (n_atoms, 3).
        reference (np.ndarray): Reference coordinates, shape (n_atoms, 3).
        weights (np.ndarray, optional): Per-atom weights (e.g. atomic masses),
            shape (n_atoms,), non-negative with a positive sum. Defaults to
            uniform weights.

    Returns:
        np.ndarray: Rotation matrix R, shape (3, 3), with determinant +1, such
            that ``(mobile - com_mobile) @ R`` best matches
            ``reference - com_reference``. The same R rotates any vector
            attached to the mobile structure (e.g. normal mode vectors) into
            the reference frame.

    Raises:
        ValueError: If the coordinate arrays differ in shape or are not
            (n_atoms, 3), or if ``weights`` has a wrong length, negative
            entries, or a non-positive sum.
    """
    mobile = np.asarray(mobile, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if mobile.shape != reference.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile/reference must both have shape (n_atoms, 3)")

    if weights is None:
        weights = np.ones(mobile.shape[0], dtype=float)

    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != mobile.shape[0]:
        raise ValueError("weights must contain one value per selected atom")
    if np.any(weights < 0.0) or weights.sum() <= 0.0:
        raise ValueError("weights must be non-negative and have a positive sum")

    weights = weights / weights.sum()

    cm = np.sum(mobile * weights[:, None], axis=0)
    cr = np.sum(reference * weights[:, None], axis=0)

    x = mobile - cm
    y = reference - cr

    cov = (x * weights[:, None]).T @ y
    u, _, vt = np.linalg.svd(cov)

    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt

    return r


def harmonize_and_transport(old_modes: np.ndarray,
                            new_modes: np.ndarray,
                            q_old: np.ndarray,
                            old_coords: np.ndarray,
                            new_coords: np.ndarray,
                            masses: np.ndarray | None = None,
                            alpha: float = 1.0) -> SubspaceResult:
    """
    Track an old active subspace inside a new search subspace and transport Q.

    ``old_modes`` defines the active subspace U_m (dimension m) and
    ``new_modes`` the new ENM search subspace V_M (dimension M >= m), both in
    the same mass-weighted coordinate space and on the same atoms. The method:

      1. Superposes the new structure onto the old one (mass-weighted Kabsch)
         and rotates the new modes into the old frame.
      2. Orthonormalizes both bases by QR and forms the m x M overlap matrix
         S = U^T V, decomposed by thin SVD (S = L Sigma R^T). The singular
         values are the cosines of the principal angles.
      3. Selects the best m-dimensional part of V_M, V_match = V R_m, and
         orients it by Procrustes harmonization, V_harm = V_match L^T. The
         selected subspace is unchanged: span(V_harm) = span(V_match).
      4. Projects Q onto span(V_harm) and renormalizes (Q_harm), choosing the
         sign for continuity with Q.
      5. Mixes, Q_final = norm[(1 - alpha) Q + alpha Q_harm].

    Args:
        old_modes (np.ndarray): Old active subspace U_m, shape (m, n_atoms, 3).
        new_modes (np.ndarray): New search subspace V_M, shape
            (M, n_atoms, 3), with the same atoms and atom order as
            ``old_modes``.
        q_old (np.ndarray): Current excitation direction to transport (the
            corrected direction Q_corr), shape (n_atoms, 3). Normalized
            internally.
        old_coords (np.ndarray): Structure the old modes refer to, shape
            (n_atoms, 3), in Å.
        new_coords (np.ndarray): Structure the new modes were computed on,
            shape (n_atoms, 3), in Å.
        masses (np.ndarray, optional): Per-atom masses used to weight the
            Kabsch superposition, shape (n_atoms,), in amu. Defaults to
            uniform weights.
        alpha (float): Mixing factor in [0, 1]. 0 keeps Q, 1 uses entirely the
            transported direction Q_harm. Default: 1.0.

    Returns:
        SubspaceResult: Final direction, harmonized basis and diagnostics
            (singular values, principal angles, Omega, rotation, matching
            coefficients). See ``SubspaceResult``.

    Raises:
        ValueError: If the arrays have inconsistent shapes or atoms, if
            M < m, if ``alpha`` is outside [0, 1], or if ``q_old`` has zero
            norm.
        RuntimeError: If the projection of Q onto the harmonized subspace is
            numerically zero (insufficient overlap with the search subspace).

    Note:
        The harmonized basis is returned in the frame of ``old_coords``, not
        in the frame of ``new_coords``. The projection of Q can reduce its
        norm when part of Q lies outside the harmonized subspace; Q_harm is
        renormalized, so with ``alpha=1`` that component is discarded.
    """
    old_modes = np.asarray(old_modes, dtype=float)
    new_modes = np.asarray(new_modes, dtype=float)
    q_old = np.asarray(q_old, dtype=float)
    old_coords = np.asarray(old_coords, dtype=float)
    new_coords = np.asarray(new_coords, dtype=float)

    if old_modes.ndim != 3 or old_modes.shape[-1] != 3:
        raise ValueError("old_modes must have shape (m, n_atoms, 3)")
    if new_modes.ndim != 3 or new_modes.shape[-1] != 3:
        raise ValueError("new_modes must have shape (M, n_atoms, 3)")
    if old_modes.shape[1:] != new_modes.shape[1:]:
        raise ValueError(
            "old_modes and new_modes must contain the same atoms in the same order"
        )

    m = old_modes.shape[0]
    M = new_modes.shape[0]

    if M < m:
        raise ValueError(f"search subspace must satisfy M >= m; got M={M}, m={m}")
    if q_old.shape != old_modes.shape[1:]:
        raise ValueError("q_old must have shape (n_atoms, 3)")
    if old_coords.shape != q_old.shape or new_coords.shape != q_old.shape:
        raise ValueError("old_coords/new_coords must have shape (n_atoms, 3)")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    r = kabsch_rotation(new_coords, old_coords, masses)
    new_modes_rot = new_modes @ r

    u = _orthonormal_columns(old_modes)   # (3N, m)
    v = _orthonormal_columns(new_modes_rot)  # (3N, M)

    s = u.T @ v                           # (m, M)

    l, sigma, rt = np.linalg.svd(s, full_matrices=False)
    r_m = rt.T                            # (M, m)

    v_match = v @ r_m                     # (3N, m)
    v_harm = v_match @ l.T                # (3N, m)

    q0 = q_old.reshape(-1)
    q0_norm = np.linalg.norm(q0)
    if q0_norm < 1e-14:
        raise ValueError("q_old has zero norm")
    q0 /= q0_norm

    q_transport = v_harm @ (v_harm.T @ q0)
    nt = np.linalg.norm(q_transport)
    if nt < 1e-12:
        raise RuntimeError(
            "transported Q is nearly zero: search subspace has insufficient overlap"
        )
    q_transport /= nt

    if np.dot(q_transport, q0) < 0.0:
        q_transport *= -1.0

    q_mix = (1.0 - alpha) * q0 + alpha * q_transport
    nq = np.linalg.norm(q_mix)
    if nq < 1e-12:
        q_mix = q_transport
        nq = 1.0

    q_new = (q_mix / nq).reshape(q_old.shape)

    sigma_clip = np.clip(sigma, 0.0, 1.0)
    angles = np.degrees(np.arccos(sigma_clip))
    omega = float(np.mean(sigma_clip ** 2))

    modes_harm = v_harm.T.reshape((m,) + old_modes.shape[1:])

    return SubspaceResult(
        q_new=q_new,
        harmonized_modes=modes_harm,
        singular_values=sigma_clip,
        principal_angles_deg=angles,
        omega=omega,
        rotation=r,
        matching_coefficients=r_m.copy(),
    )
