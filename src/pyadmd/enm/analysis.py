"""
Analytical (non-MD) predictions derived directly from a completed normal
mode analysis: mode collectivity, per-mode variance contributions, and
NMA-based RMSF/DCCM computed from the harmonic approximation.

This is distinct from ``pyadmd.analysis.analyzer.Analyzer``, which computes
RMSF/DCCM post-hoc from an actual MD trajectory; the functions here predict
the same quantities directly from the Hessian eigenspectrum, with no
simulation required. Faithfully ported from the standalone ``enm.py``
script's ``compute_collectivity``/``write_collectivity``/
``plot_mode_contributions``/``plot_atomic_fluctuations``/
``plot_residue_cross_correlation``, preserving their original indexing and
eigenvalue-array conventions exactly (mode index/label semantics and the
mixed filtered-modes/raw-eigenvalues usage in RMSF and DCCM are both
intentional in the original and are not altered here).
"""

import csv
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import openmm as mm
from openmm import app, unit
import cupy as cp

from pyadmd.console import ConsoleConfig
from pyadmd.enm.calculator import convert_hetatm_to_atom

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


class ENMAnalyzer:
    """
    Analytical structural predictions derived directly from a completed
    ENM/NMA normal-mode calculation.

    Mode selection throughout this class uses raw 0-based column indices
    into the filtered ``modes``/``frequencies`` arrays returned by
    ``ENMCalculator.compute_normal_modes`` (rigid-body modes already
    excluded), via a ``start_mode`` parameter -- matching the standalone
    ``enm.py`` script's own convention exactly. ``write_collectivity``
    and the ``-w`` mode vector/trajectory writer additionally label each
    mode in filenames/CSV rows as ``array index + 1`` (also matching the
    original script), so the *label* seen by the user is 1 more than the
    raw array index used internally.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
    """

    # Boltzmann constant in kcal/(mol*K)
    _KB_KCAL_MOL_K: float = 0.0019872041
    # Internal angular-frequency units -> cm-1
    _FREQ_TO_CM1: float = 108.58

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize ENMAnalyzer with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    @staticmethod
    def compute_collectivity(mode_vector: np.ndarray, n_atoms: int) -> float:
        """
        Compute mode collectivity (Tama & Sanejouand, 2001).

        Args:
            mode_vector (np.ndarray): Flattened mode vector, shape (3N,).
            n_atoms (int): Number of atoms in the system.

        Returns:
            float: Collectivity in (0, 1]. Values near 1 indicate the mode
                involves concerted motion of all atoms; values near 0
                indicate motion localized to a few atoms.
        """
        u = mode_vector.reshape(n_atoms, 3)
        p = np.sum(u ** 2, axis=1) + 1e-12   # guard against log(0)
        entropy = -np.sum(p * np.log(p))
        return float(np.exp(entropy) / n_atoms)

    def write_collectivity(self, frequencies: np.ndarray, modes: np.ndarray,
                           system: mm.System, output_file: str,
                           n_modes: int = 20) -> None:
        """
        Write per-mode collectivity to a CSV file.

        Args:
            frequencies (np.ndarray): Filtered mode frequencies from
                ``ENMCalculator.compute_normal_modes``.
            modes (np.ndarray): Filtered, matching mode eigenvectors,
                shape (3N, M).
            system (openmm.System): Source of particle masses, used to
                mass-weight each mode vector before computing collectivity.
            output_file (str): Output CSV path.
            n_modes (int): Number of modes to include, starting at raw
                array index 6 (matching the standalone script's default).
                Default: 20.
        """
        n_particles = system.getNumParticles()
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Mode', 'Frequency (cm-1)', 'Collectivity'])
            for i in range(6, min(6 + n_modes, modes.shape[1])):
                mw_mode = modes[:, i] * inv_sqrt_m
                mw_mode = mw_mode / np.linalg.norm(mw_mode)
                freq_cm = frequencies[i] * self._FREQ_TO_CM1
                kappa = self.compute_collectivity(mw_mode, n_particles)
                writer.writerow([i + 1, f"{freq_cm:.2f}", f"{kappa:.4f}"])

        print(f"{self.console.PGM_NAM}Saved collectivity data to "
              f"{self.console.EXT}{output_file}{self.console.STD}.")

    def plot_mode_contributions(self, eigenvalues: np.ndarray,
                                output_file: Optional[str] = None,
                                n_modes: int = 10) -> None:
        """
        Plot per-mode and cumulative variance contributions for the first
        ``n_modes`` non-rigid modes.

        Under equipartition, mode k contributes a variance proportional to
        1/eigenvalue_k to the total mean-square displacement; this is
        exactly analogous to the proportion of variance explained by a PCA
        component.

        Args:
            eigenvalues (np.ndarray): The raw (unfiltered) eigenvalue
                array -- the third value returned by
                ``ENMCalculator.compute_normal_modes`` -- from which the
                first 6 (rigid-body) entries are stripped internally.
            output_file (str, optional): Path to save the figure. If None,
                displays interactively.
            n_modes (int): Number of non-rigid modes to plot. Default: 10.
        """
        non_rigid_evals = eigenvalues[6:]
        variances = 1.0 / (np.abs(non_rigid_evals) + 1e-10)
        proportion_variance = variances / np.sum(variances)
        cumulative_variance = np.cumsum(proportion_variance[:n_modes]) * 100

        plt.figure(figsize=(12, 6))

        modes_indices = np.arange(1, n_modes + 1)

        plt.subplot(1, 2, 1)
        plt.bar(modes_indices, proportion_variance[:n_modes] * 100, alpha=0.7, color='skyblue')
        plt.title('Proportion of Variance by Mode')
        plt.xlabel('Mode Index (excluding rigid-body modes)')
        plt.ylabel('Proportion of Variance (%)')
        plt.xticks(modes_indices)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        plt.plot(modes_indices, cumulative_variance, 'o-', linewidth=2,
                markersize=8, color='#1f77b4')
        plt.title('Cumulative Proportion of Variance')
        plt.xlabel('Mode Index (excluding rigid-body modes)')
        plt.ylabel('Cumulative Variance (%)')
        plt.xticks(modes_indices)
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        for i, val in enumerate(cumulative_variance):
            plt.annotate(f'{val:.1f}%', (modes_indices[i], val), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9)

        plt.tight_layout()
        plt.figtext(
            0.5, 0.01,
            f"First {n_modes} non-rigid modes account for "
            f"{cumulative_variance[-1]:.1f}% of total variance",
            ha="center", fontsize=10, style='italic'
        )

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{self.console.PGM_NAM}Saved mode contribution plot to "
                  f"{self.console.EXT}{output_file}{self.console.STD}.")
        else:
            plt.show()

    def compute_rmsf_from_modes(self, system: mm.System, eigenvalues: np.ndarray,
                                modes: np.ndarray, topology: app.Topology,
                                output_file: Optional[str] = None,
                                temperature: float = 300.0,
                                n_modes: Optional[int] = None,
                                start_mode: int = 6) -> np.ndarray:
        """
        Predict per-residue RMSF directly from the normal modes (harmonic,
        thermal-ensemble prediction) -- no MD trajectory required.

        Follows the equipartition-weighted covariance formula::

            RMSF_i = sqrt( (kB*T/m_i) * sum_k  u_i^(k)^2 / eigenvalue_k )

        where ``u_i^(k)`` is the mass-weighted eigenvector component of
        atom i in mode k. As in the original standalone implementation,
        the mode shape comes from the filtered ``modes`` array while its
        weight comes from the raw ``eigenvalues`` array, both indexed by
        the same loop index.

        Distinct from ``pyadmd.analysis.analyzer.Analyzer``'s RMSF, which
        is computed post-hoc from an actual MD trajectory.

        Args:
            system (openmm.System): Source of particle masses.
            eigenvalues (np.ndarray): The raw (unfiltered) eigenvalue
                array -- the third value returned by
                ``ENMCalculator.compute_normal_modes``.
            modes (np.ndarray): Filtered mode eigenvectors.
            topology (openmm.app.Topology): Used to map atoms to residues
                for the plot; the returned array remains per-atom.
            output_file (str, optional): Path to save the plot. If None,
                displays interactively.
            temperature (float): Temperature in K. Default: 300.0 (matches
                the standalone script's default).
            n_modes (int, optional): Number of modes to accumulate,
                starting at ``start_mode``. Defaults to every mode
                available from ``start_mode`` onward.
            start_mode (int): First raw array index to accumulate.
                Default: 6.

        Returns:
            np.ndarray: Per-atom RMSF values in Å, shape (N,).
        """
        n_particles = system.getNumParticles()
        if n_modes is None:
            n_modes = modes.shape[1] - start_mode

        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])

        rmsf_atom = np.zeros(n_particles)
        for mode_idx in range(start_mode, start_mode + n_modes):
            if abs(eigenvalues[mode_idx]) < 1e-10:
                continue
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)
            omega_sq = eigenvalues[mode_idx]
            rmsf_atom += (self._KB_KCAL_MOL_K * temperature / omega_sq) * np.sum(mode_vector ** 2, axis=1) / masses

        rmsf_atom = np.sqrt(rmsf_atom) * 10.0   # nm -> Å

        # Average per-atom RMSF to a single representative value per residue,
        # for the plot only (the returned array stays per-atom).
        residue_rmsf: dict = {}
        for atom in topology.atoms():
            residue_rmsf.setdefault(atom.residue.id, []).append(rmsf_atom[atom.index])
        residue_ids = sorted(residue_rmsf.keys())
        residue_means = [np.mean(residue_rmsf[rid]) for rid in residue_ids]
        residue_nums = list(range(len(residue_ids)))

        plt.figure(figsize=(12, 6))
        plt.plot(residue_nums, residue_means, 'b-', linewidth=1, alpha=0.7)
        plt.fill_between(residue_nums, 0, residue_means, alpha=0.3)
        plt.xlabel('Residue Index')
        plt.ylabel('RMS Fluctuation (Å)')
        plt.title(f'NMA-Predicted Residue Fluctuations\n'
                 f'(T={temperature}K, {n_modes} modes)')
        plt.grid(True, alpha=0.3)

        avg_rmsf = float(np.mean(residue_means))
        plt.axhline(y=avg_rmsf, color='r', linestyle='--', alpha=0.7,
                   label=f'Average: {avg_rmsf:.2f} Å')
        plt.legend()

        if residue_nums:
            tick_step = max(1, len(residue_nums) // 10)
            x_ticks = np.arange(0, len(residue_nums), tick_step)
            plt.xticks(x_ticks, x_ticks)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{self.console.PGM_NAM}NMA-predicted RMSF plot saved to "
                  f"{self.console.EXT}{output_file}{self.console.STD}.")
        else:
            plt.show()

        return rmsf_atom

    def compute_dccm_from_modes(self, system: mm.System, eigenvalues: np.ndarray,
                                modes: np.ndarray, topology: app.Topology,
                                output_file: Optional[str] = None,
                                temperature: float = 300.0,
                                n_modes: Optional[int] = None,
                                start_mode: int = 6,
                                use_gpu: bool = True) -> np.ndarray:
        """
        Predict the residue dynamical cross-correlation matrix (DCCM)
        directly from the normal modes -- no MD trajectory required.

        Atomic covariances are accumulated per mode as
        ``C = (kB*T/eigenvalue_k) * U U^T`` (mass-weighted eigenvector
        outer product), then aggregated to residue level via ``R^T C R``,
        where ``R`` is the atom-to-residue assignment matrix, and finally
        normalized to a Pearson correlation coefficient in [-1, 1]. As in
        ``compute_rmsf_from_modes``, mode shape comes from the filtered
        ``modes`` array while its weight comes from the raw ``eigenvalues``
        array, both indexed by the same loop index (matching the original
        standalone implementation).

        Distinct from ``pyadmd.analysis.analyzer.Analyzer``'s DCCM, which
        is computed post-hoc from an actual MD trajectory.

        Args:
            system (openmm.System): Source of particle masses.
            eigenvalues (np.ndarray): The raw (unfiltered) eigenvalue
                array -- the third value returned by
                ``ENMCalculator.compute_normal_modes``.
            modes (np.ndarray): Filtered mode eigenvectors.
            topology (openmm.app.Topology): Used to map atoms to residues.
            output_file (str, optional): Path to save the figure (and the
                raw matrix as a sibling ``.npy`` file). If None, displays
                interactively and the matrix is not saved to disk.
            temperature (float): Temperature in K. Default: 300.0.
            n_modes (int, optional): Number of modes to accumulate,
                starting at ``start_mode``. Defaults to every mode
                available from ``start_mode`` onward.
            start_mode (int): First raw array index to accumulate.
                Default: 6.
            use_gpu (bool): Attempt GPU acceleration via CuPy, falling back
                to CPU on failure. Default: True.

        Returns:
            np.ndarray: Normalized residue cross-correlation matrix, shape
                (n_residues, n_residues), values in [-1, 1].
        """
        n_particles = system.getNumParticles()
        if n_modes is None:
            n_modes = modes.shape[1] - start_mode

        residues = list(topology.residues())
        n_residues = len(residues)
        residue_index_of = {res: i for i, res in enumerate(residues)}
        atom_to_residue = np.array(
            [residue_index_of[atom.residue] for atom in topology.atoms()], dtype=int
        )
        R = np.zeros((n_particles, n_residues))
        R[np.arange(n_particles), atom_to_residue] = 1.0

        mode_range = range(start_mode, start_mode + n_modes)
        correlation_matrix = None
        if use_gpu:
            try:
                correlation_matrix = self._dccm_gpu(
                    system, eigenvalues, modes, R, mode_range, temperature
                )
            except Exception as exc:
                print(f"{self.console.PGM_WRN}GPU acceleration failed "
                      f"({self.console.WRN}{exc}{self.console.STD}); falling back to CPU.")
        if correlation_matrix is None:
            correlation_matrix = self._dccm_cpu(
                system, eigenvalues, modes, R, mode_range, temperature
            )

        diag = np.diag(correlation_matrix)
        norm_matrix = np.sqrt(np.outer(diag, diag))
        correlation_matrix = correlation_matrix / (norm_matrix + 1e-10)

        self._plot_dccm(correlation_matrix, residues, output_file, temperature, n_modes)
        return correlation_matrix

    def _dccm_cpu(self, system, eigenvalues, modes, R, mode_range, temperature):
        """CPU accumulation path for ``compute_dccm_from_modes``."""
        n_particles = system.getNumParticles()
        n_residues = R.shape[1]
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])
        mass_factor = 1.0 / np.sqrt(masses)

        correlation_matrix = np.zeros((n_residues, n_residues))
        for mode_idx in mode_range:
            if abs(eigenvalues[mode_idx]) < 1e-10:
                continue
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)
            weighted_vectors = mode_vector * mass_factor[:, np.newaxis]
            factor = self._KB_KCAL_MOL_K * temperature / eigenvalues[mode_idx]
            atom_corr = factor * np.dot(weighted_vectors, weighted_vectors.T)
            correlation_matrix += R.T @ atom_corr @ R
        return correlation_matrix

    def _dccm_gpu(self, system, eigenvalues, modes, R, mode_range, temperature):
        """GPU (CuPy) accumulation path for ``compute_dccm_from_modes``."""
        n_particles = system.getNumParticles()
        n_residues = R.shape[1]
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])

        with cp.cuda.Device(0):
            masses_gpu = cp.array(masses)
            mass_factor_gpu = 1.0 / cp.sqrt(masses_gpu)
            R_gpu = cp.array(R)
            correlation_matrix_gpu = cp.zeros((n_residues, n_residues))

            for mode_idx in mode_range:
                if abs(eigenvalues[mode_idx]) < 1e-10:
                    continue
                mode_vector = cp.array(modes[:, mode_idx].reshape(n_particles, 3))
                weighted_vectors = mode_vector * mass_factor_gpu[:, cp.newaxis]
                factor = self._KB_KCAL_MOL_K * temperature / eigenvalues[mode_idx]
                atom_corr = factor * cp.dot(weighted_vectors, weighted_vectors.T)
                correlation_matrix_gpu += cp.dot(cp.dot(R_gpu.T, atom_corr), R_gpu)

            correlation_matrix = cp.asnumpy(correlation_matrix_gpu)
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        return correlation_matrix

    def _plot_dccm(self, correlation_matrix: np.ndarray, residues, output_file,
                   temperature: float, n_modes: int) -> None:
        """Plot (and optionally save, alongside the raw matrix) the NMA-DCCM heatmap."""
        n_residues = len(residues)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1,
                       aspect='auto', origin='lower')
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        ax.set_xlabel('Residue Index', fontsize=12)
        ax.set_ylabel('Residue Index', fontsize=12)
        ax.set_title(f'NMA-Predicted Residue Cross-Correlation Matrix\n'
                    f'(T={temperature}K, {n_modes} modes)', fontsize=13)

        tick_step = max(1, n_residues // 10)
        ticks = np.arange(0, n_residues, tick_step)
        labels = [f'{residues[i].id}' for i in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(ticks); ax.set_yticklabels(labels)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            np.save(output_file.rsplit('.', 1)[0] + '.npy', correlation_matrix)
            print(f"{self.console.PGM_NAM}NMA-predicted DCCM plot saved to "
                  f"{self.console.EXT}{output_file}{self.console.STD}.")
        else:
            plt.show()

    @staticmethod
    def parse_mode_string(mode_str: str) -> List[int]:
        """
        Convert a mode-selection string into a sorted list of unique
        mode numbers (1-based file labels, see note above).

        Accepts comma-separated tokens where each token is either a
        single integer or an inclusive range in the form ``start:end``
        (e.g. ``"7:10,34,44:50"``).

        Args:
            mode_str (str): Selection string.

        Returns:
            list[int]: Sorted list of unique mode numbers (>= 1).

        Raises:
            ValueError: On malformed input (non-integers, inverted
                ranges, numbers < 1, or an empty selection).
        """
        modes = set()
        for raw_token in mode_str.split(','):
            token = raw_token.strip()
            if not token:
                continue

            if ':' in token:
                parts = token.split(':')
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid range '{token}': expected exactly one "
                        "':' separating start and end (e.g. '7:10')."
                    )
                try:
                    start, end = int(parts[0]), int(parts[1])
                except ValueError:
                    raise ValueError(f"Non-integer value in range '{token}'.")
                if start < 1 or end < 1:
                    raise ValueError(f"Mode numbers must be >= 1 (got '{token}').")
                if start > end:
                    raise ValueError(
                        f"Range start must be <= end (got '{token}'). "
                        f"Did you mean '{end}:{start}'?"
                    )
                modes.update(range(start, end + 1))
            else:
                try:
                    m = int(token)
                except ValueError:
                    raise ValueError(f"Non-integer mode number '{token}'.")
                if m < 1:
                    raise ValueError(f"Mode numbers must be >= 1 (got '{token}').")
                modes.add(m)

        if not modes:
            raise ValueError("Empty mode selection -- provide at least one mode number.")

        return sorted(modes)

    def find_enm_output_files(self, output_folder: str) -> Tuple[str, str, str]:
        """
        Locate the three files a completed ENM run leaves behind inside
        ``output_folder``:

          * ``*_modes.npy``                                     -- filtered eigenvector matrix
          * ``*_frequencies.npy``                                -- filtered frequency array
          * ``*_ca_structure.pdb`` or ``*_heavy_structure.pdb``  -- reduced-resolution structure

        Args:
            output_folder (str): Directory to search.

        Returns:
            tuple[str, str, str]: ``(modes_file, freq_file, structure_pdb)``.

        Raises:
            FileNotFoundError: If any required file is missing, or if the
                directory contains ambiguous matches (multiple runs mixed
                in the same folder).
        """
        def _require_one(pattern, label):
            matches = glob.glob(pattern)
            if not matches:
                raise FileNotFoundError(
                    f"No {label} found matching '{pattern}'. Make sure "
                    "the output directory points to a completed ENM run."
                )
            if len(matches) > 1:
                raise FileNotFoundError(
                    f"Multiple {label} files found (ambiguous): {matches}. "
                    "Use a more specific output directory."
                )
            return matches[0]

        base = output_folder.rstrip(os.sep)

        modes_file = _require_one(os.path.join(base, '*_modes.npy'),
                                  'eigenvector file (*_modes.npy)')
        freq_file  = _require_one(os.path.join(base, '*_frequencies.npy'),
                                  'frequency file (*_frequencies.npy)')

        pdb_candidates = (
            glob.glob(os.path.join(base, '*_ca_structure.pdb')) +
            glob.glob(os.path.join(base, '*_heavy_structure.pdb'))
        )
        if not pdb_candidates:
            raise FileNotFoundError(
                f"No structure PDB found in '{output_folder}' (looked for "
                "*_ca_structure.pdb and *_heavy_structure.pdb). Make sure "
                "the ENM run completed successfully."
            )
        if len(pdb_candidates) > 1:
            raise FileNotFoundError(
                f"Multiple structure PDB files found in '{output_folder}': "
                f"{pdb_candidates}. Use a more specific output directory."
            )

        return modes_file, freq_file, pdb_candidates[0]

    def write_modes_from_files(self, output_folder: str, mode_numbers: List[int],
                               write_vectors: bool = True,
                               write_trajectories: bool = True,
                               amplitude: float = 4.0, num_frames: int = 34) -> None:
        """
        Write mode vectors (.xyz) and/or PDB trajectories, at the ENM's
        native reduced resolution, for an arbitrary selection of
        previously computed modes -- without rerunning the ENM.

        Args:
            output_folder (str): Directory containing a completed ENM run
                (see ``find_enm_output_files``).
            mode_numbers (list[int]): 1-based mode numbers to write, e.g.
                ``[26, 41]`` (as returned by ``parse_mode_string``).
            write_vectors (bool): If True, write ``.xyz`` vector files.
                Default: True.
            write_trajectories (bool): If True, write ``_traj.pdb``
                trajectory files. Independent of ``write_vectors`` (either
                or both may be enabled). Default: True.
            amplitude (float): Peak trajectory displacement amplitude in
                Å. Default: 4.0.
            num_frames (int): Number of MODEL frames per trajectory PDB.
                Must be >= 4 (four linear segments: equilibrium ->
                -amplitude -> equilibrium -> +amplitude -> equilibrium).
                Default: 34.

        Raises:
            ValueError: If ``num_frames`` is less than 4 (only checked
                when ``write_trajectories`` is True), or if the modes
                array has an unexpected shape.
        """
        if not write_vectors and not write_trajectories:
            print(f"{self.console.PGM_WRN}Both write_vectors and "
                  "write_trajectories are False; nothing to do.")
            return

        if write_trajectories and num_frames < 4:
            raise ValueError(
                f"num_frames must be >= 4 for a 4-segment oscillation "
                f"trajectory (got {num_frames})."
            )

        modes_file, freq_file, pdb_file = self.find_enm_output_files(output_folder)
        print(f"{self.console.PGM_NAM}Searching for ENM output files in "
              f"{self.console.EXT}{output_folder}{self.console.STD}...")
        print(f"{self.console.PGM_NAM}  Modes       : {self.console.EXT}"
              f"{os.path.basename(modes_file)}{self.console.STD}")
        print(f"{self.console.PGM_NAM}  Frequencies : {self.console.EXT}"
              f"{os.path.basename(freq_file)}{self.console.STD}")
        print(f"{self.console.PGM_NAM}  Structure   : {self.console.EXT}"
              f"{os.path.basename(pdb_file)}{self.console.STD}")

        modes       = np.load(modes_file)
        frequencies = np.load(freq_file).ravel()

        if modes.ndim != 2:
            raise ValueError(f"Unexpected modes array shape {modes.shape}; expected (3N, M).")

        n_stored = modes.shape[1]
        n_atoms  = modes.shape[0] // 3
        print(f"{self.console.PGM_NAM}  Stored modes: {self.console.EXT}{n_stored}"
              f"{self.console.STD}  |  Atoms: {self.console.EXT}{n_atoms}{self.console.STD}")

        rigid_body   = [m for m in mode_numbers if 1 <= m <= 6]
        out_of_range = [m for m in mode_numbers if m > n_stored]
        valid_modes  = [m for m in mode_numbers if m not in out_of_range]

        if rigid_body:
            print(f"{self.console.PGM_WRN}Modes {self.console.WRN}{rigid_body}"
                  f"{self.console.STD} correspond to near-zero rigid-body "
                  "translations/rotations and are typically not "
                  "meaningful. Proceeding anyway.")
        if out_of_range:
            print(f"{self.console.PGM_WRN}Skipping mode(s) "
                  f"{self.console.WRN}{out_of_range}{self.console.STD}: "
                  f"exceed the number of stored modes ({n_stored}).")

        if not valid_modes:
            print(f"{self.console.PGM_ERR}No valid modes to write.")
            return

        # Rebuild OpenMM topology + positions from the structure PDB
        structure = app.PDBFile(pdb_file)
        topology  = structure.topology
        positions = structure.positions   # openmm.unit.Quantity, in nm

        # Reconstruct a lightweight System carrying only particle masses
        system = mm.System()
        for atom in topology.atoms():
            if atom.element is not None:
                system.addParticle(atom.element.mass)
            else:
                system.addParticle(12.011 * unit.daltons)   # fallback: carbon

        output_prefix = modes_file[:-len('_modes.npy')]
        elements = [atom.element.symbol if atom.element else 'C'
                   for atom in topology.atoms()]

        if write_vectors and write_trajectories:
            action = 'vector + trajectory'
        elif write_vectors:
            action = 'vector only'
        else:
            action = 'trajectory only'
        print(f"{self.console.PGM_NAM}Writing {self.console.EXT}{len(valid_modes)}"
              f"{self.console.STD} mode(s) ({action}): "
              f"{self.console.EXT}{valid_modes}{self.console.STD}")

        n_particles = system.getNumParticles()
        for nm in valid_modes:
            mode_idx = nm - 1   # 0-based column index

            if write_vectors:
                freq_cm1 = frequencies[mode_idx] * self._FREQ_TO_CM1
                vec_file = f"{output_prefix}_mode_{nm}.xyz"
                with open(vec_file, 'w') as f:
                    f.write(f"{n_atoms}\n")
                    f.write(f"Normal Mode {nm}, Frequency: {freq_cm1:.2f} cm-1\n")
                    mode_vector = modes[:, mode_idx].reshape(n_atoms, 3)
                    for i in range(n_atoms):
                        x, y, z = mode_vector[i]
                        f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")

            if write_trajectories:
                self._write_reduced_mode_trajectory(
                    system, topology, positions, modes, mode_idx, nm,
                    output_prefix, n_particles, amplitude, num_frames,
                )

        print(f"{self.console.PGM_NAM}Done. Files written to "
              f"{self.console.EXT}{os.path.abspath(output_folder)}{self.console.STD}.")

    def _write_reduced_mode_trajectory(self, system: mm.System, topology: app.Topology,
                                       positions: unit.Quantity, modes: np.ndarray,
                                       mode_idx: int, nm: int, output_prefix: str,
                                       n_particles: int, amplitude: float,
                                       num_frames: int) -> None:
        """
        Write a single reduced-resolution mode trajectory PDB (helper for
        ``write_modes_from_files``). Displacement is mass-weighted and
        scaled to ``amplitude``; motion is a 4-segment piecewise-linear
        oscillation: equilibrium -> -amplitude -> equilibrium -> +amplitude
        -> equilibrium.

        Args:
            system (openmm.System): Source of particle masses.
            topology (openmm.app.Topology): Reduced-structure topology.
            positions (openmm.unit.Quantity): Equilibrium positions.
            modes (np.ndarray): Filtered mode eigenvectors, shape (3N, M).
            mode_idx (int): 0-based filtered array column index for this mode.
            nm (int): 1-based mode label (used only for filename/logging).
            output_prefix (str): Filename prefix (without ``_mode_{nm}...`` suffix).
            n_particles (int): Number of particles in ``system``.
            amplitude (float): Peak displacement amplitude in Å.
            num_frames (int): Number of MODEL frames (already validated >= 4
                by the caller).
        """
        trj_file = f"{output_prefix}_mode_{nm}_traj.pdb"

        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                           for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)
        u = modes[:, mode_idx] * inv_sqrt_m

        rms = np.linalg.norm(u) / np.sqrt(n_particles)
        if rms < 1e-10:
            print(f"{self.console.PGM_WRN}Mode {self.console.WRN}{nm}{self.console.STD}: "
                  "near-zero displacement norm -- skipping trajectory.")
            return

        scaled_disp = u.reshape(n_particles, 3) * (amplitude / rms * 0.1)   # Å -> nm
        orig_pos_np = np.array([[p.x, p.y, p.z]
                                for p in positions.value_in_unit(unit.nanometer)])

        seg1 = int(num_frames * 0.25)
        seg2 = int(num_frames * 0.25)
        seg3 = int(num_frames * 0.25)
        seg4 = num_frames - seg1 - seg2 - seg3

        with open(trj_file, 'w') as f:
            for frame in range(num_frames):
                if frame < seg1:
                    factor = -frame / seg1
                elif frame < seg1 + seg2:
                    factor = -1 + (frame - seg1) / seg2
                elif frame < seg1 + seg2 + seg3:
                    factor = (frame - seg1 - seg2) / seg3
                else:
                    factor = 1 - (frame - seg1 - seg2 - seg3) / seg4

                new_pos_np = orig_pos_np + scaled_disp * factor
                new_positions = [mm.Vec3(*new_pos_np[i]) for i in range(n_particles)]
                pos_qty = unit.Quantity(new_positions, unit.nanometer)

                f.write(f"MODEL     {frame + 1:5d}\n")
                app.PDBFile.writeFile(topology, pos_qty, f, keepIds=True)
                f.write("ENDMDL\n")

        convert_hetatm_to_atom(trj_file)
