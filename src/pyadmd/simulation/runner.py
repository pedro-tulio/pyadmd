"""Orchestrates running, restarting, and appending aMDeNM replica simulations."""

import csv
import gc
import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import MDAnalysis as mda
from openmm import unit

from pyadmd.console import ConsoleConfig
from pyadmd.constants import AKMA_VEL_TO_NM_PS
from pyadmd.geometry import _kabsch_align
from pyadmd.io.dcd import _count_dcd_frames
from pyadmd.io.namd import NAMDInputReader
from pyadmd.io.openmm_restart import OpenMMRestartReader
from pyadmd.io.state import SystemState, make_reference_universe
from pyadmd.enm.calculator import ENMCalculator
from pyadmd.modes.exciter import ModeExciter
from pyadmd.modes.subspace import harmonize_and_transport, kabsch_rotation
from pyadmd.modes.diversity import (
    harmonize_with_diversity,
    validate_diversity,
    write_diversity_spectrum,
)
from pyadmd.simulation.engine import OpenMMSimulationEngine
from pyadmd.simulation.system_builder import OpenMMSystemBuilder


class SimulationRunner:
    """
    Handles running, restarting, and appending aMDeNM simulations.

    This class provides a unified interface for managing simulation runs,
    including initialization, execution, and cleanup of replica simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
        args (argparse.Namespace): Command line arguments.
        cwd (str): Current working directory.
        input_dir (str): Input directory path.
        psffile (str): PSF topology file path.
        pdbfile (str): PDB structure file path.
        coorfile (str): Coordinate file path.
        velfile (str): Velocity file path.
        xscfile (str): Extended system configuration file path.
        strfile (str): Structure file path.
        sys_coor (mda.Universe): System structure universe.
        n_atoms (int): Total number of atoms.
        sys_mass (np.ndarray): System atomic masses.
        sel_mass (np.ndarray): Selection atomic masses.
        energy (float): Excitation energy value.
        mode_exciter (ModeExciter): Mode exciter instance.
    """

    # Representative nucleic acid beads of the Cα-model SBP representation
    # (see ``ENMCalculator._create_ca_system``).
    _SBP_BEAD_NAMES: Tuple[str, ...] = ("P", "C1'", "C2")

    def __init__(self, console: ConsoleConfig, args: Any, cwd: str, input_dir: str,
                 psffile: str, pdbfile: str, coorfile: Optional[str], velfile: Optional[str],
                 xscfile: Optional[str], strfile: Optional[str], sys_coor: mda.Universe,
                 n_atoms: int, sys_mass: np.ndarray, sel_mass: np.ndarray, energy: float,
                 mode_exciter: ModeExciter,
                 init_state: SystemState,
                 platform: str = 'auto', n_threads: Optional[int] = None) -> None:
        """
        Initialize SimulationRunner.

        Accepts a pre-built ``SystemState`` (``init_state``) so that the runner
        is engine-agnostic: both the NAMD and OpenMM input paths produce a
        ``SystemState`` before constructing this object.

        Args:
            console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
            xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
            mode_exciter: as in 3.5.0 (coorfile/velfile/xscfile/strfile may be
                None in OpenMM input mode).
            init_state (SystemState): Initial state, rotated to canonical OpenMM
                box orientation.
            platform (str): OpenMM platform ('auto', 'cuda', 'opencl', 'cpu').
            n_threads (int): CPU thread count for the CPU platform.

        Raises:
            ValueError: If the beta options are inconsistent with the run.
        """
        self.console = console
        self.args = args
        self.cwd = cwd
        self.input_dir = input_dir
        self.psffile = psffile
        self.pdbfile = pdbfile
        self.coorfile = coorfile
        self.velfile = velfile
        self.xscfile = xscfile
        self.strfile = strfile
        self.sys_coor = sys_coor
        self.n_atoms = n_atoms
        self.sys_mass = sys_mass
        self.sel_mass = sel_mass
        self.energy = energy
        self.mode_exciter = mode_exciter

        # OpenMM platform preferences (used per-replica in run_simulation)
        self._platform    = platform
        self._n_threads   = n_threads
        self._temperature = 303.15   # Kelvin
        self._nh_frequency = float(getattr(args, 'nh_frequency', 1.0))   # ps^-1

        # Optional beta path: validate before anything is built.
        beta = self._beta()
        active = [int(x) for x in str(args.modes).split(',') if x.strip()]
        search = self._get_subspace_search_modes(active)
        validate_diversity(self._alpha(), beta, active, search, self._beta_count())
        if beta:
            if not self._is_harmonized():
                raise ValueError("--subspace-beta requires -r --recalc-method harmonized")
            if str(args.model).upper() not in ('CA', 'HEAVY'):
                raise ValueError("--subspace-beta requires an ENM model (CA or HEAVY)")

        # Consume the pre-built SystemState (already rotated to canonical form)
        self._init_state = init_state

        # Cache initial positions for diagnostics and direction correction
        self._init_pos_nm: np.ndarray = self._init_state.positions_nm.copy()

        # Reference Universe: topology + real positions, engine-agnostic.
        init_pos_ang = self._init_pos_nm * 10.0   # nm → Å
        self._ref_universe: mda.Universe = make_reference_universe(psffile, init_pos_ang)

        # Derive str_box for PME sizing (.str file, else initial-state box).
        str_box = None
        if strfile:
            try:
                str_box = NAMDInputReader.parse_str_box(strfile)
                print(
                    f"{console.PGM_NAM}STR cell: "
                    f"{console.EXT}{str_box['xtltype']}{console.STD}  "
                    f"a={str_box['a']:.3f} b={str_box['b']:.3f} "
                    f"c={str_box['c']:.3f} Ang  "
                    f"alpha={str_box['alpha']:.2f} beta={str_box['beta']:.2f} "
                    f"gamma={str_box['gamma']:.2f} deg"
                )
                rst_lengths_nm = [float(np.linalg.norm(v))
                                  for v in self._init_state.box_vectors_nm]
                str_lengths_nm = [float(np.linalg.norm(v))
                                  for v in str_box['box_vectors_nm']]
                for lbl, rst_l, str_l in zip(('a', 'b', 'c'), rst_lengths_nm, str_lengths_nm):
                    if rst_l > 0:
                        diff_pct = abs(rst_l - str_l) / rst_l * 100.0
                        if diff_pct > 5.0:
                            print(
                                f"{console.PGM_WRN}WARNING: STR/initial-state box mismatch "
                                f"for {console.WRN}{lbl}{console.STD}: STR={console.WRN}{str_l*10:.3f}{console.STD} Ang, "
                                f"state={console.WRN}{rst_l*10:.3f}{console.STD} Ang ({console.WRN}{diff_pct:.1f}{console.STD}% difference)"
                            )
            except Exception as exc:
                print(f"{console.PGM_WRN}WARNING: could not parse str box "
                      f"({console.WRN}{exc}{console.STD}); falling back to box derived from initial state.")
                str_box = None

        if str_box is None:
            # Derive cell parameters from the initial state's box vectors.
            # This is the primary path for OpenMM input mode when no .str is given,
            # and also the fallback for NAMD mode when .str parsing fails.
            a_rot, b_rot, c_rot = self._init_state.box_vectors_nm
            try:
                str_box = OpenMMRestartReader.box_vectors_to_cell(a_rot, b_rot, c_rot)
                print(
                    f"{console.PGM_NAM}Cell derived from initial state: "
                    f"{console.EXT}{str_box['xtltype']}{console.STD}  "
                    f"a={str_box['a']:.3f} b={str_box['b']:.3f} "
                    f"c={str_box['c']:.3f} Ang  "
                    f"alpha={str_box['alpha']:.2f} beta={str_box['beta']:.2f} "
                    f"gamma={str_box['gamma']:.2f} deg"
                )
            except Exception as exc:
                print(f"{console.PGM_WRN}WARNING: could not derive cell from initial state "
                      f"({console.WRN}{exc}{console.STD}); using placeholder box (not recommended).")
                str_box = None

        # Build OpenMM System with real box dimensions
        toppar_dir = os.path.join(input_dir, "charmm_toppar")
        builder = OpenMMSystemBuilder(console)
        self._psf_omm, self._omm_system, self._system_type = builder.build(
            psffile, toppar_dir, temperature=self._temperature, str_box=str_box
        )

        # Sanity-check: OpenMM System particle count must match the PSF topology.
        _omm_n = self._omm_system.getNumParticles()
        _psf_n = self._psf_omm.topology.getNumAtoms()
        if _omm_n != _psf_n:
            raise RuntimeError(f"Atom count mismatch after system build: OpenMM system has {console.ERR}{_omm_n}{console.STD} "
                f"particles but the PSF topology ({console.WRN}{psffile}{console.STD}) has {console.WRN}{_psf_n}{console.STD} atoms.")
        print(f"{console.PGM_NAM}System atom count verified: "
              f"{console.EXT}{_omm_n}{console.STD} atoms in both OpenMM system and PSF.")

        # In-memory correction state (initialised per-replica)
        self._cntrl_vec:         Optional[np.ndarray] = None  # Q vector, full system
        self._exc_vel_akma:      Optional[np.ndarray] = None  # excitation velocity, AKMA
        self._correc_ref_pos_nm: Optional[np.ndarray] = None  # RMS-displacement reference
        self._align_ref_pos_nm:  Optional[np.ndarray] = None  # alignment reference
        self._curr_pos_nm:       Optional[np.ndarray] = None  # positions this cycle
        self._prev_pos_nm:       Optional[np.ndarray] = None  # positions previous cycle
        self._avg_pos_nm:        Optional[np.ndarray] = None  # last average structure

        # Energy correction thresholds
        self.top    = energy * 1.25
        self.bottom = energy * 0.75

        # Adaptive correction parameters
        self.globfreq = self.cos_alpha = self.qrms_correc = 0.5

        # ENM subspace tracking (only used with --recalc --recalc-method harmonized).
        self._enm_basis_modes:      Optional[np.ndarray] = None  # active subspace U_m, (m, n_sel, 3)
        self._enm_basis_ref_pos_nm: Optional[np.ndarray] = None  # positions the basis refers to, (N, 3) nm
        self._enm_reservoir_modes:  Optional[np.ndarray] = None  # beta reservoir, (M, n_sel, 3)

    # ------------------------------------------------------------------ #
    # Option helpers                                                      #
    # ------------------------------------------------------------------ #

    def _alpha(self) -> float:
        """Return the alpha mixing factor (default 1.0 for older parameter files)."""
        return float(getattr(self.args, 'subspace_alpha', 1.0))

    def _beta(self) -> float:
        """Return the beta amplitude (default 0.0 for older parameter files)."""
        return float(getattr(self.args, 'subspace_beta', 0.0))

    def _beta_count(self) -> int:
        """Return the number of supplementary beta directions (default 3)."""
        return int(getattr(self.args, 'subspace_beta_count', 3))

    def _diversity_enabled(self) -> bool:
        """Return True when the optional beta path is active."""
        return self._beta() > 0

    def _is_harmonized(self) -> bool:
        """
        Return whether the harmonized ENM-recalculation method is active.

        Returns:
            bool: True if ``--recalc`` is set and ``--recalc-method`` is
                ``harmonized``; False otherwise.
        """
        return (bool(getattr(self.args, 'recalc', False))
                and getattr(self.args, 'recalc_method', 'random') == 'harmonized')

    @staticmethod
    def _state_digest(path: str) -> Optional[str]:
        """SHA-256 of a file, or None if it does not exist."""
        if not os.path.isfile(path):
            return None
        with open(path, 'rb') as stream:
            return hashlib.sha256(stream.read()).hexdigest()

    def _excite_control_vector(self, q: np.ndarray, energy: float,
                               masses: np.ndarray) -> np.ndarray:
        """
        Excitation velocity (AKMA) of a control direction.

        Official scaling in the alpha path; for beta the direction is a
        mass-weighted unit vector and the velocity is converted to physical
        units, so that the kinetic energy along it equals ``energy``.

        Args:
            q (np.ndarray): Direction, shape (n_sel, 3).
            energy (float): Target kinetic energy, kcal/mol.
            masses (np.ndarray): Masses of the selected atoms, amu.

        Returns:
            np.ndarray: Excitation velocity, shape (n_sel, 3).
        """
        if not self._diversity_enabled():
            return self.mode_exciter.excite(q, energy, masses)
        q = np.asarray(q, dtype=float)
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (len(q),) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("diversification requires positive finite atomic masses")
        if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-14 or not np.isfinite(energy) or energy < 0:
            raise ValueError("excitation requires a finite nonzero direction and non-negative energy")
        return np.sqrt(2 * energy) * q / np.linalg.norm(q) / np.sqrt(masses)[:, None]

    # ------------------------------------------------------------------ #
    # Memory helpers                                                      #
    # ------------------------------------------------------------------ #

    def _release_memory(self) -> None:
        """Collect garbage and return freed heap memory to the operating system."""
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

    # ------------------------------------------------------------------ #
    # Selection and mode-basis helpers                                    #
    # ------------------------------------------------------------------ #

    def _get_subspace_search_modes(self, nm_parsed: List[int]) -> List[int]:
        """
        Return the larger ENM search space M used for subspace tracking.

        ``-nm/--modes`` defines the active subspace m. The optional
        ``--subspace-search-modes`` defines the search space M ⊇ m; when
        absent, M = m.

        Args:
            nm_parsed (list[int]): Active mode numbers (dimension m).

        Returns:
            list[int]: Search-space mode numbers (dimension M >= m).

        Raises:
            ValueError: If the list is malformed, has duplicates, is smaller
                than ``nm_parsed``, or does not contain every active mode.
        """
        raw = getattr(self.args, "subspace_search_modes", None)
        if raw is None or str(raw).strip() == "":
            return list(nm_parsed)

        try:
            search_modes = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError(
                "--subspace-search-modes must be a comma-separated list of integers"
            ) from exc

        if len(search_modes) < len(nm_parsed):
            raise ValueError(
                f"subspace search space must satisfy M >= m; "
                f"got M={len(search_modes)}, m={len(nm_parsed)}"
            )
        if len(set(search_modes)) != len(search_modes):
            raise ValueError("--subspace-search-modes contains duplicate mode indices")
        missing = [mode for mode in nm_parsed if mode not in search_modes]
        if missing:
            raise ValueError(
                "the active modes given by -nm must be contained in "
                f"--subspace-search-modes; missing: {missing}"
            )
        return search_modes

    def _enm_selection_atoms(self) -> mda.AtomGroup:
        """
        Return the excitation/correction atoms: the ENM representative atoms
        within the user-provided selection.

        Returns:
            mda.AtomGroup: Sorted, unique atoms of the reference Universe.

        Raises:
            ValueError: If the resulting selection is empty.
        """
        base = self._ref_universe.select_atoms(self.args.selection)
        if self.args.model.lower() == 'ca':
            protein_ca = base.select_atoms("protein and name CA")
            nucleic    = base.select_atoms("nucleic")
            sbp        = nucleic[np.isin(nucleic.names, self._SBP_BEAD_NAMES)]
            atoms      = protein_ca | sbp
        else:
            atoms = base.select_atoms("not name H*")

        if atoms.n_atoms == 0:
            raise ValueError(
                f"Selection '{self.args.selection}' contains no ENM representative "
                f"atoms for model '{self.args.model}'."
            )
        return atoms

    def _validate_enm_selection(self, sel_ix: np.ndarray) -> None:
        """
        Verify that every excitation atom belongs to the ENM atom set
        (``protein or nucleic``).

        Args:
            sel_ix (np.ndarray): System indices of the excitation atoms.

        Raises:
            ValueError: If any selected atom is neither protein nor nucleic.
        """
        enm_ix = set(int(i) for i in self._ref_universe.select_atoms("protein or nucleic").ix)
        outside = [int(i) for i in sel_ix if int(i) not in enm_ix]
        if outside:
            raise ValueError(
                "--recalc requires every excitation atom to be protein or nucleic; "
                f"{len(outside)} selected atom(s) are not (first indices: {outside[:10]})."
            )

    def _read_enm_basis(self, base_name: str, mode_numbers: List[int], rep_dir: str,
                        sel_ix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Read ENM mode vectors and reduce them to the excitation atom subset.

        Args:
            base_name (str): Base name of the ENM run; files are read from
                ``{rep_dir}/{base_name}_enm/``.
            mode_numbers (list[int]): Mode numbers to read, in order.
            rep_dir (str): Directory containing the ENM folder.
            sel_ix (np.ndarray, optional): System indices of the excitation atoms.

        Returns:
            np.ndarray: Array of shape (n_modes, n_atoms, 3).

        Raises:
            FileNotFoundError: If the ENM directory or a mode file is missing.
            ValueError: If a selected atom is not protein/nucleic, or a mode file
                has an unexpected atom count.
            RuntimeError: If the assembled basis has an invalid shape.
        """
        enm_dir = os.path.join(rep_dir, f"{base_name}_enm")
        prefix = "ca" if self.args.model.lower() == 'ca' else "heavy"

        if not os.path.isdir(enm_dir):
            raise FileNotFoundError(f"ENM directory not found: {enm_dir}")

        enm_ix = np.asarray(self._ref_universe.select_atoms("protein or nucleic").ix, dtype=int)
        enm_local = {int(full_ix): local_ix for local_ix, full_ix in enumerate(enm_ix)}

        mode_local_ix = None
        if sel_ix is not None:
            sel_ix = np.asarray(sel_ix, dtype=int)
            missing = [int(ix) for ix in sel_ix if int(ix) not in enm_local]
            if missing:
                raise ValueError(
                    "Some selected atoms are not protein or nucleic atoms; "
                    f"first missing full-system indices: {missing[:10]}"
                )
            mode_local_ix = np.asarray([enm_local[int(ix)] for ix in sel_ix], dtype=int)

        modes = []
        for mode_num in mode_numbers:
            fn = os.path.join(enm_dir, f"{base_name}_{prefix}_mode_{mode_num}.xyz")
            if not os.path.exists(fn):
                raise FileNotFoundError(f"ENM mode file {fn} not found")

            u_mode = mda.Universe(fn, format="XYZ")
            arr = u_mode.atoms.positions.astype(float).copy()
            n_mode_atoms = arr.shape[0]

            if sel_ix is None:
                arr_sel = arr
            elif n_mode_atoms == len(enm_ix):
                arr_sel = arr[mode_local_ix]
            elif n_mode_atoms == self.n_atoms:
                arr_sel = arr[sel_ix]
            elif n_mode_atoms == len(sel_ix):
                arr_sel = arr
            else:
                raise ValueError(
                    f"Unexpected number of atoms in ENM mode file {fn}: "
                    f"{n_mode_atoms}. Expected {len(enm_ix)} protein/nucleic atoms, "
                    f"{self.n_atoms} system atoms, or {len(sel_ix)} selected atoms."
                )
            modes.append(arr_sel)

        basis = np.asarray(modes, dtype=float)
        if basis.ndim != 3 or basis.shape[-1] != 3:
            raise RuntimeError(f"Invalid ENM mode array shape: {basis.shape}")
        if sel_ix is not None and basis.shape[1] != len(sel_ix):
            raise RuntimeError(
                f"Reduced ENM basis contains {basis.shape[1]} atoms; expected {len(sel_ix)}."
            )
        return basis

    def _read_diversity_spectrum(self, base_name: str, search_modes: List[int],
                                 sel_ix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a beta reservoir spectrum written by ``write_diversity_spectrum``.

        Args:
            base_name (str): ENM base name (``step_{loop}`` or ``subspace_initial``).
            search_modes (list[int]): Expected reservoir labels, in order.
            sel_ix (np.ndarray): System indices of the excitation atoms.

        Returns:
            tuple: (basis, stiffness) with basis of shape (M, n_sel, 3) on the
                excitation atoms and stiffness the (M, M) modal matrix.

        Raises:
            ValueError: On schema/order mismatch, incomplete selection, or a
                basis that is not orthonormal.
        """
        prefix = "ca" if self.args.model.lower() == 'ca' else "heavy"
        path = os.path.join(f"{base_name}_enm", f"{base_name}_{prefix}_diversity_spectrum.npz")
        with np.load(path, allow_pickle=False) as data:
            if int(data['schema']) != 1 or list(data['mode_numbers']) != search_modes:
                raise ValueError("diversity spectrum mode order/schema mismatch")
            enm_ix = np.asarray(self._ref_universe.select_atoms("protein or nucleic").ix)
            full_ix = enm_ix[data['atom_input_indices']]
            if len(set(full_ix)) != len(full_ix) or not np.isin(full_ix, sel_ix).all():
                raise ValueError("beta requires the whole ENM representative-atom selection")
            local = {int(full): i for i, full in enumerate(sel_ix)}
            rows = np.array([local[int(i)] for i in full_ix])
            basis = np.zeros((len(search_modes), len(sel_ix), 3))
            basis[:, rows] = data['modes']
            flat = basis.reshape(len(search_modes), -1)
            if not np.isfinite(flat).all() or not np.allclose(
                    flat @ flat.T, np.eye(len(search_modes)), atol=1e-7):
                raise ValueError("beta spectrum is not a complete orthonormal mass-weighted basis")
            return basis, data['stiffness'].copy()

    # ------------------------------------------------------------------ #
    # State persistence                                                   #
    # ------------------------------------------------------------------ #

    def _save_cycle_state(self, loop: int, cnt: int, engine: OpenMMSimulationEngine) -> None:
        """
        Persist the adaptive state; in beta mode also seal the checkpoint so
        that the state and the checkpoint describe the same instant.

        Args:
            loop (int): Current cycle number.
            cnt (int): Correction counter.
            engine (OpenMMSimulationEngine): Active engine.
        """
        if self._diversity_enabled():
            engine.save_checkpoint('checkpoint.chk')
        self._save_correction_state(loop, cnt)

    def _save_correction_state(self, loop: int, cnt: int) -> None:
        """
        Persist in-memory correction state to disk.

        Written files:
          correction_state.json, _state_cntrl_vec.npy, _state_exc_vel_akma.npy,
          _state_correc_ref_pos_nm.npy, _state_align_ref_pos_nm.npy,
          _state_init_pos_nm.npy, _state_avg_pos_nm.npy,
          _state_enm_basis_modes.npy and _state_enm_basis_ref_pos_nm.npy
          (harmonized method), and in beta mode _state_enm_reservoir_modes.npy
          plus _state_enm_diversity.json (configuration and checksums).

        Args:
            loop (int): Current cycle number.
            cnt (int): Correction counter.

        Raises:
            RuntimeError: If beta is active but no reservoir is held.
        """
        np.save("_state_cntrl_vec.npy",         self._cntrl_vec)
        np.save("_state_exc_vel_akma.npy",      self._exc_vel_akma)
        np.save("_state_correc_ref_pos_nm.npy", self._correc_ref_pos_nm)
        np.save("_state_align_ref_pos_nm.npy",  self._align_ref_pos_nm)
        np.save("_state_init_pos_nm.npy",       self._init_pos_nm)
        avg = self._avg_pos_nm if self._avg_pos_nm is not None else self._correc_ref_pos_nm
        np.save("_state_avg_pos_nm.npy",        avg)
        if self._enm_basis_modes is not None:
            np.save("_state_enm_basis_modes.npy", self._enm_basis_modes)
        if self._enm_basis_ref_pos_nm is not None:
            np.save("_state_enm_basis_ref_pos_nm.npy", self._enm_basis_ref_pos_nm)
        if self._diversity_enabled():
            if self._enm_reservoir_modes is None:
                raise RuntimeError("cannot save beta state without the transported reservoir")
            np.save("_state_enm_reservoir_modes.npy", self._enm_reservoir_modes)
            search = self._get_subspace_search_modes(
                [int(x) for x in str(self.args.modes).split(',')])
            names = ('_state_cntrl_vec.npy', '_state_exc_vel_akma.npy',
                     '_state_correc_ref_pos_nm.npy', '_state_align_ref_pos_nm.npy',
                     '_state_init_pos_nm.npy', '_state_avg_pos_nm.npy',
                     '_state_enm_basis_modes.npy', '_state_enm_basis_ref_pos_nm.npy',
                     '_state_enm_reservoir_modes.npy')
            with open("_state_enm_diversity.json", "w") as stream:
                json.dump({
                    "schema": 3,
                    "beta": self._beta(),
                    "beta_count": self._beta_count(),
                    "alpha": self._alpha(),
                    "active_modes": str(self.args.modes),
                    "search_modes": search,
                    "metric": "mass-weighted/current-md-frame-v1",
                    "cycle": loop,
                    "cnt": cnt,
                    "checkpoint_sha256": self._state_digest('checkpoint.chk'),
                    "state_sha256": {name: self._state_digest(name) for name in names},
                }, stream, indent=2)
        with open("correction_state.json", 'w') as fh:
            json.dump({'cycle': loop, 'cnt': cnt,
                       'qrms_correc': self.qrms_correc}, fh, indent=2)

    def _load_correction_state(self) -> Tuple[int, int, float]:
        """
        Restore in-memory correction state from disk.

        In beta mode the saved configuration, checksums and checkpoint are
        verified; any mismatch aborts the restart instead of silently
        reinitializing the reservoir.

        Returns:
            tuple: (loop, cnt, qrms_correc).

        Raises:
            FileNotFoundError: If a required file is missing.
            ValueError: If beta restart data are inconsistent.
        """
        self._cntrl_vec         = np.load("_state_cntrl_vec.npy")
        self._exc_vel_akma      = np.load("_state_exc_vel_akma.npy")
        self._correc_ref_pos_nm = np.load("_state_correc_ref_pos_nm.npy")
        self._align_ref_pos_nm  = np.load("_state_align_ref_pos_nm.npy")
        if os.path.exists("_state_init_pos_nm.npy"):
            self._init_pos_nm = np.load("_state_init_pos_nm.npy")
        if os.path.exists("_state_avg_pos_nm.npy"):
            try:
                self._avg_pos_nm = np.load("_state_avg_pos_nm.npy")
            except ValueError:
                self._avg_pos_nm = self._correc_ref_pos_nm.copy()
        if os.path.exists("_state_enm_basis_modes.npy"):
            self._enm_basis_modes = np.load("_state_enm_basis_modes.npy")
        if os.path.exists("_state_enm_basis_ref_pos_nm.npy"):
            self._enm_basis_ref_pos_nm = np.load("_state_enm_basis_ref_pos_nm.npy")

        state_cycle = state_cnt = None
        if self._diversity_enabled():
            if not os.path.isfile("_state_enm_diversity.json"):
                raise ValueError("beta restart requires _state_enm_diversity.json")
            if not os.path.isfile("_state_enm_reservoir_modes.npy"):
                raise ValueError("beta restart requires the saved reservoir")
            with open("_state_enm_diversity.json") as stream:
                metadata = json.load(stream)
            state_cycle = metadata.pop('cycle')
            state_cnt = metadata.pop('cnt')
            checkpoint_hash = metadata.pop('checkpoint_sha256')
            hashes = metadata.pop('state_sha256')
            for name, digest in hashes.items():
                if self._state_digest(name) != digest:
                    raise ValueError(f"beta restart contains inconsistent state: {name}")
            if checkpoint_hash is not None and self._state_digest('checkpoint.chk') != checkpoint_hash:
                raise ValueError("beta checkpoint does not match saved adaptive state")
            search = self._get_subspace_search_modes(
                [int(x) for x in str(self.args.modes).split(',')])
            expected = {"schema": 3,
                        "beta": self._beta(),
                        "beta_count": self._beta_count(),
                        "alpha": self._alpha(),
                        "active_modes": str(self.args.modes),
                        "search_modes": search,
                        "metric": "mass-weighted/current-md-frame-v1"}
            if metadata.get("schema") == 2 and len(str(self.args.modes).split(",")) == 3:
                metadata = {"beta_count": 3, **metadata, "schema": 3}
            if metadata != expected:
                raise ValueError("beta restart state differs from saved configuration")
            self._enm_reservoir_modes = np.load("_state_enm_reservoir_modes.npy")
            if self._enm_basis_modes is None or self._enm_basis_ref_pos_nm is None:
                raise ValueError("beta restart requires both active basis and reference")
            n_sel = len(self._enm_selection_atoms())
            m = len(str(self.args.modes).split(","))
            if self._enm_basis_modes.shape != (m, n_sel, 3) or \
                    not np.isfinite(self._enm_basis_modes).all():
                raise ValueError("invalid active basis in beta restart state")
            if self._enm_reservoir_modes.shape != (len(search), n_sel, 3) or \
                    not np.isfinite(self._enm_reservoir_modes).all():
                raise ValueError("invalid reservoir in beta restart state")

        with open("correction_state.json") as fh:
            cs = json.load(fh)
        if self._diversity_enabled() and (cs['cycle'] != state_cycle or cs['cnt'] != state_cnt):
            raise ValueError("beta scalar state does not match the sealed adaptive state")
        return cs['cycle'], cs['cnt'], cs['qrms_correc']

    # ------------------------------------------------------------------ #
    # Replica driver                                                      #
    # ------------------------------------------------------------------ #

    def run_simulation(self, rep: int, start_loop: int, end_loop: int,
                       correction_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run (or resume) an aMDeNM replica entirely within OpenMM.

        Args:
            rep (int): Replica index (1-based).
            start_loop (int): 0 for a fresh run; last completed cycle for restart.
            end_loop (int): Total number of cycles to reach.
            correction_state (dict, optional): Kept for call-site compatibility;
                state is loaded from disk on restart.

        Returns:
            dict with keys 'cnt' and 'qrms_correc'.
        """
        rep_dir = f"{self.cwd}/rep{rep}"

        _sel   = self._enm_selection_atoms()
        sel_ix: np.ndarray     = _sel.ix        # (n_sel,) system atom indices
        sel_masses: np.ndarray = _sel.masses    # (n_sel,) amu
        del _sel

        if getattr(self.args, 'recalc', False):
            self._validate_enm_selection(sel_ix)

        ## == NEW RUN == ##
        if start_loop == 0:
            if self.args.no_correc:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Standard MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}")
            elif self.args.fixed:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Constant MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}")
            else:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Adaptive MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}")

            os.makedirs(rep_dir, exist_ok=True)
            os.chdir(rep_dir)

            # The runner is shared by all replicas: every replica starts with the
            # initial RMS window, no average structure and no stale ENM basis.
            self.qrms_correc = self.globfreq
            self._avg_pos_nm = None
            self._enm_basis_modes = None
            self._enm_basis_ref_pos_nm = None
            self._enm_reservoir_modes = None

            vec_file = f"{self.cwd}/rep-struct-list/rep{rep}_vector.vec"
            shutil.copy(vec_file, "pff_vector.vec")   # traceability copy
            q_vec_full = NAMDInputReader.read_nm_vector(self.psffile, vec_file)
            q_vec_sel  = q_vec_full[sel_ix]           # selection subset, Å

            exc_vel_sel = self._excite_control_vector(q_vec_sel, self.energy, sel_masses)
            print(f"{self.console.PGM_NAM}Writing the excitation vector with a Ek injection of {self.console.EXT}{self.energy}{self.console.STD} kcal/mol.")

            exc_vel_nm_ps = exc_vel_sel * AKMA_VEL_TO_NM_PS
            replica_velocities_nm_ps = self._init_state.velocities_nm_ps.copy()
            replica_velocities_nm_ps[sel_ix] += exc_vel_nm_ps
            replica_init_state = SystemState(
                positions_nm=self._init_state.positions_nm,
                velocities_nm_ps=replica_velocities_nm_ps,
                box_vectors_nm=self._init_state.box_vectors_nm,
            )

            self._cntrl_vec    = np.zeros((self.n_atoms, 3))
            self._cntrl_vec[sel_ix] = q_vec_sel
            self._exc_vel_akma = np.zeros((self.n_atoms, 3))
            self._exc_vel_akma[sel_ix] = exc_vel_sel

            self.mode_exciter._write_vector(self._cntrl_vec, "cntrl_vector.vec", self.sys_coor)
            self.mode_exciter._write_vector(self._exc_vel_akma, "excitation.vel", self.sys_coor)

            # In harmonized mode the archive index N is the N-th Q_final;
            # the initial Q is vector set 1.
            if self._is_harmonized():
                shutil.copy("cntrl_vector.vec", "cntrl_vector.vec.1")
                shutil.copy("excitation.vel", "excitation.vel.1")

            self._curr_pos_nm       = self._init_pos_nm.copy()
            self._prev_pos_nm       = self._init_pos_nm.copy()
            self._correc_ref_pos_nm = self._init_pos_nm.copy()
            self._align_ref_pos_nm  = self._init_pos_nm.copy()

            if self._is_harmonized():
                nm_list = [int(s) for s in str(self.args.modes).split(',')]
                self._initialize_enm_subspace(nm_list, sel_ix)

            loop = 0
            cnt  = 1
            vp, ek, qp, rmsp = [], [], [], []

            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                platform_name=self._platform,
                n_threads=self._n_threads,
                device_index=0,
                rep_num=rep,
                is_restart=False,
                full_ener=getattr(self.args, 'full_ener', False),
                n_steps=getattr(self.args, 'n_steps', 50),
                friction_per_ps=self._nh_frequency,
            )
            engine.initialize_state(replica_init_state)

        ## == RESTART / APPEND == ##
        else:
            os.chdir(rep_dir)

            self._enm_basis_modes      = None
            self._enm_basis_ref_pos_nm = None
            self._enm_reservoir_modes  = None

            try:
                loop, cnt, self.qrms_correc = self._load_correction_state()
            except FileNotFoundError:
                if self._diversity_enabled():
                    raise RuntimeError("beta restart is incomplete; refusing legacy reinitialization")
                if correction_state and os.path.exists("correction_state.json"):
                    with open("correction_state.json") as fh:
                        cs = json.load(fh)
                    cnt              = cs.get('cnt', 1)
                    self.qrms_correc = cs.get('qrms_correc', 0.5)
                    loop             = start_loop
                    print(f"{self.console.PGM_WRN}No .npy state files found for "
                          f"replica {rep}. In-memory vectors re-initialised from "
                          f"combination file.")
                    vec_file = f"{self.cwd}/rep-struct-list/rep{rep}_vector.vec"
                    q_vec_full = NAMDInputReader.read_nm_vector(self.psffile, vec_file)
                    q_vec_sel  = q_vec_full[sel_ix]
                    exc_vel_sel = self._excite_control_vector(q_vec_sel, self.energy, sel_masses)
                    self._cntrl_vec    = np.zeros((self.n_atoms, 3))
                    self._cntrl_vec[sel_ix] = q_vec_sel
                    self._exc_vel_akma = np.zeros((self.n_atoms, 3))
                    self._exc_vel_akma[sel_ix] = exc_vel_sel
                    self._correc_ref_pos_nm = self._init_pos_nm.copy()
                    self._align_ref_pos_nm  = self._init_pos_nm.copy()
                else:
                    raise RuntimeError(
                        f"Cannot restart {self.console.ERR}Replica {rep}{self.console.STD}: correction_state.json missing."
                    )

            # DCD SYNC: one frame per cycle; the DCD header gives the frame count
            # without needing a topology.
            dcd_path = f"rep{rep}.dcd"
            if os.path.exists(dcd_path):
                n_dcd_frames = _count_dcd_frames(dcd_path)
                if n_dcd_frames > 0:
                    if self._diversity_enabled() and n_dcd_frames != loop:
                        raise ValueError("beta DCD/checkpoint/state cycles differ; "
                                         "refusing implicit synchronization")
                    if n_dcd_frames > loop:
                        print(f"{self.console.PGM_NAM}DCD sync: advancing loop from "
                              f"{self.console.WRN}{loop}{self.console.STD} to {self.console.EXT}{n_dcd_frames}{self.console.STD} "
                              f"frames found in {self.console.EXT}{dcd_path}{self.console.STD}.")
                        loop = n_dcd_frames
                    else:
                        print(f"{self.console.PGM_NAM}DCD sync: {self.console.EXT}{dcd_path}{self.console.STD} has "
                              f"{self.console.EXT}{n_dcd_frames}{self.console.STD} frames, consistent with JSON cycle "
                              f"{self.console.EXT}{loop}{self.console.STD}.")
                else:
                    print(f"{self.console.PGM_WRN}Could not read DCD header for "
                          f"{self.console.WRN}Replica {rep}{self.console.STD} ({self.console.WRN}{dcd_path}{self.console.STD}). "
                          f"Proceeding with JSON cycle {self.console.WRN}{loop}{self.console.STD}.")

            self._curr_pos_nm = self._init_pos_nm.copy()
            self._prev_pos_nm = self._init_pos_nm.copy()

            vp, ek, qp, rmsp = [], [], [], []
            for fname, lst_name in [
                ("vp-proj.out",   "vp"),
                ("ek-proj.out",   "ek"),
                ("coor-proj.out", "qp"),
                ("rms-proj.out",  "rmsp"),
            ]:
                if os.path.exists(fname):
                    with open(fname) as fh:
                        lines = fh.readlines()
                    if lst_name == "vp":   vp   = lines
                    elif lst_name == "ek": ek   = lines
                    elif lst_name == "qp": qp   = lines
                    else:                  rmsp = lines

            if self.args.no_correc:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Restarting Standard MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}"
                      f"{self.console.HGH} from cycle {self.console.EXT}{loop}{self.console.STD}")
            elif self.args.fixed:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Restarting Constant MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}"
                      f"{self.console.HGH} from cycle {self.console.EXT}{loop}{self.console.STD}")
            else:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Restarting Adaptive MDeNM "
                      f"for {self.console.EXT}Replica {rep}{self.console.STD}"
                      f"{self.console.HGH} from cycle {self.console.EXT}{loop}{self.console.STD}")

            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                platform_name=self._platform,
                n_threads=self._n_threads,
                device_index=0,
                rep_num=rep,
                is_restart=True,
                full_ener=getattr(self.args, 'full_ener', False),
                n_steps=getattr(self.args, 'n_steps', 50),
                friction_per_ps=self._nh_frequency,
            )
            chk = "checkpoint.chk"
            if not os.path.exists(chk):
                raise FileNotFoundError(
                    f"{chk} not found in {rep_dir}. Cannot restart replica {rep}."
                )
            engine.load_checkpoint(chk)

            # Align the OpenMM step counter with the cycle count so the DCD and
            # checkpoint reporters fire on the correct cycles after restart.
            _n_steps_per_cycle = getattr(self.args, 'n_steps', 50)
            _expected_step     = loop * _n_steps_per_cycle
            _chk_step          = engine.simulation.currentStep
            _chk_time_ps       = engine.simulation.context.getState().getTime().value_in_unit(unit.picosecond)
            _chk_cycle         = _chk_step // _n_steps_per_cycle
            print(f"{self.console.PGM_NAM}Checkpoint loaded: "
                  f"OpenMM step {_chk_step} ({_chk_time_ps:.2f} ps, "
                  f"~cycle {_chk_cycle}); DCD/JSON cycle is {loop}.")
            if _chk_step != _expected_step:
                if self._diversity_enabled():
                    raise ValueError("beta checkpoint step differs from adaptive state")
                engine.simulation.currentStep = _expected_step
            if self._diversity_enabled():
                loaded = engine.simulation.context.getState(getPositions=True)
                self._curr_pos_nm = np.asarray(
                    loaded.getPositions(asNumpy=True).value_in_unit(unit.nanometer)).copy()
                self._prev_pos_nm = self._curr_pos_nm.copy()

        # ═══════════════════════════════════════════════════════════════════
        # MAIN SIMULATION LOOP
        # ═══════════════════════════════════════════════════════════════════
        while loop < end_loop:
            loop += 1
            now = time.strftime("%H:%M:%S")
            print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: running "
                  f"{self.console.EXT}step {self.console.WRN}{loop}{self.console.STD}/{self.console.EXT}{end_loop}{self.console.STD}...")

            # RUN MD CYCLE
            pos_nm, vel_nm_ps, box_nm = engine.run_cycle(self.args.n_steps, rep=rep, loop=loop)

            self._prev_pos_nm = self._curr_pos_nm.copy()
            self._curr_pos_nm = pos_nm

            # DIRECTION CORRECTION CHECK
            # The rms_check measures how far the system has moved along Q since the
            # last correction reference.  When it reaches qrms_correc, _correct_excitation_direction
            # is called, which unconditionally advances the reference window and
            # conditionally replaces Q (only when motion has diverged > 60° from Q).
            # qrms_correc is incremented here (outer gate) so the window keeps growing
            # even in cycles where the angle test does not trigger a replacement.
            if not self.args.no_correc and not self.args.fixed:
                curr_sel  = self._curr_pos_nm[sel_ix]       * 10.0  # nm to Å
                ref_sel   = self._correc_ref_pos_nm[sel_ix]  * 10.0

                diff_corr   = ((curr_sel - ref_sel).T * np.sqrt(sel_masses)).T
                cntrl_sel   = self._cntrl_vec[sel_ix]

                q_proj_chk  = np.sum(diff_corr * cntrl_sel)
                rms_check   = np.sqrt((q_proj_chk ** 2) / np.sum(sel_masses))

                if rms_check >= self.qrms_correc:
                    if getattr(self.args, 'recalc', False):
                        nm_list = [int(s) for s in str(self.args.modes).split(',')]

                        if self._is_harmonized():
                            # Q is replaced only when the angular test is satisfied;
                            # 1 correction = 1 ENM recalculation = 1 transport = 1 Q_final.
                            cnt_before = cnt
                            cnt = self._correct_excitation_direction(
                                rep, loop, self._cntrl_vec, cnt,
                                sel_ix, sel_masses
                            )
                            if cnt > cnt_before:
                                self._recompute_enm_modes(rep, loop, nm_list, cnt)
                        else:
                            # Random method (3.4.0 behavior)
                            self._recompute_enm_modes(rep, loop, nm_list, cnt)
                            cnt += 1
                    else:
                        cnt = self._correct_excitation_direction(
                            rep, loop,
                            self._cntrl_vec, cnt,
                            sel_ix, sel_masses
                        )
                    self.qrms_correc += self.globfreq

            # VELOCITIES AND KINETIC ENERGY PROJECTED ONTO Q
            vel_akma    = vel_nm_ps / AKMA_VEL_TO_NM_PS
            vel_akma_mw = (vel_akma.T * np.sqrt(self.sys_mass)).T

            velo        = np.sum(vel_akma_mw * self._cntrl_vec)
            v_proj_akma = self._cntrl_vec * velo
            ek_vel      = 0.5 * np.sum(v_proj_akma ** 2)

            vp.append(f"{round(velo,   5)}\n")
            ek.append(f"{round(ek_vel, 5)}\n")

            # PROJECT THE COORDINATES ONTO Q
            curr_sel_coor = self._curr_pos_nm[sel_ix] * 10.0    # nm to Å
            init_sel_coor = self._init_pos_nm[sel_ix]  * 10.0

            diff_coor  = np.zeros((self.n_atoms, 3))
            diff_coor[sel_ix] = ((curr_sel_coor - init_sel_coor).T * np.sqrt(sel_masses)).T

            q_proj_coor = np.sum(diff_coor * self._cntrl_vec)
            mrms        = np.sqrt((q_proj_coor ** 2) / np.sum(sel_masses))
            qp.append(f"{round(q_proj_coor, 5)}\n")
            rmsp.append(f"{round(mrms,       5)}\n")

            if self.args.no_correc:
                if loop % 10 == 0:
                    self._save_cycle_state(loop, cnt, engine)
                continue

            # RESCALE KINETIC ENERGY ALONG Q (±25% band around the target)
            if (ek_vel < self.bottom) or (ek_vel > self.top):
                v_proj_akma_phys = (v_proj_akma.T / np.sqrt(self.sys_mass)).T  # Å/AKMA
                new_vel_akma     = vel_akma + (self._exc_vel_akma - v_proj_akma_phys)
                engine.set_velocities(new_vel_akma * AKMA_VEL_TO_NM_PS)

            if loop % 10 == 0:
                self._save_cycle_state(loop, cnt, engine)

        if self._diversity_enabled():
            self._save_cycle_state(loop, cnt, engine)

        for data, tag in zip((vp, ek, qp, rmsp), ("vp", "ek", "coor", "rms")):
            with open(f"{tag}-proj.out", 'w') as fh:
                fh.writelines(data)

        engine.close()
        del engine
        self._release_memory()

        return {'cnt': cnt, 'qrms_correc': self.qrms_correc}

    # ------------------------------------------------------------------ #
    # Harmonized ENM subspace tracking                                    #
    # ------------------------------------------------------------------ #

    def _enm_base_name(self) -> str:
        """
        Return the base name of the ENM computed by ``pyadmd run``.

        Returns:
            str: Stem of the NAMD ``.coor`` or OpenMM ``.rst`` file.
        """
        if str(getattr(self.args, 'source', 'NAMD')).upper() == 'NAMD':
            path = self.args.coorfile
        else:
            path = self.args.rstfile
        return os.path.splitext(os.path.basename(path))[0]

    def _initialize_enm_subspace(self, nm_parsed: List[int], sel_ix: np.ndarray) -> None:
        """
        Establish the initial active ENM subspace U_m.

        The ENM computed by ``pyadmd run`` (``inputs/{base_name}_enm/``) is
        reduced to the excitation atoms. If it is missing, the ENM is computed
        on the current structure. In beta mode the reservoir is initialized
        instead (see ``_initialize_diversity_subspace``).

        Args:
            nm_parsed (list[int]): Active mode numbers (dimension m).
            sel_ix (np.ndarray): System indices of the excitation atoms.

        Raises:
            RuntimeError: If the basis atom count does not match ``sel_ix``.
        """
        if self._diversity_enabled():
            self._initialize_diversity_subspace(nm_parsed, sel_ix)
            return
        base_name = self._enm_base_name()
        try:
            self._enm_basis_modes = self._read_enm_basis(
                base_name, nm_parsed, self.input_dir, sel_ix=sel_ix
            )
            self._enm_basis_ref_pos_nm = self._init_pos_nm.copy()
            origin = f"inputs/{base_name}_enm"
        except FileNotFoundError:
            print(f"{self.console.PGM_WRN}Initial ENM not found in "
                  f"{self.console.WRN}inputs/{base_name}_enm{self.console.STD}; "
                  "computing it on the current structure.")
            sub_name = "subspace_initial"
            ENMCalculator(self.console).compute_enm(
                positions_ang=self._curr_pos_nm * 10.0,
                base_name=sub_name,
                nm_type=self.args.model.lower(),
                nm_parsed=nm_parsed,
                input_dir=os.getcwd(),
                psffile=self.psffile,
                rtb=getattr(self.args, 'rtb', False),
            )
            self._enm_basis_modes = self._read_enm_basis(
                sub_name, nm_parsed, os.getcwd(), sel_ix=sel_ix
            )
            self._enm_basis_ref_pos_nm = self._curr_pos_nm.copy()
            origin = "computed on the current structure"

        if self._enm_basis_modes.shape[1] != len(sel_ix):
            raise RuntimeError(
                f"ENM selection mismatch: modes contain {self._enm_basis_modes.shape[1]} atoms "
                f"but adaptive selection contains {len(sel_ix)}"
            )
        print(f"{self.console.PGM_NAM}Initial ENM subspace stored ({origin}): "
              f"{self.console.EXT}{len(nm_parsed)}{self.console.STD} modes on "
              f"{self.console.EXT}{len(sel_ix)}{self.console.STD} selected atoms.")

    def _initialize_diversity_subspace(self, nm_parsed: List[int], sel_ix: np.ndarray) -> None:
        """
        Initialize the beta reservoir (M search modes) and the active basis U_m.

        Args:
            nm_parsed (list[int]): Active mode numbers.
            sel_ix (np.ndarray): System indices of the excitation atoms.

        Raises:
            ValueError: If the reservoir spectrum is unstable or near-singular.
        """
        search = self._get_subspace_search_modes(nm_parsed)
        base = "subspace_initial"
        write_diversity_spectrum(
            ENMCalculator(self.console),
            positions_ang=self._curr_pos_nm * 10.0,
            base_name=base,
            nm_type=self.args.model.lower(),
            search_modes=search,
            input_dir=os.getcwd(),
            psffile=self.psffile,
            max_modes=max(50, max(search) + 1),
            rtb=getattr(self.args, 'rtb', False),
        )
        reservoir, stiffness = self._read_diversity_spectrum(base, search, sel_ix)
        eigenvalues = np.linalg.eigvalsh(stiffness)
        if eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-8 * eigenvalues[-1]:
            raise ValueError("initial beta reservoir contains unstable or near-zero internal modes")
        self._enm_reservoir_modes = reservoir.copy()
        self._enm_basis_modes = reservoir[[search.index(i) for i in nm_parsed]].copy()
        self._enm_basis_ref_pos_nm = self._curr_pos_nm.copy()
        print(f"{self.console.PGM_NAM}Beta reservoir initialized: {len(search)} directions; "
              f"active modes {nm_parsed}; beta={self._beta()}, "
              f"beta count={self._beta_count()}.")

    def _transport_q_through_enm_subspace(self, rep: int, loop: int,
                                          nm_parsed: List[int],
                                          search_modes: List[int],
                                          cnt: int, base_name: str,
                                          sel_ix: np.ndarray,
                                          sel_masses: np.ndarray) -> None:
        """
        Track U_m inside V_M, harmonize it, and transport Q_corr.

        Reads the recomputed search modes V_M, calls ``harmonize_and_transport``
        (Kabsch alignment, SVD of the overlap S = U_m^T V_M, Procrustes
        harmonization, projection of Q_corr onto V_harm, α-mixing), updates the
        in-memory excitation state, archives Q_final, promotes V_harm to the new
        active subspace, and appends the diagnostics ``subspace_overlap.csv`` and
        ``subspace_matching_coefficients.csv``.

        Reached only after a true angular correction, so the current
        ``self._cntrl_vec`` is Q_corr.

        Args:
            rep (int): Replica number (logging/diagnostics only).
            loop (int): Current cycle number.
            nm_parsed (list[int]): Active mode numbers (dimension m).
            search_modes (list[int]): Search-space mode numbers (dimension M ≥ m).
            cnt (int): Correction counter (already incremented for this correction).
            base_name (str): Base name of the ENM run holding the V_M mode files.
            sel_ix (np.ndarray): System indices of the excitation atoms.
            sel_masses (np.ndarray): Masses of the excitation atoms (amu).

        Raises:
            RuntimeError: If the search-space atom count does not match ``sel_ix``.
            ValueError: If ``--subspace-alpha`` is outside [0, 1].
        """
        if self._diversity_enabled():
            self._transport_diversity(rep, loop, nm_parsed, search_modes, cnt,
                                      base_name, sel_ix, sel_masses)
            return
        if self._enm_basis_modes is None or self._enm_basis_ref_pos_nm is None:
            # Restart compatibility: establish U_m at the current structure
            # when no persisted basis is available.
            self._initialize_enm_subspace(nm_parsed, sel_ix)

        new_modes = self._read_enm_basis(
            base_name, search_modes, os.getcwd(), sel_ix=sel_ix
        )
        if new_modes.shape[1] != len(sel_ix):
            raise RuntimeError(
                f"ENM search-space selection mismatch: modes contain {new_modes.shape[1]} atoms "
                f"but adaptive selection contains {len(sel_ix)}"
            )

        q_old_sel = self._cntrl_vec[sel_ix]
        alpha = self._alpha()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"--subspace-alpha must be between 0 and 1; got {alpha}")

        result = harmonize_and_transport(
            old_modes=self._enm_basis_modes,
            new_modes=new_modes,
            q_old=q_old_sel,
            old_coords=self._enm_basis_ref_pos_nm[sel_ix] * 10.0,
            new_coords=self._curr_pos_nm[sel_ix] * 10.0,
            masses=sel_masses,
            alpha=alpha,
        )

        # U_m frame correction: outputs are in the stored reference frame; rotate
        # them into the current MD frame (same convention as the beta path).
        back = result.rotation.T
        result.q_new = result.q_new @ back
        result.harmonized_modes = result.harmonized_modes @ back

        q_full = np.zeros((self.n_atoms, 3))
        q_full[sel_ix] = result.q_new
        self._cntrl_vec = q_full

        exc_sel = self._excite_control_vector(result.q_new, self.energy, sel_masses)
        exc_full = np.zeros((self.n_atoms, 3))
        exc_full[sel_ix] = exc_sel
        self._exc_vel_akma = exc_full

        self.mode_exciter._write_vector(q_full, "cntrl_vector.vec", self.sys_coor)
        self.mode_exciter._write_vector(exc_full, "excitation.vel", self.sys_coor)

        # Archive Q_final
        shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")
        shutil.copy("excitation.vel", f"excitation.vel.{cnt}")

        # The harmonized m-dimensional subspace becomes U_m for the next
        # correction; the MD system is unchanged.
        self._enm_basis_modes = result.harmonized_modes.copy()
        self._enm_basis_ref_pos_nm = self._curr_pos_nm.copy()

        # Diagnostics for m -> M subspace tracking
        diag = "subspace_overlap.csv"
        new_file = not os.path.exists(diag)
        with open(diag, "a", newline="") as fh:
            w = csv.writer(fh)
            if new_file:
                w.writerow([
                    "replica", "cycle", "correction", "m", "M",
                    "active_modes", "search_modes", "omega", "sigma",
                    "angles_deg", "alpha"
                ])
            w.writerow([
                rep, loop, cnt, len(nm_parsed), len(search_modes),
                ";".join(str(x) for x in nm_parsed),
                ";".join(str(x) for x in search_modes),
                result.omega,
                ";".join(f"{x:.8f}" for x in result.singular_values),
                ";".join(f"{x:.5f}" for x in result.principal_angles_deg),
                alpha
            ])

        # Coefficients of the M search modes in the m matched directions
        coeff_diag = "subspace_matching_coefficients.csv"
        coeff_new_file = not os.path.exists(coeff_diag)
        with open(coeff_diag, "a", newline="") as fh:
            w = csv.writer(fh)
            if coeff_new_file:
                w.writerow([
                    "replica", "cycle", "correction",
                    "search_mode", "matched_direction", "coefficient"
                ])
            for i, mode_number in enumerate(search_modes):
                for j in range(result.matching_coefficients.shape[1]):
                    w.writerow([
                        rep, loop, cnt, mode_number, j + 1,
                        result.matching_coefficients[i, j]
                    ])

        print(
            f"{self.console.PGM_NAM}ENM subspace transported Q at cycle {loop}: "
            f"m={len(nm_parsed)}, M={len(search_modes)}, "
            f"Omega={result.omega:.4f}, "
            f"sigma_min={result.singular_values.min():.4f}, alpha={alpha:.3f}."
        )

    def _transport_diversity(self, rep: int, loop: int, nm_parsed: List[int],
                             search_modes: List[int], cnt: int, base_name: str,
                             sel_ix: np.ndarray, sel_masses: np.ndarray) -> None:
        """
        Beta update: transport the reservoir, add the transverse directions, and
        store the new direction, active basis and reservoir in the current frame.

        Args:
            rep (int): Replica number (diagnostics only).
            loop (int): Current cycle number.
            nm_parsed (list[int]): Active mode numbers.
            search_modes (list[int]): Reservoir mode labels.
            cnt (int): Correction counter (already incremented).
            base_name (str): ENM base name of the recomputed spectrum.
            sel_ix (np.ndarray): System indices of the excitation atoms.
            sel_masses (np.ndarray): Masses of the excitation atoms (amu).

        Raises:
            RuntimeError: If the reservoir or active basis is missing.
        """
        if self._enm_reservoir_modes is None or self._enm_basis_ref_pos_nm is None:
            raise RuntimeError("beta update requires a persisted reservoir and reference")
        new_modes, stiffness = self._read_diversity_spectrum(base_name, search_modes, sel_ix)
        result = harmonize_with_diversity(
            self._enm_basis_modes, new_modes, self._cntrl_vec[sel_ix],
            self._enm_basis_ref_pos_nm[sel_ix] * 10.0,
            self._curr_pos_nm[sel_ix] * 10.0, sel_masses,
            self._alpha(), self._beta(),
            self._enm_reservoir_modes, stiffness,
            beta_count=self._beta_count(),
        )
        q = result.transport.q_new
        excitation = self._excite_control_vector(q, self.energy, sel_masses)
        self._cntrl_vec = np.zeros((self.n_atoms, 3))
        self._exc_vel_akma = np.zeros((self.n_atoms, 3))
        self._cntrl_vec[sel_ix] = q
        self._exc_vel_akma[sel_ix] = excitation
        self._enm_basis_modes = result.transport.harmonized_modes.copy()
        self._enm_reservoir_modes = result.reservoir_modes.copy()
        self._enm_basis_ref_pos_nm = self._curr_pos_nm.copy()

        for array, filename in ((self._cntrl_vec, 'cntrl_vector.vec'),
                                (self._exc_vel_akma, 'excitation.vel')):
            self.mode_exciter._write_vector(array, filename, self.sys_coor)
            shutil.copy(filename, f"{filename}.{cnt}")

        np.savez(f"subspace_diversity_{cnt}.npz", schema=np.array(1),
                 search_mode_labels=np.asarray(search_modes),
                 selected=result.selected, excluded=result.excluded,
                 coefficients=result.coefficients,
                 projections=result.projections, transverse=result.transverse,
                 stiffness=result.stiffness, scores=result.scores,
                 overlap_active=result.overlap_active,
                 reservoir_coefficients=result.reservoir_coefficients,
                 modal_stiffness=stiffness,
                 reservoir_singular_values=result.reservoir_singular_values,
                 modal_coefficients=result.modal_coefficients)

        path = "subspace_candidates.csv"
        fresh = not os.path.exists(path)
        with open(path, 'a', newline='') as stream:
            writer = csv.writer(stream)
            if fresh:
                writer.writerow(['replica', 'cycle', 'correction', 'tracked_label',
                                 'projection_Qc', 'transverse', 'stiffness', 'score',
                                 'overlap_alpha', 'excluded', 'selected', 'coefficient'])
            coefficients = dict(zip(result.selected, result.coefficients))
            for j, label in enumerate(search_modes):
                writer.writerow([rep, loop, cnt, label, result.projections[j],
                                 result.transverse[j], result.stiffness[j], result.scores[j],
                                 result.overlap_active[j], j in result.excluded,
                                 j in result.selected, coefficients.get(j, 0.0)])

        path = "subspace_diversity.csv"
        fresh = not os.path.exists(path)
        with open(path, 'a', newline='') as stream:
            writer = csv.writer(stream)
            if fresh:
                writer.writerow(['replica', 'cycle', 'correction', 'alpha', 'beta',
                                 'excluded_labels', 'selected_labels', 'coefficients',
                                 'omega', 'sigma', 'angles_deg', 'deviation_deg',
                                 'cosine_continuity', 'mixture_stiffness', 'cross_term',
                                 'modal_participation', 'collectivity'])
            labels = lambda indices: ';'.join(str(search_modes[i]) for i in indices)
            values = lambda array: ';'.join(f'{x:.12g}' for x in array)
            writer.writerow([rep, loop, cnt, self._alpha(), self._beta(),
                             labels(result.excluded), labels(result.selected),
                             values(result.coefficients), result.transport.omega,
                             values(result.transport.singular_values),
                             values(result.transport.principal_angles_deg),
                             result.deviation_deg, result.cosine_continuity,
                             result.mixture_stiffness, result.mixture_cross_term,
                             result.modal_participation, result.collectivity])

        path = "subspace_overlap.csv"
        fresh = not os.path.exists(path)
        with open(path, 'a', newline='') as stream:
            writer = csv.writer(stream)
            if fresh:
                writer.writerow(['replica', 'cycle', 'correction', 'm', 'M', 'active_modes',
                                 'search_modes', 'omega', 'sigma', 'angles_deg', 'alpha'])
            writer.writerow([rep, loop, cnt, len(nm_parsed), len(search_modes),
                             ';'.join(map(str, nm_parsed)), ';'.join(map(str, search_modes)),
                             result.transport.omega,
                             values(result.transport.singular_values),
                             values(result.transport.principal_angles_deg), self._alpha()])

        print(f"{self.console.PGM_NAM}Beta transport at cycle {loop}: "
              f"alpha={self._alpha()}, beta={self._beta()}, "
              f"selected tracked labels={labels(result.selected)}, "
              f"deviation={result.deviation_deg:.3f} deg, "
              f"cos(Qnew,Qc)={result.cosine_continuity:.6f}.")

    # ------------------------------------------------------------------ #
    # ENM recalculation                                                   #
    # ------------------------------------------------------------------ #

    def _recompute_enm_modes(self, rep: int, loop: int, nm_parsed: List[int], cnt: int) -> None:
        """
        Recompute ENM modes and update Q, then release the memory used.

        Thin wrapper around ``_recompute_enm_modes_impl`` that brackets the
        recalculation with the opt-in memory log and returns freed memory to
        the operating system afterwards (see ``_release_memory``).

        Args:
            rep (int): Replica number (logging only).
            loop (int): Current simulation cycle number.
            nm_parsed (list[int]): Active mode numbers.
            cnt (int): Current correction counter.
        """
        try:
            self._recompute_enm_modes_impl(rep, loop, nm_parsed, cnt)
        finally:
            self._release_memory()

    def _recompute_enm_modes_impl(self, rep: int, loop: int, nm_parsed: List[int], cnt: int) -> None:
        """
        Recompute ENM modes from the current structure and update Q.

        Two methods, selected by ``--recalc-method``:

          - ``random`` (default, 3.4.0 behavior): recompute the active modes
            and draw a new random unit-vector combination of them
            (``_generate_new_excitation_vector``).
          - ``harmonized``: recompute the search space M ⊇ m and transport
            Q_corr through the harmonized subspace
            (``_transport_q_through_enm_subspace``).

        Both honor ``--rtb`` in the ENM diagonalization. The structure used is
        saved by ``ENMCalculator.compute_enm`` in
        ``step_{loop}_enm/step_{loop}.pdb``.

        Args:
            rep (int): Replica number (logging only).
            loop (int): Current simulation cycle number.
            nm_parsed (list[int]): Active mode numbers.
            cnt (int): Current correction counter.

        Raises:
            RuntimeError: In beta mode, if the ENM update fails (the reservoir
                is never discarded silently).

        Note:
            On failure the ``random`` method falls back to the standard
            direction correction; the ``harmonized`` method keeps Q_corr
            (already computed by the caller) and archives it.
        """
        console = ConsoleConfig()
        now = time.strftime("%H:%M:%S")
        harmonized = self._is_harmonized()
        method = 'harmonized' if harmonized else 'random'

        if harmonized:
            modes_to_compute = self._get_subspace_search_modes(nm_parsed)
            print(
                f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: "
                f"recomputing ENM modes at step {console.EXT}{loop}{console.STD} "
                f"[method=harmonized; active m={len(nm_parsed)}: {nm_parsed}; "
                f"search M={len(modes_to_compute)}: {modes_to_compute}]..."
            )
        else:
            modes_to_compute = list(nm_parsed)
            print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: "
                  f"recomputing ENM modes at step {console.EXT}{loop}{console.STD} "
                  f"[method=random; active modes={nm_parsed}]...")

        # compute_enm saves the structure it uses (protein/nucleic atoms) as
        # step_{loop}_enm/step_{loop}.pdb; no separate full-system PDB is written.
        current_pos_ang = self._curr_pos_nm * 10.0   # nm to Å
        enm_calc = ENMCalculator(self.console)
        base_name = f"step_{loop}"

        try:
            if self._diversity_enabled():
                write_diversity_spectrum(
                    enm_calc,
                    positions_ang=current_pos_ang,
                    base_name=base_name,
                    nm_type=self.args.model.lower(),
                    search_modes=modes_to_compute,
                    input_dir=os.getcwd(),
                    psffile=self.psffile,
                    max_modes=max(50, max(modes_to_compute) + 1),
                    rtb=getattr(self.args, 'rtb', False),
                )
            else:
                enm_calc.compute_enm(
                    positions_ang=current_pos_ang,
                    base_name=base_name,
                    nm_type=self.args.model.lower(),
                    nm_parsed=modes_to_compute,
                    input_dir=os.getcwd(),
                    psffile=self.psffile,
                    rtb=getattr(self.args, 'rtb', False)
                )

            if harmonized:
                sel_sub = self._enm_selection_atoms()
                self._transport_q_through_enm_subspace(
                    rep, loop, nm_parsed, modes_to_compute, cnt, base_name,
                    sel_sub.ix, sel_sub.masses
                )
                print(f"{console.PGM_NAM}ENM recomputation + harmonized m/M subspace "
                      f"transport completed for {console.EXT}Replica {rep}{console.STD}.\n")
            else:
                self._generate_new_excitation_vector(rep, loop, nm_parsed, os.getcwd(), cnt, base_name)
                print(f"{console.PGM_NAM}ENM recomputation completed for "
                      f"{console.EXT}Replica {rep}{console.STD}.\n")

        except Exception as e:
            if self._diversity_enabled():
                raise RuntimeError("beta ENM update failed; refusing to erase reservoir memory") from e
            print(f"{self.console.PGM_ERR}ENM recomputation failed [method={method}]: "
                  f"{self.console.ERR}{e}{self.console.STD}")
            if harmonized:
                # Q_corr was already produced by the angular test: keep it and
                # archive it, so every true correction leaves one archived vector.
                for src in ("cntrl_vector.vec", "excitation.vel"):
                    if os.path.exists(src):
                        shutil.copy(src, f"{src}.{cnt}")
            else:
                # Fall back to standard direction correction
                _sel_fb = self._enm_selection_atoms()
                self._correct_excitation_direction(
                    rep, loop, self._cntrl_vec, cnt,
                    _sel_fb.ix, _sel_fb.masses
                )
            return

    def _generate_new_excitation_vector(self, rep: int, loop: int, nm_parsed: List[int],
                                        rep_dir: str, cnt: int, base_name: str) -> None:
        """
        Build a new excitation vector from recomputed ENM mode files (random method).

        Args:
            rep (int): Replica number (logging only).
            loop (int): Current simulation cycle (logging only).
            nm_parsed (list[int]): Mode numbers to combine.
            rep_dir (str): Absolute path to the replica working directory.
            cnt (int): Current correction counter, used to archive old vector files.
            base_name (str): Base name of the recomputed ENM run.

        Raises:
            FileNotFoundError: If a mode file is missing.
            RuntimeError: If the combined vector has zero norm.

        Note:
            The combined vector is built from the *written mode vector files*,
            which are already in full Cartesian space whether or not the ENM
            used RTB (see ``ENMCalculator.compute_enm``).
        """
        console = ConsoleConfig()
        print(f"{console.PGM_NAM}Generating new random factors for ENM recombination.")

        enm_dir = os.path.join(rep_dir, f"{base_name}_enm")

        sel_atoms  = self._enm_selection_atoms()
        sel_ix     = sel_atoms.ix
        sel_masses = sel_atoms.masses
        basis = self._read_enm_basis(base_name, nm_parsed, rep_dir, sel_ix=sel_ix)

        n_modes = len(nm_parsed)
        factors = np.random.normal(size=n_modes)
        factors = factors / np.linalg.norm(factors)

        comb_vec = np.tensordot(factors, basis, axes=(0, 0))

        comb_norm = np.linalg.norm(comb_vec)
        if comb_norm < 1.0e-15:
            raise RuntimeError("Random ENM recombination produced a zero excitation vector.")
        comb_vec /= comb_norm

        if os.path.exists("excitation.vel"):
            shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
        if os.path.exists("cntrl_vector.vec"):
            shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

        self._cntrl_vec = np.zeros((self.n_atoms, 3))
        self._cntrl_vec[sel_ix] = comb_vec
        self.mode_exciter._write_vector(self._cntrl_vec, "cntrl_vector.vec", self.sys_coor)

        exc_vec_sel = self._excite_control_vector(comb_vec, self.energy, sel_masses)
        exc_vec_full = np.zeros((self.n_atoms, 3))
        exc_vec_full[sel_ix] = exc_vec_sel
        self._exc_vel_akma = exc_vec_full.copy()
        self.mode_exciter._write_vector(exc_vec_full, "excitation.vel", self.sys_coor)

        factors_csv = os.path.join(enm_dir, f"recalc_factors_{cnt}.csv")
        with open(factors_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Mode', 'Factor'])
            for mode_num, factor in zip(nm_parsed, factors):
                writer.writerow([mode_num, factor])
        print(f"{console.PGM_NAM}New excitation vector written; factors saved to {console.EXT}{factors_csv}{console.STD}.")

    def _correct_excitation_direction(self, rep: int, loop: int,
                                      cntrl_vec: np.ndarray, cnt: int,
                                      sel_ix: np.ndarray,
                                      sel_masses: np.ndarray) -> int:
        """
        Update the excitation direction from the observed structural displacement.

        Computes the mass-weighted average structure over the last two frames,
        aligns it to the reference, and projects the displacement onto the
        current control vector Q. If the cosine falls below ``cos_alpha``, the
        direction is replaced by the normalized displacement (Q_corr). In beta
        mode the displacement is rotated back into the MD frame first.

        Args:
            rep (int): Replica number (logging only).
            loop (int): Current cycle number (logging only).
            cntrl_vec (np.ndarray): Full-system Q vector, shape (N, 3).
            cnt (int): Correction counter; incremented when Q is replaced.
            sel_ix (np.ndarray): Indices of the selection subset.
            sel_masses (np.ndarray): Masses of the selected atoms (amu).

        Returns:
            int: Updated correction counter.
        """
        n_atoms = self.sys_coor.atoms.n_atoms

        avg_sel_nm  = (self._curr_pos_nm[sel_ix] + self._prev_pos_nm[sel_ix]) / 2.0
        avg_full_nm = self._curr_pos_nm.copy()
        avg_full_nm[sel_ix] = avg_sel_nm
        self._avg_pos_nm = avg_full_nm

        ref_positions_ang = self._align_ref_pos_nm[sel_ix] * 10.0   # nm to Å
        avg_sel_ang       = avg_sel_nm * 10.0                       # nm to Å
        aligned_avg_ang   = _kabsch_align(avg_sel_ang, ref_positions_ang, sel_masses)

        diff_sub = ((aligned_avg_ang - ref_positions_ang).T * np.sqrt(sel_masses)).T
        if self._diversity_enabled():
            # Beta convention: the direction is stored in the current MD frame.
            rotation = kabsch_rotation(avg_sel_ang, ref_positions_ang, sel_masses)
            diff_sub = diff_sub @ rotation.T

        diff = np.zeros((n_atoms, 3))
        diff[sel_ix] = diff_sub
        norm_diff = np.linalg.norm(diff)

        self._align_ref_pos_nm = avg_full_nm.copy()

        if norm_diff < 1e-10:
            return cnt

        diff_norm = diff / norm_diff
        dotp = np.sum(diff_norm * cntrl_vec)

        if dotp <= self.cos_alpha:
            self._correc_ref_pos_nm = self._curr_pos_nm.copy()

            if not self._is_harmonized():
                if os.path.exists("excitation.vel"):
                    shutil.copy("excitation.vel",   f"excitation.vel.{cnt}")
                if os.path.exists("cntrl_vector.vec"):
                    shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

            self.mode_exciter._write_vector(diff_norm, "cntrl_vector.vec", self.sys_coor)
            now = time.strftime("%H:%M:%S")
            print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}"
                  f"{self.console.STD}: wrote corrected excitation vector "
                  f"(cosα={dotp:.3f}, cnt={cnt}).")

            exc_vec_sub  = self._excite_control_vector(diff_sub, self.energy, sel_masses)
            exc_vec_full = np.zeros((n_atoms, 3))
            exc_vec_full[sel_ix] = exc_vec_sub
            self.mode_exciter._write_vector(exc_vec_full, "excitation.vel", self.sys_coor)

            self._cntrl_vec    = diff_norm.copy()
            self._exc_vel_akma = exc_vec_full.copy()

            cnt += 1
            self.qrms_correc = 0
        return cnt
