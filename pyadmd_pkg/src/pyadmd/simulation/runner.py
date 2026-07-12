"""Orchestrates running, restarting, and appending aMDeNM replica simulations."""

import csv
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

    def __init__(self, console: ConsoleConfig, args: 'Any', cwd: str, input_dir: str,
                 psffile: str, pdbfile: str, coorfile: Optional[str], velfile: Optional[str],
                 xscfile: Optional[str], strfile: Optional[str], sys_coor: mda.Universe,
                 n_atoms: int, sys_mass: np.ndarray, sel_mass: np.ndarray, energy: float,
                 mode_exciter: 'ModeExciter',
                 init_state: 'SystemState',
                 platform: str = 'auto', n_threads: Optional[int] = None) -> None:
        """
        Initialize SimulationRunner.

        Accepts a pre-built ``SystemState`` (``init_state``) so that the runner
        is engine-agnostic: both the NAMD and OpenMM input paths produce a
        ``SystemState`` before constructing this object, and no further
        engine-specific I/O occurs after this point.

        Args:
            console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
            xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
            mode_exciter: same as before (coorfile/velfile/xscfile/strfile may
                be None in OpenMM input mode).
            init_state (SystemState): Pre-built initial state (positions in nm,
                velocities in nm/ps, box vectors in nm) already rotated to the
                canonical OpenMM box orientation.
            platform (str): OpenMM platform ('auto', 'cuda', 'opencl', 'cpu').
            n_threads (int): CPU thread count for the CPU platform.
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

        # Consume the pre-built SystemState (already rotated to canonical form)
        self._init_state = init_state

        # Cache initial positions for diagnostics and direction correction
        self._init_pos_nm: np.ndarray = self._init_state.positions_nm.copy()

        # Build a reference MDAnalysis Universe from the initial positions.
        # This replaces all downstream patterns of the form
        #   mda.Universe(psffile, coorfile, format='NAMDBIN')
        # which assumed NAMD binary input.  The reference Universe is topology
        # + real positions; no engine-specific data is needed.
        init_pos_ang = self._init_pos_nm * 10.0   # nm → Å
        self._ref_universe: mda.Universe = make_reference_universe(psffile, init_pos_ang)

        # Derive str_box for PME sizing.
        # Priority:
        #   1. Parse from .str file (NAMD mode or optional in OpenMM mode).
        #   2. Derive from .rst box vectors via box_vectors_to_cell (OpenMM mode,
        #      no .str supplied).
        #   3. Fall back to None (legacy placeholder — not recommended).
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
                # Cross-check str lengths against init_state box vectors (warn if >5%)
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
                                f"for {lbl}: STR={str_l*10:.3f} Ang, "
                                f"state={rst_l*10:.3f} Ang ({diff_pct:.1f}% difference)"
                            )
            except Exception as exc:
                print(f"{console.PGM_WRN}WARNING: could not parse str box "
                      f"({exc}); falling back to box derived from initial state.")
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
                      f"({exc}); using placeholder box (not recommended).")
                str_box = None

        # Build OpenMM System with real box dimensions from .str file
        toppar_dir = os.path.join(input_dir, "charmm_toppar")
        builder = OpenMMSystemBuilder(console)
        self._psf_omm, self._omm_system, self._system_type = builder.build(
            psffile, toppar_dir, temperature=self._temperature, str_box=str_box
        )

        # Sanity-check: OpenMM System particle count must match the PSF topology.
        _omm_n = self._omm_system.getNumParticles()
        _psf_n = self._psf_omm.topology.getNumAtoms()
        if _omm_n != _psf_n:
            raise RuntimeError(f"Atom count mismatch after system build: OpenMM system has {_omm_n} "
                f"particles but the PSF topology ({psffile}) has {_psf_n} atoms.")
        print(f"{console.PGM_NAM}System atom count verified: "
              f"{console.EXT}{_omm_n}{console.STD} atoms in both OpenMM system and PSF.")

        # In-memory correction state (initialised per-replica)
        self._cntrl_vec:         Optional[np.ndarray] = None  # Q vector, Å convention
        self._exc_vel_akma:      Optional[np.ndarray] = None  # excitation velocity, AKMA
        self._correc_ref_pos_nm: Optional[np.ndarray] = None  # RMS-displacement reference
        self._align_ref_pos_nm:  Optional[np.ndarray] = None  # alignment reference
        self._curr_pos_nm:       Optional[np.ndarray] = None  # positions this cycle
        self._prev_pos_nm:       Optional[np.ndarray] = None  # positions previous cycle
        self._avg_pos_nm:        Optional[np.ndarray] = None  # last average structure (mirrors average_{loop}.coor)
        self._cntrl_vec_history:    List[np.ndarray] = []
        self._exc_vel_akma_history: List[np.ndarray] = []

        # Energy correction thresholds
        self.top    = energy * 1.25
        self.bottom = energy * 0.75

        # Adaptive correction parameters
        self.globfreq = self.cos_alpha = self.qrms_correc = 0.5

    def _save_correction_state(self, loop: int, cnt: int) -> None:
        """
        Persist in-memory correction state to disk every 10 cycles.

        Written files:
          correction_state.json       — scalar state (cycle, cnt, qrms_correc)
          _state_cntrl_vec.npy        — current Q direction vector (Å, full system)
          _state_exc_vel_akma.npy     — current excitation velocity (AKMA, full system)
          _state_correc_ref_pos_nm.npy— RMS-displacement reference positions (nm)
          _state_align_ref_pos_nm.npy — Kabsch alignment reference positions (nm)
          _state_init_pos_nm.npy      — replica initial positions used for coordinate
                                        projection; needed to keep coor-proj.out
                                        consistent across restarts/appends
          _state_avg_pos_nm.npy       — most recent average-structure positions (nm);
                                        mirrors the original correc_ref.coor / average_{loop}.coor
                                        files so that restarts can resume the adaptive
                                        direction-correction without re-loading NAMD binaries

        Args:
            loop (int): Current cycle number.
            cnt (int): Correction counter.
        """
        np.save("_state_cntrl_vec.npy",         self._cntrl_vec)
        np.save("_state_exc_vel_akma.npy",       self._exc_vel_akma)
        np.save("_state_correc_ref_pos_nm.npy",  self._correc_ref_pos_nm)
        np.save("_state_align_ref_pos_nm.npy",   self._align_ref_pos_nm)
        np.save("_state_init_pos_nm.npy",        self._init_pos_nm)
        # Average position is only meaningful after at least one correction step;
        # fall back to correc_ref if not yet set separately. Use correc_ref as
        # fallback when _avg_pos_nm is None (no correction step has fired yet).
        avg = self._avg_pos_nm if self._avg_pos_nm is not None else self._correc_ref_pos_nm
        np.save("_state_avg_pos_nm.npy",         avg)
        with open("correction_state.json", 'w') as fh:
            json.dump({'cycle': loop, 'cnt': cnt,
                       'qrms_correc': self.qrms_correc}, fh, indent=2)

    def _load_correction_state(self) -> Tuple[int, int, float]:
        """
        Restore in-memory correction state from disk.

        Returns:
            (loop, cnt, qrms_correc) scalar triple.

        Raises:
            FileNotFoundError: If any required file is missing.
        """
        self._cntrl_vec         = np.load("_state_cntrl_vec.npy")
        self._exc_vel_akma      = np.load("_state_exc_vel_akma.npy")
        self._correc_ref_pos_nm = np.load("_state_correc_ref_pos_nm.npy")
        self._align_ref_pos_nm  = np.load("_state_align_ref_pos_nm.npy")
        # Restore the replica's initial positions (used for coordinate projection).
        if os.path.exists("_state_init_pos_nm.npy"):
            self._init_pos_nm = np.load("_state_init_pos_nm.npy")
        if os.path.exists("_state_avg_pos_nm.npy"):
            try:
                self._avg_pos_nm = np.load("_state_avg_pos_nm.npy")
            except ValueError:
                # Legacy file was saved as a None object array (before first
                # correction step).  Fall back to correc_ref_pos_nm.
                self._avg_pos_nm = self._correc_ref_pos_nm.copy()
        with open("correction_state.json") as fh:
            cs = json.load(fh)
        return cs['cycle'], cs['cnt'], cs['qrms_correc']

    def run_simulation(self, rep: int, start_loop: int, end_loop: int,
                       correction_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run (or resume) an aMDeNM replica entirely within OpenMM.

        All MD state lives in the OpenMM Context between cycles.
        Per-cycle coordinate/velocity/box files are never written; the DCD
        trajectory and OpenMM checkpoints serve as the persistent record.

        Args:
            rep (int): Replica index (1-based).
            start_loop (int): 0 for a fresh run; last completed cycle for restart.
            end_loop (int): Total number of cycles to reach.
            correction_state (dict, optional): Ignored — state is loaded from disk on restart.

        Returns:
            dict with keys 'cnt' and 'qrms_correc' (backward-compat with main()).
        """
        rep_dir = f"{self.cwd}/rep{rep}"

        # Atom selection for projections/corrections
        if self.args.model.lower() == 'ca':
            sel_type = self.args.selection + " and name CA"
        else:
            sel_type = self.args.selection + " and not name H*"

        # Pre-compute selection indices and masses (constant for this replica)
        # Use the pre-built reference Universe instead of re-reading NAMD binary.
        _sel   = self._ref_universe.select_atoms(sel_type)
        sel_ix: np.ndarray     = _sel.ix        # (n_sel,) system atom indices
        sel_masses: np.ndarray = _sel.masses    # (n_sel,) amu
        n_sel: int             = len(sel_ix)
        del _sel

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

            # Read combination vector
            vec_file = f"{self.cwd}/rep-struct-list/rep{rep}_vector.vec"
            shutil.copy(vec_file, "pff_vector.vec")   # traceability copy
            q_vec_full = NAMDInputReader.read_nm_vector(self.psffile, vec_file)
            q_vec_sel  = q_vec_full[sel_ix]           # selection subset, Å

            # Compute excitation velocity (AKMA units, selection subset)
            exc_vel_sel = self.mode_exciter.excite(q_vec_sel, self.energy, sel_masses)
            print(f"{self.console.PGM_NAM}Writing the excitation vector with a Ek injection of {self.console.EXT}{self.energy}{self.console.STD} kcal/mol.")

            # Convert AKMA to nm/ps and ADD to initial velocities
            exc_vel_nm_ps = exc_vel_sel * AKMA_VEL_TO_NM_PS
            self._init_state.velocities_nm_ps[sel_ix] += exc_vel_nm_ps

            # Store full-system excitation vector (AKMA) for potential rescaling
            self._cntrl_vec    = np.zeros((self.n_atoms, 3))
            self._cntrl_vec[sel_ix] = q_vec_sel
            self._exc_vel_akma = np.zeros((self.n_atoms, 3))
            self._exc_vel_akma[sel_ix] = exc_vel_sel

            # Write the combination and the excited vector
            # cntrl_vector.vec: unit direction vector Q (used for projections each loop)
            # excitation.vel:   velocity-scaled vector actually added to the NAMD velocities
            self.mode_exciter._write_vector(q_vec_sel, "cntrl_vector.vec", self.sys_coor)
            self.mode_exciter._write_vector(exc_vel_sel, "excitation.vel", self.sys_coor)

            # Initialise position caches and correction references
            self._curr_pos_nm       = self._init_pos_nm.copy()
            self._prev_pos_nm       = self._init_pos_nm.copy()
            self._correc_ref_pos_nm = self._init_pos_nm.copy()
            self._align_ref_pos_nm  = self._init_pos_nm.copy()

            loop = 0
            cnt  = 1
            vp, ek, qp, rmsp = [], [], [], []

            # Create a fresh OpenMM engine for this replica
            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                platform_name=self._platform,
                n_threads=self._n_threads,
                device_index=0,          # use GPU 0 for all replicas
                rep_num=rep,
                is_restart=False,
                full_ener=getattr(self.args, 'full_ener', False),
                n_steps=getattr(self.args, 'n_steps', 50),
            )
            engine.initialize_state(self._init_state)

        ## == RESTART / APPEND == ##
        else:
            os.chdir(rep_dir)

            try:
                loop, cnt, self.qrms_correc = self._load_correction_state()
            except FileNotFoundError:
                # Fallback: scalar-only state (legacy JSON without .npy files)
                if correction_state and os.path.exists("correction_state.json"):
                    with open("correction_state.json") as fh:
                        cs = json.load(fh)
                    cnt              = cs.get('cnt', 1)
                    self.qrms_correc = cs.get('qrms_correc', 0.5)
                    loop             = start_loop
                    print(f"{self.console.PGM_WRN}No .npy state files found for "
                          f"replica {rep}. In-memory vectors re-initialised from "
                          f"combination file.")
                    # Re-initialise vectors from vec file using the reference Universe
                    vec_file = f"{self.cwd}/rep-struct-list/rep{rep}_vector.vec"
                    q_vec_full = NAMDInputReader.read_nm_vector(self.psffile, vec_file)
                    q_vec_sel  = q_vec_full[sel_ix]
                    exc_vel_sel = self.mode_exciter.excite(q_vec_sel, self.energy, sel_masses)
                    self._cntrl_vec    = np.zeros((self.n_atoms, 3))
                    self._cntrl_vec[sel_ix] = q_vec_sel
                    self._exc_vel_akma = np.zeros((self.n_atoms, 3))
                    self._exc_vel_akma[sel_ix] = exc_vel_sel
                    self._correc_ref_pos_nm = self._init_pos_nm.copy()
                    self._align_ref_pos_nm  = self._init_pos_nm.copy()
                else:
                    raise RuntimeError(
                        f"Cannot restart replica {rep}: correction_state.json missing."
                    )

            # DCD SYNC
            # Read frame count directly from the DCD binary header (topology-independent)
            # to avoid atom-count mismatch errors from mda.Universe.
            dcd_path = f"rep{rep}.dcd"
            if os.path.exists(dcd_path):
                n_dcd_frames = _count_dcd_frames(dcd_path)
                if n_dcd_frames > 0:
                    if n_dcd_frames > loop:
                        print(f"{self.console.PGM_NAM}DCD sync: advancing loop from "
                              f"{loop} to {n_dcd_frames} frames found in {dcd_path}.")
                        loop = n_dcd_frames
                    else:
                        print(f"{self.console.PGM_NAM}DCD sync: {dcd_path} has "
                              f"{n_dcd_frames} frames, consistent with JSON cycle {loop}.")
                else:
                    print(f"{self.console.PGM_WRN}Could not read DCD header for "
                          f"replica {rep} ({dcd_path}). Proceeding with JSON cycle {loop}.")

            self._curr_pos_nm = self._init_pos_nm.copy()
            self._prev_pos_nm = self._init_pos_nm.copy()

            # Reload projection lists accumulated in the previous run
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
                device_index=0,          # use GPU 0 for all replicas
                rep_num=rep,
                is_restart=True,
                full_ener=getattr(self.args, 'full_ener', False),
                n_steps=getattr(self.args, 'n_steps', 50),
            )
            chk = "checkpoint.chk"
            if not os.path.exists(chk):
                raise FileNotFoundError(
                    f"{chk} not found in {rep_dir}. Cannot restart replica {rep}."
                )
            engine.load_checkpoint(chk)

            # DCD APPEND
            # Force currentStep to exactly loop * n_steps so the reporter
            # fires on the very first simulation.step() call of the restart.
            _n_steps_per_cycle = getattr(self.args, 'n_steps', 50)
            _expected_step     = loop * _n_steps_per_cycle
            _chk_step          = engine.simulation.currentStep
            _chk_time_ps       = engine.simulation.context.getState().getTime().value_in_unit(unit.picosecond)
            _chk_cycle         = _chk_step // _n_steps_per_cycle
            print(f"{self.console.PGM_NAM}Checkpoint loaded: "
                  f"OpenMM step {_chk_step} ({_chk_time_ps:.2f} ps, "
                  f"~cycle {_chk_cycle}); DCD/JSON cycle is {loop}.")
            if _chk_step != _expected_step:
                engine.simulation.currentStep = _expected_step

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

                # Compute the difference and mass-weight the selected atoms only
                # (qcurr - qref) * sqrt(m)
                diff_corr   = ((curr_sel - ref_sel).T * np.sqrt(sel_masses)).T

                # Read the excitation vector and expand to full-system shape
                cntrl_sel   = self._cntrl_vec[sel_ix]

                # Project the current coordinates onto Q
                q_proj_chk  = np.sum(diff_corr * cntrl_sel)
                rms_check   = np.sqrt((q_proj_chk ** 2) / np.sum(sel_masses))

                # Correct the excitation direction or recompute ENM modes
                if rms_check >= self.qrms_correc:
                    # If --recalc flag is set, recompute ENM from the current structure
                    # rather than deriving the new Q from the structural displacement.
                    # This is more expensive but allows the mode subspace itself to adapt.
                    if hasattr(self.args, 'recalc') and self.args.recalc:
                        nm_list = [int(s) for s in self.args.modes.split(',')]
                        self._recompute_enm_modes(rep, loop, nm_list, cnt)
                        cnt += 1
                    else:
                        # Otherwise, correct the Q vector
                        cnt = self._correct_excitation_direction(
                            rep, loop,
                            self._cntrl_vec, cnt,
                            sel_ix, sel_masses
                        )
                    # Update the RMS threshold after a triggered window
                    self.qrms_correc += self.globfreq

            # OBTAIN THE VELOCITIES AND KINETIC ENERGY PROJECTED ONTO Q
            # Open the current velocities
            vel_akma    = vel_nm_ps / AKMA_VEL_TO_NM_PS
            vel_akma_mw = (vel_akma.T * np.sqrt(self.sys_mass)).T

            # Compute the scalar projection of velocity onto Q
            velo        = np.sum(vel_akma_mw * self._cntrl_vec)

            # Compute the vectorial projection of velocity onto Q
            v_proj_akma = self._cntrl_vec * velo

            # Kinetic energy along the excitation direction
            ek_vel      = 0.5 * np.sum(v_proj_akma ** 2)

            vp.append(f"{round(velo,   5)}\n")
            ek.append(f"{round(ek_vel, 5)}\n")

            # PROJECT THE COORDINATES ONTO Q
            # Open the current and initial coordinates
            curr_sel_coor = self._curr_pos_nm[sel_ix] * 10.0    # nm to Å
            init_sel_coor = self._init_pos_nm[sel_ix]  * 10.0

            # Mass-weight the displacement using only the sel_type atom masses
            diff_coor  = np.zeros((self.n_atoms, 3))
            diff_coor[sel_ix] = ((curr_sel_coor - init_sel_coor).T * np.sqrt(sel_masses)).T

            # Scalar projection of displacement onto Q and RMS displacement
            q_proj_coor = np.sum(diff_coor * self._cntrl_vec)
            mrms        = np.sqrt((q_proj_coor ** 2) / np.sum(sel_masses))
            qp.append(f"{round(q_proj_coor, 5)}\n")
            rmsp.append(f"{round(mrms,       5)}\n")

            # Skip EK rescaling for standard MDeNM
            if self.args.no_correc:
                if loop % 10 == 0:
                    self._save_correction_state(loop, cnt)
                continue

            # RESCALE KINETIC ENERGY ACCORDING TO VALUES PROJECTED ONTO VECTOR Q
            # Re-excite the NM vector when ek is below the lower threshold (bottom),
            # meaning the system has lost energy along Q (e.g. damped by friction).
            # Reduce the injected energy when ek exceeds the upper threshold (top),
            # preventing over-excitation that could distort the protein structure.
            # Both thresholds are set to ±25% of the target excitation energy.
            if (ek_vel < self.bottom) or (ek_vel > self.top):
                v_proj_akma_phys = (v_proj_akma.T / np.sqrt(self.sys_mass)).T  # Å/AKMA
                # Compute the difference between the projected and the excitation velocities
                # and then sum to the current velocities: Vnew = Vdyna + (VQ - Vp)
                # This injects exactly the missing energy along Q without altering
                # the orthogonal velocity components that drive thermal motion.
                new_vel_akma     = vel_akma + (self._exc_vel_akma - v_proj_akma_phys)
                engine.set_velocities(new_vel_akma * AKMA_VEL_TO_NM_PS)

            if loop % 10 == 0:
                self._save_correction_state(loop, cnt)

        # Write projections into files
        for data, tag in zip((vp, ek, qp, rmsp), ("vp", "ek", "coor", "rms")):
            with open(f"{tag}-proj.out", 'w') as fh:
                fh.writelines(data)

        return {'cnt': cnt, 'qrms_correc': self.qrms_correc}

    def _recompute_enm_modes(self, rep: int, loop: int, nm_parsed: List[int], cnt: int) -> None:
        """
        Recompute ENM modes from the current structure (in‑memory coordinates) and
        generate a new random excitation vector.

        This method is called when `--recalc` is set and the displacement threshold
        is reached. It writes a temporary PDB file from the current positions,
        runs a full ENM calculation on that structure, then creates a new linear
        combination of the recomputed modes with random unit‑vector coefficients.

        Args:
            rep (int): Replica number, used only for logging.
            loop (int): Current simulation cycle number (used for logging).
            nm_parsed (List[int]): List of mode numbers to include in the new ENM.
            cnt (int): Current correction counter, incremented after a successful
                       recomputation; used to archive old vector files.

        Raises:
            RuntimeError: If the temporary PDB file cannot be written or the ENM
                          calculation fails to produce the expected mode files.
        """
        console = ConsoleConfig()
        now = time.strftime("%H:%M:%S")
        print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: "
              f"recomputing ENM modes at step {console.EXT}{loop}{console.STD}...")

        # Write current positions to a temporary PDB file
        current_pos_ang = self._curr_pos_nm * 10.0   # nm to Å
        temp_pdb = f"step_{loop}.pdb"
        u_temp = mda.Universe(self.psffile)
        u_temp.atoms.positions = current_pos_ang
        u_temp.atoms.write(temp_pdb)

        # Initialize ENM calculator
        enm_calc = ENMCalculator(self.console)
        base_name = os.path.splitext(os.path.basename(temp_pdb))[0]

        try:
            enm_calc.compute_enm(
                positions_ang=current_pos_ang,
                base_name=base_name,
                nm_type=self.args.model.lower(),
                nm_parsed=nm_parsed,
                input_dir=os.getcwd(),
                psffile=self.psffile
            )

            # Generate new combination using RANDOM factors
            rep_dir = os.getcwd()
            base_name = os.path.splitext(os.path.basename(temp_pdb))[0]
            self._generate_new_excitation_vector(rep, loop, nm_parsed, rep_dir, cnt, base_name)
            print(f"{console.PGM_NAM}ENM recomputation completed for {console.EXT}Replica {rep}{console.STD}.\n")

        except Exception as e:
            print(f"{self.console.PGM_ERR}ENM recomputation failed: {self.console.ERR}{e}{self.console.STD}")
            # Fall back to standard direction correction using pre-computed sel_ix/sel_masses
            _u_fb  = make_reference_universe(self.psffile, self._curr_pos_nm * 10.0)
            sel_type_fb = self.args.selection
            if self.args.model.lower() == 'ca':
                sel_type_fb += " and name CA"
            else:
                sel_type_fb += " and not name H*"
            _sel_fb = _u_fb.atoms.select_atoms(sel_type_fb)
            self._correct_excitation_direction(
                rep, loop, self._cntrl_vec, cnt,
                _sel_fb.ix, _sel_fb.masses
            )
            del _u_fb, _sel_fb
            return

        # No temp_pdb to clean up — compute_enm receives positions directly.

    def _generate_new_excitation_vector(self, rep: int, loop: int, nm_parsed: List[int],
                                        rep_dir: str, cnt: int, base_name: str) -> None:
        """
        Build a new excitation vector from recomputed ENM mode files.

        Reads the mode XYZ files from the ENM output directory, combines them with
        a random unit vector, normalises the result, and writes the new control
        vector (`cntrl_vector.vec`) and excitation velocity (`excitation.vel`)
        to disk. Updates the in‑memory state (`_cntrl_vec`, `_exc_vel_akma`,
        `_correc_ref_pos_nm`) and archives previous vector files.

        Args:
            rep (int): Replica number (used only for logging).
            loop (int): Current simulation cycle (used for logging and to form
                        the ENM output directory name).
            nm_parsed (List[int]): Mode numbers to combine (e.g. [7,8,9]).
            rep_dir (str): Absolute path to the replica working directory.
            cnt (int): Current correction counter, used to archive old vector files.
            base_name (str): Base name of the temporary PDB that was used for ENM
                            recomputation (e.g. "temp_recalc_42"). The ENM output
                            folder is expected to be `{base_name}_enm`.

        Raises:
            FileNotFoundError: If the first mode file is missing.
            RuntimeError: If the combined vector is zero after normalisation.
        """
        console = ConsoleConfig()
        # Generate new random factors for this recombination
        print(f"{console.PGM_NAM}Generating new random factors for ENM recombination.")

        # Build the ENM output directory and the prefix used for mode XYZ files.
        enm_dir = os.path.join(rep_dir, f"{base_name}_enm")
        prefix = "ca" if self.args.model.lower() == 'ca' else "heavy"

        # Generate random unit‑vector coefficients (N‑dimensional hypersphere).
        n_modes = len(nm_parsed)
        factors = np.random.normal(size=n_modes)
        factors = factors / np.linalg.norm(factors)

        # Load the first mode to get the number of atoms in the ENM‑reduced system.
        first_mode_file = os.path.join(enm_dir, f"{base_name}_{prefix}_mode_{nm_parsed[0]}.xyz")
        if not os.path.exists(first_mode_file):
            raise FileNotFoundError(f"ENM mode file {first_mode_file} not found.")
        u_first = mda.Universe(first_mode_file, format="XYZ")
        n_enm_atoms = u_first.atoms.n_atoms
        comb_vec = np.zeros((n_enm_atoms, 3))

        # Sum the mode vectors weighted by the random factors.
        for i, mode_num in enumerate(nm_parsed):
            mode_file = os.path.join(enm_dir, f"{base_name}_{prefix}_mode_{mode_num}.xyz")
            if not os.path.exists(mode_file):
                print(f"{console.PGM_WRN}Mode file {mode_file} missing, skipping.")
                continue
            u_mode = mda.Universe(mode_file, format="XYZ")
            comb_vec += u_mode.atoms.positions * factors[i]

        # Normalise the combined vector.
        comb_vec /= np.linalg.norm(comb_vec)

        # Rename previous excitation vector files
        if os.path.exists("excitation.vel"):
            shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
        if os.path.exists("cntrl_vector.vec"):
            shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

        # Write new control vector to disk
        self.mode_exciter._write_vector(comb_vec, "cntrl_vector.vec", self.sys_coor)

        # Map ENM-reduced atoms to the full system using the reference Universe
        sel_type_full = self.args.selection
        if self.args.model.lower() == 'ca':
            sel_type_full += " and name CA"
        else:
            sel_type_full += " and not name H*"
        sel_atoms  = self._ref_universe.select_atoms(sel_type_full)
        sel_ix     = sel_atoms.ix
        sel_masses = sel_atoms.masses

        self._cntrl_vec = np.zeros((self.n_atoms, 3))
        self._cntrl_vec[sel_ix] = comb_vec

        # Compute excitation velocity (AKMA units)
        exc_vec_sel = self.mode_exciter.excite(comb_vec, self.energy, sel_masses)
        exc_vec_full = np.zeros((self.n_atoms, 3))
        exc_vec_full[sel_ix] = exc_vec_sel
        self._exc_vel_akma = exc_vec_full.copy()
        self.mode_exciter._write_vector(exc_vec_full, "excitation.vel", self.sys_coor)

        # Save the combination factors to a CSV file
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
        Update the excitation vector direction based on the observed structural displacement.

        Computes the mass-weighted average structure between the previous and current steps,
        aligns it to the reference, and projects the resulting displacement onto the current
        control vector Q. If the cosine of the angle between the displacement and Q falls
        below the adaptive threshold (cos_alpha), the excitation direction is replaced by
        the normalized displacement vector and the energy injection is re-applied.

        This is the core aMDeNM adaptive correction step: it steers the excitation along
        the direction the protein is actually moving, rather than persisting with the
        original mode combination.

        Args:
            rep (int): Replica number (logging only).
            loop (int): Current cycle number (logging only).
            cntrl_vec (np.ndarray): Full-system Q vector (Å, shape N×3).
            cnt (int): Correction counter; incremented when Q is replaced.
            sel_ix (np.ndarray): Pre-computed atom indices of the selection subset.
            sel_masses (np.ndarray): Pre-computed masses of selected atoms (amu).

        Returns:
            cnt (int): Updated correction counter.
        """
        n_atoms = self.sys_coor.atoms.n_atoms

        # Compute the average structure of the last excitation using sel_type atoms only.
        # Averaging two consecutive frames reduces single-step noise before alignment.
        avg_sel_nm  = (self._curr_pos_nm[sel_ix] + self._prev_pos_nm[sel_ix]) / 2.0
        avg_full_nm = self._curr_pos_nm.copy()
        avg_full_nm[sel_ix] = avg_sel_nm
        self._avg_pos_nm = avg_full_nm

        # Align the averaged current structure onto the correction reference frame
        # (Kabsch rotation) to remove rigid-body drift before computing displacement.
        ref_positions_ang = self._align_ref_pos_nm[sel_ix] * 10.0   # nm to Å
        avg_sel_ang       = avg_sel_nm * 10.0                       # nm to Å
        aligned_avg_ang   = _kabsch_align(avg_sel_ang, ref_positions_ang, sel_masses)

        # Mass-weighted displacement (unnormalised, selection-subset)
        diff_sub = ((aligned_avg_ang - ref_positions_ang).T * np.sqrt(sel_masses)).T

        # Full-system vector for dot-product with the full-system cntrl_vec
        diff = np.zeros((n_atoms, 3))
        diff[sel_ix] = diff_sub
        norm_diff = np.linalg.norm(diff)

        # Set the average structure as the new reference for the next steps
        self._align_ref_pos_nm = avg_full_nm.copy()

        if norm_diff < 1e-10:
            return cnt   # no meaningful displacement

        diff_norm = diff / norm_diff

        # Project onto current control vector
        dotp = np.sum(diff_norm * cntrl_vec)

        # dotp is the cosine of the angle between the observed displacement and Q.
        # If it falls below cos_alpha (default 0.5, i.e. >60°), the protein is moving
        # away from the current excitation direction and a correction is needed.
        if dotp <= self.cos_alpha:
            # Advance the RMS-gate reference
            self._correc_ref_pos_nm = self._curr_pos_nm.copy()

            # Archive previous disk files for traceability
            if os.path.exists("excitation.vel"):
                shutil.copy("excitation.vel",   f"excitation.vel.{cnt}")
            if os.path.exists("cntrl_vector.vec"):
                shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

            # Write the corrected Q (normalised full-system vector)
            self.mode_exciter._write_vector(diff_norm, "cntrl_vector.vec", self.sys_coor)
            now = time.strftime("%H:%M:%S")
            print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}"
                  f"{self.console.STD}: wrote corrected excitation vector "
                  f"(cosα={dotp:.3f}, cnt={cnt}).")

            # Compute new excitation velocity
            exc_vec_sub  = self.mode_exciter.excite(diff_sub, self.energy, sel_masses)
            exc_vec_full = np.zeros((n_atoms, 3))
            exc_vec_full[sel_ix] = exc_vec_sub
            self.mode_exciter._write_vector(exc_vec_full, "excitation.vel", self.sys_coor)

            # Update in-memory state; EK-rescaling in the main loop will inject
            # the new velocity into the OpenMM Context on the next out-of-band cycle.
            self._cntrl_vec    = diff_norm.copy()
            self._exc_vel_akma = exc_vec_full.copy()

            cnt += 1                # advance the correction counter
            self.qrms_correc = 0    # reset threshold window

        return cnt
