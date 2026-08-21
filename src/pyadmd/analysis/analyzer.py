"""Post-hoc structural analysis (RMSD, RoG, SASA, hydrophobic exposure, RMSF, DCCM/LMI, DSSP)."""

import glob
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import align
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from pyadmd.console import ConsoleConfig
from pyadmd.analysis.completion import check_pyadmd_completion
from pyadmd.fel.completion import check_fel_completion

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Time per MD cycle (n_steps * 2 fs timestep)
_PS_PER_STEP: float = 0.002

# Cap on simultaneously in-flight multiprocessing.Pool tasks, expressed
# as a multiplier of num_cores. 2x gives workers a little slack so a
# core is never left idle waiting for the consumer.
_MAX_INFLIGHT_MULTIPLIER: int = 2

# Number of units each pool worker process handles before it is killed
# and respawned by multiprocessing.Pool.
_MAXTASKSPERCHILD: int = 1


class Analyzer:
    """
    Analyzes simulation results and generates plots.

    This class handles computation and visualization of various structural
    properties from simulation trajectories including RMSD, radius of
    gyration, SASA, hydrophobic exposure, secondary structure, RMSF, and
    DCCM/LMI residue-residue correlation matrices.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output
        param_file (str): Path to parameter JSON file
        rough (bool): If True, analyze every 5ps instead of every frame
        source (str): Trajectory source: 'pyadmd' (rep{N}.dcd replicas) or
            'fel' (centroid production trajectories).
        unit_col (str): Column/key name used to identify an analysis unit
            ('replica' for pyadmd, 'centroid_frame' for fel).
        unit_label (str): Human-readable label for an analysis unit
            ('Replica' for pyadmd, 'Centroid frame' for fel).
        skip_dccm (bool): If True, skip DCCM calculation.
        skip_lmi (bool): If True, skip LMI calculation.
    """
    def __init__(self, console: ConsoleConfig, param_file: str = "pyAdMD_params.json", rough: bool = False,
                 no_rmsd: bool = False, no_rg: bool = False, no_sasa: bool = False, no_rmsf: bool = False,
                 no_dssp: bool = False, no_dccm: bool = False, no_lmi: bool = False,
                 source: str = "pyadmd") -> None:
        """
        Initializes Analyzer with configuration and parameters.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output
            param_file (str): Path to parameter JSON file
            rough (bool): If True, analyze every 5ps instead of every frame
            no_rmsd (bool): If True, skip RMSD calculation
            no_rg (bool): If True, skip radius of gyration calculation
            no_sasa (bool): If True, skip SASA calculation
            no_rmsf (bool): If True, skip RMSF calculation
            no_dssp (bool): If True, skip secondary structure (DSSP) calculation
            no_dccm (bool): If True, skip DCCM (dynamic cross-correlation
                matrix) calculation
            no_lmi (bool): If True, skip LMI (Linear Mutual Information)
                calculation
            source (str): Trajectory source to analyze: 'pyadmd' (default) for
                rep{N}.dcd replica trajectories, or 'fel' for centroid
                production trajectories from a completed 'fel' run.
        """
        self.console = console
        self.param_file = param_file
        self.rough = rough
        self.skip_rmsd = no_rmsd
        self.skip_rg = no_rg
        self.skip_sasa = no_sasa
        self.skip_rmsf = no_rmsf
        self.skip_dssp = no_dssp
        self.skip_dccm = no_dccm
        self.skip_lmi = no_lmi
        self.source = source
        self.params = self._load_parameters()

        # Analysis unit terminology and output directory depend on source.
        if self.source == "fel":
            self.analysis_dir = os.path.join("analysis", "fel")
            self.unit_col     = "centroid_frame"
            self.unit_label   = "Centroid frame"
        else:
            self.analysis_dir = "analysis"
            self.unit_col     = "replica"
            self.unit_label   = "Replica"

        # Create analysis directory (creates parent 'analysis/' too, if needed)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100

        # Determine number of CPU cores to use
        self.num_cores = mp.cpu_count()
        print(f"{self.console.PGM_NAM}Using {self.console.EXT}{self.num_cores}{self.console.STD} CPU cores for parallel processing...")

    def _load_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Loads simulation parameters from JSON file.

        Returns:
            dict: Dictionary of loaded parameters or None if loading fails
        """
        if not os.path.exists(self.param_file):
            print(f"{self.console.PGM_ERR}Parameter file {self.console.ERR}{self.param_file}{self.console.STD} not found.")
            return None

        try:
            with open(self.param_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error loading parameters: {self.console.ERR}{e}{self.console.STD}")
            return None

    def _warn_incomplete(self, incomplete: List[Tuple[int, int, int]]) -> None:
        """
        Print a non-fatal warning listing units that have not reached their
        target cycle count.

        Replaces the previous hard-gate behavior (abort via ``sys.exit(1)``):
        analysis now proceeds on whatever units exist, complete or not, and
        this method only surfaces the shortfall to the user.

        Args:
            incomplete (list[tuple[int, int, int]]): One
                ``(unit_id, last_completed_cycle, target_cycle)`` tuple per
                incomplete unit, as returned by ``check_pyadmd_completion``/
                ``check_fel_completion``.
        """
        if not incomplete:
            return
        print(f"{self.console.PGM_WRN}{self.console.WRN}{len(incomplete)}{self.console.STD} "
              f"{self.unit_label.lower()}(s) have not reached the target cycle count; "
              "analyzing them anyway, up to their last completed cycle:")
        for unit_id, last_cycle, target in incomplete:
            print(f"{self.console.PGM_WRN}  {self.unit_label} {self.console.WRN}{unit_id}{self.console.STD}: "
                  f"{self.console.WRN}{last_cycle}{self.console.STD}/{self.console.EXT}{target}{self.console.STD} cycles completed")

    def _run_pool_bounded(self, pool: mp.Pool, worker_fn: Any,
                          args_list: List[Tuple[Any, ...]],
                          update_progress: Any):
        """
        Submit ``args_list`` to ``pool`` and yield each result as soon as
        it is consumed, keeping at most ``num_cores *
        _MAX_INFLIGHT_MULTIPLIER`` tasks simultaneously submitted-but-not-
        yet-finished at any time.

        The sliding window here submits an initial batch capped at
        ``max_in_flight``, then for each result finished submits exactly
        one more task if any remain. This bounds backlog to ``max_in_flight``
        regardless of how many total units are queued.

        Args:
            pool (multiprocessing.pool.Pool): Already-open pool to submit
                tasks to.
            worker_fn: Worker function to call via ``pool.apply_async``
                (e.g. ``self._analyze_replica_parallel`` or
                ``self._analyze_centroid_parallel``); called with each
                element of ``args_list`` unpacked as positional arguments,
                exactly as the previous ``apply_async(worker_fn, args)``
                call did.
            args_list (list[tuple]): One positional-argument tuple per
                unit to analyze.
            update_progress: Callback passed through to ``apply_async`` as
                its ``callback=`` argument, unchanged from the previous
                per-task progress printer.

        Yields:
            The unpacked return value of each worker call
                (``worker_fn(*args)``'s result), in the same submission
                order ``args_list`` was given in
        """
        max_in_flight = max(1, min(self.num_cores * _MAX_INFLIGHT_MULTIPLIER, len(args_list)))

        pending_idx = 0
        window: 'deque' = deque()

        # Prime the window with the first batch of tasks.
        while pending_idx < len(args_list) and len(window) < max_in_flight:
            args = args_list[pending_idx]
            window.append(pool.apply_async(worker_fn, args, callback=update_progress))
            pending_idx += 1

        # Drain the oldest result, refill the window by one task, repeat.
        while window:
            res = window.popleft()
            yield res.get()
            if pending_idx < len(args_list):
                args = args_list[pending_idx]
                window.append(pool.apply_async(worker_fn, args, callback=update_progress))
                pending_idx += 1

    def _unit_output_complete(self, out_dir: str) -> bool:
        """
        Return whether a unit's analysis output already exists and is
        complete, so its worker task can be skipped and its data recovered
        from disk instead of recomputed.

        Args:
            out_dir (str): This unit's own analysis output directory
                (e.g. ``analysis/rep{N}`` or
                ``analysis/fel/centroid_frame{F}``).

        Returns:
            bool: True if ``{out_dir}/analysis_results.csv`` exists and is
                non-empty.
        """
        csv_path = os.path.join(out_dir, "analysis_results.csv")
        return os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def _recover_unit_from_disk(self, out_dir: str, unit_id: int) -> Tuple[
            List[Dict[str, Any]], List[Dict[str, Any]],
            Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Reconstruct one unit's worker-return-shaped data by reading back
        the files ``_analyze_trajectory`` already wrote to ``out_dir`` on
        a previous (complete) run, instead of recomputing it.

        Reads exactly the on-disk artifacts a completed unit is guaranteed
        to have (see ``_unit_output_complete``):
            - ``analysis_results.csv``   -> per-frame data rows
            - ``rmsf.csv``               -> per-residue RMSF rows (if present;
                                             absent when ``--no_rmsf`` was
                                             used for the original run)
            - ``dccm_matrix.npy``        -> DCCM matrix (if present)
            - ``lmi_matrix.npy``         -> LMI matrix (if present)

        The returned shape matches ``_analyze_trajectory``'s return value
        exactly, so callers can feed a recovered unit through the same
        ``_append_csv_rows``/``_update_rmsf_running_sum``/
        ``_update_correlation_running_sum`` aggregation calls used for
        freshly-computed units, with no branching required downstream.

        Args:
            out_dir (str): This unit's own analysis output directory.
            unit_id (int): This unit's identifier, used only in warning
                messages.

        Returns:
            tuple: ``(data, rmsf_data, dccm, lmi)``, same shape as
                ``_analyze_trajectory``'s return value. Falls back to
                empty/``None`` for any file that is missing or unreadable,
                with a warning printed -- a recovery failure for one
                sub-artifact does not block recovery of the others.
        """
        data: List[Dict[str, Any]] = []
        rmsf_data: List[Dict[str, Any]] = []
        dccm: Optional[np.ndarray] = None
        lmi: Optional[np.ndarray] = None

        results_path = os.path.join(out_dir, "analysis_results.csv")
        try:
            data = pd.read_csv(results_path).to_dict('records')
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not recover analysis_results.csv "
                  f"for {self.console.WRN}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

        rmsf_path = os.path.join(out_dir, "rmsf.csv")
        if not self.skip_rmsf and os.path.exists(rmsf_path):
            try:
                rmsf_data = pd.read_csv(rmsf_path).to_dict('records')
            except Exception as e:
                print(f"{self.console.PGM_WRN}Could not recover rmsf.csv "
                      f"for {self.console.WRN}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

        dccm_path = os.path.join(out_dir, "dccm_matrix.npy")
        if not self.skip_dccm and os.path.exists(dccm_path):
            try:
                dccm = np.load(dccm_path)
            except Exception as e:
                print(f"{self.console.PGM_WRN}Could not recover dccm_matrix.npy "
                      f"for {self.console.WRN}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

        lmi_path = os.path.join(out_dir, "lmi_matrix.npy")
        if not self.skip_lmi and os.path.exists(lmi_path):
            try:
                lmi = np.load(lmi_path)
            except Exception as e:
                print(f"{self.console.PGM_WRN}Could not recover lmi_matrix.npy "
                      f"for {self.console.WRN}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

        return data, rmsf_data, dccm, lmi

    def analyze_all_replicas(self) -> None:
        """
        Analyze all replicas and generate plots.

        This method processes all replica directories, computes structural
        properties, generates visualizations, and creates summary reports.

        Every replica's completion status is checked against its target
        cycle count (see ``check_pyadmd_completion``) before analysis. Units
        that have not reached the target are analyzed anyway, using only the
        cycles/frames they actually completed; a warning is printed and the
        same information is included in the HTML summary.
        """
        t0 = time.time()

        if self.params is None:
            return

        try:
            incomplete = check_pyadmd_completion(self.params)
        except ValueError as exc:
            print(f"{self.console.PGM_ERR}Cannot verify replica completion: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            sys.exit(1)

        self._warn_incomplete(incomplete)

        cwd = self.params.get('cwd', os.getcwd())
        args = self.params['args']
        replicas = args.get('replicas', 10)
        sim_time = args.get('time', 250)  # Total (target) simulation time in ps
        n_steps  = args.get('n_steps', 100)  # MD steps per cycle (0.2 ps/cycle by default)
        cycle_ps = n_steps * _PS_PER_STEP

        # Combined outputs are streamed incrementally to disk as each worker's result arrives
        results_csv   = f"{self.analysis_dir}/analysis_results.csv"
        rmsf_csv      = f"{self.analysis_dir}/rmsf.csv"
        results_header_written = False
        rmsf_header_written    = False
        any_data       = False
        running_rmsf: Dict[int, List[float]] = {}
        dccm_sum: Optional[np.ndarray] = None
        dccm_count = 0
        lmi_sum: Optional[np.ndarray]  = None
        lmi_count  = 0

        # Log skipped analyses
        skipped = []
        if self.skip_rmsd:   skipped.append("RMSD")
        if self.skip_rg:     skipped.append("Radius of Gyration")
        if self.skip_sasa:   skipped.append("SASA and Hydrophobic Exposure")
        if self.skip_rmsf:   skipped.append("RMSF")
        if self.skip_dssp:   skipped.append("Secondary Structure (DSSP)")
        if self.skip_dccm:   skipped.append("DCCM")
        if self.skip_lmi:    skipped.append("LMI")
        if skipped:
            print(f"{self.console.PGM_WRN}Skipping analyses: {self.console.WRN}{', '.join(skipped)}{self.console.STD}\n")

        # Prepare arguments for parallel processing. Recovered units are
        # read back directly in the parent process instead of being recomputed,
        # so a crash never forces redoing already-finished work.
        replica_args = []
        replica_dirs = []
        recovered_units: List[Tuple[int, str]] = []   # (rep_num, rep_analysis_dir)
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                print(f"{self.console.PGM_WRN}{self.console.WRN}{self.unit_label} {rep}{self.console.STD} directory not found, skipping.")
                continue

            # Create replica-specific analysis directory
            rep_analysis_dir = f"{self.analysis_dir}/rep{rep}"
            os.makedirs(rep_analysis_dir, exist_ok=True)

            if self._unit_output_complete(rep_analysis_dir):
                recovered_units.append((rep, rep_analysis_dir))
                continue

            replica_args.append((rep_dir, rep, sim_time, rep_analysis_dir, cycle_ps))
            replica_dirs.append(rep_dir)

        if recovered_units:
            print(f"{self.console.PGM_NAM}Recovering {self.console.EXT}{len(recovered_units)}"
                  f"{self.console.STD} {self.unit_label.lower()}(s) with already-complete "
                  "output from a previous run (skipping recomputation): "
                  f"{self.console.EXT}{', '.join(str(r) for r, _ in recovered_units)}{self.console.STD}")
            for rep_num, rep_analysis_dir in recovered_units:
                rec_data, rec_rmsf, rec_dccm, rec_lmi = self._recover_unit_from_disk(
                    rep_analysis_dir, rep_num)
                if rec_data:
                    any_data = True
                    results_header_written = self._append_csv_rows(
                        rec_data, results_csv, results_header_written)
                if rec_rmsf:
                    rmsf_header_written = self._append_csv_rows(
                        rec_rmsf, rmsf_csv, rmsf_header_written)
                    self._update_rmsf_running_sum(running_rmsf, rec_rmsf)
                if rec_dccm is not None and rec_dccm.size:
                    dccm_sum, dccm_count = self._update_correlation_running_sum(
                        dccm_sum, dccm_count, rec_dccm, 'dccm', rep_num)
                if rec_lmi is not None and rec_lmi.size:
                    lmi_sum, lmi_count = self._update_correlation_running_sum(
                        lmi_sum, lmi_count, rec_lmi, 'lmi', rep_num)

        # Print analysis settings once
        if replica_dirs:
            print(f"{self.console.PGM_NAM}Analyzing {self.console.EXT}{len(replica_dirs)}{self.console.STD} replicas in parallel using CPU...\n")
            if self.rough:
                # Estimate frame step from first replica using its DCD trajectory
                try:
                    first_rep_dir = replica_dirs[0]
                    psf_file = f"{first_rep_dir}/../inputs/{self.params['args']['psffile'].split('/')[-1]}"
                    dcd_files = sorted(glob.glob(f"{first_rep_dir}/rep*.dcd"))
                    if dcd_files:
                        u = mda.Universe(psf_file, dcd_files[0], format="DCD")
                        n_frames = len(u.trajectory)
                        frame_step = max(1, int(5 / (sim_time / n_frames)))
                        print(f"{self.console.PGM_NAM}Using rough analysis: analyzing every {self.console.EXT}{frame_step}{self.console.STD} "
                              f"frames ({self.console.EXT}{frame_step * (sim_time/n_frames):.1f}{self.console.STD} ps)\n")
                except:
                    pass

        # Process replicas in parallel using CPU cores
        if replica_args:
            # Use multiprocessing for CPU-bound tasks.
            with mp.Pool(processes=min(self.num_cores, len(replica_args)),
                        maxtasksperchild=_MAXTASKSPERCHILD) as pool:
                # Create a progress tracking function
                completed = 0
                def update_progress(result: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]) -> None:
                    nonlocal completed
                    completed += 1
                    rep_num, _, _, _, _ = result  # Unpack the result tuple
                    print(f"{self.console.PGM_NAM}Completed analysis of {self.console.EXT}{self.unit_label} {rep_num}{self.console.STD}"
                          f" [{self.console.WRN}{completed}{self.console.STD}/{self.console.EXT}{len(replica_args)}{self.console.STD}].")

                # Submit tasks through a bounded sliding window.
                for rep_num, rep_data, rep_rmsf_data, rep_dccm, rep_lmi in self._run_pool_bounded(
                        pool, self._analyze_replica_parallel, replica_args, update_progress):
                    if rep_data:
                        any_data = True
                        results_header_written = self._append_csv_rows(
                            rep_data, results_csv, results_header_written)
                    if rep_rmsf_data:
                        rmsf_header_written = self._append_csv_rows(
                            rep_rmsf_data, rmsf_csv, rmsf_header_written)
                        self._update_rmsf_running_sum(running_rmsf, rep_rmsf_data)
                    if rep_dccm is not None and rep_dccm.size:
                        dccm_sum, dccm_count = self._update_correlation_running_sum(
                            dccm_sum, dccm_count, rep_dccm, 'dccm', rep_num)
                    if rep_lmi is not None and rep_lmi.size:
                        lmi_sum, lmi_count = self._update_correlation_running_sum(
                            lmi_sum, lmi_count, rep_lmi, 'lmi', rep_num)

        elif not recovered_units:
            print(f"{self.console.PGM_WRN}No replicas found for analysis.")

        if not any_data:
            print(f"{self.console.PGM_WRN}No analysis data was generated.")
            return

        if not self.skip_rmsf and rmsf_header_written:
            print(f"\n{self.console.PGM_NAM}Average RMSF results saved to {self.console.EXT}{rmsf_csv}{self.console.STD}.")

        print(f"{self.console.PGM_NAM}Analysis results saved to {self.console.EXT}{results_csv}{self.console.STD}.")

        # Re-read the combined CSV once for plotting/HTML.
        df = pd.read_csv(results_csv)

        # Generate plots
        self._generate_plots(df, sim_time)

        # Generate RMSF plots (only if RMSF was computed)
        if not self.skip_rmsf and running_rmsf:
            self._generate_rmsf_avg_plot(running_rmsf)

        # Generate average DCCM/LMI plots (only if computed)
        if not self.skip_dccm and dccm_sum is not None:
            self._generate_correlation_avg_plot(dccm_sum, dccm_count, kind='dccm')
        if not self.skip_lmi and lmi_sum is not None:
            self._generate_correlation_avg_plot(lmi_sum, lmi_count, kind='lmi')

        # Generate HTML summary
        self._generate_html_summary(df, sim_time, incomplete=incomplete)

        print(f"\n{self.console.PGM_NAM}Analysis complete in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")
        print(f"{self.console.PGM_NAM}Results saved into {self.console.EXT}{self.analysis_dir}{self.console.STD} folder.")

    def analyze_all_centroids(self) -> None:
        """
        Analyze all fel centroid production trajectories and generate plots.

        Mirrors ``analyze_all_replicas``, but the analysis units are
        fel centroids (``fel/centroids/centroid_frame{F}/prod.dcd``)
        instead of pyadmd replicas, and the shared (target) time axis is the
        ``production_ps`` value already recorded in
        ``fel/run_metadata.json`` (playing the role
        ``pyAdMD_params.json``'s ``time`` plays for ``analyze_all_replicas``).

        Raises:
            SystemExit: If ``fel/clustering_summary.csv`` or
                ``fel/run_metadata.json`` cannot be found/read, or if
                the shared PSF file cannot be located.
        """
        t0 = time.time()

        if self.params is None:
            return

        cwd = self.params.get('cwd', os.getcwd())

        try:
            incomplete = check_fel_completion(cwd)
        except FileNotFoundError as exc:
            print(f"{self.console.PGM_ERR}Cannot verify centroid completion: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            sys.exit(1)

        self._warn_incomplete(incomplete)

        # PSF path is shared across all centroids (same file used for the whole run)
        psf_file = f"{cwd}/inputs/{self.params['args']['psffile'].split('/')[-1]}"
        if not os.path.exists(psf_file):
            print(f"{self.console.PGM_ERR}PSF file not found: "
                  f"{self.console.ERR}{psf_file}{self.console.STD}.")
            return

        # Shared (target) time axis: production_ps from fel/run_metadata.json
        # (same role args['time'] plays for analyze_all_replicas)
        run_metadata_path = f"{cwd}/fel/run_metadata.json"
        try:
            with open(run_metadata_path) as fh:
                run_metadata = json.load(fh)
            sim_time = run_metadata['production_ps']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            print(f"{self.console.PGM_ERR}Could not read 'production_ps' from "
                  f"{self.console.ERR}{run_metadata_path}{self.console.STD}: "
                  f"{self.console.ERR}{exc}{self.console.STD}.")
            sys.exit(1)

        # Per-cycle time.
        n_steps  = 100
        cycle_ps = n_steps * _PS_PER_STEP

        # Combined outputs are streamed incrementally to disk as each worker's result arrives
        results_csv   = f"{self.analysis_dir}/analysis_results.csv"
        rmsf_csv      = f"{self.analysis_dir}/rmsf.csv"
        results_header_written = False
        rmsf_header_written    = False
        any_data       = False
        running_rmsf: Dict[int, List[float]] = {}
        dccm_sum: Optional[np.ndarray] = None
        dccm_count = 0
        lmi_sum: Optional[np.ndarray]  = None
        lmi_count  = 0

        # Log skipped analyses
        skipped = []
        if self.skip_rmsd:   skipped.append("RMSD")
        if self.skip_rg:     skipped.append("Radius of Gyration")
        if self.skip_sasa:   skipped.append("SASAand Hydrophobic Exposure")
        if self.skip_rmsf:   skipped.append("RMSF")
        if self.skip_dssp:   skipped.append("Secondary Structure (DSSP)")
        if self.skip_dccm:   skipped.append("DCCM")
        if self.skip_lmi:    skipped.append("LMI")
        if skipped:
            print(f"{self.console.PGM_WRN}Skipping analyses: {self.console.WRN}{', '.join(skipped)}{self.console.STD}\n")

        # Discover centroid production trajectories, sorted by frame index
        centroid_pattern = re.compile(r"centroid_frame(\d+)")
        centroid_dcds = sorted(
            glob.glob(f"{cwd}/fel/centroids/centroid_frame*/prod.dcd"),
            key=lambda p: int(centroid_pattern.search(p).group(1))
        )

        # Prepare arguments for parallel processing. Recovered units are
        # read back directly in the parent process instead of being recomputed,
        # so a crash never forces redoing already-finished work.
        centroid_args = []
        recovered_units: List[Tuple[int, str]] = []   # (frame_idx, out_dir)
        for dcd_path in centroid_dcds:
            frame_idx = int(centroid_pattern.search(dcd_path).group(1))

            out_dir = f"{self.analysis_dir}/centroid_frame{frame_idx}"
            os.makedirs(out_dir, exist_ok=True)

            if self._unit_output_complete(out_dir):
                recovered_units.append((frame_idx, out_dir))
                continue

            centroid_args.append((dcd_path, psf_file, frame_idx, sim_time, out_dir, cycle_ps))

        if recovered_units:
            print(f"{self.console.PGM_NAM}Recovering {self.console.EXT}{len(recovered_units)}"
                  f"{self.console.STD} {self.unit_label.lower()}(s) with already-complete "
                  "output from a previous run (skipping recomputation): "
                  f"{self.console.EXT}{', '.join(str(f) for f, _ in recovered_units)}{self.console.STD}")
            for frame_idx, out_dir in recovered_units:
                rec_data, rec_rmsf, rec_dccm, rec_lmi = self._recover_unit_from_disk(
                    out_dir, frame_idx)
                if rec_data:
                    any_data = True
                    results_header_written = self._append_csv_rows(
                        rec_data, results_csv, results_header_written)
                if rec_rmsf:
                    rmsf_header_written = self._append_csv_rows(
                        rec_rmsf, rmsf_csv, rmsf_header_written)
                    self._update_rmsf_running_sum(running_rmsf, rec_rmsf)
                if rec_dccm is not None and rec_dccm.size:
                    dccm_sum, dccm_count = self._update_correlation_running_sum(
                        dccm_sum, dccm_count, rec_dccm, 'dccm', frame_idx)
                if rec_lmi is not None and rec_lmi.size:
                    lmi_sum, lmi_count = self._update_correlation_running_sum(
                        lmi_sum, lmi_count, rec_lmi, 'lmi', frame_idx)

        # Print analysis settings once
        if centroid_args:
            print(f"{self.console.PGM_NAM}Analyzing {self.console.EXT}{len(centroid_args)}{self.console.STD} centroids in parallel using CPU...\n")
        elif not recovered_units:
            print(f"{self.console.PGM_WRN}No centroid production trajectories found for analysis.")

        # Process centroids in parallel using CPU cores
        if centroid_args:
            # Use multiprocessing for CPU-bound tasks.
            with mp.Pool(processes=min(self.num_cores, len(centroid_args)),
                        maxtasksperchild=_MAXTASKSPERCHILD) as pool:
                # Create a progress tracking function
                completed = 0
                def update_progress(result: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]) -> None:
                    nonlocal completed
                    completed += 1
                    frame_idx, _, _, _, _ = result  # Unpack the result tuple
                    print(f"{self.console.PGM_NAM}Completed analysis of {self.console.EXT}{self.unit_label} {frame_idx}{self.console.STD}"
                          f" [{self.console.EXT}{completed}{self.console.STD}/{self.console.WRN}{len(centroid_args)}{self.console.STD}].")

                # Submit tasks through a bounded sliding window.
                for frame_idx, frame_data, frame_rmsf_data, frame_dccm, frame_lmi in self._run_pool_bounded(
                        pool, self._analyze_centroid_parallel, centroid_args, update_progress):
                    if frame_data:
                        any_data = True
                        results_header_written = self._append_csv_rows(
                            frame_data, results_csv, results_header_written)
                    if frame_rmsf_data:
                        rmsf_header_written = self._append_csv_rows(
                            frame_rmsf_data, rmsf_csv, rmsf_header_written)
                        self._update_rmsf_running_sum(running_rmsf, frame_rmsf_data)
                    if frame_dccm is not None and frame_dccm.size:
                        dccm_sum, dccm_count = self._update_correlation_running_sum(
                            dccm_sum, dccm_count, frame_dccm, 'dccm', frame_idx)
                    if frame_lmi is not None and frame_lmi.size:
                        lmi_sum, lmi_count = self._update_correlation_running_sum(
                            lmi_sum, lmi_count, frame_lmi, 'lmi', frame_idx)

        if not any_data:
            print(f"{self.console.PGM_WRN}No analysis data was generated.")
            return

        if not self.skip_rmsf and rmsf_header_written:
            print(f"\n{self.console.PGM_NAM}Average RMSF results saved to {self.console.EXT}{rmsf_csv}{self.console.STD}.")

        print(f"{self.console.PGM_NAM}Analysis results saved to {self.console.EXT}{results_csv}{self.console.STD}.")

        # Re-read the combined CSV once for plotting/HTML.
        df = pd.read_csv(results_csv)

        # Generate plots
        self._generate_plots(df, sim_time)

        # Generate RMSF plots (only if RMSF was computed)
        if not self.skip_rmsf and running_rmsf:
            self._generate_rmsf_avg_plot(running_rmsf)

        # Generate average DCCM/LMI plots (only if computed)
        if not self.skip_dccm and dccm_sum is not None:
            self._generate_correlation_avg_plot(dccm_sum, dccm_count, kind='dccm')
        if not self.skip_lmi and lmi_sum is not None:
            self._generate_correlation_avg_plot(lmi_sum, lmi_count, kind='lmi')

        # Generate HTML summary
        self._generate_html_summary(df, sim_time, incomplete=incomplete)

        print(f"\n{self.console.PGM_NAM}Analysis complete in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")
        print(f"{self.console.PGM_NAM}Results saved into {self.console.EXT}{self.analysis_dir}{self.console.STD} folder.")

    def _analyze_centroid_parallel(self, dcd_file: str, psf_file: str, frame_idx: int,
                                   sim_time: float, out_dir: str, cycle_ps: float) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Thin parallel wrapper that calls ``_analyze_trajectory`` for a single
        fel centroid and prepends the centroid frame index.

        Designed for use with multiprocessing.Pool.apply_async, mirroring
        ``_analyze_replica_parallel``.

        Args:
            dcd_file (str): Absolute path to the centroid's prod.dcd file.
            psf_file (str): Absolute path to the shared PSF topology file.
            frame_idx (int): Centroid frame index, echoed in the return value.
            sim_time (float): Shared target production time in picoseconds
                (``production_ps`` from ``fel/run_metadata.json``);
                used for plot axis scaling only.
            out_dir (str): Output directory for this centroid's analysis files.
            cycle_ps (float): Elapsed time per MD cycle in picoseconds, used
                to compute this centroid's actual per-frame timestamps from
                its real (possibly incomplete) frame count.

        Returns:
            tuple: A 5-element tuple of (frame_idx, data, rmsf_data, dccm,
                lmi), same shape as ``_analyze_replica_parallel``'s return
                value.
        """
        try:
            print(f"{self.console.PGM_NAM}Starting analysis of {self.console.EXT}{self.unit_label} {frame_idx}{self.console.STD}...")
            data, rmsf_data, dccm, lmi = self._analyze_trajectory(psf_file, dcd_file, frame_idx, sim_time, out_dir, cycle_ps)
            return (frame_idx, data, rmsf_data, dccm, lmi)
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {frame_idx}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            return (frame_idx, [], [], None, None)

    def _analyze_replica_parallel(self, rep_dir: str, rep_num: int, sim_time: int, rep_analysis_dir: str, cycle_ps: float) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Thin parallel wrapper that calls analyze_replica and prepends the replica number.

        Designed for use with multiprocessing.Pool.apply_async: the callback receives
        the returned tuple to track progress and collect results.

        Args:
            rep_dir (str): Absolute path to the replica directory.
            rep_num (int): Replica number identifier, echoed in the return value.
            sim_time (int): Total (target) simulation time in picoseconds;
                used for plot axis scaling only.
            rep_analysis_dir (str): Output directory for replica-specific analysis files.
            cycle_ps (float): Elapsed time per MD cycle in picoseconds, used
                to compute this replica's actual per-frame timestamps from
                its real (possibly incomplete) frame count.

        Returns:
            tuple: A 5-element tuple of (rep_num, data, rmsf_data, dccm, lmi) where:
                - rep_num (int): Replica identifier passed through for pool callback tracking.
                - data (list[dict]): Per-frame structural property dictionaries; empty on failure.
                - rmsf_data (list[dict]): Per-residue RMSF dictionaries; empty on failure.
                - dccm (np.ndarray or None): (n_ca, n_ca) DCCM matrix; None if skipped/failed.
                - lmi (np.ndarray or None): (n_ca, n_ca) LMI matrix; None if not requested/failed.
        """
        try:
            print(f"{self.console.PGM_NAM}Starting analysis of {self.console.EXT}{self.unit_label} {rep_num}{self.console.STD}...")
            result = self.analyze_replica(rep_dir, rep_num, sim_time, rep_analysis_dir, cycle_ps)
            return (rep_num, result[0], result[1], result[2], result[3])  # Return rep_num along with data
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {rep_num}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            return (rep_num, [], [], None, None)  # Return rep_num even on error

    def analyze_replica(self, rep_dir: str, rep_num: int, sim_time: int, rep_analysis_dir: str,
                        cycle_ps: float = 100 * _PS_PER_STEP) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Resolve the DCD/PSF paths for a single pyAdMD replica, then delegate
        the actual per-frame computation to ``_analyze_trajectory``.

        Args:
            rep_dir (str): Absolute path to the replica directory containing the rep{N}.dcd file.
            rep_num (int): Replica number identifier used in output labels and filenames.
            sim_time (int): Total (target) simulation time in picoseconds,
                used for plot axis scaling only (see ``_analyze_trajectory``).
            rep_analysis_dir (str): Output directory for replica-specific plots and CSV files.
            cycle_ps (float): Elapsed time per MD cycle in picoseconds
                (defaults to the standard 100-step/0.2 ps cycle), used to
                compute this replica's actual per-frame timestamps from its
                real (possibly incomplete) frame count.

        Returns:
            tuple: A 4-element tuple of (data, rmsf_data, dccm, lmi) where:
                - data (list[dict]): One dictionary per analyzed frame with keys:
                  replica, time, rmsd, radius_gyration, sasa, hydrophobic_exposure,
                  helix, sheet, coil, turn, other.
                - rmsf_data (list[dict]): One dictionary per Cα atom with keys:
                  replica, residue_index, residue_name, rmsf.
                - dccm (np.ndarray or None): (n_ca, n_ca) DCCM matrix.
                - lmi (np.ndarray or None): (n_ca, n_ca) LMI matrix.
                All are empty/None if the PSF or DCD file is not found.
        """
        # Find the DCD trajectory file for this replica
        dcd_files = sorted(glob.glob(f"{rep_dir}/rep{rep_num}.dcd"))
        if not dcd_files:
            # Fallback: accept any rep*.dcd present in the directory
            dcd_files = sorted(glob.glob(f"{rep_dir}/rep*.dcd"))
        if not dcd_files:
            print(f"{self.console.PGM_WRN}No DCD trajectory found for {self.console.WRN}{self.unit_label.lower()} {rep_num}{self.console.STD}.")
            return [], [], None, None

        dcd_file = dcd_files[0]

        # Load PSF file
        psf_file = f"{rep_dir}/../inputs/{self.params['args']['psffile'].split('/')[-1]}"
        if not os.path.exists(psf_file):
            print(f"{self.console.PGM_ERR}PSF file not found for {self.unit_label.lower()} {self.console.ERR}{rep_num}{self.console.STD}.")
            return [], [], None, None

        return self._analyze_trajectory(psf_file, dcd_file, rep_num, sim_time, rep_analysis_dir, cycle_ps)

    def _analyze_trajectory(self, psf_file: str, dcd_file: str, unit_id: int,
                            sim_time: float, out_dir: str,
                            cycle_ps: float = 100 * _PS_PER_STEP) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Analyze every frame of a single trajectory.

        Loads the given DCD trajectory into an MDAnalysis Universe using the
        given PSF as topology, then computes per-frame RMSD, radius of
        gyration, SASA, hydrophobic exposure, and secondary structure
        content, and calculates per-residue RMSF plus (unless skipped) the
        DCCM residue-residue correlation matrix and (if requested) LMI.
        Plots and CSV files are written to ``out_dir``. This method
        contains the computation logic shared by both the pyadmd
        (``analyze_replica``) and fel (``analyze_all_centroids``)
        source paths.

        RMSF, DCCM, and LMI share a single per-frame Kabsch alignment pass
        (performed once inside ``_calc_rmsf``) rather than each re-aligning
        the trajectory independently.

        The per-frame time axis is built from this unit's *actual* frame
        count and ``cycle_ps`` (``n_frames_actual - 1) * cycle_ps``), not
        from ``sim_time``. This keeps timestamps correct for units that
        have not reached their target cycle count (analysis no longer
        aborts on incomplete units -- see ``analyze_all_replicas`` /
        ``analyze_all_centroids``); ``sim_time`` is used only downstream
        for plot axis limits and labeling, so multiple units (complete and
        incomplete) remain visually comparable on the same combined plot.

        Args:
            psf_file (str): Absolute path to the PSF topology file.
            dcd_file (str): Absolute path to the DCD trajectory file.
            unit_id (int): Analysis unit identifier (replica number, or
                centroid frame index), used in output labels/filenames.
            sim_time (float): Total (target) simulation time in
                picoseconds; used only for plot axis scaling, not for
                per-frame timestamps.
            out_dir (str): Output directory for unit-specific plots and CSV
                files.
            cycle_ps (float): Elapsed time per MD cycle in picoseconds
                (``n_steps * 0.002``), used to compute this unit's actual
                elapsed time from its real frame count.

        Returns:
            tuple: A 4-element tuple of (data, rmsf_data, dccm, lmi), same
                shape as ``analyze_replica``'s return value. ``data`` and
                ``rmsf_data`` are empty lists, ``dccm``/``lmi`` are None,
                on failure.
        """
        try:
            # Create universe from PSF topology + DCD trajectory
            u = mda.Universe(psf_file, dcd_file, format="DCD")

            # Get the actual number of frames
            n_frames = len(u.trajectory)

            # Actual elapsed time for this unit, derived from its real frame count
            actual_time = max(0, n_frames - 1) * cycle_ps
            time_points = np.linspace(0, actual_time, n_frames) if n_frames > 0 else np.array([])

            # Determine frame step for rough analysis
            frame_step = 1
            if self.rough and n_frames > 1 and actual_time > 0:
                # Calculate step to get approximately 5ps intervals
                frame_step = max(1, int(5 / (actual_time / n_frames)))

            # Store reference positions from first frame
            u.trajectory[0]

            # Get consistent atom selection for RMSD and Rg calculations
            try:
                # Try to select protein or nucleic acid atoms
                selection = u.select_atoms("protein or nucleic")
                if len(selection) == 0:
                    # If no protein or nucleic, use all non-water atoms
                    selection = u.select_atoms("not water")
            except:
                # Fallback to all atoms
                selection = u.select_atoms("all")

            # Reused for hpSASA too
            hydrophobic_sel_str = (
                "protein and (resname "
                + ' '.join(self._HYDROPHOBIC_RESNAMES) + ")"
            )
            try:
                hydrophobic_selection = u.select_atoms(hydrophobic_sel_str)
            except:
                hydrophobic_selection = u.select_atoms("none")

            ref_positions = selection.positions.copy()

            data = []

            for i, ts in enumerate(u.trajectory):
                # Skip frames if rough analysis is enabled
                if self.rough and i % frame_step != 0:
                    continue

                # Ensure that the time_points array bounds are not exceeded
                if i >= len(time_points):
                    break

                frame_data = {
                    self.unit_col: unit_id,
                    'time': time_points[i],
                }

                if not self.skip_rmsd:
                    frame_data['rmsd'] = self._calc_rmsd(selection, ref_positions)
                if not self.skip_rg:
                    frame_data['radius_gyration'] = self._calc_rog(selection)
                if not self.skip_sasa:
                    total_sasa, hydrophobic_sasa = self._calc_sasa_metrics(
                        selection, hydrophobic_selection)
                    frame_data['sasa'] = total_sasa
                    frame_data['hydrophobic_exposure'] = hydrophobic_sasa

                # If not skipped, calculate secondary structure for selected frames only)
                if not self.skip_dssp:
                    if not self.rough or i % (frame_step * 5) == 0:  # Less frequent for SS to save time
                        ss_data = self._calc_ss(selection, unit_id, i)
                        frame_data.update(ss_data)
                    else:
                        # Use previous frame's SS data for rough analysis
                        if data and 'helix' in data[-1]:
                            for key in ['helix', 'sheet', 'coil', 'turn', 'other']:
                                frame_data[key] = data[-1].get(key, 0)
                        else:
                            # Default values if no previous data
                            frame_data.update({'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0})

                data.append(frame_data)

            # RMSF, DCCM, and LMI share one aligned-trajectory pass
            rmsf_data: List[Dict[str, Any]] = []
            dccm: Optional[np.ndarray] = None
            lmi: Optional[np.ndarray] = None
            need_alignment = not (self.skip_rmsf and self.skip_dccm and self.skip_lmi)
            if need_alignment:
                rmsf_data, aligned_disp = self._calc_rmsf(u, unit_id)
                if self.skip_rmsf:
                    rmsf_data = []  # alignment ran for DCCM/LMI, but RMSF itself wasn't requested

                if aligned_disp is not None:
                    if not self.skip_dccm:
                        dccm = self._calc_dccm(aligned_disp)
                        if dccm.size:
                            np.save(f"{out_dir}/dccm_matrix.npy", dccm)
                            self._plot_correlation_matrix(
                                dccm, f"{out_dir}/dccm_plot.png",
                                f"DCCM \u2014 {self.unit_label} {unit_id}", kind='dccm'
                            )
                    if not self.skip_lmi:
                        lmi = self._calc_lmi(aligned_disp)
                        if lmi.size:
                            np.save(f"{out_dir}/lmi_matrix.npy", lmi)
                            self._plot_correlation_matrix(
                                lmi, f"{out_dir}/lmi_plot.png",
                                f"LMI \u2014 {self.unit_label} {unit_id}", kind='lmi'
                            )

            # Generate unit-specific plots
            self._generate_replica_plots(data, rmsf_data, sim_time, out_dir, unit_id)

            return data, rmsf_data, dccm, lmi
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            traceback.print_exc()
            return [], [], None, None

    def _calc_rmsd(self, selection: mda.AtomGroup, ref_positions: np.ndarray) -> float:
        """
        Calculates RMSD against reference positions.

        Args:
            selection (mda.AtomGroup): Atom selection to calculate RMSD for
            ref_positions (numpy.ndarray): Reference positions for comparison

        Returns:
            float: Calculated RMSD value in Angstroms
        """
        try:
            if len(selection) == 0:
                return 0

            # Ensure that the same number of atoms is being compared
            if len(selection.positions) != len(ref_positions):
                return 0

            # Calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((selection.positions - ref_positions) ** 2, axis=1)))
            return rmsd
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate RMSD: {self.console.WRN}{e}{self.console.STD}")
            return 0

    def _calc_rog(self, selection: mda.AtomGroup) -> float:
        """
        Calculates radius of gyration.

        Args:
            selection (mda.AtomGroup): Atom selection to calculate Rg for

        Returns:
            float: Calculated radius of gyration in Angstroms
        """
        try:
            if len(selection) == 0:
                return 0

            # Get coordinates
            coordinates = selection.positions

            # Calculate center of geometry
            cog = np.mean(coordinates, axis=0)

            # Calculate squared distances from center
            squared_distances = np.sum((coordinates - cog) ** 2, axis=1)

            # Calculate radius of gyration
            Rg = np.sqrt(np.mean(squared_distances))
            return Rg
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate radius of gyration: {self.console.WRN}{e}{self.console.STD}")
            return 0

    _HYDROPHOBIC_RESNAMES = ('ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO')

    def _calc_sasa_metrics(self, protein: mda.AtomGroup, hydrophobic: mda.AtomGroup) -> Tuple[float, float]:
        """
        Calculates total protein SASA and hydrophobic-residue SASA.

        Args:
            protein (mda.AtomGroup): Pre-built protein atom selection
                (or an ``"all"``-atoms fallback if the Universe has no
                protein atoms), built once by the caller before its frame
                loop and reused across every frame.
            hydrophobic (mda.AtomGroup): Pre-built hydrophobic-residue
                atom selection (subset of ``protein``), built once by the
                caller and reused across every frame. Used only by the
                fallback atom-count estimator if the Bio.PDB SASA
                computation itself fails.

        Returns:
            tuple: ``(total_sasa, hydrophobic_sasa)``, both in Å².
                ``hydrophobic_sasa`` is the subset of ``total_sasa``
                contributed by atoms belonging to hydrophobic residues, so
                ``hydrophobic_sasa <= total_sasa`` always holds.
        """
        try:
            # Write temporary PDB file for this frame (protein atoms only)
            positions = protein.positions
            temp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w')
            temp_pdb.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")
            for j, atom in enumerate(protein.atoms):
                x, y, z = positions[j]
                element = atom.element if hasattr(atom, 'element') else atom.name[0]
                res_seq = str(atom.resid % 10000).rjust(4)
                line = (f"ATOM  {j+1:5d} {atom.name:<4s} {atom.resname:<3s} A{res_seq}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}\n")
                temp_pdb.write(line)
            temp_pdb.write("TER\n")
            temp_pdb.close()

            # Parse with Biopython.
            parser = PDBParser()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PDBConstructionWarning)
                structure = parser.get_structure('temp', temp_pdb.name)

            # Calculate SASA using Shrake-Rupley algorithm
            sasa_calculator = ShrakeRupley()
            sasa_calculator.compute(structure, level="S")

            # Sum total SASA, and separately the subset belonging to
            # hydrophobic residues.
            total_sasa = 0.0
            hydrophobic_sasa = 0.0
            for atom in structure.get_atoms():
                total_sasa += atom.sasa
                residue = atom.get_parent()
                if residue is not None and residue.get_resname() in self._HYDROPHOBIC_RESNAMES:
                    hydrophobic_sasa += atom.sasa

            # Clean up
            os.unlink(temp_pdb.name)
            return total_sasa, hydrophobic_sasa
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate SASA: {self.console.WRN}{e}{self.console.STD}")
            # Fallback to simple estimation when Biopython SASA fails.
            # Using ~15 Å² per atom as a rough per-atom SASA heuristic.
            try:
                return len(protein) * 15, len(hydrophobic) * 15
            except:
                return 0, 0

    def _calc_rmsf(self, universe: mda.Universe, rep_num: int) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Calculate per-residue RMSF for Cα atoms over the full trajectory,
        and return the shared per-frame aligned Cα displacement array used
        by DCCM/LMI.

        Selects Cα atoms, aligns each frame to the first-frame reference via a
        rotation matrix, accumulates squared deviations, and returns the
        root-mean value per residue. The same trajectory is returned so that
        ``_calc_dccm``/``_calc_lmi`` can be computed.

        Args:
            universe (mda.Universe): MDAnalysis Universe object containing the trajectory.
            rep_num (int): Replica number identifier, stored in the returned records.

        Returns:
            tuple: (rmsf_data, aligned_disp) where:
                - rmsf_data (list[dict]): One dictionary per Cα atom with keys:
                    - replica (int): Replica number.
                    - residue_index (int): Residue sequence number (resid).
                    - residue_name (str): Three-letter residue name.
                    - rmsf (float): Root-mean-square fluctuation in Å.
                  Empty list if no Cα atoms are found or an error occurs.
                - aligned_disp (np.ndarray or None): Shape
                  (n_frames, n_ca, 3), per-frame Kabsch-superposed Cα
                  coordinates minus the mean structure (mean-centered), in
                  Å. None if no Cα atoms are found or an error occurs.
        """
        try:
            # Select Cα atoms
            calphas = universe.select_atoms("protein and name CA")
            if len(calphas) == 0:
                print(f"{self.console.PGM_WRN}No Cα atoms found for RMSF calculation.")
                return [], None

            n_frames = len(universe.trajectory)

            # Store the first-frame positions as the alignment reference
            ref_coords = calphas.positions.copy()
            aligned_coords = np.empty((n_frames, len(calphas), 3), dtype=np.float64)

            for i, ts in enumerate(universe.trajectory):
                # Align each frame to the reference
                mobile_coords = calphas.positions
                R, rmsd = align.rotation_matrix(mobile_coords, ref_coords)
                calphas.positions = np.dot(mobile_coords, R.T)
                aligned_coords[i] = calphas.positions

            # Mean-center: subtract the average (aligned) structure.
            mean_coords = aligned_coords.mean(axis=0)
            aligned_disp = aligned_coords - mean_coords[np.newaxis, :, :]

            # sqrt of mean squared deviation over all frames
            rmsf_values = np.sqrt(np.mean(np.sum(aligned_disp ** 2, axis=2), axis=0))

            rmsf_data = []
            for i, atom in enumerate(calphas):
                rmsf_data.append({
                    self.unit_col: rep_num,
                    'residue_index': atom.residue.resid,
                    'residue_name': atom.residue.resname,
                    'rmsf': rmsf_values[i]
                })

            return rmsf_data, aligned_disp

        except Exception as e:
            print(f"{self.console.PGM_ERR}Error calculating RMSF: {self.console.ERR}{e}{self.console.STD}")
            return [], None

    def _calc_dccm(self, aligned_disp: np.ndarray) -> np.ndarray:
        """
        Compute the linear (Pearson) dynamic cross-correlation matrix
        (DCCM) from a shared aligned/mean-centered Cα displacement
        trajectory.

            C_ij = <dr_i . dr_j> / sqrt(<dr_i^2> <dr_j^2>)

        Values range from +1 (fully correlated motion) through 0
        (uncorrelated) to -1 (fully anti-correlated motion).

        Reference: Ichiye & Karplus, Proteins 1991, 11:205-217,
        DOI: 10.1002/prot.340110305.

        Args:
            aligned_disp (np.ndarray): Shape (n_frames, n_ca, 3), per-frame
                Kabsch-superposed, mean-centered Cα displacements in Å (as
                returned by ``_calc_rmsf``).

        Returns:
            np.ndarray: Shape (n_ca, n_ca), symmetric, values in [-1, 1],
                unit diagonal. Empty (0, 0) array on failure.
        """
        try:
            # Dot product of displacement vectors between every pair of
            # atoms, averaged over frames: (n_ca, n_ca). einsum sums over
            # frames (f) and Cartesian components (k).
            cross = np.einsum('fik,fjk->ij', aligned_disp, aligned_disp) / aligned_disp.shape[0]

            # <dr_i^2> is the diagonal of `cross`
            variances = np.diag(cross).copy()
            variances[variances < 1e-12] = 1e-12  # guard against zero-fluctuation atoms

            norm = np.sqrt(np.outer(variances, variances))
            dccm = cross / norm

            # Numerical safety: clip to [-1, 1] and force exact symmetry
            dccm = np.clip(dccm, -1.0, 1.0)
            dccm = 0.5 * (dccm + dccm.T)
            np.fill_diagonal(dccm, 1.0)
            return dccm

        except Exception as e:
            print(f"{self.console.PGM_ERR}Error calculating DCCM: {self.console.ERR}{e}{self.console.STD}")
            return np.zeros((0, 0))

    def _calc_lmi(self, aligned_disp: np.ndarray) -> np.ndarray:
        """
        Compute the Linear Mutual Information (LMI) matrix from a shared
        aligned/mean-centered Cα displacement trajectory, using the
        Gaussian approximation of generalized correlation.

            I_ij = 0.5 * [ln(det(Sigma_ii)) + ln(det(Sigma_jj)) - ln(det(Sigma_combined))]
            LMI_ij = sqrt(1 - exp(-2/3 * I_ij))

        where Sigma_ii, Sigma_jj are the 3x3 covariance matrices of atoms
        i, j and Sigma_combined is the 6x6 joint covariance matrix of
        (i, j). Unlike DCCM, LMI is signless (it measures total coupling
        strength, not direction) and ranges over [0, 1].

        Reference: Lange & Grubmüller, Proteins 2006, 62:1053-1061,
        DOI: 10.1002/prot.20784.

        Args:
            aligned_disp (np.ndarray): Shape (n_frames, n_ca, 3), per-frame
                Kabsch-superposed, mean-centered Cα displacements in Å (as
                returned by ``_calc_rmsf``).

        Returns:
            np.ndarray: Shape (n_ca, n_ca), symmetric, values in [0, 1],
                unit diagonal. Empty (0, 0) array on failure.
        """
        try:
            n_frames, n_ca, _ = aligned_disp.shape
            disp = aligned_disp  # (F, N, 3)

            # Per-atom 3x3 covariance matrices: (n_ca, 3, 3)
            cov_ii = np.einsum('fik,fil->ikl', disp, disp) / n_frames

            # log(det(Sigma_ii)) per atom, shape (N,). A small Tikhonov
            # regularization guards against near-singular covariance
            eps_reg = 1e-10
            eye3 = np.eye(3)
            cov_ii_reg = cov_ii + eps_reg * eye3[np.newaxis, :, :]
            sign_ii, logdet_ii = np.linalg.slogdet(cov_ii_reg)

            lmi = np.zeros((n_ca, n_ca), dtype=np.float64)

            # Only the joint 6x6 covariance genuinely needs a pairwise
            # loop (it mixes atoms i and j); everything single-atom is
            # precomputed above.
            for i in range(n_ca):
                di = disp[:, i, :]  # (F, 3)
                for j in range(i + 1, n_ca):
                    dj = disp[:, j, :]  # (F, 3)
                    joint = np.concatenate([di, dj], axis=1)  # (F, 6)
                    cov_ij = (joint.T @ joint) / n_frames      # (6, 6)
                    cov_ij_reg = cov_ij + eps_reg * np.eye(6)
                    sign_c, logdet_c = np.linalg.slogdet(cov_ij_reg)

                    if sign_ii[i] <= 0 or sign_ii[j] <= 0 or sign_c <= 0:
                        continue  # degenerate covariance; leave as 0

                    I_ij = 0.5 * (logdet_ii[i] + logdet_ii[j] - logdet_c)
                    I_ij = max(I_ij, 0.0)  # numerical noise can make this slightly negative
                    val = np.sqrt(max(0.0, 1.0 - np.exp(-2.0 / 3.0 * I_ij)))
                    lmi[i, j] = val
                    lmi[j, i] = val

            np.fill_diagonal(lmi, 1.0)
            return lmi

        except Exception as e:
            print(f"{self.console.PGM_ERR}Error calculating LMI: {self.console.ERR}{e}{self.console.STD}")
            return np.zeros((0, 0))

    def _plot_correlation_matrix(self, matrix: np.ndarray, out_path: str, title: str,
                                 kind: str = 'dccm') -> None:
        """
        Plot and save a correlation-matrix heatmap. Shared by DCCM/LMI,
        for both per-unit and cross-unit average matrices.

        Args:
            matrix (np.ndarray): (n_ca, n_ca) correlation matrix.
            out_path (str): Output PNG path.
            title (str): Plot title.
            kind (str): 'dccm' (diverging colormap, red = +1 fully
                correlated, white = 0 uncorrelated, blue = -1 fully
                anti-correlated, range [-1, 1]) or 'lmi' (sequential
                colormap, range [0, 1], since LMI has no sign).
        """
        if matrix.size == 0:
            return
        try:
            fig, ax = plt.subplots(figsize=(7, 6))
            if kind == 'lmi':
                cmap = 'viridis'
                vmin, vmax = 0.0, 1.0
                cbar_label = 'LMI'
            else:
                cmap = 'RdBu_r'
                vmin, vmax = -1.0, 1.0
                cbar_label = 'Correlation coefficient'

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)
            ax.set_xlabel('Residue index', fontsize=12)
            ax.set_ylabel('Residue index', fontsize=12)
            ax.set_title(title, fontsize=13)
            plt.tight_layout()
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create {self.console.WRN}{kind.upper()}{self.console.STD} plot: "
                  f"{self.console.WRN}{e}{self.console.STD}")

    def _generate_correlation_avg_plot(self, sum_matrix: Optional[np.ndarray], count: int,
                                       kind: str = 'dccm') -> None:
        """
        Save the cross-unit average correlation matrix (element-wise mean
        over all replicas/centroids), plus its heatmap, from a running sum
        accumulated incrementally as each unit's result arrived (see
        ``_update_correlation_running_sum``) rather than from a retained
        list of every unit's full matrix.

        Args:
            sum_matrix (np.ndarray or None): Element-wise sum of every
                contributing unit's (n_ca, n_ca) matrix. ``None`` if no
                unit contributed (e.g. all were skipped or none computed).
            count (int): Number of units folded into ``sum_matrix``.
                Shape-mismatched units are excluded from both
                ``sum_matrix`` and ``count`` by
                ``_update_correlation_running_sum``, so no shape check is
                needed here.
            kind (str): 'dccm' or 'lmi'; controls output filenames,
                colormap, and value range (see ``_plot_correlation_matrix``).
        """
        if sum_matrix is None or count == 0:
            print(f"{self.console.PGM_WRN}No {kind.upper()} data to generate average plot.")
            return
        try:
            avg_matrix = sum_matrix / count
            np.save(f"{self.analysis_dir}/{kind}_average.npy", avg_matrix)
            self._plot_correlation_matrix(
                avg_matrix, f"{self.analysis_dir}/{kind}_average.png",
                f"Average {kind.upper()} ({count} {self.unit_label.lower()}s)",
                kind=kind
            )
            print(f"{self.console.PGM_NAM}Average {kind.upper()} saved to "
                  f"{self.console.EXT}{self.analysis_dir}/{kind}_average.png"
                  f"{self.console.STD}.")
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create average {self.console.WRN}{kind.upper()}{self.console.STD} plot: "
                  f"{self.console.WRN}{e}{self.console.STD}")

    def _update_correlation_running_sum(self, sum_matrix: Optional[np.ndarray], count: int,
                                        new_matrix: Optional[np.ndarray],
                                        kind: str, unit_id: int) -> Tuple[Optional[np.ndarray], int]:
        """
        Fold one unit's correlation matrix into a running element-wise sum,
        instead of appending it to a list retained for the whole method
        call.

        Args:
            sum_matrix (np.ndarray or None): Running sum so far, or
                ``None`` if no unit has contributed yet.
            count (int): Number of units folded into ``sum_matrix`` so far.
            new_matrix (np.ndarray or None): This unit's (n_ca, n_ca)
                matrix, or ``None``/empty if this unit had none (skipped or
                failed) -- in which case ``sum_matrix``/``count`` are
                returned unchanged.
            kind (str): 'dccm' or 'lmi', used only in the mismatch warning.
            unit_id (int): This unit's identifier, used only in the
                mismatch warning.

        Returns:
            tuple: Updated ``(sum_matrix, count)``.

        Note:
            If ``new_matrix``'s shape differs from the running sum's shape
            (e.g. a different Cα count for that unit), this unit is
            skipped from the average with a warning, rather than aborting
            the average for every unit.
        """
        if new_matrix is None or not new_matrix.size:
            return sum_matrix, count
        if sum_matrix is not None and new_matrix.shape != sum_matrix.shape:
            print(f"{self.console.PGM_WRN}{kind.upper()} matrix for "
                  f"{self.unit_label.lower()} {unit_id} has shape "
                  f"{new_matrix.shape}, expected {sum_matrix.shape}; "
                  "excluding it from the average (likely a different Cα "
                  "count for this unit).")
            return sum_matrix, count
        if sum_matrix is None:
            sum_matrix = new_matrix.copy()
        else:
            sum_matrix = sum_matrix + new_matrix
        return sum_matrix, count + 1

    def _calc_ss(self, protein: mda.AtomGroup, rep_num: int, frame_idx: int) -> Dict[str, int]:
        """
        Calculates secondary structure content using DSSP.

        Args:
            protein (mda.AtomGroup): Pre-built protein atom selection,
                built once by the caller before its frame loop and reused
                across every frame.
            rep_num (int): Replica number identifier
            frame_idx (int): Frame index number

        Returns:
            dict: Dictionary with secondary structure counts
        """
        try:
            if len(protein) == 0:
                print(f"{self.console.PGM_WRN}No protein atoms found for secondary structure analysis.")
                return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0}

            # Create temporary PDB file
            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False, mode='w') as temp_pdb:
                pdb_path = temp_pdb.name

                # Write a dummy header to avoid DSSP error
                temp_pdb.write(f"HEADER     MDANALYSIS FRAME {frame_idx}: Created by PDBWriter\n")
                temp_pdb.write("CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")

                # Write protein atoms to the PDB file
                for i, atom in enumerate(protein.atoms):
                    # Format atom record according to PDB specification
                    record = "ATOM  "
                    serial = str(i+1).rjust(5)
                    name = atom.name.ljust(4)
                    alt_loc = " "
                    res_name = atom.resname.ljust(3)
                    chain_id = "A"
                    res_seq = str(atom.resid).rjust(4)
                    i_code = " "
                    x = "{:8.3f}".format(atom.position[0])
                    y = "{:8.3f}".format(atom.position[1])
                    z = "{:8.3f}".format(atom.position[2])
                    occupancy = "  1.00"
                    temp_factor = "  0.00"
                    element = atom.element.rjust(2) if hasattr(atom, 'element') else "  "
                    charge = "  "

                    atom_line = f"{record}{serial} {name}{alt_loc}{res_name} {chain_id}{res_seq}{i_code}   {x}{y}{z}{occupancy}{temp_factor}          {element}{charge}\n"
                    temp_pdb.write(atom_line)

                # Add TER record at the end
                temp_pdb.write("TER\n")

            # Use DSSP command directly
            try:
                # Create temporary DSSP output file
                with tempfile.NamedTemporaryFile(suffix='.dssp', delete=False) as temp_dssp:
                    dssp_path = temp_dssp.name

                # Run DSSP command
                cmd = f"dssp --output-format dssp {pdb_path} {dssp_path}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                # Clean up PDB file
                os.unlink(pdb_path)

                if result.returncode != 0:
                    print(f"{self.console.PGM_WRN}DSSP command failed: {result.stderr}")
                    # Clean up DSSP file
                    os.unlink(dssp_path)
                    return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0}

                # Parse DSSP output
                ss_data = self._parse_dssp_output(dssp_path)

                # Clean up DSSP file
                os.unlink(dssp_path)

                return ss_data

            except Exception as dssp_error:
                print(f"{self.console.PGM_WRN}DSSP calculation failed: {dssp_error}")
                # Clean up files if they exist
                if os.path.exists(pdb_path):
                    os.unlink(pdb_path)
                if os.path.exists(dssp_path):
                    os.unlink(dssp_path)
                return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0}

        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate secondary structure: {self.console.WRN}{e}{self.console.STD}")
            return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0}

    def _parse_dssp_output(self, dssp_path: str) -> Dict[str, int]:
        """
        Parses DSSP output file and counts secondary structure types.

        Args:
            dssp_path (str): Path to DSSP output file

        Returns:
            dict: Dictionary with secondary structure counts
        """
        ss_data = {
            'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0
        }

        try:
            with open(dssp_path, 'r') as f:
                lines = f.readlines()

            # DSSP format: secondary structure assignment is at position 16 (0-based index)
            # Skip header lines (look for the line that starts with "  #")
            start_parsing = False
            processed_residues = set()

            for line in lines:
                if line.startswith("  #"):
                    start_parsing = True
                    continue

                if start_parsing and len(line) > 16:
                    # Extract residue identifier to avoid double-counting
                    residue_id = line[5:10].strip()  # Residue number
                    chain_id = line[10:12].strip()   # Chain identifier
                    unique_id = f"{chain_id}_{residue_id}"

                    # Skip if we've already processed this residue
                    if unique_id in processed_residues:
                        continue

                    processed_residues.add(unique_id)

                    ss_type = line[16]

                    # DSSP codes: H=α-helix, G=3(10)-helix, I=π-helix to helix
                    #             E=β-strand, B=β-bridge to sheet
                    #             T=hydrogen-bonded turn, ' '=random coil
                    if ss_type in ['H', 'G', 'I']:
                        ss_data['helix'] += 1
                    elif ss_type in ['E', 'B']:
                        ss_data['sheet'] += 1
                    elif ss_type == 'T':
                        ss_data['turn'] += 1
                    elif ss_type == ' ':
                        ss_data['coil'] += 1
                    else:
                        ss_data['other'] += 1

            return ss_data

        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not parse DSSP output: {self.console.WRN}{e}{self.console.STD}")
            return {'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0}

    def _plot_ss_rep(self, df: pd.DataFrame, sim_time: int, rep_analysis_dir: str, rep_num: int) -> None:
        """
        Creates secondary structure plot for a single replica.

        Args:
            df (pandas.DataFrame): DataFrame containing analysis data
            sim_time (int): Total simulation time in picoseconds
            rep_analysis_dir (str): Output directory for plots
            rep_num (int): Replica number identifier
        """
        try:
            plt.figure(figsize=(6, 6))

            # Create stacked area plot for this replica
            plt.stackplot(df['time'],
                         df['helix'],
                         df['sheet'],
                         df['coil'],
                         df['turn'],
                         df['other'],
                         labels=['Helix', 'Sheet', 'Coil', 'Turn', 'Other'],
                         alpha=0.8)

            plt.xlabel('Time (ps)')
            plt.ylabel('Number of Residues')
            plt.title(f'Secondary Structure Evolution - {self.unit_label} {rep_num}')
            plt.xlim(0, sim_time)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f"{rep_analysis_dir}/secondary_structure.png", bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create secondary structure plot for "
                  f"{self.console.WRN}{self.unit_label.lower()} {rep_num}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

    def _append_csv_rows(self, rows: List[Dict[str, Any]], csv_file: str,
                         header_written: bool) -> bool:
        """
        Append one unit's rows to a combined CSV incrementally, instead of
        accumulating every unit's rows in memory and writing the whole
        file once at the end. Used for the parent-process combined
        ``analysis_results.csv``/``rmsf.csv`` so a unit's data can be
        dropped from memory as soon as its worker result has been
        processed.

        Args:
            rows (list[dict]): This unit's rows (one dict per frame, or per
                Cα atom for RMSF). No-op if empty.
            csv_file (str): Combined CSV path. Rows are appended in the
                order this method is called, matching the previous
                ``extend()``-based accumulation order (unit-major, in
                worker-submission order).
            header_written (bool): Whether the header has already been
                written to ``csv_file`` by an earlier call in this same
                combined-CSV sequence.

        Returns:
            bool: Updated ``header_written`` (``True`` after this call if
                ``rows`` was non-empty; unchanged otherwise).
        """
        if not rows:
            return header_written
        df = pd.DataFrame(rows)
        mode = 'a' if header_written else 'w'
        df.to_csv(csv_file, mode=mode, header=not header_written, index=False)
        return True

    def _update_rmsf_running_sum(self, running: Dict[int, List[float]],
                                 rmsf_data: List[Dict[str, Any]]) -> None:
        """
        Fold one unit's per-residue RMSF rows into a running
        ``{residue_index: [sum_rmsf, count]}`` accumulator, in place,
        instead of retaining every unit's per-atom RMSF rows in memory for
        the whole method call. Used to compute the average-RMSF plot
        (``_generate_rmsf_avg_plot``) from a small dict (one entry per Cα
        atom, not one per unit) rather than a list that grows with the
        number of units.

        Args:
            running (dict[int, list[float]]): Accumulator, mutated in
                place. Keys are ``residue_index``; values are
                ``[running_sum, running_count]`` (a 2-element list so it
                can be updated in place without dict-of-tuple
                reassignment).
            rmsf_data (list[dict]): This unit's RMSF rows, as returned by
                ``_calc_rmsf`` (keys include ``residue_index`` and
                ``rmsf``). No-op if empty.
        """
        for row in rmsf_data:
            idx = row['residue_index']
            val = row['rmsf']
            if idx in running:
                running[idx][0] += val
                running[idx][1] += 1
            else:
                running[idx] = [val, 1]

    def _plot_rmsf_rep(self, rmsf_data: List[Dict[str, Any]], rep_analysis_dir: str, rep_num: int) -> None:
        """
        Creates RMSF plot for a single replica.

        Args:
            rmsf_data (list): List of RMSF data dictionaries
            rep_analysis_dir (str): Output directory for plots
            rep_num (int): Replica number identifier
        """
        try:
            rmsf_df = pd.DataFrame(rmsf_data)

            plt.figure(figsize=(10, 6))
            plt.plot(rmsf_df['residue_index'], rmsf_df['rmsf'], 'b-', linewidth=1.5)
            plt.xlabel('Residue Index')
            plt.ylabel('RMSF (Å)')
            plt.title(f'RMSF per Residue (Cα) - {self.unit_label} {rep_num}')
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f"{rep_analysis_dir}/rmsf_plot.png", bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create RMSF plot for "
                  f"{self.console.WRN}{self.unit_label.lower()} {rep_num}{self.console.STD}: {self.console.WRN}{e}{self.console.STD}")

    def _generate_html_summary(self, df: pd.DataFrame, sim_time: int,
                               incomplete: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """
        Generates an HTML summary of the analysis results.
        Only includes sections for analyses that were actually computed
        (i.e. not disabled via skip flags).

        Args:
            df (pandas.DataFrame): Combined per-frame analysis data for
                every unit, as read back from the incrementally-written
                ``analysis_results.csv`` (see ``analyze_all_replicas`` /
                ``analyze_all_centroids``) rather than built from an
                in-memory list of every unit's rows.
            sim_time (int): Total simulation time in picoseconds
            incomplete (list[tuple[int, int, int]], optional): One
                ``(unit_id, last_completed_cycle, target_cycle)`` tuple per
                unit that had not reached its target cycle count at
                analysis time, as returned by ``check_pyadmd_completion``/
                ``check_fel_completion``. When non-empty, a note and
                table are added to the summary; each such unit's own data
                reflects only the cycles it actually completed.
        """
        if df.empty:
            print(f"{self.console.PGM_WRN}No data to generate HTML summary.")
            return

        # Build the list of (column, label, is_max) tuples for computed stats only
        stat_specs = []
        if not self.skip_rmsd:
            stat_specs.append(('rmsd', 'Max RMSD (Å)', True))
        if not self.skip_rg:
            stat_specs.append(('radius_gyration', 'Final Radius of Gyration (Å)', False))
        if not self.skip_sasa:
            stat_specs.append(('sasa', 'Final SASA (Å²)', False))
            stat_specs.append(('hydrophobic_exposure', 'Max Hydrophobic Exposure (Å²)', True))
        if not self.skip_dssp:
            stat_specs += [
                ('helix', 'Final Helix (residues)', False),
                ('sheet', 'Final Sheet (residues)', False),
                ('coil',  'Final Coil (residues)',  False),
                ('turn',  'Final Turn (residues)',  False),
                ('other', 'Final Other (residues)', False),
            ]

        # Calculate statistics for each analysis unit
        summary_data = {}
        for unit_val in df[self.unit_col].unique():
            rep_data = df[df[self.unit_col] == unit_val]
            replica_summary = {}
            for col, label, use_max in stat_specs:
                if col in rep_data.columns:
                    val = rep_data[col].max() if use_max else rep_data[col].iloc[-1]
                    replica_summary[label] = val if not rep_data.empty else 0
            summary_data[f'{self.unit_label} {unit_val}'] = replica_summary

        # Calculate averages across all analysis units
        avg_summary = {}
        for col, label, use_max in stat_specs:
            if col in df.columns:
                if use_max:
                    avg_summary[f'Average Max {col.upper()}'] = df.groupby(self.unit_col)[col].max().mean()
                else:
                    avg_summary[f'Average Final {col.upper()}'] = df.groupby(self.unit_col)[col].last().mean()

        # Build conditional notes
        notes_items = [f"<li>For detailed analysis, see the files in the {self.unit_label.lower()}-specific subdirectories</li>"]
        if not self.skip_dssp:
            notes_items.insert(0, "<li>Secondary structure content is calculated using DSSP</li>")
            notes_items.insert(1, "<li>Values represent the number of residues in each secondary structure type</li>")
        if not self.skip_sasa:
            notes_items.insert(-1, "<li>SASA is calculated using Bio.PDB.SASA (Shrake-Rupley algorithm)</li>")
        if not self.skip_dccm:
            notes_items.insert(-1, "<li>DCCM (dynamic cross-correlation matrix) uses linear Pearson "
                                    "correlation of Kabsch-aligned Cα displacements; +1 = fully "
                                    "correlated, 0 = uncorrelated, -1 = fully anti-correlated</li>")
        if not self.skip_lmi:
            notes_items.insert(-1, "<li>LMI (Linear Mutual Information) is signless (range [0, 1]) and "
                                    "reports total coupling strength regardless of correlation direction</li>")
        notes_html = "\n                    ".join(notes_items)

        # Source note: which trajectories this summary was generated from
        n_units = df[self.unit_col].nunique()
        if self.source == 'fel':
            source_note = (f"fel centroid production trajectories "
                          f"({n_units} centroids, {sim_time} ps production each)")
        else:
            source_note = f"pyAdMD replica runs ({n_units} replicas, {sim_time} ps each)"

        # Incomplete-units note/table: only rendered when there is at least
        # one unit that had not reached its target cycle count at analysis
        # time (see analyze_all_replicas / analyze_all_centroids).
        incomplete_html = ""
        if incomplete:
            rows = "\n                    ".join(
                f"<tr><td>{self.unit_label} {unit_id}</td><td>{last_cycle}</td>"
                f"<td>{target}</td></tr>"
                for unit_id, last_cycle, target in incomplete
            )
            incomplete_html = f"""
                <h2>Incomplete Units</h2>
                <p>The following {len(incomplete)} {self.unit_label.lower()}(s) had not
                reached their target cycle count at analysis time. They were
                analyzed anyway, using only the cycles they actually
                completed; their per-frame time axis reflects their real
                elapsed simulation time, not the target.</p>
                <table>
                    <tr><th>{self.unit_label}</th><th>Cycles completed</th><th>Target cycles</th></tr>
                    {rows}
                </table>
            """

        # Generate HTML file with escaped curly braces in CSS
        html_file = f"{self.analysis_dir}/analysis_summary.html"
        with open(html_file, 'w') as f:
            html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>PyAdMD Analysis Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .plot-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
                    .plot-item {{ text-align: center; }}
                    .plot-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>PyAdMD Analysis Summary</h1>
                <p>Generated on: {date}</p>
                <p>Source: {source_note}</p>
                {incomplete_html}

                <h2>Summary Statistics</h2>
                <h3>Individual {unit_label} Results</h3>
                {replica_tables}

                <h3>Average Across All {unit_label_plural}</h3>
                <table>
                    {avg_table_rows}
                </table>

                <h2>Analysis Plots</h2>
                <div class="plot-grid">
                    {plot_items}
                </div>

                <h2>Notes</h2>
                <ul>
                    {notes_html}
                </ul>
            </body>
            </html>
            """
            f.write(html_template.format(
                date=time.strftime("%Y-%m-%d %H:%M:%S"),
                source_note=source_note,
                incomplete_html=incomplete_html,
                unit_label=self.unit_label,
                unit_label_plural=f"{self.unit_label}s",
                replica_tables=self._html_rep_tables(summary_data),
                avg_table_rows=self._html_summary_avg_table(avg_summary),
                plot_items=self._html_summary_plots(),
                notes_html=notes_html
            ))

        print(f"{self.console.PGM_NAM}HTML summary saved to {self.console.EXT}{html_file}{self.console.STD}")

    def _generate_plots(self, df: pd.DataFrame, sim_time: int) -> None:
        """
        Generates plots from analysis data.

        Args:
            df (pandas.DataFrame): Combined per-frame analysis data for
                every unit, as read back from the incrementally-written
                ``analysis_results.csv`` rather than built from an
                in-memory list of every unit's rows.
            sim_time (int): Total simulation time in picoseconds
        """
        if df.empty:
            print(f"{self.console.PGM_WRN}No data to generate plots.")
            return

        # Create individual plots for each enabled property
        properties = []
        if not self.skip_rmsd:
            properties.append(('rmsd', 'RMSD (Å)', 'RMSD'))
        if not self.skip_rg:
            properties.append(('radius_gyration', 'Radius of Gyration (Å)', 'Radius of Gyration'))
        if not self.skip_sasa:
            properties.append(('sasa', 'SASA (Å²)', 'SASA'))
            properties.append(('hydrophobic_exposure', 'Hydrophobic Exposure (Å²)', 'Hydrophobic Exposure'))

        for prop, ylabel, title in properties:
            plt.figure(figsize=(8, 6))

            for unit_val in df[self.unit_col].unique():
                rep_data = df[df[self.unit_col] == unit_val]
                plt.plot(rep_data['time'], rep_data[prop], label=f'{self.unit_label} {unit_val}', alpha=0.7, linewidth=1.5)

            plt.xlabel('Time (ps)')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xlim(0, sim_time)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save individual plot
            plt.savefig(f"{self.analysis_dir}/{prop}_plot.png", bbox_inches='tight', dpi=300)
            plt.close()

        # Create average secondary structure plot (only if DSSP was computed)
        if not self.skip_dssp:
            self._generate_ss_avg_plot(df, sim_time)

    def _generate_rmsf_avg_plot(self, running_rmsf: Dict[int, List[float]]) -> None:
        """
        Generates average RMSF plot across all replicas, from a running
        ``{residue_index: [sum_rmsf, count]}`` accumulator built
        incrementally by ``_update_rmsf_running_sum`` as each unit's
        result arrived, rather than from a retained list of every unit's
        per-atom RMSF rows.

        Args:
            running_rmsf (dict[int, list[float]]): Accumulator as produced
                by ``_update_rmsf_running_sum``; keys are
                ``residue_index``, values are ``[running_sum,
                running_count]``.
        """
        if not running_rmsf:
            print(f"{self.console.PGM_WRN}No RMSF data to generate plots.")
            return

        residue_indices = sorted(running_rmsf.keys())
        avg_values = [running_rmsf[idx][0] / running_rmsf[idx][1] for idx in residue_indices]

        # Create average RMSF plot across all replicas
        plt.figure(figsize=(10, 6))

        plt.plot(residue_indices, avg_values, 'b-', linewidth=2, label='Average')
        plt.xlabel('Residue Index')
        plt.ylabel('RMSF (Å)')
        plt.title('Average RMSF per Residue (Cα)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save average plot
        plt.savefig(f"{self.analysis_dir}/rmsf_average.png", bbox_inches='tight', dpi=300)
        plt.close()

    def _generate_ss_avg_plot(self, df: pd.DataFrame, sim_time: int) -> None:
        """
        Creates stacked area plot for average secondary structure.

        Args:
            df (pandas.DataFrame): DataFrame containing analysis data
            sim_time (int): Total simulation time in picoseconds
        """
        try:
            plt.figure(figsize=(8, 6))

            # Group by time and calculate averages
            time_groups = df.groupby('time')
            avg_data = time_groups[['helix', 'sheet', 'coil', 'turn', 'other']].mean()

            # Create stacked area plot
            plt.stackplot(avg_data.index,
                         avg_data['helix'],
                         avg_data['sheet'],
                         avg_data['coil'],
                         avg_data['turn'],
                         avg_data['other'],
                         labels=['Helix', 'Sheet', 'Coil', 'Turn', 'Other'],
                         alpha=0.8)

            plt.xlabel('Time (ps)')
            plt.ylabel('Number of Residues')
            plt.title('Average Secondary Structure Evolution')
            plt.xlim(0, sim_time)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f"{self.analysis_dir}/secondary_structure_average.png", bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create average secondary structure plot: {self.console.WRN}{e}{self.console.STD}")

    def _html_rep_tables(self, summary_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Generates HTML tables for replica summaries.

        Args:
            summary_data (dict): Dictionary containing replica summary data

        Returns:
            str: HTML string containing replica tables
        """
        html_tables = ""
        for replica, stats in summary_data.items():
            html_tables += f"""
            <h4>{replica}</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            for stat, value in stats.items():
                if 'residues' in stat:
                    html_tables += f"<tr><th>{stat}</th><td>{value:.0f}</td></tr>"
                else:
                    html_tables += f"<tr><th>{stat}</th><td>{value:.2f}</td></tr>"
            html_tables += "</table>"
        return html_tables

    def _html_summary_avg_table(self, avg_summary: Dict[str, float]) -> str:
        """
        Generates HTML table for average summary.

        Args:
            avg_summary (dict): Dictionary containing average summary data

        Returns:
            str: HTML string containing average summary table
        """
        html_rows = ""
        for stat, value in avg_summary.items():
            if 'HELIX' in stat or 'SHEET' in stat or 'COIL' in stat or 'TURN' in stat or 'OTHER' in stat:
                html_rows += f"<tr><th>{stat}</th><td>{value:.0f}</td></tr>"
            else:
                html_rows += f"<tr><th>{stat}</th><td>{value:.2f}</td></tr>"
        return html_rows

    def _html_summary_plots(self) -> str:
        """
        Generates HTML img elements for plots.

        Returns:
            str: HTML string containing plot images
        """
        plot_files = [
            "rmsd_plot.png", "radius_gyration_plot.png",
            "sasa_plot.png", "hydrophobic_exposure_plot.png",
            "rmsf_average.png", "secondary_structure_average.png",
            "dccm_average.png", "lmi_average.png",
        ]

        plot_items = ""
        for plot_file in plot_files:
            if os.path.exists(f"{self.analysis_dir}/{plot_file}"):
                plot_items += f"""
                <div class="plot-item">
                    <img src="{plot_file}" alt="{plot_file.replace('_', ' ').replace('.png', '')}">
                    <p>{plot_file.replace('_', ' ').replace('.png', '')}</p>
                </div>
                """
        return plot_items

    def _generate_replica_plots(self, data: List[Dict[str, Any]], rmsf_data: List[Dict[str, Any]],
                                sim_time: int, rep_analysis_dir: str, rep_num: int) -> None:
        """
        Generates plots for a single replica analysis.

        Args:
            data (list): List of analysis data dictionaries
            rmsf_data (list): List of RMSF data dictionaries
            sim_time (int): Total simulation time in picoseconds
            rep_analysis_dir (str): Output directory for plots
            rep_num (int): Replica number identifier
        """
        if not data:
            return

        df = pd.DataFrame(data)

        # Create individual plots for each enabled property
        properties = []
        if not self.skip_rmsd:
            properties.append(('rmsd', 'RMSD (Å)', 'RMSD'))
        if not self.skip_rg:
            properties.append(('radius_gyration', 'Radius of Gyration (Å)', 'Radius of Gyration'))
        if not self.skip_sasa:
            properties.append(('sasa', 'SASA (Å²)', 'SASA'))
            properties.append(('hydrophobic_exposure', 'Hydrophobic Exposure (Å²)', 'Hydrophobic Exposure'))

        for prop, ylabel, title in properties:
            plt.figure(figsize=(6, 6))
            plt.plot(df['time'], df[prop], label=f'{self.unit_label} {rep_num}', color='blue', linewidth=2)
            plt.xlabel('Time (ps)')
            plt.ylabel(ylabel)
            plt.title(f'{title} - {self.unit_label} {rep_num}')
            plt.xlim(0, sim_time)
            plt.grid(True, alpha=0.3)

            # Save individual plot
            plt.savefig(f"{rep_analysis_dir}/{prop}_plot.png", bbox_inches='tight', dpi=300)
            plt.close()

        # Create RMSF plot for this replica (only if computed)
        if rmsf_data and not self.skip_rmsf:
            self._plot_rmsf_rep(rmsf_data, rep_analysis_dir, rep_num)

        # Create secondary structure plot for this replica (only if DSSP was computed)
        if not self.skip_dssp:
            self._plot_ss_rep(df, sim_time, rep_analysis_dir, rep_num)

        # Save replica data to CSV
        csv_file = f"{rep_analysis_dir}/analysis_results.csv"
        df.to_csv(csv_file, index=False)

        # Save RMSF data to CSV (only if computed)
        if rmsf_data and not self.skip_rmsf:
            rmsf_df = pd.DataFrame(rmsf_data)
            rmsf_csv_file = f"{rep_analysis_dir}/rmsf.csv"
            rmsf_df.to_csv(rmsf_csv_file, index=False)
