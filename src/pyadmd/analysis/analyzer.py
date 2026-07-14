"""Post-hoc structural analysis (RMSD, RoG, SASA, hydrophobic exposure, RMSF, DSSP)."""

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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import MDAnalysis as mda
from MDAnalysis.analysis import align
from Bio.PDB import PDBParser, ShrakeRupley

from pyadmd.console import ConsoleConfig
from pyadmd.analysis.completion import check_pyadmd_completion
from pyadmd.freeenergy.completion import check_freeenergy_completion


class Analyzer:
    """
    Analyzes simulation results and generates plots.

    This class handles computation and visualization of various structural
    properties from simulation trajectories including RMSD, radius of gyration,
    SASA, hydrophobic exposure, secondary structure, and RMSF.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output
        param_file (str): Path to parameter JSON file
        rough (bool): If True, analyze every 5ps instead of every frame
        source (str): Trajectory source: 'pyadmd' (rep{N}.dcd replicas) or
            'freeenergy' (centroid production trajectories).
        unit_col (str): Column/key name used to identify an analysis unit
            ('replica' for pyadmd, 'centroid_frame' for freeenergy).
        unit_label (str): Human-readable label for an analysis unit
            ('Replica' for pyadmd, 'Centroid frame' for freeenergy).
    """
    def __init__(self, console: ConsoleConfig, param_file: str = "pyAdMD_params.json", rough: bool = False,
                 no_rmsd: bool = False, no_rg: bool = False, no_sasa: bool = False,
                 no_hp: bool = False, no_rmsf: bool = False, no_dssp: bool = False,
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
            no_hp (bool): If True, skip hydrophobic exposure calculation
            no_rmsf (bool): If True, skip RMSF calculation
            no_dssp (bool): If True, skip secondary structure (DSSP) calculation
            source (str): Trajectory source to analyze: 'pyadmd' (default) for
                rep{N}.dcd replica trajectories, or 'freeenergy' for centroid
                production trajectories from a completed 'freeenergy' run.
        """
        self.console = console
        self.param_file = param_file
        self.rough = rough
        self.skip_rmsd = no_rmsd
        self.skip_rg = no_rg
        self.skip_sasa = no_sasa
        self.skip_hp = no_hp
        self.skip_rmsf = no_rmsf
        self.skip_dssp = no_dssp
        self.source = source
        self.params = self._load_parameters()

        # Analysis unit terminology and output directory depend on source.
        # NOTE (Phase 1): only the 'pyadmd' data path is wired up so far;
        # 'freeenergy' plumbing (analysis_dir/unit_col/unit_label) is set
        # here but the actual centroid-trajectory analysis path is not yet
        # implemented (see analyze_all_centroids, added in a later phase).
        if self.source == "freeenergy":
            self.analysis_dir = os.path.join("analysis", "freeenergy")
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

    def analyze_all_replicas(self) -> None:
        """
        Analyze all replicas and generate plots.

        This method processes all replica directories, computes structural
        properties, generates visualizations, and creates summary reports.

        Before any analysis runs, every replica is verified to have reached
        its target cycle count (see ``check_pyadmd_completion``). If any
        replica is incomplete, this method aborts (``sys.exit(1)``) with a
        message listing the incomplete replicas, instead of silently
        analyzing a partial trajectory against a time axis scaled for the
        full target length.

        Raises:
            SystemExit: If one or more replicas have not reached their
                target cycle count, or if completion cannot be verified
                (e.g. ``end_loop`` missing from ``pyAdMD_params.json``).
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

        if incomplete:
            print(f"{self.console.PGM_ERR}Cannot analyze: "
                  f"{self.console.ERR}{len(incomplete)}{self.console.STD} "
                  f"{self.unit_label.lower()}(s) have not reached the target "
                  "cycle count:")
            for rep, last_cycle, end_loop in incomplete:
                print(f"{self.console.PGM_ERR}  {self.unit_label} {rep}: "
                      f"{self.console.ERR}{last_cycle}/{end_loop}"
                      f"{self.console.STD} cycles completed")
            print(f"{self.console.PGM_ERR}Run 'pyadmd restart' or "
                  "'pyadmd append' to complete them first, then re-run "
                  "'analyze'.")
            sys.exit(1)

        cwd = self.params.get('cwd', os.getcwd())
        args = self.params['args']
        replicas = args.get('replicas', 10)
        sim_time = args.get('time', 250)  # Total simulation time in ps

        all_data = []
        all_rmsf_data = []  # Store RMSF data per residue (only if not skipped)

        # Log skipped analyses
        skipped = []
        if self.skip_rmsd:   skipped.append("RMSD")
        if self.skip_rg:     skipped.append("Radius of Gyration")
        if self.skip_sasa:   skipped.append("SASA")
        if self.skip_hp:     skipped.append("Hydrophobic Exposure")
        if self.skip_rmsf:   skipped.append("RMSF")
        if self.skip_dssp:   skipped.append("Secondary Structure (DSSP)")
        if skipped:
            print(f"{self.console.PGM_WRN}Skipping analyses: {self.console.WRN}{', '.join(skipped)}{self.console.STD}\n")

        # Prepare arguments for parallel processing
        replica_args = []
        replica_dirs = []
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                print(f"{self.console.PGM_WRN}{self.console.WRN}{self.unit_label} {rep}{self.console.STD} directory not found, skipping.")
                continue

            # Create replica-specific analysis directory
            rep_analysis_dir = f"{self.analysis_dir}/rep{rep}"
            os.makedirs(rep_analysis_dir, exist_ok=True)

            replica_args.append((rep_dir, rep, sim_time, rep_analysis_dir))
            replica_dirs.append(rep_dir)

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
                        print(f"{self.console.PGM_NAM}Using rough analysis: analyzing every {self.console.EXT}{frame_step}{self.console.STD}."
                              f" frames ({frame_step * (sim_time/n_frames):.1f} ps)\n")
                except:
                    pass

        # Process replicas in parallel using CPU cores
        if replica_args:
            # Use multiprocessing for CPU-bound tasks
            with mp.Pool(processes=min(self.num_cores, len(replica_args))) as pool:
                # Create a progress tracking function
                completed = 0
                def update_progress(result: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]) -> None:
                    nonlocal completed
                    completed += 1
                    rep_num, _, _ = result  # Unpack the result tuple
                    print(f"{self.console.PGM_NAM}Completed analysis of {self.console.EXT}{self.unit_label} {rep_num}{self.console.STD}"
                          f" [{self.console.EXT}{completed}{self.console.STD}/{self.console.WRN}{len(replica_args)}{self.console.STD}].")

                results = []

                # Submit all tasks
                for args in replica_args:
                    res = pool.apply_async(self._analyze_replica_parallel, args, callback=update_progress)
                    results.append(res)

                # Wait for all results
                for res in results:
                    rep_num, rep_data, rep_rmsf_data = res.get()  # Unpack all three values
                    if rep_data:
                        all_data.extend(rep_data)
                    if rep_rmsf_data:
                        all_rmsf_data.extend(rep_rmsf_data)

        else:
            print(f"{self.console.PGM_WRN}No replicas found for analysis.")

        if not all_data:
            print(f"{self.console.PGM_WRN}No analysis data was generated.")
            return

        # Save RMSF data to separate CSV (only if RMSF was computed)
        if not self.skip_rmsf and all_rmsf_data:
            csv_file = f"{self.analysis_dir}/rmsf.csv"
            self._save_to_csv(all_rmsf_data, csv_file)
            print(f"\n{self.console.PGM_NAM}Average RMSF results saved to {self.console.EXT}{csv_file}{self.console.STD}.")

        # Save all data to CSV
        csv_file = f"{self.analysis_dir}/analysis_results.csv"
        self._save_to_csv(all_data, csv_file)
        print(f"{self.console.PGM_NAM}Analysis results saved to {self.console.EXT}{csv_file}{self.console.STD}.")

        # Generate plots
        self._generate_plots(all_data, sim_time)

        # Generate RMSF plots (only if RMSF was computed)
        if not self.skip_rmsf and all_rmsf_data:
            self._generate_rmsf_avg_plot(all_rmsf_data)

        # Generate HTML summary
        self._generate_html_summary(all_data, sim_time)

        print(f"\n{self.console.PGM_NAM}Analysis complete in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")
        print(f"{self.console.PGM_NAM}Results saved into {self.console.EXT}{self.analysis_dir}{self.console.STD} folder.")

    def analyze_all_centroids(self) -> None:
        """
        Analyze all freeenergy centroid production trajectories and generate plots.

        Mirrors ``analyze_all_replicas``, but the analysis units are
        freeenergy centroids (``freeenergy/centroids/centroid_frame{F}/prod.dcd``)
        instead of pyadmd replicas, and the shared time axis is the target
        ``production_ps`` value already recorded in
        ``freeenergy/run_metadata.json`` (playing the role
        ``pyAdMD_params.json``'s ``time`` plays for ``analyze_all_replicas``).

        Before any analysis runs, every centroid is verified to have
        reached its target production cycle count (see
        ``check_freeenergy_completion``). If any centroid is incomplete,
        this method aborts (``sys.exit(1)``) with a message listing the
        incomplete centroids.

        Raises:
            SystemExit: If one or more centroids have not reached their
                target production cycle count, if
                ``freeenergy/clustering_summary.csv`` or
                ``freeenergy/run_metadata.json`` cannot be found/read, or
                if the shared PSF file cannot be located.
        """
        t0 = time.time()

        if self.params is None:
            return

        cwd = self.params.get('cwd', os.getcwd())

        try:
            incomplete = check_freeenergy_completion(cwd)
        except FileNotFoundError as exc:
            print(f"{self.console.PGM_ERR}Cannot verify centroid completion: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            sys.exit(1)

        if incomplete:
            print(f"{self.console.PGM_ERR}Cannot analyze: "
                  f"{self.console.ERR}{len(incomplete)}{self.console.STD} "
                  f"{self.unit_label.lower()}(s) have not reached the target "
                  "production cycle count:")
            for frame, done, target in incomplete:
                print(f"{self.console.PGM_ERR}  {self.unit_label} {frame}: "
                      f"{self.console.ERR}{done}/{target}"
                      f"{self.console.STD} cycles completed")
            print(f"{self.console.PGM_ERR}Re-run 'pyadmd freeenergy' to "
                  "complete production for these centroids first, then "
                  "re-run 'analyze'.")
            sys.exit(1)

        # PSF path is shared across all centroids (same file used for the whole run)
        psf_file = f"{cwd}/inputs/{self.params['args']['psffile'].split('/')[-1]}"
        if not os.path.exists(psf_file):
            print(f"{self.console.PGM_ERR}PSF file not found: "
                  f"{self.console.ERR}{psf_file}{self.console.STD}.")
            return

        # Shared time axis: production_ps from freeenergy/run_metadata.json
        # (plays the role args['time'] plays for analyze_all_replicas)
        run_metadata_path = f"{cwd}/freeenergy/run_metadata.json"
        try:
            with open(run_metadata_path) as fh:
                run_metadata = json.load(fh)
            sim_time = run_metadata['production_ps']
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            print(f"{self.console.PGM_ERR}Could not read 'production_ps' from "
                  f"{self.console.ERR}{run_metadata_path}{self.console.STD}: "
                  f"{self.console.ERR}{exc}{self.console.STD}.")
            sys.exit(1)

        all_data = []
        all_rmsf_data = []  # Store RMSF data per residue (only if not skipped)

        # Log skipped analyses
        skipped = []
        if self.skip_rmsd:   skipped.append("RMSD")
        if self.skip_rg:     skipped.append("Radius of Gyration")
        if self.skip_sasa:   skipped.append("SASA")
        if self.skip_hp:     skipped.append("Hydrophobic Exposure")
        if self.skip_rmsf:   skipped.append("RMSF")
        if self.skip_dssp:   skipped.append("Secondary Structure (DSSP)")
        if skipped:
            print(f"{self.console.PGM_WRN}Skipping analyses: {self.console.WRN}{', '.join(skipped)}{self.console.STD}\n")

        # Discover centroid production trajectories, sorted by frame index
        centroid_pattern = re.compile(r"centroid_frame(\d+)")
        centroid_dcds = sorted(
            glob.glob(f"{cwd}/freeenergy/centroids/centroid_frame*/prod.dcd"),
            key=lambda p: int(centroid_pattern.search(p).group(1))
        )

        # Prepare arguments for parallel processing
        centroid_args = []
        for dcd_path in centroid_dcds:
            frame_idx = int(centroid_pattern.search(dcd_path).group(1))

            out_dir = f"{self.analysis_dir}/centroid_frame{frame_idx}"
            os.makedirs(out_dir, exist_ok=True)

            centroid_args.append((dcd_path, psf_file, frame_idx, sim_time, out_dir))

        # Print analysis settings once
        if centroid_args:
            print(f"{self.console.PGM_NAM}Analyzing {self.console.EXT}{len(centroid_args)}{self.console.STD} centroids in parallel using CPU...\n")
        else:
            print(f"{self.console.PGM_WRN}No centroid production trajectories found for analysis.")

        # Process centroids in parallel using CPU cores
        if centroid_args:
            with mp.Pool(processes=min(self.num_cores, len(centroid_args))) as pool:
                # Create a progress tracking function
                completed = 0
                def update_progress(result: Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]) -> None:
                    nonlocal completed
                    completed += 1
                    frame_idx, _, _ = result  # Unpack the result tuple
                    print(f"{self.console.PGM_NAM}Completed analysis of {self.console.EXT}{self.unit_label} {frame_idx}{self.console.STD}"
                          f" [{self.console.EXT}{completed}{self.console.STD}/{self.console.WRN}{len(centroid_args)}{self.console.STD}].")

                results = []

                # Submit all tasks
                for c_args in centroid_args:
                    res = pool.apply_async(self._analyze_centroid_parallel, c_args, callback=update_progress)
                    results.append(res)

                # Wait for all results
                for res in results:
                    frame_idx, frame_data, frame_rmsf_data = res.get()  # Unpack all three values
                    if frame_data:
                        all_data.extend(frame_data)
                    if frame_rmsf_data:
                        all_rmsf_data.extend(frame_rmsf_data)

        if not all_data:
            print(f"{self.console.PGM_WRN}No analysis data was generated.")
            return

        # Save RMSF data to separate CSV (only if RMSF was computed)
        if not self.skip_rmsf and all_rmsf_data:
            csv_file = f"{self.analysis_dir}/rmsf.csv"
            self._save_to_csv(all_rmsf_data, csv_file)
            print(f"\n{self.console.PGM_NAM}Average RMSF results saved to {self.console.EXT}{csv_file}{self.console.STD}.")

        # Save all data to CSV
        csv_file = f"{self.analysis_dir}/analysis_results.csv"
        self._save_to_csv(all_data, csv_file)
        print(f"{self.console.PGM_NAM}Analysis results saved to {self.console.EXT}{csv_file}{self.console.STD}.")

        # Generate plots
        self._generate_plots(all_data, sim_time)

        # Generate RMSF plots (only if RMSF was computed)
        if not self.skip_rmsf and all_rmsf_data:
            self._generate_rmsf_avg_plot(all_rmsf_data)

        # Generate HTML summary
        self._generate_html_summary(all_data, sim_time)

        print(f"\n{self.console.PGM_NAM}Analysis complete in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")
        print(f"{self.console.PGM_NAM}Results saved into {self.console.EXT}{self.analysis_dir}{self.console.STD} folder.")

    def _analyze_centroid_parallel(self, dcd_file: str, psf_file: str, frame_idx: int,
                                   sim_time: float, out_dir: str) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Thin parallel wrapper that calls ``_analyze_trajectory`` for a single
        freeenergy centroid and prepends the centroid frame index.

        Designed for use with multiprocessing.Pool.apply_async, mirroring
        ``_analyze_replica_parallel``.

        Args:
            dcd_file (str): Absolute path to the centroid's prod.dcd file.
            psf_file (str): Absolute path to the shared PSF topology file.
            frame_idx (int): Centroid frame index, echoed in the return value.
            sim_time (float): Shared production time in picoseconds
                (``production_ps`` from ``freeenergy/run_metadata.json``).
            out_dir (str): Output directory for this centroid's analysis files.

        Returns:
            tuple: A 3-element tuple of (frame_idx, data, rmsf_data), same
                shape as ``_analyze_replica_parallel``'s return value.
        """
        try:
            print(f"{self.console.PGM_NAM}Starting analysis of {self.console.EXT}{self.unit_label} {frame_idx}{self.console.STD}...")
            data, rmsf_data = self._analyze_trajectory(psf_file, dcd_file, frame_idx, sim_time, out_dir)
            return (frame_idx, data, rmsf_data)
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {frame_idx}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            return (frame_idx, [], [])

    def _analyze_replica_parallel(self, rep_dir: str, rep_num: int, sim_time: int, rep_analysis_dir: str) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Thin parallel wrapper that calls analyze_replica and prepends the replica number.

        Designed for use with multiprocessing.Pool.apply_async: the callback receives
        the returned tuple to track progress and collect results.

        Args:
            rep_dir (str): Absolute path to the replica directory.
            rep_num (int): Replica number identifier, echoed in the return value.
            sim_time (int): Total simulation time in picoseconds.
            rep_analysis_dir (str): Output directory for replica-specific analysis files.

        Returns:
            tuple: A 3-element tuple of (rep_num, data, rmsf_data) where:
                - rep_num (int): Replica identifier passed through for pool callback tracking.
                - data (list[dict]): Per-frame structural property dictionaries; empty on failure.
                - rmsf_data (list[dict]): Per-residue RMSF dictionaries; empty on failure.
        """
        try:
            print(f"{self.console.PGM_NAM}Starting analysis of {self.console.EXT}{self.unit_label} {rep_num}{self.console.STD}...")
            result = self.analyze_replica(rep_dir, rep_num, sim_time, rep_analysis_dir)
            return (rep_num, result[0], result[1])  # Return rep_num along with data
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {rep_num}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            return (rep_num, [], [])  # Return rep_num even on error

    def analyze_replica(self, rep_dir: str, rep_num: int, sim_time: int, rep_analysis_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Resolve the DCD/PSF paths for a single pyAdMD replica, then delegate
        the actual per-frame computation to ``_analyze_trajectory``.

        Args:
            rep_dir (str): Absolute path to the replica directory containing the rep{N}.dcd file.
            rep_num (int): Replica number identifier used in output labels and filenames.
            sim_time (int): Total simulation time in picoseconds, used to build the time axis.
            rep_analysis_dir (str): Output directory for replica-specific plots and CSV files.

        Returns:
            tuple: A 2-element tuple of (data, rmsf_data) where:
                - data (list[dict]): One dictionary per analyzed frame with keys:
                  replica, time, rmsd, radius_gyration, sasa, hydrophobic_exposure,
                  helix, sheet, coil, turn, other.
                - rmsf_data (list[dict]): One dictionary per Cα atom with keys:
                  replica, residue_index, residue_name, rmsf.
                Both lists are empty if the PSF or DCD file is not found.
        """
        # Find the DCD trajectory file for this replica
        dcd_files = sorted(glob.glob(f"{rep_dir}/rep{rep_num}.dcd"))
        if not dcd_files:
            # Fallback: accept any rep*.dcd present in the directory
            dcd_files = sorted(glob.glob(f"{rep_dir}/rep*.dcd"))
        if not dcd_files:
            print(f"{self.console.PGM_WRN}No DCD trajectory found for {self.console.WRN}{self.unit_label.lower()} {rep_num}{self.console.STD}.")
            return [], []

        dcd_file = dcd_files[0]

        # Load PSF file
        psf_file = f"{rep_dir}/../inputs/{self.params['args']['psffile'].split('/')[-1]}"
        if not os.path.exists(psf_file):
            print(f"{self.console.PGM_ERR}PSF file not found for {self.unit_label.lower()} {self.console.ERR}{rep_num}{self.console.STD}.")
            return [], []

        return self._analyze_trajectory(psf_file, dcd_file, rep_num, sim_time, rep_analysis_dir)

    def _analyze_trajectory(self, psf_file: str, dcd_file: str, unit_id: int,
                            sim_time: float, out_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Analyze every frame of a single trajectory (source-agnostic core).

        Loads the given DCD trajectory into an MDAnalysis Universe using the
        given PSF as topology, then computes per-frame RMSD, radius of
        gyration, SASA, hydrophobic exposure, and secondary structure
        content, and calculates per-residue RMSF. Plots and CSV files are
        written to ``out_dir``. This method contains the computation logic
        shared by both the pyadmd (``analyze_replica``) and freeenergy
        (``analyze_all_centroids``) source paths -- only trajectory/PSF
        path resolution and output-directory naming differ between them.

        Args:
            psf_file (str): Absolute path to the PSF topology file.
            dcd_file (str): Absolute path to the DCD trajectory file.
            unit_id (int): Analysis unit identifier (replica number, or
                centroid frame index), used in output labels/filenames.
            sim_time (float): Total simulation time in picoseconds, used to
                build the time axis.
            out_dir (str): Output directory for unit-specific plots and CSV
                files.

        Returns:
            tuple: A 2-element tuple of (data, rmsf_data), same shape as
                ``analyze_replica``'s return value. Both lists are empty on
                failure.
        """
        try:
            # Create universe from PSF topology + DCD trajectory
            u = mda.Universe(psf_file, dcd_file, format="DCD")

            # Get the actual number of frames
            n_frames = len(u.trajectory)

            # Calculate time points (assuming each step is 0.2 ps)
            time_points = np.linspace(0, sim_time, n_frames)

            # Determine frame step for rough analysis
            frame_step = 1
            if self.rough:
                # Calculate step to get approximately 5ps intervals
                frame_step = max(1, int(5 / (sim_time / n_frames)))

            # Store reference positions from first frame
            u.trajectory[0]

            # Get consistent atom selection for RMSD and Rg calculations
            try:
                # Try to select protein atoms
                selection = u.select_atoms("protein")
                if len(selection) == 0:
                    # If no protein, use all atoms
                    selection = u.select_atoms("all")
            except:
                # Fallback to all atoms
                selection = u.select_atoms("all")

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
                    frame_data['sasa'] = self._calc_sasa(u, unit_id, i)
                if not self.skip_hp:
                    frame_data['hydrophobic_exposure'] = self._calculate_hp(u)

                # If not skipped, calculate secondary structure for selected frames only)
                if not self.skip_dssp:
                    if not self.rough or i % (frame_step * 5) == 0:  # Less frequent for SS to save time
                        ss_data = self._calc_ss(u, unit_id, i)
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

            # If not skipped, calculate RMSF per residue (Cα atoms)
            rmsf_data = []
            if not self.skip_rmsf:
                rmsf_data = self._calc_rmsf(u, unit_id)

            # Generate unit-specific plots
            self._generate_replica_plots(data, rmsf_data, sim_time, out_dir, unit_id)

            return data, rmsf_data
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}{self.unit_label.lower()} {unit_id}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}.")
            traceback.print_exc()
            return [], []

    def _process_frame(self, u: mda.Universe, selection: mda.AtomGroup, ref_positions: np.ndarray,
                       frame_idx: int, time_val: float, rep_num: int, frame_step: int) -> Dict[str, Any]:
        """
        Process a single frame and compute all properties.

        Args:
            u (mda.Universe): MDAnalysis Universe object
            selection (mda.AtomGroup): Atom selection for analysis
            ref_positions (numpy.ndarray): Reference positions for RMSD
            frame_idx (int): Frame index
            time_val (float): Time value for this frame
            rep_num (int): Replica number
            frame_step (int): Frame step for rough analysis

        Returns:
            dict: Frame data dictionary
        """
        u.trajectory[frame_idx]

        frame_data = {
            self.unit_col: rep_num,
            'time': time_val,
        }

        if not self.skip_rmsd:
            frame_data['rmsd'] = self._calc_rmsd(selection, ref_positions)
        if not self.skip_rg:
            frame_data['radius_gyration'] = self._calc_rog(selection)
        if not self.skip_sasa:
            frame_data['sasa'] = self._calc_sasa(u, rep_num, frame_idx)
        if not self.skip_hp:
            frame_data['hydrophobic_exposure'] = self._calculate_hp(u)

        # Calculate secondary structure for selected frames only (if not skipped)
        if not self.skip_dssp:
            if not self.rough or frame_idx % (frame_step * 5) == 0:  # Less frequent for SS to save time
                ss_data = self._calc_ss(u, rep_num, frame_idx)
                frame_data.update(ss_data)
            else:
                # Default values if no SS calculation
                frame_data.update({'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0})

        return frame_data

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
            print(f"{self.console.PGM_WRN}Could not calculate RMSD: {e}")
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
            print(f"{self.console.PGM_WRN}Could not calculate radius of gyration: {e}")
            return 0

    def _calc_sasa(self, u: mda.Universe, rep_num: int, frame_idx: int) -> float:
        """
        Calculates solvent accessible surface area using Bio.PDB.SASA.

        Args:
            u (mda.Universe): MDAnalysis Universe object
            rep_num (int): Replica number identifier
            frame_idx (int): Frame index number

        Returns:
            float: Total SASA value in square Angstroms
        """
        try:
            # Write temporary PDB file for this frame
            temp_pdb = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
            u.atoms.write(temp_pdb.name)
            temp_pdb.close()

            # Parse with Biopython
            parser = PDBParser()
            structure = parser.get_structure('temp', temp_pdb.name)

            # Calculate SASA using Shrake-Rupley algorithm
            sasa_calculator = ShrakeRupley()
            sasa_calculator.compute(structure, level="S")

            # Get total SASA
            total_sasa = 0
            for atom in structure.get_atoms():
                total_sasa += atom.sasa

            # Clean up
            os.unlink(temp_pdb.name)
            return total_sasa
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate SASA: {e}")
            # Fallback to simple estimation when Biopython SASA fails.
            # Using ~15 Å² per atom as a rough per-atom SASA heuristic.
            try:
                selection = u.select_atoms("protein")
                if len(selection) == 0:
                    selection = u.select_atoms("all")
                return len(selection) * 15  # Approximate 15 Å² per atom
            except:
                return 0

    def _calculate_hp(self, universe: mda.Universe) -> float:
        """
        Calculates percentage of hydrophobic residues in the protein.

        Args:
            universe (mda.Universe): MDAnalysis Universe object

        Returns:
            float: Percentage of hydrophobic residues
        """
        try:
            # Define hydrophobic residues
            hydrophobic_residues = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO']

            # Select hydrophobic atoms
            hydrophobic_sel = f"protein and (resname {' '.join(hydrophobic_residues)})"
            hydrophobic_atoms = universe.select_atoms(hydrophobic_sel)

            # Simple metric: ratio of hydrophobic atoms to total protein atoms
            total_protein_atoms = len(universe.select_atoms("protein"))
            exposure_ratio = len(hydrophobic_atoms) / total_protein_atoms if total_protein_atoms > 0 else 0
            return exposure_ratio * 100  # Return as percentage
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate hydrophobic exposure: {e}")
            return 0

    def _calc_rmsf(self, universe: mda.Universe, rep_num: int) -> List[Dict[str, Any]]:
        """
        Calculate per-residue RMSF for Cα atoms over the full trajectory.

        Selects Cα atoms, aligns each frame to the first-frame reference via a
        rotation matrix, accumulates squared deviations, and returns the root-mean
        value per residue.

        Args:
            universe (mda.Universe): MDAnalysis Universe object containing the trajectory.
            rep_num (int): Replica number identifier, stored in the returned records.

        Returns:
            list[dict]: One dictionary per Cα atom with keys:
                - replica (int): Replica number.
                - residue_index (int): Residue sequence number (resid).
                - residue_name (str): Three-letter residue name.
                - rmsf (float): Root-mean-square fluctuation in Å.
            Returns an empty list if no Cα atoms are found or an error occurs.
        """
        try:
            # Select Cα atoms
            calphas = universe.select_atoms("protein and name CA")
            if len(calphas) == 0:
                print(f"{self.console.PGM_WRN}No Cα atoms found for RMSF calculation.")
                return []

            # Store the first-frame positions as the alignment reference
            ref_coords = calphas.positions.copy()
            rmsf_values = np.zeros(len(calphas))

            for ts in universe.trajectory:
                # Align each frame to the reference and accumulate squared deviations
                mobile_coords = calphas.positions
                R, rmsd = align.rotation_matrix(mobile_coords, ref_coords)
                calphas.positions = np.dot(mobile_coords, R.T)
                rmsf_values += np.sum((calphas.positions - ref_coords) ** 2, axis=1)

            # sqrt of mean squared deviation over all frames
            rmsf_values = np.sqrt(rmsf_values / len(universe.trajectory))

            rmsf_data = []
            for i, atom in enumerate(calphas):
                rmsf_data.append({
                    self.unit_col: rep_num,
                    'residue_index': atom.residue.resid,
                    'residue_name': atom.residue.resname,
                    'rmsf': rmsf_values[i]
                })

            return rmsf_data

        except Exception as e:
            print(f"{self.console.PGM_ERR}Error calculating RMSF: {e}")
            return []

    def _calc_ss(self, u: mda.Universe, rep_num: int, frame_idx: int) -> Dict[str, int]:
        """
        Calculates secondary structure content using DSSP.

        Args:
            u (mda.Universe): MDAnalysis Universe object
            rep_num (int): Replica number identifier
            frame_idx (int): Frame index number

        Returns:
            dict: Dictionary with secondary structure counts
        """
        try:
            # Select only protein atoms
            protein = u.select_atoms("protein")
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
                cmd = f"dssp {pdb_path} {dssp_path}"
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
            print(f"{self.console.PGM_WRN}Could not calculate secondary structure: {e}")
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
            print(f"{self.console.PGM_WRN}Could not parse DSSP output: {e}")
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
            print(f"{self.console.PGM_WRN}Could not create secondary structure plot for {self.unit_label.lower()} {rep_num}: {e}")

    def _save_to_csv(self, data: List[Dict[str, Any]], csv_file: str) -> None:
        """
        Saves analysis data to CSV file.

        Args:
            data (list): List of data dictionaries to save
            csv_file (str): Output CSV file path
        """
        if not data:
            print(f"{self.console.PGM_WRN}No data to save to CSV.")
            return

        # Convert to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

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
            print(f"{self.console.PGM_WRN}Could not create RMSF plot for {self.unit_label.lower()} {rep_num}: {e}")

    def _generate_html_summary(self, data: List[Dict[str, Any]], sim_time: int) -> None:
        """
        Generates an HTML summary of the analysis results.
        Only includes sections for analyses that were actually computed
        (i.e. not disabled via skip flags).

        Args:
            data (list): List of analysis data dictionaries
            sim_time (int): Total simulation time in picoseconds
        """
        if not data:
            print(f"{self.console.PGM_WRN}No data to generate HTML summary.")
            return

        df = pd.DataFrame(data)

        # Build the list of (column, label, is_max) tuples for computed stats only
        stat_specs = []
        if not self.skip_rmsd:
            stat_specs.append(('rmsd', 'Max RMSD (Å)', True))
        if not self.skip_rg:
            stat_specs.append(('radius_gyration', 'Final Radius of Gyration (Å)', False))
        if not self.skip_sasa:
            stat_specs.append(('sasa', 'Final SASA (Å²)', False))
        if not self.skip_hp:
            stat_specs.append(('hydrophobic_exposure', 'Max Hydrophobic Exposure (%)', True))
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
        notes_html = "\n                    ".join(notes_items)

        # Source note: which trajectories this summary was generated from
        n_units = df[self.unit_col].nunique()
        if self.source == 'freeenergy':
            source_note = (f"freeenergy centroid production trajectories "
                          f"({n_units} centroids, {sim_time} ps production each)")
        else:
            source_note = f"pyAdMD replica runs ({n_units} replicas, {sim_time} ps each)"

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
                unit_label=self.unit_label,
                unit_label_plural=f"{self.unit_label}s",
                replica_tables=self._html_rep_tables(summary_data),
                avg_table_rows=self._html_summary_avg_table(avg_summary),
                plot_items=self._html_summary_plots(),
                notes_html=notes_html
            ))

        print(f"{self.console.PGM_NAM}HTML summary saved to {self.console.EXT}{html_file}{self.console.STD}")

    def _generate_plots(self, data: List[Dict[str, Any]], sim_time: int) -> None:
        """
        Generates plots from analysis data.

        Args:
            data (list): List of analysis data dictionaries
            sim_time (int): Total simulation time in picoseconds
        """
        if not data:
            print(f"{self.console.PGM_WRN}No data to generate plots.")
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
        if not self.skip_hp:
            properties.append(('hydrophobic_exposure', 'Hydrophobic Exposure (%)', 'Hydrophobic Exposure'))

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

    def _generate_rmsf_avg_plot(self, rmsf_data: List[Dict[str, Any]]) -> None:
        """
        Generates average RMSF plot across all replicas.

        Args:
            rmsf_data (list): List of RMSF data dictionaries
        """
        if not rmsf_data:
            print(f"{self.console.PGM_WRN}No RMSF data to generate plots.")
            return

        rmsf_df = pd.DataFrame(rmsf_data)

        # Create average RMSF plot across all replicas
        plt.figure(figsize=(10, 6))

        # Group by residue index and calculate average RMSF
        avg_rmsf = rmsf_df.groupby('residue_index')['rmsf'].mean().reset_index()

        plt.plot(avg_rmsf['residue_index'], avg_rmsf['rmsf'], 'b-', linewidth=2, label='Average')
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
            print(f"{self.console.PGM_WRN}Could not create average secondary structure plot: {e}")

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
            "rmsf_average.png", "secondary_structure_average.png"
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
        if not self.skip_hp:
            properties.append(('hydrophobic_exposure', 'Hydrophobic Exposure (%)', 'Hydrophobic Exposure'))

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
