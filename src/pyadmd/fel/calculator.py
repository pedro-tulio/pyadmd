"""Two-stage free energy landscape protocol (Costa et al. JCTC 2015)."""

import json
import math
import os
import sys
import time
import traceback
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp

import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors

import openmm as mm
from openmm import app, unit, XmlSerializer

from pyadmd.io.namd import NAMDInputReader
from pyadmd.io.openmm_restart import OpenMMRestartReader
from pyadmd.io.state import SystemState, load_reference_state
from pyadmd.io.dcd import _count_dcd_frames
from pyadmd.modes.projection import get_projection_setup, compute_mode_projections_df
from pyadmd.simulation.engine import OpenMMSimulationEngine
from pyadmd.simulation.system_builder import OpenMMSystemBuilder

# Get rid of MDAnalysis deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='MDAnalysis.coordinates.DCD')


def _merge_replica_dcds(console, cwd: str, replicas: int, psffile: str) -> mda.Universe:
    """
    Concatenate all replica DCD files into a single MDAnalysis Universe.

    Scans ``rep1`` through ``rep{replicas}`` for a ``rep{N}.dcd`` trajectory
    file, skipping any replica whose DCD is missing, and loads the found
    files as frames of a single merged pseudo-trajectory.

    Args:
        console (ConsoleConfig): Console configuration object for formatted output.
        cwd (str): Working directory containing the ``rep{N}/`` replica folders.
        replicas (int): Number of replicas to scan for.
        psffile (str): Path to the shared PSF topology file.

    Returns:
        MDAnalysis.Universe: Universe built from the PSF topology and the
            concatenated replica DCD files.

    Raises:
        FileNotFoundError: If no replica DCD files are found.
    """
    dcd_files = []
    for rep in range(1, replicas + 1):
        dcd = f"{cwd}/rep{rep}/rep{rep}.dcd"
        if os.path.exists(dcd):
            dcd_files.append(dcd)
        else:
            print(f"{console.PGM_WRN}DCD not found for {console.WRN} replica {rep}{console.STD}, skipping.")
    if not dcd_files:
        raise FileNotFoundError("No DCD trajectory files found in any replica directory.")
    print(f"{console.PGM_NAM}Merging {console.EXT}{len(dcd_files)}"
          f"{console.STD} replica DCD files...")
    u = mda.Universe(psffile, dcd_files, format="DCD")
    print(f"{console.PGM_NAM}Pseudo-trajectory: "
          f"{console.EXT}{len(u.trajectory)}{console.STD} frames total.")
    return u


def export_cluster(console, params: Dict[str, Any], args_fe) -> bool:
    """
    Export the member-frame list (and optionally per-member PDBs) for one
    already-computed GROMOS cluster, without running the FEL protocol.

    Args:
        console (ConsoleConfig): Console configuration object for formatted output.
        params (dict): Persisted simulation parameters loaded via
            ``ParameterStorage.load_parameters()`` (same as passed to
            ``FreeEnergyCalculator``).
        args_fe (argparse.Namespace): Parsed ``fel`` CLI arguments;
            uses ``export_cluster`` (int, the target centroid's global
            frame index) and ``dump_pdb`` (bool).

    Returns:
        bool: True on success, False if the cluster/cache could not be
            found or read (caller is expected to exit non-zero on False).
    """
    run_args = params['args']
    cwd      = params['cwd']
    replicas = getattr(run_args, 'replicas', 10)
    psffile  = f"{cwd}/inputs/{getattr(run_args, 'psffile', '').split('/')[-1]}"
    out_dir  = f"{cwd}/fel"
    cache_path = f"{out_dir}/clustering_clusters_cache.json"

    if not os.path.exists(cache_path):
        print(f"{console.PGM_ERR}{console.ERR}{cache_path}{console.STD} not found. "
              "Run 'pyadmd fel' at least once first.")
        return False

    try:
        with open(cache_path) as fh:
            cache = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"{console.PGM_ERR}Could not read {console.ERR}{cache_path}"
              f"{console.STD}: {console.ERR}{console.WRN}{exc}{console.STD}{console.STD}.")
        return False

    clusters = cache.get('clusters', [])
    target   = args_fe.export_cluster
    match    = next((c for c in clusters if c.get('centroid') == target), None)

    if match is None:
        available = sorted(c.get('centroid') for c in clusters)
        print(f"{console.PGM_ERR}No cluster with centroid frame "
              f"{console.ERR}{target}{console.STD} found. Available centroid "
              f"frames: {console.EXT}{available}{console.STD} (see also "
              f"{console.EXT}fel/clustering_summary.csv{console.STD}).")
        return False

    if 'members' not in match:
        print(f"{console.PGM_ERR}This clustering cache predates membership "
              "export support. Re-run 'pyadmd fel' once (with the "
              "same -s/-c/--max_centroids as before) to regenerate it, "
              "then retry --export-cluster.")
        return False

    members     = sorted(int(m) for m in match['members'])
    export_dir  = f"{out_dir}/exports/cluster_frame{target}"
    os.makedirs(export_dir, exist_ok=True)

    members_csv = f"{export_dir}/members.csv"
    df = pd.DataFrame({
        'member_frame': members,
        'is_centroid':  [m == target for m in members],
    })
    df.to_csv(members_csv, index=False)
    print(f"{console.PGM_NAM}Cluster centroid frame {console.EXT}{target}"
          f"{console.STD}: {console.EXT}{len(members)}{console.STD} member "
          f"frames written to {console.EXT}{members_csv}{console.STD}.")

    if getattr(args_fe, 'dump_pdb', False):
        merged_u = _merge_replica_dcds(console, cwd, replicas, psffile)
        pdbs_dir = f"{export_dir}/pdbs"
        os.makedirs(pdbs_dir, exist_ok=True)
        for m in members:
            merged_u.trajectory[m]
            merged_u.atoms.write(f"{pdbs_dir}/frame{m}.pdb", file_format="PDB")
        print(f"{console.PGM_NAM}{console.EXT}{len(members)}{console.STD} "
              f"PDB file(s) written to {console.EXT}{pdbs_dir}{console.STD}.")

    return True


class FreeEnergyCalculator:
    """
    Implements the two-stage free energy protocol.

    Protocol:
      1. Merge all replica DCD trajectories into a single pseudo-trajectory.
      2. GROMOS clustering on Cα RMSD → centroid structures.
      3. Short standard OpenMM MD per centroid: de-excitation phase (discarded,
         Langevin thermostat) followed by production phase (kept, Nosé-Hoover
         thermostat by default).
      4. Project each production frame onto individual original mode vectors (MRMS displacement).
      5. Compute the FEL via population histogram: 1D per mode and 2D for user-specified mode pairs.

    Reference:
        Costa et al., J. Chem. Theory Comput. 2015, 11, 2395-2408.
        DOI: 10.1021/acs.jctc.5b00003
    """

    def __init__(self, console, params, args_fe):
        """
        Initialize the FreeEnergyCalculator and build the shared OpenMM system.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
            params (dict): Persisted simulation parameters loaded via
                ParameterStorage.load_parameters().
            args_fe (argparse.Namespace): Command line arguments for the
                free-energy subcommand (cutoff, deexcite, production, bins,
                max_centroids, sel, modes, modes_2d, temp).
        """
        self.console = console
        self.params  = params
        self.args_fe = args_fe

        run_args = params['args']
        self.cwd       = params['cwd']
        self.input_dir = f"{self.cwd}/inputs"
        self.nm_parsed = params['nm_parsed']
        self.nm_type   = getattr(run_args, 'model', 'CA').lower()
        self.replicas  = getattr(run_args, 'replicas', 10)
        self._temperature = float(getattr(args_fe, 'temp', 303.15))
        self.n_steps      = 100   # steps per cycle, same as run phase

        # Derive input-engine type from saved parameters
        self._input_engine = getattr(run_args, 'source', 'NAMD').upper()

        self.psffile = f"{self.input_dir}/{getattr(run_args, 'psffile', '').split('/')[-1]}"

        if self._input_engine == 'NAMD':
            self.coorfile = f"{self.input_dir}/{getattr(run_args, 'coorfile', '').split('/')[-1]}"
            self.xscfile  = f"{self.input_dir}/{getattr(run_args, 'xscfile',  '').split('/')[-1]}"
            self.rstfile  = None
        else:
            # OPENMM mode: coorfile/xscfile are absent; use saved reference state.
            self.coorfile = None
            self.xscfile  = None
            rstfile_raw   = getattr(run_args, 'rstfile', None)
            self.rstfile  = (f"{self.input_dir}/{rstfile_raw.split('/')[-1]}"
                             if rstfile_raw else None)

        strfile_raw = getattr(run_args, 'strfile', None)
        self.strfile = (f"{self.input_dir}/{strfile_raw.split('/')[-1]}"
                        if strfile_raw else None)

        # Load saved reference positions and box
        try:
            self._ref_positions_ang, self._ref_box_nm, _ = load_reference_state(self.input_dir)
        except FileNotFoundError as exc:
            print(f"{console.PGM_WRN}Persisted reference state not found ({console.WRN}{exc}{console.STD}). "
                  "Falling back to engine-specific file read for reference positions.")
            self._ref_positions_ang = None
            self._ref_box_nm        = None

        self.cutoff          = float(getattr(args_fe, 'cutoff',       0.8))
        self.n_deexcite_ps   = int(getattr(args_fe, 'deexcite',      200))
        self.n_prod_ps       = int(getattr(args_fe, 'production',    800))
        self.bins            = int(getattr(args_fe, 'bins',           50))
        self.max_centroids   = int(getattr(args_fe, 'max_centroids',  50))
        self.cluster_sel_str = str(getattr(args_fe, 'sel', 'protein and name CA'))

        # Compare against the previous fel run (if any) and resolve
        # effective max_centroids/production_ps
        self._resolve_and_gate_parameters()

        self.n_deexcite_cycles = max(1, int(self.n_deexcite_ps / (self.n_steps * 0.002)))
        self.n_prod_cycles     = max(1, int(self.n_prod_ps     / (self.n_steps * 0.002)))

        modes_arg = getattr(args_fe, 'modes', None)
        self.fe_modes = ([int(x) for x in modes_arg.split(',')]
                         if modes_arg else list(self.nm_parsed))

        modes_2d_arg = getattr(args_fe, 'modes_2d', None)
        if modes_2d_arg:
            self.pairs_2d = []
            for pair_str in modes_2d_arg.split():
                parts = pair_str.split(',')
                if len(parts) == 2:
                    self.pairs_2d.append((int(parts[0]), int(parts[1])))
        else:
            self.pairs_2d = list(combinations(self.fe_modes, 2))

        self.out_dir = f"{self.cwd}/fel"
        os.makedirs(f"{self.out_dir}/centroids", exist_ok=True)

        # Build OpenMM system
        toppar_dir = os.path.join(self.input_dir, "charmm_toppar")
        str_box = None
        if self.strfile and os.path.exists(self.strfile):
            try:
                str_box = NAMDInputReader.parse_str_box(self.strfile)
            except Exception as exc:
                print(f"{console.PGM_WRN}Could not parse STR box ({console.WRN}{exc}{console.STD}); "
                      "falling back to saved box vectors.")
        if str_box is None and self._ref_box_nm is not None:
            try:
                str_box = OpenMMRestartReader.box_vectors_to_cell(
                    self._ref_box_nm[0], self._ref_box_nm[1], self._ref_box_nm[2]
                )
                print(f"{console.PGM_NAM}FreeEnergyCalculator: using saved "
                      "box vectors for PME system construction.")
            except Exception as exc:
                print(f"{console.PGM_WRN}Could not derive cell from saved box "
                      f"({console.WRN}{exc}{console.STD}); using placeholder.")
        builder = OpenMMSystemBuilder(console)
        self._psf_omm, self._omm_system, _ = builder.build(
            self.psffile, toppar_dir, temperature=self._temperature, str_box=str_box,
        )

    # Saving and reading parameters metadata

    def _run_metadata_path(self) -> str:
        """Path to the saved run-parameter record (may not exist yet)."""
        return f"{self.cwd}/fel/run_metadata.json"

    def _load_run_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load the previous run's parameters, if present.

        Returns:
            dict or None: Parsed ``run_metadata.json`` contents, or ``None``
                if the file is absent or unreadable (treated as a first run).
        """
        path = self._run_metadata_path()
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"{self.console.PGM_WRN}Could not read "
                  f"{self.console.WRN}run_metadata.json{self.console.STD} "
                  f"({self.console.WRN}{exc}{self.console.STD}); treating this as a first run.")
            return None

    def _save_run_metadata(self) -> None:
        """Save the effective parameters for this run."""
        os.makedirs(f"{self.cwd}/fel", exist_ok=True)
        meta = {
            'cluster_sel':   self.cluster_sel_str,
            'temperature':   self._temperature,
            'cutoff':        self.cutoff,
            'deexcite_ps':   self.n_deexcite_ps,
            'max_centroids': self.max_centroids,
            'production_ps': self.n_prod_ps,
            'timestamp':     time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self._run_metadata_path(), 'w') as fh:
            json.dump(meta, fh, indent=2)

    def _resolve_and_gate_parameters(self) -> None:
        """
        Compare this call's parameters against the previous ``fel``
        run (if any) and resolve the effective values.

        Raises:
            SystemExit: If ``-s/--sel`` or ``-T/--temp`` differ from the
                previous run.
        """
        prev = self._load_run_metadata()
        self._is_append_run = prev is not None   # used only for reporting
        if prev is None:
            self._save_run_metadata()
            return

        # Selection and temperature must match
        mismatches = []
        if prev.get('cluster_sel') != self.cluster_sel_str:
            mismatches.append(
                f"-s/--sel: previous='{prev.get('cluster_sel')}' "
                f"vs current='{self.cluster_sel_str}'"
            )
        prev_temp = prev.get('temperature')
        if prev_temp is None or not math.isclose(
                prev_temp, self._temperature, rel_tol=1e-9, abs_tol=1e-6):
            mismatches.append(
                f"-T/--temp: previous={prev_temp} K "
                f"vs current={self._temperature} K"
            )
        if mismatches:
            print(f"{self.console.PGM_ERR}The following parameter(s) differ "
                  f"from the previous run recorded in "
                  f"{self.console.ERR}fel/run_metadata.json"
                  f"{self.console.STD}:")
            for m in mismatches:
                print(f"{self.console.PGM_ERR}  {m}")
            print(f"{self.console.PGM_ERR}Mixing different selections or "
                  "temperatures inside one pooled free energy landscape is "
                  "not physically valid. Remove or rename the existing "
                  f"{self.console.ERR}fel/{self.console.STD} "
                  "directory to start a fresh calculation with the new "
                  "parameters.")
            sys.exit(1)

        # Never shrink already-completed work
        prev_max_centroids = int(prev.get('max_centroids', self.max_centroids))
        if self.max_centroids < prev_max_centroids:
            print(f"{self.console.PGM_WRN}--max_centroids="
                  f"{self.console.WRN}{self.max_centroids}{self.console.STD} is smaller than the previous run's "
                  f"{self.console.EXT}{prev_max_centroids}{self.console.STD}. Existing centroids are never "
                  f"discarded; using {self.console.EXT}{prev_max_centroids}"
                  f"{self.console.STD}.")
            self.max_centroids = prev_max_centroids

        prev_production_ps = int(prev.get('production_ps', self.n_prod_ps))
        if self.n_prod_ps < prev_production_ps:
            print(f"{self.console.PGM_WRN}-p/--production={self.console.WRN}{self.n_prod_ps}{self.console.STD} "
                  f"is smaller than the previous run's {self.console.EXT}{prev_production_ps}{self.console.STD} "
                  f"ps. Existing production trajectories are never "
                  f"truncated; using {self.console.EXT}{prev_production_ps}"
                  f"{self.console.STD} ps.")
            self.n_prod_ps = prev_production_ps

        # Informational only: no gating nor overriding
        prev_cutoff = prev.get('cutoff')
        if prev_cutoff is not None and not math.isclose(
                prev_cutoff, self.cutoff, rel_tol=1e-9, abs_tol=1e-9):
            print(f"{self.console.PGM_NAM}Note: -c/--cutoff changed from "
                  f"{self.console.WRN}{prev_cutoff}{self.console.STD} to "
                  f"{self.console.EXT}{self.cutoff}{self.console.STD} Å since the previous "
                  "run. The cached pairwise-RMSD matrix is still reused; "
                  "clusters are simply re-thresholded with the new cutoff.")
        prev_deexcite = prev.get('deexcite_ps')
        if prev_deexcite is not None and prev_deexcite != self.n_deexcite_ps:
            print(f"{self.console.PGM_NAM}Note: -d/--deexcite changed from "
                  f"{self.console.WRN}{prev_deexcite}{self.console.STD} to "
                  f"{self.console.EXT}{self.n_deexcite_ps}{self.console.STD} ps since the "
                  "previous run. This only affects newly-created centroids; "
                  "existing centroids keep their original de-excitation and "
                  "are simply extended in production.")

        self._save_run_metadata()

    # Step 1: merge trajectories

    def merge_trajectories(self):
        """
        Concatenate all replica DCD files into a single MDAnalysis Universe.

        Returns:

            MDAnalysis.Universe: Universe built from the PSF topology and the
                concatenated replica DCD files.

        Raises:
            FileNotFoundError: If no replica DCD files are found.
        """
        return _merge_replica_dcds(self.console, self.cwd, self.replicas, self.psffile)

    def _count_merged_frames_cheap(self) -> int:
        """
        Return the total merged-trajectory frame count without building an
        MDAnalysis Universe.

        Returns:
            int: Total frame count across all existing replica DCDs.
        """
        total = 0
        for rep in range(1, self.replicas + 1):
            dcd = f"{self.cwd}/rep{rep}/rep{rep}.dcd"
            if os.path.exists(dcd):
                total += _count_dcd_frames(dcd)
        return total

    # Step 2: GROMOS clustering

    def cluster_gromos(self, merged_u):
        """
        Cluster frames using the GROMOS algorithm on Cα RMSD.

        Args:
            merged_u (MDAnalysis.Universe): Merged pseudo-trajectory produced
                by ``merge_trajectories``.

        Returns:
            list[dict]: One dict per cluster, each containing:
                - centroid (int): Original frame index in merged_u.
                - size (int): Number of subsampled members.
                - _sampled_idx (int): Subsampled index (used by MaxMin selection).
                - members (list[int]): Global frame indices belonging to
                  this cluster, including the centroid.
        """
        rmsd_matrix, frame_indices = self._get_or_build_rmsd_matrix(merged_u)

        print(f"{self.console.PGM_NAM}Running GROMOS clustering "
              f"(cutoff = {self.console.EXT}{self.cutoff}{self.console.STD} Å)...")
        clusters = self._gromos_threshold(rmsd_matrix, frame_indices)

        n_clusters = len(clusters)
        print(f"{self.console.PGM_NAM}Found "
              f"{self.console.EXT}{n_clusters}{self.console.STD} clusters.")

        if n_clusters > self.max_centroids:
            print(f"{self.console.PGM_WRN}{n_clusters} clusters exceed "
                  f"{self.console.WRN}max_centroids={self.max_centroids}{self.console.STD}. Selecting "
                  f"{self.console.EXT}{self.max_centroids}{self.console.STD} "
                  "maximally diverse centroids via greedy farthest-point "
                  "(MaxMin) sampling...")
            clusters = self._select_diverse_centroids(
                clusters, rmsd_matrix, self.max_centroids
            )

        return clusters

    def _gromos_threshold(self, rmsd_matrix, frame_indices):
        """
        GROMOS neighbor-counting/greedy-pick clustering over an pairwise RMSD matrix.

        Args:
            rmsd_matrix (numpy.ndarray): (n_sampled, n_sampled) pairwise
                RMSD matrix in Å, as produced by
                ``_compute_rmsd_matrix_batched``.
            frame_indices (numpy.ndarray): (n_sampled,) original
                merged-trajectory frame index for each row/column of
                ``rmsd_matrix``.

        Returns:
            list[dict]: Raw (pre-MaxMin) cluster list sorted by size
                (largest first), one dict per cluster with keys
                ``centroid``, ``_sampled_idx``, ``size``, ``members`` (list
                of global merged-trajectory frame indices belonging to this
                cluster, including the centroid itself -- consumed by
                ``export_cluster`` via the persisted clusters cache).
        """
        n_sampled = len(frame_indices)

        # Boolean adjacency.
        adjacency = rmsd_matrix < self.cutoff   # (n_sampled, n_sampled) bool

        active          = np.ones(n_sampled, dtype=bool)
        neighbor_counts = adjacency.sum(axis=1).astype(np.int64)

        clusters    = []
        n_remaining = n_sampled
        while n_remaining > 0:
            # argmax restricted to the active pool
            masked_counts    = np.where(active, neighbor_counts, -1)
            centroid_sampled = int(np.argmax(masked_counts))
            centroid_global  = int(frame_indices[centroid_sampled])

            member_mask = active & adjacency[centroid_sampled]
            members     = np.flatnonzero(member_mask)

            clusters.append({
                'centroid':     centroid_global,
                '_sampled_idx': centroid_sampled,   # kept for MaxMin selection
                'size':         int(members.size),
                'members':      frame_indices[members].tolist(),   # global frame indices
            })

            # Decrement remaining points' neighbor counts by however many of
            # the just-removed members they were counting as neighbors
            neighbor_counts -= adjacency[:, members].sum(axis=1)
            active[members]  = False
            # deactivate the removed members.
            n_remaining     -= members.size

        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters

    def _get_or_build_rmsd_matrix(self, merged_u):
        """
        Return the pairwise RMSD matrix over subsampled frames, reusing a
        cached one when it is still valid, otherwise computing and caching
        a fresh one.

        Args:
            merged_u (MDAnalysis.Universe): Merged pseudo-trajectory produced
                by ``merge_trajectories``.

        Returns:
            rmsd_matrix (numpy.ndarray): (n_sampled, n_sampled) float32
                pairwise RMSD matrix in Å.
            frame_indices (numpy.ndarray): (n_sampled,) original
                merged-trajectory frame index for each row/column.
        """
        n_frames = len(merged_u.trajectory)

        cached = self._load_rmsd_cache(n_frames)
        if cached is not None:
            return cached

        sel      = merged_u.select_atoms(self.cluster_sel_str)
        n_sel    = sel.n_atoms

        # Compute number of sampled frames using the stride
        n_sampled = (n_frames + self._CLUSTER_STRIDE - 1) // self._CLUSTER_STRIDE

        print(f"{self.console.PGM_NAM}Accumulating positions: "
              f"{self.console.EXT}{n_sampled}{self.console.STD} frames "
              f"(every {self._CLUSTER_STRIDE} of "
              f"{self.console.EXT}{n_frames}{self.console.STD} total, "
              f"{self.console.EXT}{n_sel}{self.console.STD} atoms)...")
        positions = np.empty((n_sampled, n_sel, 3), dtype=np.float32)
        frame_indices = np.empty(n_sampled, dtype=np.int64)

        # Use slicing to iterate only over every _CLUSTER_STRIDE-th frame
        for i, ts in enumerate(merged_u.trajectory[::self._CLUSTER_STRIDE]):
            frame_indices[i] = ts.frame
            positions[i] = sel.positions.copy()

        print(f"{self.console.PGM_NAM}Computing pairwise RMSD matrix "
              f"({self.console.EXT}{n_sampled}{self.console.STD}"
              f"×{self.console.EXT}{n_sampled}{self.console.STD})...")
        rmsd_matrix = self._compute_rmsd_matrix_batched(positions)

        self._save_rmsd_cache(rmsd_matrix, frame_indices, n_frames)
        return rmsd_matrix, frame_indices

    def _load_rmsd_cache(self, n_merged_frames):
        """
        Load the cached pairwise RMSD matrix if it is valid for the current
        clustering selection and merged-trajectory frame count.

        Args:
            n_merged_frames (int): Current ``len(merged_u.trajectory)``,
                used to detect a changed set of replica DCDs.

        Returns:
            tuple or None: ``(rmsd_matrix, frame_indices)`` if the cache is
                present and valid, otherwise ``None`` (caller recomputes).
        """
        meta_path = f"{self.out_dir}/clustering_rmsd_cache.json"
        npz_path  = f"{self.out_dir}/clustering_rmsd_cache.npz"
        if not (os.path.exists(meta_path) and os.path.exists(npz_path)):
            return None

        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"{self.console.PGM_WRN}Could not read RMSD cache metadata "
                  f"({self.console.WRN}{exc}{self.console.STD}); recomputing.")
            return None

        if meta.get('cluster_sel') != self.cluster_sel_str:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix was built with a "
                  f"different clustering selection ('{self.console.WRN}{meta.get('cluster_sel')}{self.console.STD}' "
                  f"vs current '{self.console.EXT}{self.cluster_sel_str}{self.console.STD}'); recomputing.")
            return None
        if meta.get('n_merged_frames') != n_merged_frames:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix was built from "
                  f"{self.console.WRN}{meta.get('n_merged_frames')}{self.console.STD} merged frames, but "
                  f"{self.console.EXT}{n_merged_frames}{self.console.STD} are present now (replica DCDs changed); "
                  "recomputing.")
            return None
        if meta.get('stride') != self._CLUSTER_STRIDE:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix used a different "
                  f"subsampling stride ({self.console.WRN}{meta.get('stride')}{self.console.STD} vs "
                  f"{self.console.EXT}{self._CLUSTER_STRIDE}{self.console.STD}); recomputing.")
            return None

        try:
            data = np.load(npz_path)
            rmsd_matrix   = data['rmsd_matrix']
            frame_indices = data['frame_indices']
        except Exception as exc:
            print(f"{self.console.PGM_WRN}Could not load cached RMSD matrix "
                  f"({self.console.WRN}{exc}{self.console.STD}); recomputing.")
            return None

        print(f"{self.console.PGM_NAM}Reusing cached pairwise RMSD matrix "
              f"({self.console.EXT}{meta.get('n_sampled')}"
              f"{self.console.STD}×{self.console.EXT}{meta.get('n_sampled')}"
              f"{self.console.STD}, saved {meta.get('timestamp', '?')}); "
              "skipping recomputation.")
        return rmsd_matrix, frame_indices

    def _save_rmsd_cache(self, rmsd_matrix, frame_indices, n_merged_frames):
        """
        Save the pairwise RMSD matrix and its metadata.

        Args:
            rmsd_matrix (numpy.ndarray): (n_sampled, n_sampled) pairwise
                RMSD matrix in Å.
            frame_indices (numpy.ndarray): (n_sampled,) original
                merged-trajectory frame index for each row/column.
            n_merged_frames (int): ``len(merged_u.trajectory)`` at
                computation time, used for later cache-validity checks.
        """
        npz_path  = f"{self.out_dir}/clustering_rmsd_cache.npz"
        meta_path = f"{self.out_dir}/clustering_rmsd_cache.json"

        np.savez_compressed(
            npz_path,
            rmsd_matrix=rmsd_matrix.astype(np.float32),
            frame_indices=frame_indices.astype(np.int64),
        )
        meta = {
            'cluster_sel':     self.cluster_sel_str,
            'n_merged_frames': int(n_merged_frames),
            'stride':          self._CLUSTER_STRIDE,
            'n_sampled':       int(len(frame_indices)),
            'timestamp':       time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f"{self.console.PGM_NAM}Pairwise RMSD matrix cached to "
              f"{self.console.EXT}{npz_path}{self.console.STD} ")

    def _clusters_cache_path(self) -> str:
        """Path to the cached final-cluster-list sidecar file."""
        return f"{self.out_dir}/clustering_clusters_cache.json"

    def _load_clusters_cache(self, n_merged_frames: int) -> Optional[List[Dict[str, Any]]]:
        """
        Return the cached final cluster list.

        Args:
            n_merged_frames (int): Current total merged-trajectory frame
                count, as returned by ``_count_merged_frames_cheap()``.

        Returns:
            list[dict] or None: Cached
                ``[{'centroid': int, 'size': int, 'members': list[int]}, ...]``
                if present and valid, otherwise ``None`` (caller recomputes).
                Cache files written before ``members`` was added will yield
                dicts without that key; ``export_cluster`` detects and
                reports this rather than failing silently.
        """
        path = self._clusters_cache_path()
        if not os.path.exists(path):
            return None

        try:
            with open(path) as fh:
                cache = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"{self.console.PGM_WRN}Could not read clusters cache "
                  f"({self.console.WRN}{exc}{self.console.STD}); recomputing.")
            return None

        if cache.get('cluster_sel') != self.cluster_sel_str:
            return None
        if not math.isclose(cache.get('cutoff', -1.0), self.cutoff,
                            rel_tol=1e-9, abs_tol=1e-9):
            return None
        if cache.get('max_centroids') != self.max_centroids:
            return None
        if cache.get('n_merged_frames') != n_merged_frames:
            return None
        if cache.get('stride') != self._CLUSTER_STRIDE:
            return None

        clusters = cache.get('clusters')
        if not clusters:
            return None

        print(f"{self.console.PGM_NAM}Reusing cached clustering "
              f"({self.console.EXT}{len(clusters)}{self.console.STD} "
              f"centroids, cutoff={self.console.EXT}{self.cutoff}{self.console.STD} Å, "
              f"max_centroids={self.console.EXT}{self.max_centroids}{self.console.STD}, "
              f"saved {cache.get('timestamp', '?')}); skipping "
              "trajectory merge and clustering entirely.")
        return clusters

    def _save_clusters_cache(self, clusters: List[Dict[str, Any]],
                             n_merged_frames: int) -> None:
        """
        Save the final cluster list.

        Args:
            clusters (list[dict]): Final cluster list as returned by
                ``cluster_gromos``. ``centroid``/``size``/``members`` are
                persisted; ``_sampled_idx`` is not (internal-only to
                MaxMin selection).
            n_merged_frames (int): Total merged-trajectory frame count at
                computation time, used for later cache-validity checks.
        """
        cache = {
            'cluster_sel':     self.cluster_sel_str,
            'cutoff':          self.cutoff,
            'max_centroids':   self.max_centroids,
            'n_merged_frames': int(n_merged_frames),
            'stride':          self._CLUSTER_STRIDE,
            'clusters':        [{'centroid': int(c['centroid']),
                                 'size':     int(c['size']),
                                 'members':  [int(m) for m in c['members']]}
                                for c in clusters],
            'timestamp':       time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self._clusters_cache_path(), 'w') as fh:
            json.dump(cache, fh, indent=2)
        print(f"{self.console.PGM_NAM}Clustering cached to "
              f"{self.console.EXT}{self._clusters_cache_path()}{self.console.STD}.")

    def _select_diverse_centroids(self, clusters, rmsd_matrix, max_n):
        """
        Select ``max_n`` maximally diverse centroids from a larger cluster list
        using greedy farthest-point (MaxMin) sampling.

        Args:
            clusters:    Full list of cluster dicts sorted by size (largest first).
                         Each dict must contain a ``'_sampled_idx'`` key.
            rmsd_matrix: (n_sampled, n_sampled) float32 pairwise RMSD array
                         (same array used in GROMOS).
            max_n:       Number of centroids to select.

        Returns:
            List of ``max_n`` cluster dicts sorted by cluster size.
        """
        n            = len(clusters)
        sampled_idxs = np.array([c['_sampled_idx'] for c in clusters], dtype=np.int32)

        # Centroid-centroid RMSD sub-matrix  (n_clusters × n_clusters)
        cc_rmsd = rmsd_matrix[np.ix_(sampled_idxs, sampled_idxs)]

        # Greedy MaxMin - O(n × max_n)
        selected  = [0]                    # seed: largest cluster
        remaining = list(range(1, n))

        while len(selected) < max_n and remaining:
            sel_arr   = np.array(selected, dtype=np.int32)
            rem_arr   = np.array(remaining, dtype=np.int32)
            # Minimum distance from each remaining centroid to the selected set
            min_dists = cc_rmsd[np.ix_(rem_arr, sel_arr)].min(axis=1)
            best_local  = int(np.argmax(min_dists))
            best_global = remaining[best_local]
            selected.append(best_global)
            remaining.pop(best_local)

        chosen = [clusters[i] for i in selected]
        chosen.sort(key=lambda c: c['size'], reverse=True)
        return chosen

    def _compute_rmsd_matrix_batched(self, positions, batch_size=1024):
        """
        Build the symmetric pairwise RMSD matrix using the Gram-matrix
        (GEMM) formulation.

        Args:
            positions (numpy.ndarray): (n_frames, n_atoms, 3) array of
                subsampled frame coordinates in Å.
            batch_size (int): Number of frames per row-batch GEMM call.
                Defaults to 1024.

        Returns:
            numpy.ndarray: (n_frames, n_frames) symmetric float32 pairwise
                RMSD matrix in Å.
        """
        n, n_atoms, _ = positions.shape
        flat  = positions.reshape(n, -1).astype(np.float32)
        scale = np.float32(1.0 / np.sqrt(n_atoms))

        try:
            rmsd_matrix = self._rmsd_gram_gpu(flat, scale, n, batch_size)
        except Exception as exc:
            print(f"{self.console.PGM_WRN}GPU RMSD matrix computation "
                  f"unavailable ({self.console.WRN}{exc}{self.console.STD}); falling back to CPU (BLAS).")
            rmsd_matrix = self._rmsd_gram_cpu(flat, scale, n, batch_size)

        print()
        return rmsd_matrix

    def _rmsd_gram_gpu(self, flat, scale, n, batch_size):
        """
        GPU implementation of the Gram-matrix RMSD computation (CuPy/cuBLAS).

        Args:
            flat (numpy.ndarray): (n_frames, 3*n_atoms) flattened, float32
                frame coordinates.
            scale (numpy.float32): 1/sqrt(n_atoms) RMSD normalization factor.
            n (int): Number of frames.
            batch_size (int): Number of frames per row-batch GEMM call.

        Returns:
            numpy.ndarray: (n, n) symmetric float32 pairwise RMSD matrix.

        Raises:
            Exception: Propagated if no CUDA device / CuPy runtime is
                available, so the caller can fall back to CPU.
        """
        rmsd_matrix = np.zeros((n, n), dtype=np.float32)
        mem_pool         = cp.get_default_memory_pool()
        pinned_mem_pool  = cp.get_default_pinned_memory_pool()

        with cp.cuda.Device(0):
            flat_gpu = cp.asarray(flat)
            sq_norms = cp.sum(flat_gpu ** 2, axis=1)   # (n,)

            for i in range(0, n, batch_size):
                i_end = min(i + batch_size, n)
                bi    = flat_gpu[i:i_end]
                gram  = bi @ flat_gpu.T                # (bi, n) GEMM on GPU
                d2    = sq_norms[i:i_end, None] + sq_norms[None, :] - 2.0 * gram
                cp.clip(d2, 0.0, None, out=d2)          # guard against fp noise
                rmsd_matrix[i:i_end, :] = cp.asnumpy(cp.sqrt(d2) * scale)
                print(f"{self.console.PGM_NAM}RMSD matrix (GPU): "
                      f"{self.console.WRN}{i_end}{self.console.STD}/"
                      f"{self.console.EXT}{n}{self.console.STD} rows computed", end='\r')

            del flat_gpu, sq_norms
            mem_pool.free_all_blocks()
            pinned_mem_pool.free_all_blocks()

        # Symmetrize to cancel float32 GEMM round-off between the (i,j) and (j,i) paths
        rmsd_matrix = 0.5 * (rmsd_matrix + rmsd_matrix.T)
        np.fill_diagonal(rmsd_matrix, 0.0)
        return rmsd_matrix

    def _rmsd_gram_cpu(self, flat, scale, n, batch_size):
        """
        CPU (BLAS-backed) implementation of the Gram-matrix RMSD computation,
        used when no GPU is available.

        Args:
            flat (numpy.ndarray): (n_frames, 3*n_atoms) flattened, float32
                frame coordinates.
            scale (numpy.float32): 1/sqrt(n_atoms) RMSD normalization factor.
            n (int): Number of frames.
            batch_size (int): Number of frames per row-batch GEMM call.

        Returns:
            numpy.ndarray: (n, n) symmetric float32 pairwise RMSD matrix.
        """
        rmsd_matrix = np.zeros((n, n), dtype=np.float32)
        sq_norms    = np.sum(flat ** 2, axis=1)   # (n,)

        for i in range(0, n, batch_size):
            i_end = min(i + batch_size, n)
            bi    = flat[i:i_end]
            gram  = bi @ flat.T                    # (bi, n) GEMM, multi-threaded BLAS
            d2    = sq_norms[i:i_end, None] + sq_norms[None, :] - 2.0 * gram
            np.clip(d2, 0.0, None, out=d2)
            rmsd_matrix[i:i_end, :] = np.sqrt(d2) * scale
            print(f"{self.console.PGM_NAM}RMSD matrix (CPU): "
                  f"{self.console.WRN}{i_end}{self.console.STD}/"
                  f"{self.console.EXT}{n}{self.console.STD} rows computed", end='\r')

        rmsd_matrix = 0.5 * (rmsd_matrix + rmsd_matrix.T)
        np.fill_diagonal(rmsd_matrix, 0.0)
        return rmsd_matrix

    # Step 3: centroid MD

    def extract_centroid_state(self, merged_u, frame_idx):
        """
        Build a SystemState from a specific DCD frame.

        Args:
            merged_u (MDAnalysis.Universe): Merged pseudo-trajectory produced
                by ``merge_trajectories``.
            frame_idx (int): Index of the frame to extract from merged_u.

        Returns:
            SystemState: Positions (nm) and box vectors (nm) for the selected
                frame, with velocities_nm_ps set to None.

        Raises:
            RuntimeError: If no periodic box information can be determined
                from the DCD frame, the saved reference box, or the XSC
                file.
        """
        ts     = merged_u.trajectory[frame_idx]
        has_box = (ts.dimensions is not None
                  and len(ts.dimensions) >= 6
                  and np.all(ts.dimensions[:3] > 0.0))

        if has_box:
            # Wrap whole molecules back into the primary cell
            # using bonded connectivity from the PSF
            merged_u.atoms.wrap(compound='fragments')

        pos_nm = merged_u.atoms.positions.copy() * 0.1   # Å → nm
        if has_box:
            vecs_ang = triclinic_vectors(ts.dimensions)   # (3, 3) Å
            box_nm   = [vecs_ang[k] * 0.1 for k in range(3)]
        elif self._ref_box_nm is not None:
            # Use the saved initial-run box vectors (engine-agnostic fallback).
            box_nm = [v.copy() for v in self._ref_box_nm]
        elif self._input_engine == 'NAMD' and self.xscfile and os.path.exists(self.xscfile):
            box_nm = NAMDInputReader.read_xsc(self.xscfile)
        else:
            raise RuntimeError(
                "Cannot determine periodic box for centroid extraction: "
                "DCD frame has no box info, and no saved reference box or "
                "XSC file is available."
            )
        return SystemState(positions_nm=pos_nm, velocities_nm_ps=None,
                           box_vectors_nm=box_nm)

    # Positional restraint de-excitation schedule
    # Each tuple is (k_backbone, k_sidechain) in kcal/mol/Å²
    # The total de-excitation time is divided equally among the 4 phases
    _RESTRAINT_SCHEDULE: List[Tuple[float, float]] = [
        (5.0,   2.5  ),   # phase 1 - heavy restraint
        (2.5,   1.125),   # phase 2
        (1.0,   0.25 ),   # phase 3
        (0.1,   0.0  ),   # phase 4 - nearly free
    ]

    # Mid-phase strain-relief dip target
    _RELIEF_TARGETS: List[Tuple[float, float]] = [
        (3.5,  1.75 ),   # phase 1 dip
        (1.75, 0.875),   # phase 2 dip
        (0.6,  0.25 ),   # phase 3 dip
        (0.05, 0.0  ),   # phase 4 dip
    ]

    # Relief routine: 30% nominal / 20% dip / 30% nominal / 20% dip
    _RELIEF_NOMINAL_FRACTION: float = 0.30   # each of the two nominal segments
    _RELIEF_DIP_FRACTION: float = 0.20       # each of the two relief dips

    # 1 kcal/mol/Å² → kJ/mol/nm²  (OpenMM internal units)
    _KCAL_A2_TO_KJ_NM2: float = 418.4

    # Frame stride used when accumulating positions for GROMOS clustering
    _CLUSTER_STRIDE: int = 3

    # Exact production-end checkpoint
    _PROD_CHECKPOINT_FILE: str = "prod_checkpoint.chk"

    # Maximum number of alternative member frames to retry if
    # de-excitation fails
    _MAX_MEMBER_RETRIES: int = 4

    # Reinforced integrator settings for the escalating last-resort pass
    _REINFORCED_TIMESTEP_FS: float = 1.0        # finer integrator
    _REINFORCED_FRICTION_PER_PS: float = 5.0    # stronger friction

    # Protein backbone heavy-atom names (CHARMM naming convention)
    _BACKBONE_ATOM_NAMES: set = {'CA', 'C', 'N', 'O', 'OT1', 'OT2', 'OXT'}

    # Residue names that should NOT receive positional restraints
    _SKIP_RESNAMES: set = (
        {'HOH', 'TIP3', 'WAT', 'TIP4', 'TIP5', 'SPC', 'TIP3P'}       # water
        | {'SOD', 'CLA', 'POT', 'MG', 'CAL', 'CES', 'ZN',             # ions (CHARMM)
           'NA', 'CL', 'K', 'NA+', 'CL-', 'K+', 'CA2+', 'MG2+'}
        | OpenMMSystemBuilder.LIPID_RESIDUES                            # lipids
    )

    def _build_restrained_system(self, ref_pos_nm: np.ndarray) -> mm.System:
        """
        Return a per-centroid deep copy of the shared OpenMM system with two
        ``CustomExternalForce`` restraints added:

        * ``k_bb`` - applied to protein backbone heavy atoms (CA, C, N, O, OXT).
        * ``k_sc`` - applied to protein sidechain heavy atoms.

        Args:
            ref_pos_nm: (N_atoms, 3) array of centroid positions in nm.

        Returns:
            A new ``mm.System`` with the two restraint forces appended and
            any barostat removed (see note below).

        Note:
            Combining a Monte Carlo barostat with strong positional
            restraints is a known source of instability: the barostat's
            volume-move acceptance is driven by total system energy,
            including the restraint terms, which do not reflect real
            physical pressure. This can transiently compress the box while
            restrained atoms cannot relocate to relieve the resulting
            overlap, producing a sudden nonbonded clash. Restrained
            de-excitation is therefore run at constant volume (NVT); any
            ``MonteCarloBarostat``/``MonteCarloMembraneBarostat`` present
            on the shared system is stripped from this per-centroid copy
            only. The unrestrained production phase is built separately
            from ``self._omm_system`` (see ``_run_centroid_md``) and keeps
            its barostat, remaining NPT as before.
        """

        # Independent copy - leaves self._omm_system untouched
        system_copy = XmlSerializer.deserialize(
            XmlSerializer.serialize(self._omm_system)
        )

        # Remove any barostat: restrained de-excitation runs NVT (see note above)
        for force_idx in reversed(range(system_copy.getNumForces())):
            force = system_copy.getForce(force_idx)
            if isinstance(force, (mm.MonteCarloBarostat, mm.MonteCarloMembraneBarostat)):
                system_copy.removeForce(force_idx)

        # Backbone restraint
        bb_force = mm.CustomExternalForce(
            "k_bb*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        bb_force.addGlobalParameter("k_bb", 0.0)
        bb_force.addPerParticleParameter("x0")
        bb_force.addPerParticleParameter("y0")
        bb_force.addPerParticleParameter("z0")

        # Sidechain restraint
        sc_force = mm.CustomExternalForce(
            "k_sc*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        sc_force.addGlobalParameter("k_sc", 0.0)
        sc_force.addPerParticleParameter("x0")
        sc_force.addPerParticleParameter("y0")
        sc_force.addPerParticleParameter("z0")

        n_bb = n_sc = 0
        for atom in self._psf_omm.topology.atoms():
            # Skip water, ions, lipids
            if atom.residue.name.upper() in self._SKIP_RESNAMES:
                continue
            # Skip hydrogens
            if atom.element == app.element.hydrogen:
                continue

            x0 = float(ref_pos_nm[atom.index, 0])
            y0 = float(ref_pos_nm[atom.index, 1])
            z0 = float(ref_pos_nm[atom.index, 2])

            if atom.name.upper() in self._BACKBONE_ATOM_NAMES:
                bb_force.addParticle(atom.index, [x0, y0, z0])
                n_bb += 1
            else:
                sc_force.addParticle(atom.index, [x0, y0, z0])
                n_sc += 1

        system_copy.addForce(bb_force)
        system_copy.addForce(sc_force)

        print(f"{self.console.PGM_NAM}Positional restraints: "
              f"{self.console.EXT}{n_bb}{self.console.STD} backbone atoms, "
              f"{self.console.EXT}{n_sc}{self.console.STD} sidechain atoms.")
        return system_copy

    def _centroid_dir(self, frame_idx: int) -> str:
        """
        Return the stable, frame-index-keyed directory for one centroid's
        MD output.

        Args:
            frame_idx (int): Centroid frame index in the merged
                pseudo-trajectory (``cluster['centroid']``).

        Returns:
            str: Path to ``fel/centroids/centroid_frame{frame_idx}``.
        """
        return f"{self.out_dir}/centroids/centroid_frame{frame_idx}"

    def _centroid_prod_dcd_name(self, frame_idx: int) -> str:
        """
        Return the filename (no directory) of a centroid's production DCD.

        Args:
            frame_idx (int): Centroid frame index in the merged
                pseudo-trajectory (``cluster['centroid']``).

        Returns:
            str: ``centroid_{frame_idx}.dcd``.
        """
        return f"centroid_{frame_idx}.dcd"

    def _centroid_prod_dcd_path(self, frame_idx: int) -> str:
        """
        Return the path to a centroid's production DCD.

        Args:
            frame_idx (int): Centroid frame index in the merged
                pseudo-trajectory (``cluster['centroid']``).

        Returns:
            str: Path to ``{centroid_dir}/centroid_{frame_idx}.dcd``.
        """
        return f"{self._centroid_dir(frame_idx)}/{self._centroid_prod_dcd_name(frame_idx)}"

    def _select_fallback_members(self, cluster: Dict[str, Any], original_frame: int,
                                 n: int) -> List[int]:
        """
        Select up to n alternative member frames from a GROMOS cluster to
        retry centroid MD with, if the original centroid frame's
        restrained de-excitation fails.

        Args:
            cluster (dict): Cluster dict as returned by cluster_gromos,
                with a 'members' key (list of global merged-trajectory
                frame indices, including the centroid itself).
            original_frame (int): The cluster's centroid frame, excluded
                from the candidate pool.
            n (int): Maximum number of fallback frames to return.

        Returns:
            list[int]: Up to n alternative frame indices, evenly spaced
                through the member list. Empty if the cluster has no
                other members.
        """
        candidates = [m for m in cluster.get('members', []) if m != original_frame]
        if not candidates:
            return []
        if len(candidates) <= n:
            return candidates
        step = len(candidates) / n
        return [candidates[int(i * step)] for i in range(n)]

    def _run_phase_with_relief(self, engine: OpenMMSimulationEngine, phase_cycles: int,
                               phase_idx: int, k_bb_kj: float, k_sc_kj: float,
                               timestep_fs: float = 2.0) -> None:
        """
        Step through one restrained de-excitation phase as a fixed
        4-segment pattern dipping to a gentler restraint level twice
        per phase for a burst of real dynamics before restoring the
        phase's nominal level.

        Args:
            engine (OpenMMSimulationEngine): Engine to step; its context's
                "k_bb"/"k_sc" global parameters are toggled between the
                nominal and relief levels.
            phase_cycles (int): This phase's nominal excitation cycles
                (excitation-cycle units, i.e. multiplied by ``self.n_steps``
                to get the phase's total step count at the reference 2 fs
                timestep).
            phase_idx (int): 0-based phase index into ``_RESTRAINT_SCHEDULE``/
                ``_RELIEF_TARGETS``.
            k_bb_kj (float): This phase's nominal backbone restraint
                constant, kJ/mol/nm².
            k_sc_kj (float): This phase's nominal sidechain restraint
                constant, kJ/mol/nm².
            timestep_fs (float): The ACTUAL timestep the passed-in
                ``engine`` was built with (femtoseconds). Defaults to 2.0,
                the standard value, in which case the total step count
                reduces to exactly ``phase_cycles * self.n_steps`` as
                before. When ``engine`` uses a smaller timestep, the step
                count is scaled up proportionally so this phase still covers
                the same physical duration in picoseconds.
        """
        total_steps = round(phase_cycles * self.n_steps * (2.0 / timestep_fs))
        if total_steps <= 0:
            return

        context = engine.simulation.context

        seg_nominal = round(self._RELIEF_NOMINAL_FRACTION * total_steps)
        seg_dip     = round(self._RELIEF_DIP_FRACTION * total_steps)
        # Second dip absorbs any rounding remainder so the four segments
        # always sum to exactly total_steps.
        seg_dip_2   = total_steps - (2 * seg_nominal) - seg_dip

        k_bb_relief_kcal, k_sc_relief_kcal = self._RELIEF_TARGETS[phase_idx]
        k_bb_relief_kj = k_bb_relief_kcal * self._KCAL_A2_TO_KJ_NM2
        k_sc_relief_kj = k_sc_relief_kcal * self._KCAL_A2_TO_KJ_NM2

        segments = [
            (seg_nominal, k_bb_kj,        k_sc_kj       ),   # 30% nominal
            (seg_dip,     k_bb_relief_kj, k_sc_relief_kj),   # 20% dip
            (seg_nominal, k_bb_kj,        k_sc_kj       ),   # 30% nominal
            (seg_dip_2,   k_bb_relief_kj, k_sc_relief_kj),   # 20% dip (+ rounding remainder)
        ]
        for steps, k_bb_val, k_sc_val in segments:
            if steps <= 0:
                continue
            context.setParameter("k_bb", k_bb_val)
            context.setParameter("k_sc", k_sc_val)
            engine.simulation.step(steps)

    def _run_centroid_md(self, centroid_state: 'SystemState',
                         frame_idx: int, reinforced: bool = False) -> Optional[str]:
        """
        Run 4-phase restrained de-excitation followed by unrestrained production
        MD from a single centroid structure.

        De-excitation protocol:
          Phase 1 - k_bb = 5.0, k_sc = 2.5  kcal/mol/Å²
          Phase 2 - k_bb = 2.5, k_sc = 1.125 kcal/mol/Å²
          Phase 3 - k_bb = 1.0, k_sc = 0.25  kcal/mol/Å²
          Phase 4 - k_bb = 0.1, k_sc = 0.0   kcal/mol/Å²

        Each phase spans (n_deexcite_ps / 4) ps, internally split into a
        30% nominal / 20% dip / 30% nominal / 20% dip pattern (see
        ``_run_phase_with_relief``) rather than held flat throughout.
        Restraint reference positions are the centroid coordinates
        themselves. No DCD frames are written during de-excitation.

        The de-excitation engine is always built with
        ``thermostat='langevin'``: the stochastic Langevin thermostat is
        specifically relied upon here to damp local strain during the
        restraint-relief schedule. The subsequent unrestrained production
        engine uses the class default (Nosé-Hoover) instead, since
        production is not subject to this localized-strain concern.

        Args:
            centroid_state: SystemState with centroid positions and box.
            frame_idx: Centroid's merged-trajectory frame index - the
                stable identifier used to name its output directory.
            reinforced (bool): If True, the DE-EXCITATION engine only
                is built with
                ``_REINFORCED_TIMESTEP_FS``/``_REINFORCED_FRICTION_PER_PS``
                instead of the standard 2 fs / 1.0 ps⁻¹, for
                ``run()``'s escalating last-resort pass on centroids that
                still fail after exhausting all standard-settings
                cluster-member fallback attempts.

        Returns:
            Absolute path to the production DCD, or None on failure.
        """
        centroid_dir  = self._centroid_dir(frame_idx)
        os.makedirs(centroid_dir, exist_ok=True)
        prev_dir      = os.getcwd()
        os.chdir(centroid_dir)
        prod_dcd_name = self._centroid_prod_dcd_name(frame_idx)

        timestep_fs     = self._REINFORCED_TIMESTEP_FS if reinforced else 2.0
        friction_per_ps = self._REINFORCED_FRICTION_PER_PS if reinforced else 1.0

        try:
            # Build a per-centroid system copy with positional restraint forces
            restrained_system = self._build_restrained_system(
                centroid_state.positions_nm
            )

            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, restrained_system,
                self._temperature,
                rep_num=frame_idx,
                is_restart=False, full_ener=False, n_steps=self.n_steps,
                attach_main_dcd=False,   # no frames written during de-excitation
                file_prefix='centroid_',
                timestep_fs=timestep_fs,
                thermostat='langevin',   # stochastic thermostat for the restrained,
                                         # strain-relief de-excitation schedule
            )
            # initialize_state assigns MB velocities when velocities_nm_ps is None
            engine.initialize_state(centroid_state)

            # Restrained Energy minimization
            k_bb_init = self._RESTRAINT_SCHEDULE[0][0] * self._KCAL_A2_TO_KJ_NM2
            k_sc_init = self._RESTRAINT_SCHEDULE[0][1] * self._KCAL_A2_TO_KJ_NM2
            engine.simulation.context.setParameter("k_bb", k_bb_init)
            engine.simulation.context.setParameter("k_sc", k_sc_init)
            print(f"{self.console.PGM_NAM}Minimizing energy under initial restraints (max 500 iterations)...")
            engine.simulation.minimizeEnergy(maxIterations=500)
            # Reassign MB velocities after minimization
            engine.simulation.context.setVelocitiesToTemperature(
                self._temperature * unit.kelvin
            )

            # 4-phase restrained de-excitation
            n_phases        = len(self._RESTRAINT_SCHEDULE)
            # Distribute de-excitation cycles evenly; remainder goes to last phase
            cycles_per_phase = self.n_deexcite_cycles // n_phases
            remainder_cycles = self.n_deexcite_cycles - cycles_per_phase * n_phases

            for phase_idx, (k_bb_kcal, k_sc_kcal) in enumerate(self._RESTRAINT_SCHEDULE):
                k_bb_kj = k_bb_kcal * self._KCAL_A2_TO_KJ_NM2
                k_sc_kj = k_sc_kcal * self._KCAL_A2_TO_KJ_NM2
                engine.simulation.context.setParameter("k_bb", k_bb_kj)
                engine.simulation.context.setParameter("k_sc", k_sc_kj)

                phase_cycles = (cycles_per_phase
                                + (remainder_cycles if phase_idx == n_phases - 1 else 0))
                phase_ps     = phase_cycles * self.n_steps * 0.002

                print(f"{self.console.PGM_NAM}De-excitation phase {self.console.WRN}{phase_idx + 1}{self.console.STD}/{self.console.EXT}4{self.console.STD}: "
                      f"k_bb={self.console.EXT}{k_bb_kcal:.3f}{self.console.STD}, k_sc={self.console.EXT}{k_sc_kcal:.3f}{self.console.STD} kcal/mol/Å² "
                      f"({self.console.EXT}{phase_ps:.1f}{self.console.STD} ps{', REINFORCED' if reinforced else ''})...")
                self._run_phase_with_relief(engine, phase_cycles, phase_idx, k_bb_kj, k_sc_kj,
                                            timestep_fs=timestep_fs)

            # Carry the de-excited state (positions, velocities, box) over to a
            # fresh, fully unrestrained Simulation
            pos_nm, vel_nm_ps, box_nm = engine.get_state()
            engine.close()
            del engine

            production_state = SystemState(
                positions_nm=pos_nm,
                velocities_nm_ps=vel_nm_ps,
                box_vectors_nm=box_nm,
            )

            prod_engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                rep_num=frame_idx,
                is_restart=True, full_ener=False, n_steps=self.n_steps,
                attach_main_dcd=False,   # production frames go to the dedicated centroid_{frame_idx}.dcd below
                file_prefix='centroid_',
                # thermostat omitted: defaults to 'nose_hoover' for production
            )
            prod_engine.initialize_state(production_state)

            # Unrestrained production
            print(f"{self.console.PGM_NAM}Performing unrestrained MD on centroids for FEL computation...")
            prod_engine.simulation.reporters.append(
                app.DCDReporter(prod_dcd_name, self.n_steps,
                                append=False, enforcePeriodicBox=False)
            )
            prod_engine.simulation.step(self.n_prod_cycles * self.n_steps)
            # Final checkpoint
            prod_engine.save_checkpoint(self._PROD_CHECKPOINT_FILE)
            prod_engine.close()
            return os.path.join(centroid_dir, prod_dcd_name)

        except Exception as exc:
            print(f"{self.console.PGM_ERR}Centroid (frame {self.console.ERR}{frame_idx}{self.console.STD}) MD failed: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            traceback.print_exc()
            return None
        finally:
            os.chdir(prev_dir)

    def _load_last_frame_as_state(self, dcd_filename: str) -> 'SystemState':
        """
        Build a SystemState (positions + box, no velocities) from the last
        frame of a centroid's own production DCD.

        Args:
            dcd_filename: DCD filename, resolved relative to the current
                working directory. Callers are expected to have already
                ``os.chdir``'d into the centroid's own directory.

        Returns:
            SystemState: positions (nm) and box vectors (nm) from the last
                frame; ``velocities_nm_ps`` is ``None``.

        Raises:
            RuntimeError: If no periodic box information can be determined
                from the last frame or the saved reference box.
        """
        u  = mda.Universe(self.psffile, dcd_filename, format="DCD")
        ts = u.trajectory[-1]
        has_box = (ts.dimensions is not None
                  and len(ts.dimensions) >= 6
                  and np.all(ts.dimensions[:3] > 0.0))

        if has_box:
            u.atoms.wrap(compound='fragments')

        pos_nm = u.atoms.positions.copy() * 0.1   # Å → nm
        if has_box:
            vecs_ang = triclinic_vectors(ts.dimensions)   # (3, 3) Å
            box_nm   = [vecs_ang[k] * 0.1 for k in range(3)]
        elif self._ref_box_nm is not None:
            box_nm = [v.copy() for v in self._ref_box_nm]
        else:
            raise RuntimeError(
                "Cannot determine periodic box for production-DCD fallback "
                "continuation: last frame has no box info, and no saved "
                "reference box is available."
            )
        return SystemState(positions_nm=pos_nm, velocities_nm_ps=None,
                           box_vectors_nm=box_nm)

    def _extend_centroid_production(self, frame_idx: int,
                                    additional_cycles: int) -> Optional[str]:
        """
        Extend an existing centroid's production MD by
        ``additional_cycles``, appending frames to its existing
        production DCD.

        Continuation state is restored, in order of preference:
          1. ``prod_checkpoint.chk`` - the exact positions/velocities/box/
             RNG state saved at the end of this centroid's last production
             run (by ``_run_centroid_md`` or a previous call to this
             method). Gives a bit-identical continuation.
          2. Last frame of the production DCD (via
             ``_load_last_frame_as_state``) - used only when no checkpoint
             is available. Positions and box only; velocities are
             re-assigned at Maxwell-Boltzmann. Appending is then physically
             valid but not bit-identical.

        This engine uses the class default thermostat (Nosé-Hoover), same
        as the production engine in ``_run_centroid_md``. Loading a
        checkpoint that was itself written by a Nosé-Hoover engine restores
        its thermostat chain state exactly; a checkpoint written by an
        older, Langevin-based version of pyAdMD is not loadable here (see
        ``OpenMMSimulationEngine``'s breaking-change note).

        Args:
            frame_idx: Centroid's merged-trajectory frame index - the
                stable identifier used to locate its output directory.
            additional_cycles: Number of additional ``n_steps``-step cycles
                to run.

        Returns:
            Absolute path to the (extended) production DCD, or None on
            failure.
        """
        centroid_dir  = self._centroid_dir(frame_idx)
        prod_dcd_name = self._centroid_prod_dcd_name(frame_idx)
        prod_dcd_path = self._centroid_prod_dcd_path(frame_idx)
        prev_dir      = os.getcwd()
        os.chdir(centroid_dir)

        try:
            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                rep_num=frame_idx,
                is_restart=True, full_ener=False, n_steps=self.n_steps,
                attach_main_dcd=False,   # appended frames go to the dedicated centroid_{frame_idx}.dcd below
                file_prefix='centroid_',
                # thermostat omitted: defaults to 'nose_hoover', matching
                # the production engine that originally wrote this centroid
            )

            if os.path.exists(self._PROD_CHECKPOINT_FILE):
                print(f"{self.console.PGM_NAM}Resuming centroid (frame "
                      f"{self.console.WRN}{frame_idx}{self.console.STD}) from {self.console.EXT}"
                      f"{self._PROD_CHECKPOINT_FILE}{self.console.STD}.")
                engine.load_checkpoint(self._PROD_CHECKPOINT_FILE)
            else:
                print(f"{self.console.PGM_WRN}No production checkpoint "
                      f"found for centroid (frame {self.console.WRN}{frame_idx}{self.console.STD}); falling "
                      f"back to the last frame of {self.console.WRN}"
                      f"{prod_dcd_name}{self.console.STD}.")
                fallback_state = self._load_last_frame_as_state(prod_dcd_name)
                engine.initialize_state(fallback_state)

            engine.simulation.reporters.append(
                app.DCDReporter(prod_dcd_name, self.n_steps,
                                append=True, enforcePeriodicBox=False)
            )

            additional_ps = additional_cycles * self.n_steps * 0.002
            print(f"{self.console.PGM_NAM}Extending production for "
                  f"centroid (frame {self.console.EXT}{frame_idx}{self.console.STD}) by "
                  f"{self.console.EXT}{additional_ps:.1f}{self.console.STD} "
                  f"ps ({self.console.WRN}{additional_cycles}{self.console.STD} cycles)...")
            engine.simulation.step(additional_cycles * self.n_steps)

            engine.save_checkpoint(self._PROD_CHECKPOINT_FILE)
            engine.close()
            return prod_dcd_path

        except Exception as exc:
            print(f"{self.console.PGM_ERR}Extending centroid (frame "
                  f"{self.console.WRN}{frame_idx}{self.console.STD}) production failed: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            traceback.print_exc()
            return None
        finally:
            os.chdir(prev_dir)

    # Step 4: mode projection

    def _get_projection_setup(self):
        """
        Resolve Cα selection, reference positions, and normalised mode vectors.

        Thin wrapper around
        ``pyadmd.modes.projection.get_projection_setup`` -- the same
        function ``Analyzer``'s mode-projection analysis uses, so both
        code paths agree on the reference structure / mode-loading /
        Cα-extraction logic. See that function's docstring for full
        parameter/return details.

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
            RuntimeError: If no saved reference positions or NAMD coorfile
                are available to build the reference structure, or if no valid
                mode vectors could be loaded.
        """
        return get_projection_setup(
            psffile=self.psffile, input_dir=self.input_dir,
            ref_positions_ang=self._ref_positions_ang,
            coorfile=self.coorfile, rstfile=self.rstfile,
            input_engine=self._input_engine, nm_type=self.nm_type,
            modes=self.fe_modes, console=self.console,
        )

    def compute_mode_projections(self, unit_dcd_pairs, ca_ix_full, ca_masses,
                                 M_ca, ref_pos_ca_ang, mode_vectors_ca):
        """
        Compute signed MRMS displacement of every production frame along each
        individual mode vector:

            d_j = (1/√M) Σ_i √m_i · (r_i − r₀ᵢ) · q_{ij}

        Sign is preserved so that FEL plots distinguish both directions.
        Thin wrapper around
        ``pyadmd.modes.projection.compute_mode_projections_df`` (the same
        function ``Analyzer``'s mode-projection analysis uses), keyed on
        ``'centroid_frame'`` and this run's per-cycle time step.

        Args:
            unit_dcd_pairs (list[tuple[int, str or None]]): One
                ``(centroid_frame, dcd_path)`` pair per centroid. Entries
                with ``dcd_path is None`` (failed centroid MD runs) are
                skipped.
            ca_ix_full (numpy.ndarray): (n_ca,) global Cα atom indices.
            ca_masses (numpy.ndarray): (n_ca,) Cα atomic masses in amu.
            M_ca (float): Total Cα mass.
            ref_pos_ca_ang (numpy.ndarray): (n_ca, 3) reference Cα positions
                in Å.
            mode_vectors_ca (dict): {mode_num: (n_ca, 3) normalised mode
                vector}.

        Returns:
            pandas.DataFrame: Columns ``['centroid_frame', 'time',
                'mode_{n1}', 'mode_{n2}', ...]``, one row per analyzed
                production frame across all centroids. Empty (with the
                correct columns) if no centroid had a usable DCD.
        """
        cycle_ps = self.n_steps * 0.002
        return compute_mode_projections_df(
            psffile=self.psffile, unit_dcd_pairs=unit_dcd_pairs,
            unit_col='centroid_frame', cycle_ps=cycle_ps,
            ca_ix_full=ca_ix_full, ca_masses=ca_masses, M_ca=M_ca,
            ref_pos_ca_ang=ref_pos_ca_ang, mode_vectors_ca=mode_vectors_ca,
        )

    # Step 5: FEL computation

    def compute_fel_1d(self, proj):
        """
        Compute the 1D free energy landscape along a single mode coordinate.

        ΔGα = −kBT ln[ P(qα) / Pmax(q) ]. Empty bins are returned as NaN.

        Args:
            proj (numpy.ndarray): 1D array of mode projections in Å.

        Returns:
            tuple: (bin_centers, delta_G) where bin_centers (numpy.ndarray)
                are the histogram bin centers in Å and delta_G
                (numpy.ndarray) is the free energy in kcal/mol.
        """
        kBT         = 0.001987204 * self._temperature
        hist, edges = np.histogram(proj, bins=self.bins, density=False)
        centers     = 0.5 * (edges[:-1] + edges[1:])
        hf          = hist.astype(float)
        hf[hf == 0] = np.nan
        dG          = -kBT * np.log(hf / np.nanmax(hf))
        return centers, dG

    def compute_fel_2d(self, proj_x, proj_y):
        """
        Compute the 2D free energy landscape from a joint probability
        histogram.

        Args:
            proj_x (numpy.ndarray): 1D array of mode projections in Å for
                the first mode.
            proj_y (numpy.ndarray): 1D array of mode projections in Å for
                the second mode.

        Returns:
            tuple: (xc, yc, delta_G) where xc and yc (numpy.ndarray) are the
                histogram bin centers in Å for each mode and delta_G
                (numpy.ndarray) is the free energy in kcal/mol (NaN for
                empty bins).
        """
        kBT = 0.001987204 * self._temperature
        h2d, xedges, yedges = np.histogram2d(proj_x, proj_y,
                                              bins=self.bins, density=False)
        xc  = 0.5 * (xedges[:-1] + xedges[1:])
        yc  = 0.5 * (yedges[:-1] + yedges[1:])
        hf  = h2d.astype(float)
        hf[hf == 0] = np.nan
        dG  = -kBT * np.log(hf / np.nanmax(hf))
        return xc, yc, dG

    # Output generation

    def generate_outputs(self, fel_1d, fel_2d, projections_df, clusters,
                        centroid_records=None):
        """
        Write clustering CSV, mode projections CSV, plots, and HTML summary.

        Args:
            fel_1d (dict): {mode_num: (bin_centers, delta_G)} from
                compute_fel_1d.
            fel_2d (dict): {(mode1, mode2): (xc, yc, delta_G)} from
                compute_fel_2d.
            projections_df (pandas.DataFrame): Combined per-frame mode
                projections from ``compute_mode_projections``, columns
                ``['centroid_frame', 'time', 'mode_{n}', ...]``. Saved
                once to ``fel/mode_projections.csv`` -- replaces the
                former per-mode ``projections_mode{N}.npy`` files, and is
                reusable by ``pyadmd analyze -src fel`` without
                recomputation whenever it already covers every mode that
                run needs.
            clusters (list[dict]): Cluster list returned by ``cluster_gromos``.
            centroid_records (list[dict], optional): Per-centroid status
                collected during ``run()``'s centroid MD loop (keys
                ``frame``, ``status``, ``cycles_before``), used to report
                which centroids were fresh, extended, or skipped this call.
        """
        self._save_clustering_summary(clusters, centroid_records)

        if not projections_df.empty:
            projections_csv = f"{self.out_dir}/mode_projections.csv"
            projections_df.to_csv(projections_csv, index=False)
            print(f"{self.console.PGM_NAM}Mode projections saved to "
                  f"{self.console.EXT}{projections_csv}{self.console.STD}.")

        for mode_num, (centers, dG) in fel_1d.items():
            pd.DataFrame({'coordinate_A': centers,
                          'delta_G_kcalmol': dG}).to_csv(
                f"{self.out_dir}/fel_mode{mode_num}.csv", index=False)
            self._plot_fel_1d(centers, dG, mode_num)

        for (m1, m2), (xc, yc, dG2d) in fel_2d.items():
            self._plot_fel_2d(xc, yc, dG2d, m1, m2)

        self._generate_fel_html(fel_1d, clusters, centroid_records)

        print(f"{self.console.PGM_NAM}Free energy results saved to "
              f"{self.console.EXT}{self.out_dir}{self.console.STD}.")

    def _save_clustering_summary(self, clusters, centroid_records=None):
        """
        Write per-cluster centroid frame index, size, and production status
        to a CSV file.

        Args:
            clusters (list[dict]): Cluster list returned by ``cluster_gromos``.
            centroid_records (list[dict], optional): Per-centroid status
                from ``run()`` (keys ``frame``, ``status``, and, for
                brand-new centroids that went through the cluster-member
                fallback path, ``source_frame_used``, ``md_attempts``, the
                total number of attempts across both the standard and
                reinforced passes, and ``reinforced``, whether the
                successful attempt needed the reinforced last-resort pass
                (reduced timestep, elevated friction; see
                ``_run_centroid_md``'s ``reinforced`` argument)). Records
                from the 'skipped'/'extended' branches don't carry these
                keys; they default to the centroid frame itself,
                ``md_attempts=1``, and ``reinforced=False`` via
                ``.get()``, since no substitution/escalation is possible
                on those paths. When ``centroid_records`` is omitted
                entirely, ``status`` is reported as ``'n/a'`` (e.g. when
                this is called outside the normal ``run()`` flow).
        """
        record_by_frame = {r['frame']: r for r in (centroid_records or [])}
        rows = []
        for i, c in enumerate(clusters):
            frame  = c['centroid']
            record = record_by_frame.get(frame, {})
            done_cycles = self._centroid_done_cycles(frame)
            rows.append({
                'cluster_id':               i + 1,
                'centroid_frame':           frame,
                'size':                     c['size'],
                'status':                   record.get('status', 'n/a'),
                'source_frame_used':        record.get('source_frame_used', frame),
                'md_attempts':              record.get('md_attempts', 1),
                'reinforced':               record.get('reinforced', False),
                'production_cycles_done':   done_cycles,
                'production_cycles_target': self.n_prod_cycles,
                'production_ps_done':       round(done_cycles * self.n_steps * 0.002, 3),
            })
        pd.DataFrame(rows).to_csv(
            f"{self.out_dir}/clustering_summary.csv", index=False)

    def _centroid_done_cycles(self, frame_idx: int) -> int:
        """
        Return the number of production cycles already completed for a
        centroid, read directly from its production DCD's frame count.

        Args:
            frame_idx: Centroid's merged-trajectory frame index - the
                stable identifier used to locate its output directory.

        Returns:
            int: Number of completed production cycles (0 if no production
                DCD exists yet).
        """
        prod_dcd = self._centroid_prod_dcd_path(frame_idx)
        if not os.path.exists(prod_dcd):
            return 0
        return _count_dcd_frames(prod_dcd)

    def _centroid_is_complete(self, frame_idx: int) -> bool:
        """
        Return True if this centroid's production has already reached (or
        exceeded) the *current* target production length
        (``self.n_prod_cycles``).

        Args:
            frame_idx: Centroid's merged-trajectory frame index - the
                stable identifier used to locate its output directory.

        Returns:
            bool: True if completed production cycles meet or exceed
                ``self.n_prod_cycles``.
        """
        return self._centroid_done_cycles(frame_idx) >= self.n_prod_cycles

    def _plot_fel_1d(self, centers, dG, mode_num):
        """
        Plot and save the 1D free energy landscape for a single mode.

        Args:
            centers (numpy.ndarray): Histogram bin centers in Å.
            dG (numpy.ndarray): Free energy values in kcal/mol (NaN for
                empty bins).
            mode_num (int): Mode number, used in the plot title and filename.
        """
        valid = ~np.isnan(dG)
        if not valid.any():
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.plot(centers[valid], dG[valid], 'b-', linewidth=2)
        ax.fill_between(centers[valid], dG[valid], alpha=0.15, color='blue')
        ax.set_xlabel(f'Mode {mode_num} coordinate (\u00c5)', fontsize=12)
        ax.set_ylabel('\u0394G (kcal/mol)', fontsize=12)
        ax.set_title(f'Free Energy Landscape \u2014 Mode {mode_num}', fontsize=13)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/fel_mode{mode_num}_plot.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_fel_2d(self, xc, yc, dG2d, m1, m2):
        """
        Plot and save the 2D free energy landscape for a pair of modes.

        Args:
            xc (numpy.ndarray): Histogram bin centers in Å for the first mode.
            yc (numpy.ndarray): Histogram bin centers in Å for the second
                mode.
            dG2d (numpy.ndarray): Free energy values in kcal/mol (NaN for
                empty bins).
            m1 (int): First mode number, used in the plot title and filename.
            m2 (int): Second mode number, used in the plot title and
                filename.
        """
        if np.all(np.isnan(dG2d)):
            print(f"{self.console.PGM_WRN}2D FEL for modes {self.console.WRN}{m1}{self.console.STD}×"
                  f"{self.console.WRN}{m2}{self.console.STD} has no populated bins; skipping plot.")
            return
        dG_plot    = np.ma.masked_invalid(dG2d.T.copy())
        finite_max = np.nanmax(dG2d)
        X, Y       = np.meshgrid(xc, yc)
        levels     = np.linspace(0.0, finite_max, 21)
        cmap       = plt.get_cmap('RdYlBu_r').copy()
        cmap.set_bad('white')
        fig, ax    = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        cf  = ax.contourf(X, Y, dG_plot, levels=levels, cmap=cmap,
                           extend='neither')
        ax.contour(X, Y, dG_plot, levels=levels, colors='k',
                   linewidths=0.3, alpha=0.4)
        cbar = fig.colorbar(cf, ax=ax)
        cbar.set_label('\u0394G (kcal/mol)', fontsize=11)
        ax.set_xlabel(f'Mode {m1} coordinate (\u00c5)', fontsize=12)
        ax.set_ylabel(f'Mode {m2} coordinate (\u00c5)', fontsize=12)
        ax.set_title(f'2D Free Energy Landscape \u2014 Modes {m1} \u00d7 {m2}',
                     fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/fel_2d_mode{m1}_mode{m2}.png",
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _generate_fel_html(self, fel_1d, clusters, centroid_records=None):
        """
        Build and write the fel_summary.html report.

        Args:
            fel_1d (dict): {mode_num: (bin_centers, delta_G)} from
                compute_fel_1d.
            clusters (list[dict]): Cluster list returned by ``cluster_gromos``.
            centroid_records (list[dict], optional): Per-centroid status
                from ``run()`` (keys ``frame``, ``status``), used to build
                the "Centroid Production Status" table. When omitted, the
                table falls back to reporting only frame/size/current
                on-disk cycle counts with status ``'n/a'``.
        """
        n_clusters     = len(clusters)
        n_frames_total = sum(c['size'] for c in clusters)
        run_mode       = ("extended previous calculation"
                          if getattr(self, '_is_append_run', False)
                          else "first run")

        mode_rows = ""
        for mode_num, (centers, dG) in fel_1d.items():
            valid = ~np.isnan(dG)
            if not valid.any():
                continue
            coord_min = float(centers[np.nanargmin(dG)])
            dG_max    = float(np.nanmax(dG[valid]))
            mode_rows += (
                f"    <tr><td>{mode_num}</td><td>{coord_min:.3f}</td>"
                f"<td>0.00</td><td>{dG_max:.2f}</td></tr>\n"
            )

        status_by_frame = {r['frame']: r['status'] for r in (centroid_records or [])}
        centroid_rows = ""
        for i, c in enumerate(clusters):
            frame       = c['centroid']
            done_cycles = self._centroid_done_cycles(frame)
            done_ps     = round(done_cycles * self.n_steps * 0.002, 1)
            status      = status_by_frame.get(frame, 'n/a')
            centroid_rows += (
                f"    <tr><td>{i + 1}</td><td>{frame}</td><td>{c['size']}</td>"
                f"<td>{status}</td>"
                f"<td>{done_cycles}/{self.n_prod_cycles}</td>"
                f"<td>{done_ps}</td></tr>\n"
            )

        plots_1d = "".join(
            f'    <div class="plot-item"><img src="fel_mode{m}_plot.png"'
            f' alt="FEL mode {m}"><p>Mode {m}</p></div>\n'
            for m in fel_1d
            if os.path.exists(f"{self.out_dir}/fel_mode{m}_plot.png")
        )
        plots_2d = "".join(
            f'    <div class="plot-item"><img src="fel_2d_mode{m1}_mode{m2}.png"'
            f' alt="2D FEL {m1}x{m2}"><p>Modes {m1} \u00d7 {m2}</p></div>\n'
            for (m1, m2) in self.pairs_2d
            if os.path.exists(f"{self.out_dir}/fel_2d_mode{m1}_mode{m2}.png")
        )

        html = (
            '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            '  <meta charset="UTF-8">\n'
            '  <title>pyAdMD Free Energy Analysis</title>\n'
            '  <style>\n'
            '    body { font-family: Arial, sans-serif; margin: 40px; }\n'
            '    h1 { color: #2c3e50; } h2 { color: #34495e; border-bottom: 1px solid #ccc; padding-bottom:4px; }\n'
            '    table { border-collapse: collapse; width: 70%; margin-bottom: 20px; }\n'
            '    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n'
            '    th { background-color: #f2f2f2; }\n'
            '    .plot-grid { display: flex; flex-wrap: wrap; gap: 20px; margin: 12px 0; }\n'
            '    .plot-item { text-align: center; }\n'
            '    .plot-item img { max-width: 480px; border: 1px solid #ccc; border-radius:4px; }\n'
            '    .plot-item p { font-size: 13px; color: #555; margin: 4px 0; }\n'
            '  </style>\n</head>\n<body>\n'
            '  <h1>pyAdMD Free Energy Analysis</h1>\n'
            f'  <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>\n'
            '  <h2>Protocol Summary</h2>\n  <table>\n'
            '    <tr><th>Parameter</th><th>Value</th></tr>\n'
            f'    <tr><td>Run mode</td><td>{run_mode}</td></tr>\n'
            f'    <tr><td>Total frames (merged pseudo-trajectory)</td><td>{n_frames_total}</td></tr>\n'
            f'    <tr><td>GROMOS RMSD cutoff (\u00c5)</td><td>{self.cutoff}</td></tr>\n'
            f'    <tr><td>Number of clusters (centroids)</td><td>{n_clusters}</td></tr>\n'
            f'    <tr><td>De-excitation MD per centroid (ps)</td><td>{self.n_deexcite_ps}</td></tr>\n'
            f'    <tr><td>Production MD per centroid (ps)</td><td>{self.n_prod_ps}</td></tr>\n'
            f'    <tr><td>Temperature (K)</td><td>{self._temperature}</td></tr>\n'
            f'    <tr><td>Histogram bins</td><td>{self.bins}</td></tr>\n'
            f'    <tr><td>Modes projected</td><td>{self.fe_modes}</td></tr>\n'
            '  </table>\n'
            '  <h2>1D FEL Summary</h2>\n  <table>\n'
            '    <tr><th>Mode</th><th>Min-energy coord (\u00c5)</th>'
            '<th>\u0394G min (kcal/mol)</th><th>\u0394G max (kcal/mol)</th></tr>\n'
            f'{mode_rows}'
            '  </table>\n'
            '  <h2>Centroid Production Status</h2>\n  <table>\n'
            '    <tr><th>#</th><th>Frame</th><th>Cluster size</th>'
            '<th>Status this run</th><th>Cycles done/target</th>'
            '<th>Production (ps)</th></tr>\n'
            f'{centroid_rows}'
            '  </table>\n'
            '  <h2>1D Free Energy Landscapes</h2>\n'
            f'  <div class="plot-grid">\n{plots_1d}  </div>\n'
            '  <h2>2D Free Energy Landscapes</h2>\n'
            f'  <div class="plot-grid">\n{plots_2d}  </div>\n'
            '</body>\n</html>\n'
        )
        with open(f"{self.out_dir}/fel_summary.html", 'w') as fh:
            fh.write(html)

    # Top-level orchestrator

    def run(self):
        """
        Execute the full free energy protocol.
        """

        t0 = time.time()

        # 1-2. Merge trajectories + cluster — GROMOS + MaxMin, or reuse a
        # cached cluster list.
        n_merged_frames_cheap = self._count_merged_frames_cheap()
        clusters = self._load_clusters_cache(n_merged_frames_cheap)
        merged_u = None
        if clusters is None:
            merged_u = self.merge_trajectories()
            clusters = self.cluster_gromos(merged_u)
            self._save_clusters_cache(clusters, len(merged_u.trajectory))

        # 3. Projection setup (load mode vectors once before centroid MD loop)
        print(f"\n{self.console.PGM_NAM}Loading mode vectors for modes "
              f"{self.console.EXT}{self.fe_modes}{self.console.STD}...")
        (ca_ix_full, ca_masses, M_ca,
         ref_pos_ca_ang, mode_vectors_ca) = self._get_projection_setup()

        # 4. Centroid MD
        n_centroids      = len(clusters)
        prod_dcd_files   = []
        centroid_records = []   # per-centroid status, for reporting only
        n_pending        = sum(1 for c in clusters
                               if not self._centroid_is_complete(c['centroid']))
        print(f"\n{self.console.PGM_NAM}Centroid MD: "
              f"{self.console.EXT}{n_centroids}{self.console.STD} total, "
              f"{self.console.WRN}{n_pending}{self.console.STD} pending "
              f"({self.console.WRN}{n_centroids - n_pending}{self.console.STD} already complete, target "
              f"{self.console.EXT}{self.n_prod_cycles}{self.console.STD} cycles / "
              f"{self.console.EXT}{self.n_prod_ps}{self.console.STD} ps).")

        for i, cluster in enumerate(clusters):
            display_idx = i + 1                # display-only; no on-disk meaning
            frame_idx   = cluster['centroid']   # stable identity for all paths
            now         = time.strftime("%H:%M:%S")
            done_cycles = self._centroid_done_cycles(frame_idx)

            if done_cycles >= self.n_prod_cycles:
                # Already meets (or exceeds) the current target - reuse as-is.
                prod_dcd = self._centroid_prod_dcd_path(frame_idx)
                print(f"{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} (frame {self.console.WRN}{frame_idx}{self.console.STD}): already complete "
                      f"({self.console.WRN}{done_cycles}{self.console.STD}/{self.console.EXT}{self.n_prod_cycles}{self.console.STD} "
                      "cycles), skipping.")
                prod_dcd_files.append(prod_dcd)
                centroid_records.append({'frame': frame_idx, 'status': 'skipped',
                                         'cycles_before': done_cycles})

            elif done_cycles == 0:
                # Brand-new centroid: full de-excitation + full target production.
                # Tries the original centroid frame first; on failure, retries
                # with up to _MAX_MEMBER_RETRIES alternative member frames from
                # the same cluster. If ALL of those fail, the same candidate
                # frames are tried once more with reinforced settings.
                if merged_u is None:
                    # Clusters came from cache, so the trajectory merge was
                    # skipped above; a fresh centroid needs it now.
                    print(f"\n{self.console.PGM_NAM}Fresh centroid needs its "
                          "source structure; merging replica trajectories now...")
                    merged_u = self.merge_trajectories()
                print(f"\n{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} "
                      f"(frame {self.console.EXT}{frame_idx}{self.console.STD}, "
                      f"cluster size {self.console.WRN}{cluster['size']}{self.console.STD})...")

                fallback_frames  = self._select_fallback_members(
                    cluster, frame_idx, self._MAX_MEMBER_RETRIES)
                candidate_frames = [frame_idx] + fallback_frames

                dcd_path          = None
                source_frame_used = frame_idx
                reinforced_used   = False
                attempts          = 0

                for reinforced_pass in (False, True):
                    if dcd_path is not None:
                        break
                    if reinforced_pass:
                        print(f"{self.console.PGM_WRN}Centroid (frame "
                              f"{self.console.WRN}{frame_idx}{self.console.STD}) failed all "
                              f"{self.console.WRN}{attempts}{self.console.STD} standard-settings "
                              "attempt(s); retrying the same candidate frame(s) with "
                              f"{self.console.EXT}reinforced{self.console.STD} integrator settings "
                              f"(timestep={self.console.EXT}{self._REINFORCED_TIMESTEP_FS}{self.console.STD} fs, "
                              f"friction={self.console.EXT}{self._REINFORCED_FRICTION_PER_PS}{self.console.STD} /ps)...")
                    for candidate_frame in candidate_frames:
                        attempts += 1
                        if candidate_frame != frame_idx or reinforced_pass:
                            print(f"{self.console.PGM_WRN}Centroid (frame "
                                  f"{self.console.WRN}{frame_idx}{self.console.STD}) MD attempt "
                                  f"{self.console.WRN}{attempts}{self.console.STD}: frame "
                                  f"{self.console.EXT}{candidate_frame}{self.console.STD}"
                                  f"{', reinforced' if reinforced_pass else ''}...")
                        state    = self.extract_centroid_state(merged_u, candidate_frame)
                        dcd_path = self._run_centroid_md(state, frame_idx, reinforced=reinforced_pass)
                        if dcd_path is not None:
                            source_frame_used = candidate_frame
                            reinforced_used   = reinforced_pass
                            break

                prod_dcd_files.append(dcd_path)
                if dcd_path is None:
                    print(f"{self.console.PGM_ERR}Centroid (frame "
                          f"{self.console.ERR}{frame_idx}{self.console.STD}) failed after "
                          f"{self.console.ERR}{attempts}{self.console.STD} total attempt(s) "
                          "(standard + reinforced passes over the original and all "
                          "substitute member frame(s)).")
                    centroid_records.append({'frame': frame_idx, 'status': 'failed',
                                             'cycles_before': done_cycles,
                                             'source_frame_used': None,
                                             'md_attempts': attempts,
                                             'reinforced': False})
                else:
                    tag = 'reinforced, ' if reinforced_used else ''
                    status = (f'fresh ({tag}original)' if reinforced_used and source_frame_used == frame_idx
                             else 'fresh' if source_frame_used == frame_idx
                             else f'fresh ({tag}substitute frame {source_frame_used})')
                    centroid_records.append({'frame': frame_idx, 'status': status,
                                             'cycles_before': done_cycles,
                                             'source_frame_used': source_frame_used,
                                             'md_attempts': attempts,
                                             'reinforced': reinforced_used})

            else:
                # Partially complete - append production via checkpoint
                # continuation, independent of de-excitation.
                additional_cycles = self.n_prod_cycles - done_cycles
                print(f"\n{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} (frame {self.console.WRN}{frame_idx}{self.console.STD}): extending "
                      f"from {self.console.WRN}{done_cycles}{self.console.STD} to "
                      f"{self.console.EXT}{self.n_prod_cycles}{self.console.STD} cycles...")
                dcd_path = self._extend_centroid_production(frame_idx, additional_cycles)
                prod_dcd_files.append(dcd_path)
                centroid_records.append({'frame': frame_idx,
                                         'status': f'extended (+{additional_cycles})',
                                         'cycles_before': done_cycles})

        # 5. Mode projections
        n_ok = sum(p is not None and os.path.exists(p)
                   for p in prod_dcd_files)
        print(f"\n{self.console.PGM_NAM}Computing mode projections on "
              f"{self.console.EXT}{n_ok}{self.console.STD} production trajectories...")
        unit_dcd_pairs = [(cluster['centroid'], dcd_path)
                          for cluster, dcd_path in zip(clusters, prod_dcd_files)]
        projections_df = self.compute_mode_projections(
            unit_dcd_pairs, ca_ix_full, ca_masses, M_ca,
            ref_pos_ca_ang, mode_vectors_ca,
        )
        if not projections_df.empty:
            print(f"{self.console.PGM_NAM}Total production frames projected: "
                  f"{self.console.EXT}{len(projections_df)}{self.console.STD}.")

        # 6. 1D FEL
        fel_1d = {}
        for mode_num in sorted(mode_vectors_ca.keys()):
            col  = f'mode_{mode_num}'
            proj = projections_df[col].values if col in projections_df.columns else np.array([])
            if len(proj) > 0:
                fel_1d[mode_num] = self.compute_fel_1d(proj)

        # 7. 2D FEL
        fel_2d = {}
        for m1, m2 in self.pairs_2d:
            col1, col2 = f'mode_{m1}', f'mode_{m2}'
            if (not projections_df.empty and col1 in projections_df.columns
                    and col2 in projections_df.columns):
                fel_2d[(m1, m2)] = self.compute_fel_2d(
                    projections_df[col1].values, projections_df[col2].values
                )

        # 8. All outputs
        self.generate_outputs(fel_1d, fel_2d, projections_df, clusters, centroid_records)

        print(f"\n{self.console.PGM_NAM}Free energy analysis complete in "
              f"{self.console.EXT}{time.time() - t0:.1f}{self.console.STD} s.")
