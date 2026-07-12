"""Two-stage free energy landscape protocol (Costa et al. JCTC 2015/2023)."""

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
from pyadmd.io.state import SystemState, load_reference_state, make_reference_universe
from pyadmd.io.dcd import _count_dcd_frames
from pyadmd.simulation.engine import OpenMMSimulationEngine
from pyadmd.simulation.system_builder import OpenMMSystemBuilder


class FreeEnergyCalculator:
    """
    Implements the two-stage free energy protocol.

    Protocol:
      1. Merge all replica DCD trajectories into a single pseudo-trajectory.
      2. GROMOS clustering on Cα RMSD → centroid structures.
      3. Short standard OpenMM MD per centroid: de-excitation phase (discarded)
         followed by production phase (kept).
      4. Project each production frame onto individual original mode vectors (MRMS displacement).
      5. Compute the FEL via population histogram: 1D per mode and 2D for user-specified mode pairs.

    Reference:
        Costa et al., J. Chem. Theory Comput. 2015, 11, 2395-2408.
        DOI: 10.1021/acs.jctc.5b00003
    """

    def __init__(self, console, params, args_fe):
        """
        Initialize the FreeEnergyCalculator and build the shared OpenMM system.

        Reads engine-agnostic run parameters from the saved params dict,
        resolves input file paths, loads saved reference positions/box
        when available, and constructs the OpenMM system once so it can be
        reused (via ``_build_restrained_system``) across all centroid MD runs.

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

        # Derive input-engine type from saved parameters (default NAMD for
        # backward compatibility with runs produced before -src was added).
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
            print(f"{console.PGM_WRN}Persisted reference state not found ({exc}). "
                  "Falling back to engine-specific file read for reference positions.")
            self._ref_positions_ang = None
            self._ref_box_nm        = None

        self.cutoff          = float(getattr(args_fe, 'cutoff',       0.8))
        self.n_deexcite_ps   = int(getattr(args_fe, 'deexcite',      200))
        self.n_prod_ps       = int(getattr(args_fe, 'production',    800))
        self.bins            = int(getattr(args_fe, 'bins',           50))
        self.max_centroids   = int(getattr(args_fe, 'max_centroids',  50))
        self.cluster_sel_str = str(getattr(args_fe, 'sel', 'protein and name CA'))

        # Compare against the previous freeenergy run (if any) and resolve
        # effective max_centroids/production_ps; hard-exits on a mismatched
        # selection or temperature. May overwrite self.max_centroids and
        # self.n_prod_ps, so this must run before cycle counts are derived.
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

        self.out_dir = f"{self.cwd}/freeenergy"
        os.makedirs(f"{self.out_dir}/centroids", exist_ok=True)

        # Build OpenMM system once (shared across all centroid MDs).
        # str_box priority: (1) .str file, (2) saved box vectors, (3) None.
        toppar_dir = os.path.join(self.input_dir, "charmm_toppar")
        str_box = None
        if self.strfile and os.path.exists(self.strfile):
            try:
                str_box = NAMDInputReader.parse_str_box(self.strfile)
            except Exception as exc:
                print(f"{console.PGM_WRN}Could not parse STR box ({exc}); "
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
                      f"({exc}); using placeholder.")
        builder = OpenMMSystemBuilder(console)
        self._psf_omm, self._omm_system, _ = builder.build(
            self.psffile, toppar_dir, temperature=self._temperature, str_box=str_box,
        )

    # Saving and reading parameters metadata

    def _run_metadata_path(self) -> str:
        """Path to the saved run-parameter record (may not exist yet)."""
        return f"{self.cwd}/freeenergy/run_metadata.json"

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
                  f"({exc}); treating this as a first run.")
            return None

    def _save_run_metadata(self) -> None:
        """Save the effective parameters for this run."""
        os.makedirs(f"{self.cwd}/freeenergy", exist_ok=True)
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
        Compare this call's parameters against the previous ``freeenergy``
        run (if any) and resolve the effective values.

        - ``-s/--sel`` and ``-T/--temp`` are hard-gated: a mismatch prints a
          clear diff and exits, since mixing selections or temperatures in
          one pooled free energy landscape is not physically valid.
        - ``--max_centroids`` and ``-p/--production`` are soft-gated:
          existing work is never shrunk, so the larger of the previous and
          current value is used, with a warning if the requested value was
          smaller.
        - ``-c/--cutoff`` and ``-d/--deexcite`` are informational only: a
          note is printed if they changed, but nothing is blocked or
          overridden (cutoff only affects the re-thresholding of the cached
          RMSD matrix; deexcite only affects newly-created centroids going
          forward).

        On a first run (no previous metadata), simply save the current
        parameters and returns. Otherwise save the resolved parameters
        at the end.

        Raises:
            SystemExit: If ``-s/--sel`` or ``-T/--temp`` differ from the
                previous run.
        """
        prev = self._load_run_metadata()
        self._is_append_run = prev is not None   # used only for reporting
        if prev is None:
            self._save_run_metadata()
            return

        # Selection and temperature must match exactly (temperature compared
        # with a small tolerance to survive JSON round-tripping of the float).
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
                  f"{self.console.ERR}freeenergy/run_metadata.json"
                  f"{self.console.STD}:")
            for m in mismatches:
                print(f"{self.console.PGM_ERR}  {m}")
            print(f"{self.console.PGM_ERR}Mixing different selections or "
                  "temperatures inside one pooled free energy landscape is "
                  "not physically valid. Remove or rename the existing "
                  f"{self.console.ERR}freeenergy/{self.console.STD} "
                  "directory to start a fresh calculation with the new "
                  "parameters.")
            sys.exit(1)

        # Never shrink already-completed work
        prev_max_centroids = int(prev.get('max_centroids', self.max_centroids))
        if self.max_centroids < prev_max_centroids:
            print(f"{self.console.PGM_WRN}--max_centroids="
                  f"{self.max_centroids} is smaller than the previous run's "
                  f"{prev_max_centroids}. Existing centroids are never "
                  f"discarded; using {self.console.EXT}{prev_max_centroids}"
                  f"{self.console.STD}.")
            self.max_centroids = prev_max_centroids

        prev_production_ps = int(prev.get('production_ps', self.n_prod_ps))
        if self.n_prod_ps < prev_production_ps:
            print(f"{self.console.PGM_WRN}-p/--production={self.n_prod_ps} "
                  f"is smaller than the previous run's {prev_production_ps} "
                  f"ps. Existing production trajectories are never "
                  f"truncated; using {self.console.EXT}{prev_production_ps}"
                  f"{self.console.STD} ps.")
            self.n_prod_ps = prev_production_ps

        # Informational only: no gating nor overriding
        prev_cutoff = prev.get('cutoff')
        if prev_cutoff is not None and not math.isclose(
                prev_cutoff, self.cutoff, rel_tol=1e-9, abs_tol=1e-9):
            print(f"{self.console.PGM_NAM}Note: -c/--cutoff changed from "
                  f"{prev_cutoff} to {self.cutoff} Å since the previous "
                  "run. The cached pairwise-RMSD matrix is still reused; "
                  "clusters are simply re-thresholded with the new cutoff.")
        prev_deexcite = prev.get('deexcite_ps')
        if prev_deexcite is not None and prev_deexcite != self.n_deexcite_ps:
            print(f"{self.console.PGM_NAM}Note: -d/--deexcite changed from "
                  f"{prev_deexcite} to {self.n_deexcite_ps} ps since the "
                  "previous run. This only affects newly-created centroids; "
                  "existing centroids keep their original de-excitation and "
                  "are simply extended in production.")

        self._save_run_metadata()

    # Step 1: merge trajectories

    def merge_trajectories(self):
        """
        Concatenate all replica DCD files into a single MDAnalysis Universe.

        Scans ``rep1`` through ``rep{self.replicas}`` for a ``rep{N}.dcd``
        trajectory file, skipping any replica whose DCD is missing, and loads
        the found files as frames of a single merged pseudo-trajectory.

        Returns:
            MDAnalysis.Universe: Universe built from the PSF topology and the
                concatenated replica DCD files.

        Raises:
            FileNotFoundError: If no replica DCD files are found.
        """
        dcd_files = []
        for rep in range(1, self.replicas + 1):
            dcd = f"{self.cwd}/rep{rep}/rep{rep}.dcd"
            if os.path.exists(dcd):
                dcd_files.append(dcd)
            else:
                print(f"{self.console.PGM_WRN}DCD not found for replica {rep}, skipping.")
        if not dcd_files:
            raise FileNotFoundError("No DCD trajectory files found in any replica directory.")
        print(f"{self.console.PGM_NAM}Merging {self.console.EXT}{len(dcd_files)}"
              f"{self.console.STD} replica DCD files...")
        u = mda.Universe(self.psffile, dcd_files, format="DCD")
        print(f"{self.console.PGM_NAM}Pseudo-trajectory: "
              f"{self.console.EXT}{len(u.trajectory)}{self.console.STD} frames total.")
        return u

    # Step 2: GROMOS clustering

    def cluster_gromos(self, merged_u):
        """
        Cluster frames using the GROMOS algorithm on Cα RMSD.

        This is a thin orchestrator over three stages, split so that the
        expensive part (the pairwise RMSD matrix) can be cached and reused
        across ``freeenergy`` invocations even when ``--cutoff`` or
        ``--max_centroids`` change:

          1. ``_get_or_build_rmsd_matrix``: reuse the cached pairwise RMSD
             matrix when valid for the current clustering selection and
             merged-trajectory frame count, otherwise compute and cache it.
          2. ``_gromos_threshold``: neighbor-counting/greedy-pick
             clustering over the (cached or fresh) matrix, always re-run
             with the *current* ``self.cutoff``.
          3. ``_select_diverse_centroids``: greedy farthest-point (MaxMin)
             selection, always re-run with the *current* ``self.max_centroids``,
             only if needed.

        Args:
            merged_u (MDAnalysis.Universe): Merged pseudo-trajectory produced
                by ``merge_trajectories``.

        Returns:
            list[dict]: One dict per cluster, each containing:
                - centroid (int): Original frame index in merged_u.
                - size (int): Number of subsampled members.
                - members (list): Subsampled indices (pool management only).
                - _sampled_idx (int): Subsampled index (used by MaxMin selection).
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
                  f"max_centroids={self.max_centroids}. Selecting "
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

        Contains no distance computation so it is inexpensive to re-run on every
        ``freeenergy`` call with whatever ``self.cutoff`` is currently set, even when
        the RMSD matrix itself came from a cache built under a different cutoff.

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
                ``centroid``, ``_sampled_idx``, ``size``, ``members``.
        """
        n_sampled = len(frame_indices)
        pool      = list(range(n_sampled))
        clusters  = []
        while pool:
            pool_arr         = np.array(pool, dtype=np.int32)
            sub_rmsd         = rmsd_matrix[np.ix_(pool_arr, pool_arr)]
            neighbor_counts  = np.sum(sub_rmsd < self.cutoff, axis=1)
            best_local       = int(np.argmax(neighbor_counts))
            centroid_sampled = int(pool_arr[best_local])
            centroid_global  = int(frame_indices[centroid_sampled])
            member_mask      = sub_rmsd[best_local] < self.cutoff
            members          = pool_arr[member_mask].tolist()
            clusters.append({
                'centroid':     centroid_global,
                '_sampled_idx': centroid_sampled,   # kept for MaxMin selection
                'size':         len(members),
                'members':      [int(m) for m in members],
            })
            pool = [f for f in pool if f not in set(members)]

        clusters.sort(key=lambda c: c['size'], reverse=True)
        return clusters

    def _get_or_build_rmsd_matrix(self, merged_u):
        """
        Return the pairwise RMSD matrix over subsampled frames, reusing a
        cached one when it is still valid, otherwise computing and caching
        a fresh one.

        The matrix depends only on the clustering selection
        (``self.cluster_sel_str``), the subsampling stride
        (``self._CLUSTER_STRIDE``), and the set of frames in the merged
        pseudo-trajectory — it does **not** depend on ``--cutoff`` or
        ``--max_centroids``, both of which are applied afterwards on the
        cheap thresholding/selection stages. This lets ``--cutoff`` change
        between ``freeenergy`` calls without repeating the RMSD computation.

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

        frame_indices = np.arange(0, n_frames, self._CLUSTER_STRIDE)
        n_sampled     = len(frame_indices)

        print(f"{self.console.PGM_NAM}Accumulating positions: "
              f"{self.console.EXT}{n_sampled}{self.console.STD} frames "
              f"(every {self._CLUSTER_STRIDE} of "
              f"{self.console.EXT}{n_frames}{self.console.STD} total, "
              f"{self.console.EXT}{n_sel}{self.console.STD} atoms)...")
        positions = np.empty((n_sampled, n_sel, 3), dtype=np.float32)
        for i, orig_idx in enumerate(frame_indices):
            merged_u.trajectory[orig_idx]
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

        Validity is intentionally independent of ``--cutoff`` and
        ``--max_centroids`` (see ``_get_or_build_rmsd_matrix``).

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
                  f"({exc}); recomputing.")
            return None

        if meta.get('cluster_sel') != self.cluster_sel_str:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix was built with a "
                  f"different clustering selection ('{meta.get('cluster_sel')}' "
                  f"vs current '{self.cluster_sel_str}'); recomputing.")
            return None
        if meta.get('n_merged_frames') != n_merged_frames:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix was built from "
                  f"{meta.get('n_merged_frames')} merged frames, but "
                  f"{n_merged_frames} are present now (replica DCDs changed); "
                  "recomputing.")
            return None
        if meta.get('stride') != self._CLUSTER_STRIDE:
            print(f"{self.console.PGM_WRN}Cached RMSD matrix used a different "
                  f"subsampling stride ({meta.get('stride')} vs "
                  f"{self._CLUSTER_STRIDE}); recomputing.")
            return None

        try:
            data = np.load(npz_path)
            rmsd_matrix   = data['rmsd_matrix']
            frame_indices = data['frame_indices']
        except Exception as exc:
            print(f"{self.console.PGM_WRN}Could not load cached RMSD matrix "
                  f"({exc}); recomputing.")
            return None

        print(f"{self.console.PGM_NAM}Reusing cached pairwise RMSD matrix "
              f"({self.console.EXT}{meta.get('n_sampled')}"
              f"{self.console.STD}×{self.console.EXT}{meta.get('n_sampled')}"
              f"{self.console.STD}, saved {meta.get('timestamp', '?')}); "
              "skipping recomputation.")
        return rmsd_matrix, frame_indices

    def _save_rmsd_cache(self, rmsd_matrix, frame_indices, n_merged_frames):
        """
        Save the pairwise RMSD matrix and its metadata so future
        ``freeenergy`` calls with a different ``--cutoff`` or
        ``--max_centroids`` can skip the RMSD re-computation.

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

    def _select_diverse_centroids(self, clusters, rmsd_matrix, max_n):
        """
        Select ``max_n`` maximally diverse centroids from a larger cluster list
        using greedy farthest-point (MaxMin) sampling.

        The algorithm seeds with the most-populated centroid (index 0 in the
        size-sorted list) and then iteratively adds the centroid whose minimum
        RMSD distance to all already-selected centroids is largest.  This
        approximates the optimal max-min diverse subset to within a factor of 2.

        After selection the returned list is re-sorted by cluster size (largest
        first) so that downstream centroid MD follows population order.

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

        # Greedy MaxMin — O(n × max_n)
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
        (GEMM) formulation, GPU-accelerated via CuPy with an automatic
        CPU/BLAS fallback.

        Because this RMSD has no Kabsch superposition, it reduces to a
        scaled Euclidean distance between flattened per-frame coordinate
        vectors:

            RMSD_ij = || flat(pos_i) - flat(pos_j) ||_2 / sqrt(n_atoms)

        and the full squared-distance matrix can be obtained from a single
        matrix multiplication per row-batch (||a-b||^2 = ||a||^2 + ||b||^2
        - 2 a.b) instead of an O(n^2) elementwise loop. This lets the heavy
        lifting run as batched GEMM calls, which cuBLAS/BLAS parallelize
        far more efficiently than the previous nested-loop broadcasting.

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
                  f"unavailable ({exc}); falling back to CPU (BLAS).")
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

        Positions come from the frame; velocities are None so that
        initialize_state() assigns Maxwell-Boltzmann velocities at the
        simulation temperature. The periodic box is taken from the DCD
        frame with a fallback to the original XSC file.

        Replica DCDs are written with ``enforcePeriodicBox=False``, so atoms
        can drift arbitrarily far from the primary unit cell over the course
        of a multi-ns excitation run. Whole molecules (fragments, determined
        from PSF bonds) are wrapped back into the box before positions are
        extracted, which keeps absolute coordinates numerically well-behaved
        for the centroid MD that follows.

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
        (5.0,   2.5  ),   # phase 1 — heavy restraint
        (2.5,   1.125),   # phase 2
        (1.0,   0.25 ),   # phase 3
        (0.1,   0.0  ),   # phase 4 — nearly free
    ]

    # 1 kcal/mol/Å² → kJ/mol/nm²  (OpenMM internal units)
    _KCAL_A2_TO_KJ_NM2: float = 418.4

    # Frame stride used when accumulating positions for GROMOS clustering
    # Every _CLUSTER_STRIDE-th frame is kept, reducing the RMSD matrix by
    # the stride² without significant loss of conformational coverage
    _CLUSTER_STRIDE: int = 2

    # Exact production-end checkpoint
    _PROD_CHECKPOINT_FILE: str = "prod_checkpoint.chk"

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

        * ``k_bb`` — applied to protein backbone heavy atoms (CA, C, N, O, OXT).
        * ``k_sc`` — applied to protein sidechain heavy atoms.

        Both global parameters are initialised to **zero**.  Update them via
        ``context.setParameter("k_bb", value)`` before each de-excitation phase.
        Force constants are in kJ/mol/nm² (OpenMM internal units).

        Converting user-facing kcal/mol/Å² to kJ/mol/nm²:
            k_internal = k_user × 418.4

        The reference positions (``ref_pos_nm``) are the centroid atom
        coordinates in nm and serve as the equilibrium positions for the
        harmonic restraints.

        Args:
            ref_pos_nm: (N_atoms, 3) array of centroid positions in nm.

        Returns:
            A new ``mm.System`` with the two restraint forces appended.
        """

        # Independent copy — leaves self._omm_system untouched
        system_copy = XmlSerializer.deserialize(
            XmlSerializer.serialize(self._omm_system)
        )

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

        Keying by the centroid's merged-trajectory frame index keeps identity
        stable across ``freeenergy`` calls even when ``--max_centroids`` or
        ``--cutoff`` change and reshuffle that ordering.

        Args:
            frame_idx (int): Centroid frame index in the merged
                pseudo-trajectory (``cluster['centroid']``).

        Returns:
            str: Path to ``freeenergy/centroids/centroid_frame{frame_idx}``.
        """
        return f"{self.out_dir}/centroids/centroid_frame{frame_idx}"

    def _centroid_prod_dcd_path(self, frame_idx: int) -> str:
        """
        Return the path to a centroid's production DCD.

        Args:
            frame_idx (int): Centroid frame index in the merged
                pseudo-trajectory (``cluster['centroid']``).

        Returns:
            str: Path to ``{centroid_dir}/prod.dcd``.
        """
        return f"{self._centroid_dir(frame_idx)}/prod.dcd"

    def _run_centroid_md(self, centroid_state: 'SystemState',
                         frame_idx: int) -> Optional[str]:
        """
        Run 4-phase restrained de-excitation followed by unrestrained production
        MD from a single centroid structure.

        De-excitation protocol:
          Phase 1 — k_bb = 5.0, k_sc = 2.5  kcal/mol/Å²
          Phase 2 — k_bb = 2.5, k_sc = 1.125 kcal/mol/Å²
          Phase 3 — k_bb = 1.0, k_sc = 0.25  kcal/mol/Å²
          Phase 4 — k_bb = 0.1, k_sc = 0.0   kcal/mol/Å²

        Each phase spans (n_deexcite_ps / 4) ps. Restraint reference positions
        are the centroid coordinates themselves. No DCD frames are written
        during de-excitation.

        After de-excitation, the final positions/velocities/box are carried
        over into a fresh ``Simulation`` built directly from the shared,
        restraint-free ``self._omm_system``. A  checkpoint (``prod_checkpoint.chk``)
        is saved immediately after production stepping ends, enabling later appending
        via ``_extend_centroid_production`` if a subsequent ``freeenergy`` call
        requests a longer production time.

        Args:
            centroid_state: SystemState with centroid positions and box.
            frame_idx: Centroid's merged-trajectory frame index — the
                stable identifier used to name its output directory.

        Returns:
            Absolute path to the production DCD, or None on failure.
        """
        centroid_dir  = self._centroid_dir(frame_idx)
        os.makedirs(centroid_dir, exist_ok=True)
        prev_dir      = os.getcwd()
        os.chdir(centroid_dir)
        prod_dcd_name = "prod.dcd"

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
            )
            # initialize_state assigns MB velocities when velocities_nm_ps is None
            engine.initialize_state(centroid_state)

            # Detach the main DCD reporter immediately (no frames during de-excitation)
            engine.detach_main_dcd()

            # Restrained Energy minimization
            # Excited-trajectory frames can contain atom clashes or atoms
            # displaced far from equilibrium, causing NaN forces when MD begins.
            # A brief energy minimization with the maximum restraints applied
            # removes these clashes while keeping the structure near the centroid
            # geometry. Velocities are then reset to Maxwell-Boltzmann at the
            # target temperature because minimization does not update them.
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

                print(f"{self.console.PGM_NAM}De-excitation phase {self.console.WRN}{phase_idx + 1}{self.console.EXT}/{self.console.EXT}4{self.console.STD}: "
                      f"k_bb={self.console.EXT}{k_bb_kcal:.3f}{self.console.STD}, k_sc={self.console.EXT}{k_sc_kcal:.3f}{self.console.STD} kcal/mol/Å² "
                      f"({self.console.EXT}{phase_ps:.1f}{self.console.STD} ps)...")
                engine.simulation.step(phase_cycles * self.n_steps)

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
            )
            prod_engine.initialize_state(production_state)

            # No frames from the (unused) default rep{N}.dcd for this engine;
            # production frames go to the dedicated prod.dcd below (not an elegant solution).
            prod_engine.detach_main_dcd()

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
            print(f"{self.console.PGM_ERR}Centroid (frame {frame_idx}) MD failed: "
                  f"{self.console.ERR}{exc}{self.console.STD}")
            traceback.print_exc()
            return None
        finally:
            os.chdir(prev_dir)

    def _load_last_frame_as_state(self, dcd_filename: str) -> 'SystemState':
        """
        Build a SystemState (positions + box, no velocities) from the last
        frame of a centroid's own production DCD.

        Fallback used by ``_extend_centroid_production`` when no
        ``prod_checkpoint.chk`` is available (e.g. a centroid produced
        before this feature existed). Velocities are intentionally left as
        ``None`` so that ``initialize_state()`` assigns Maxwell-Boltzmann
        velocities at ``self._temperature``. The appending is then
        physically valid MD, just not bit-identical to what a checkpoint
        would give.

        Whole molecules are wrapped back into the primary cell first: like
        the excited-replica DCDs, the production DCD is written with
        ``enforcePeriodicBox=False``, so atoms can drift outside the box
        over a long production run (same reasoning as
        ``extract_centroid_state``).

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
        ``prod.dcd``.

        Continuation state is restored, in order of preference:
          1. ``prod_checkpoint.chk`` — the exact positions/velocities/box/
             RNG state saved at the end of this centroid's last production
             run (by ``_run_centroid_md`` or a previous call to this
             method). Gives a bit-identical continuation.
          2. Last frame of ``prod.dcd`` (via ``_load_last_frame_as_state``)
             — used only when no checkpoint is available. Positions and box
             only; velocities are re-assigned at Maxwell-Boltzmann.
             Appending is then physically valid but not bit-identical.

        Args:
            frame_idx: Centroid's merged-trajectory frame index — the
                stable identifier used to locate its output directory.
            additional_cycles: Number of additional ``n_steps``-step cycles
                to run.

        Returns:
            Absolute path to the (extended) production DCD, or None on
            failure.
        """
        centroid_dir  = self._centroid_dir(frame_idx)
        prod_dcd_path = self._centroid_prod_dcd_path(frame_idx)
        prev_dir      = os.getcwd()
        os.chdir(centroid_dir)

        try:
            engine = OpenMMSimulationEngine(
                self.console, self._psf_omm, self._omm_system,
                self._temperature,
                rep_num=frame_idx,
                is_restart=True, full_ener=False, n_steps=self.n_steps,
            )
            # No frames into the (unused) default rep{N}.dcd;
            # appended frames go to the dedicated prod.dcd below (not an elegant solution).
            engine.detach_main_dcd()

            if os.path.exists(self._PROD_CHECKPOINT_FILE):
                print(f"{self.console.PGM_NAM}Resuming centroid (frame "
                      f"{frame_idx}) from {self.console.EXT}"
                      f"{self._PROD_CHECKPOINT_FILE}{self.console.STD} "
                      "(bit-identical continuation).")
                engine.load_checkpoint(self._PROD_CHECKPOINT_FILE)
            else:
                print(f"{self.console.PGM_WRN}No production checkpoint "
                      f"found for centroid (frame {self.console.WRN}{frame_idx}{self.console.STD}); falling "
                      f"back to the last frame of {self.console.WRN}"
                      f"prod.dcd{self.console.STD}.")
                fallback_state = self._load_last_frame_as_state("prod.dcd")
                engine.initialize_state(fallback_state)

            engine.simulation.reporters.append(
                app.DCDReporter("prod.dcd", self.n_steps,
                                append=True, enforcePeriodicBox=False)
            )

            additional_ps = additional_cycles * self.n_steps * 0.002
            print(f"{self.console.PGM_NAM}Extending production for "
                  f"centroid (frame {self.console.EXT}{frame_idx}{self.console.STD}) by "
                  f"{self.console.EXT}{additional_ps:.1f}{self.console.STD} "
                  f"ps ({additional_cycles} cycles)...")
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

        Reference positions are taken from the saved
        ``init_reference_positions_ang.npy`` (written at ``run`` time) when
        available. This makes the FEL projection engine-agnostic: both NAMD
        and OpenMM input paths produce the same reference file, so no
        NAMD-binary read is required here. A NAMD-binary fallback is retained
        for backward compatibility with runs that pre-date the saved
        reference state feature.

        Mode vector files contain one entry per protein atom in PSF order
        (written by write_nm_vectors / wrt-nm.mdu). The Cα component is
        extracted by mapping global Cα indices to positions within the
        protein-only ordering.

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
        # Build reference Universe from saved positions if available,
        # otherwise fall back to engine-specific file read.
        if self._ref_positions_ang is not None:
            u_ref = make_reference_universe(self.psffile, self._ref_positions_ang)
        elif self._input_engine == 'NAMD' and self.coorfile:
            u_ref = mda.Universe(self.psffile, self.coorfile, format='NAMDBIN')
        else:
            raise RuntimeError(
                "Cannot load reference positions for FEL projection: "
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

        # Map global Cα index → position within protein-only ordering
        prot_to_pos   = {int(gix): pos for pos, gix in enumerate(prot_atoms.ix)}
        ca_ix_in_prot = np.array([prot_to_pos[int(gix)] for gix in ca_ix_full])

        mode_vectors_ca = {}
        for mode_num in self.fe_modes:
            try:
                vec_full = self._load_single_mode_vector(mode_num)  # (n_prot, 3) Å
                q_ca     = vec_full[ca_ix_in_prot]                  # (n_ca, 3)
                norm     = np.linalg.norm(q_ca)
                if norm < 1e-10:
                    print(f"{self.console.PGM_WRN}Mode {self.console.WRN}{mode_num}{self.console.STD} Cα vector is "
                          "near-zero after extraction; skipping.")
                    continue
                mode_vectors_ca[mode_num] = q_ca / norm
            except FileNotFoundError as exc:
                print(f"{self.console.PGM_WRN}Mode file not found for mode "
                      f"{self.console.WRN}{mode_num}{self.console.STD}: {exc}. Skipping.")

        if not mode_vectors_ca:
            raise RuntimeError("No valid mode vectors could be loaded for FEL projection.")

        return ca_ix_full, ca_masses, M_ca, ref_pos_ca_ang, mode_vectors_ca

    def _load_single_mode_vector(self, mode_num):
        """
        Load a mode vector file and return per-atom Cartesian displacements.

        Args:
            mode_num (int): Mode number to load.

        Returns:
            numpy.ndarray: (n_prot_atoms, 3) mode vector positions in Å.

        Raises:
            FileNotFoundError: If the mode vector file does not exist.
        """
        if self.nm_type == 'charmm':
            path = f"{self.input_dir}/mode_nm{mode_num}.crd"
            u    = mda.Universe(path, format="CRD")
        else:
            # base_name is derived from the PSF filename
            base_name = os.path.splitext(os.path.basename(self.psffile))[0]
            # Recover the actual base_name used at ENM-generation time from the saved
            # coorfile (NAMD) or rstfile (OpenMM) path
            if self._input_engine == 'NAMD' and self.coorfile:
                base_name = os.path.splitext(os.path.basename(self.coorfile))[0]
            elif self._input_engine != 'NAMD' and self.rstfile:
                base_name = os.path.splitext(os.path.basename(self.rstfile))[0]
            prefix = "ca" if self.nm_type == 'ca' else "heavy"
            path   = (f"{self.input_dir}/{base_name}_enm/"
                      f"{base_name}_{prefix}_mode_{mode_num}.xyz")
            u      = mda.Universe(path, format="XYZ")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return u.atoms.positions.copy()

    def compute_mode_projections(self, prod_dcd_files, ca_ix_full, ca_masses,
                                 M_ca, ref_pos_ca_ang, mode_vectors_ca):
        """
        Compute signed MRMS displacement of every production frame along each
        individual mode vector:

            d_j = (1/√M) Σ_i √m_i · (r_i − r₀ᵢ) · q_{ij}

        Sign is preserved so that FEL plots distinguish both directions.

        Args:
            prod_dcd_files (list[str]): Paths to per-centroid production DCD
                files (entries may be None for failed centroid MD runs).
            ca_ix_full (numpy.ndarray): (n_ca,) global Cα atom indices.
            ca_masses (numpy.ndarray): (n_ca,) Cα atomic masses in amu.
            M_ca (float): Total Cα mass.
            ref_pos_ca_ang (numpy.ndarray): (n_ca, 3) reference Cα positions
                in Å.
            mode_vectors_ca (dict): {mode_num: (n_ca, 3) normalised mode
                vector}.

        Returns:
            dict: {mode_num: numpy.ndarray (n_frames_total,)} signed MRMS
                displacements in Å.
        """
        projections = {m: [] for m in mode_vectors_ca}
        sqrt_M_ca   = float(np.sqrt(M_ca))
        sqrt_masses = np.sqrt(ca_masses)          # (n_ca,) pre-computed

        for dcd_file in prod_dcd_files:
            if dcd_file is None or not os.path.exists(dcd_file):
                continue
            u = mda.Universe(self.psffile, dcd_file, format="DCD")
            for ts in u.trajectory:
                curr_ca = u.atoms.positions[ca_ix_full]        # (n_ca, 3) Å
                disp    = curr_ca - ref_pos_ca_ang              # (n_ca, 3) Å
                mw_disp = (disp.T * sqrt_masses).T             # mass-weighted
                for mode_num, q_ca in mode_vectors_ca.items():
                    mrms = float(np.sum(mw_disp * q_ca)) / sqrt_M_ca
                    projections[mode_num].append(mrms)

        return {k: np.array(v) for k, v in projections.items()}

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

    def generate_outputs(self, fel_1d, fel_2d, projections, clusters,
                        centroid_records=None):
        """
        Write clustering CSV, projection .npy files, plots, and HTML summary.

        Args:
            fel_1d (dict): {mode_num: (bin_centers, delta_G)} from
                compute_fel_1d.
            fel_2d (dict): {(mode1, mode2): (xc, yc, delta_G)} from
                compute_fel_2d.
            projections (dict): {mode_num: numpy.ndarray} mode projections
                from compute_mode_projections.
            clusters (list[dict]): Cluster list returned by ``cluster_gromos``.
            centroid_records (list[dict], optional): Per-centroid status
                collected during ``run()``'s centroid MD loop (keys
                ``frame``, ``status``, ``cycles_before``), used to report
                which centroids were fresh, extended, or skipped this call.
        """
        self._save_clustering_summary(clusters, centroid_records)

        for mode_num, (centers, dG) in fel_1d.items():
            np.save(f"{self.out_dir}/projections_mode{mode_num}.npy",
                    projections[mode_num])
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

        ``production_cycles_done``/``production_ps_done`` are queried
        post-hoc via ``_centroid_done_cycles`` rather than tracked through
        the run loop, so they reflect the true on-disk state regardless of
        whether a centroid's MD succeeded or failed this call.

        Args:
            clusters (list[dict]): Cluster list returned by ``cluster_gromos``.
            centroid_records (list[dict], optional): Per-centroid status
                from ``run()`` (keys ``frame``, ``status``). When omitted,
                ``status`` is reported as ``'n/a'`` (e.g. when this is
                called outside the normal ``run()`` flow).
        """
        status_by_frame = {r['frame']: r['status'] for r in (centroid_records or [])}
        rows = []
        for i, c in enumerate(clusters):
            frame = c['centroid']
            done_cycles = self._centroid_done_cycles(frame)
            rows.append({
                'cluster_id':               i + 1,
                'centroid_frame':           frame,
                'size':                     c['size'],
                'status':                   status_by_frame.get(frame, 'n/a'),
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

        One DCD frame corresponds to exactly one production cycle (the
        DCDReporter period equals ``self.n_steps``), consistent with the
        DCD-header-driven crash-recovery convention used elsewhere in this
        module (e.g. ``_count_dcd_frames``, ``find_last_completed_cycle``).

        Args:
            frame_idx: Centroid's merged-trajectory frame index — the
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

        Note this is a target-relative check, not merely "has some
        completed production": once a later ``freeenergy`` call raises
        ``-p/--production``, a centroid that was previously "complete" can
        become incomplete again, signaling that it needs to be topped up
        via ``_extend_centroid_production`` rather than treated as done.

        Args:
            frame_idx: Centroid's merged-trajectory frame index — the
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

        Note:
            Three mechanisms let ``run`` be safely re-invoked after a
            partial failure or with different (larger) parameters:

            - RMSD-matrix cache: ``freeenergy/clustering_rmsd_cache.npz``
              (+ a small ``.json`` metadata sidecar) is written immediately
              after the pairwise RMSD matrix is computed. On re-entry, if
              it is still valid for the current clustering selection and
              merged-trajectory frame count, the O(n²) RMSD computation is
              skipped entirely and only the cheap GROMOS thresholding and
              MaxMin selection are re-run — with whatever ``--cutoff`` and
              ``--max_centroids`` are passed in the current call. Delete
              the cache files to force full recomputation (e.g. after
              changing ``--sel``, or adding/removing replica DCDs).
            - Centroid identity/completion: each centroid is keyed by its
              stable merged-trajectory frame index (``centroid_frame{F}/``,
              see ``_centroid_dir``), not by its position in the current
              call's selection, so identity survives ``--cutoff``/
              ``--max_centroids`` changes. Completion is target-relative
              (``_centroid_is_complete``): a centroid whose production
              already meets the current ``-p/--production`` target is
              skipped untouched.
            - Checkpoint-based extension: a centroid with *some* but
              insufficient production is topped up by
              ``_extend_centroid_production``, which resumes from an exact
              ``prod_checkpoint.chk`` (bit-identical) or, failing that,
              the last frame of its ``prod.dcd`` (physically valid, not
              bit-identical), and appends the additional cycles to the
              existing production trajectory. A centroid with no
              production yet runs fresh via ``_run_centroid_md`` for the
              *full* current target.
        """
        t0 = time.time()

        # 1. Merge trajectories
        merged_u = self.merge_trajectories()

        # 2. Cluster — GROMOS + MaxMin
        clusters = self.cluster_gromos(merged_u)

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
              f"({n_centroids - n_pending} already complete, target "
              f"{self.n_prod_cycles} cycles / {self.n_prod_ps} ps).")

        for i, cluster in enumerate(clusters):
            display_idx = i + 1                # display-only; no on-disk meaning
            frame_idx   = cluster['centroid']   # stable identity for all paths
            now         = time.strftime("%H:%M:%S")
            done_cycles = self._centroid_done_cycles(frame_idx)

            if done_cycles >= self.n_prod_cycles:
                # Already meets (or exceeds) the current target — reuse as-is.
                prod_dcd = self._centroid_prod_dcd_path(frame_idx)
                print(f"{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} (frame {frame_idx}): already "
                      f"complete ({done_cycles}/{self.n_prod_cycles} "
                      "cycles), skipping.")
                prod_dcd_files.append(prod_dcd)
                centroid_records.append({'frame': frame_idx, 'status': 'skipped',
                                         'cycles_before': done_cycles})

            elif done_cycles == 0:
                # Brand-new centroid: full de-excitation + full target production.
                print(f"\n{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} "
                      f"(frame {self.console.EXT}{frame_idx}{self.console.STD}, "
                      f"cluster size {self.console.WRN}{cluster['size']}{self.console.STD})...")
                state    = self.extract_centroid_state(merged_u, frame_idx)
                dcd_path = self._run_centroid_md(state, frame_idx)
                prod_dcd_files.append(dcd_path)
                centroid_records.append({'frame': frame_idx, 'status': 'fresh',
                                         'cycles_before': done_cycles})

            else:
                # Partially complete — append production via checkpoint
                # continuation, independent of de-excitation.
                additional_cycles = self.n_prod_cycles - done_cycles
                print(f"\n{self.console.PGM_NAM}{now} Centroid "
                      f"{self.console.WRN}{display_idx}{self.console.STD}/{self.console.EXT}{n_centroids}"
                      f"{self.console.STD} (frame {frame_idx}): extending "
                      f"from {done_cycles} to {self.n_prod_cycles} cycles...")
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
        projections = self.compute_mode_projections(
            prod_dcd_files, ca_ix_full, ca_masses, M_ca,
            ref_pos_ca_ang, mode_vectors_ca,
        )
        if projections:
            n_proj = len(next(iter(projections.values())))
            print(f"{self.console.PGM_NAM}Total production frames projected: "
                  f"{self.console.EXT}{n_proj}{self.console.STD}.")

        # 6. 1D FEL
        fel_1d = {}
        for mode_num, proj in projections.items():
            if len(proj) > 0:
                fel_1d[mode_num] = self.compute_fel_1d(proj)

        # 7. 2D FEL
        fel_2d = {}
        for m1, m2 in self.pairs_2d:
            if m1 in projections and m2 in projections:
                fel_2d[(m1, m2)] = self.compute_fel_2d(
                    projections[m1], projections[m2]
                )

        # 8. All outputs
        self.generate_outputs(fel_1d, fel_2d, projections, clusters, centroid_records)

        print(f"\n{self.console.PGM_NAM}Free energy analysis complete in "
              f"{self.console.EXT}{time.time() - t0:.1f}{self.console.STD} s.")
