"""
The Adaptive Molecular Dynamics with Excited Normal Modes (aMDeNM) method applies a kinetic excitation of normal modes (NMs)
to enhance molecular dynamics simulations sampling. This technique consists in injecting additional atomic velocities along
a combinations of NM vectors, creating an effective coupling between slow and fast molecular motions. The motions described
by preselected directions of low-frequency NMs are dynamically adjusted throughout the simulation. By coupling low-frequency
NM excitation with adaptive directional adjustments, aMDeNM facilitates extensive exploration of the energy landscape,
overcoming the constraints of fixed, rectilinear displacements and alleviating structural stresses and environmental resistance.
Importantly, aMDeNM requires only an initial structure without the need to specify predefined target states, distinguishing
it from many biased sampling techniques that rely on predefined target conformations.
"""

import argparse
import os
import zipfile
import subprocess
import shutil
import sys
import time
import csv
import json
import glob
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import seaborn as sns
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import math
import re
import xml.etree.ElementTree as ET
import struct

# Ignore MDAnalysis attribute warnings during ENM computation
warnings.filterwarnings("ignore", module='MDAnalysis')
# Ignore Bio deprecation warnings from MDAnalysis calls
warnings.filterwarnings("ignore", module='Bio')

# Third-party imports
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh
    from scipy.sparse import diags, coo_array, issparse
    from scipy.sparse.linalg import eigsh
    import cupy as cp
    import numba
    from numba import float64, int32

    # Bio/chemistry-specific imports
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from MDAnalysis.analysis.align import rotation_matrix as mda_rotation_matrix
    from MDAnalysis.coordinates.memory import MemoryReader
    from MDAnalysis.lib.mdamath import triclinic_vectors
    from Bio.PDB import PDBParser, ShrakeRupley
    import openmm as mm
    from openmm import app, unit, Platform, XmlSerializer
    from openmm.app import CharmmPsfFile, CharmmParameterSet
except ImportError as e:
    print(f"Required libraries not found: {e}")
    sys.exit(1)

# UNIT-CONVERSION CONSTANTS (NAMD to OpenMM)
# NAMD .vel binary files store velocities in AKMA units:
AKMA_VEL_TO_NM_PS: float = 2.04548          # AKMA to nm/ps  (into OpenMM Context)
NM_PS_TO_AKMA_VEL: float = 1.0 / 2.04548    # nm/ps to AKMA  (out of OpenMM Context)


class ConsoleConfig:
    """
    Configuration class for PyAdMD application providing console styling and messages.

    This class contains ANSI escape codes for console text styling, formatted program
    output prefixes, and application metadata such as version and citation information.

    Attributes:
        BLK (str): ANSI code for blinking cyan text
        TLE (str): ANSI code for light background title style
        HGH (str): ANSI code for bold highlighted text
        WRN (str): ANSI code for warning (yellow) text
        ERR (str): ANSI code for error (red) text
        EXT (str): ANSI code for success/emphasis (green) text
        STD (str): ANSI code to reset text styling
        PGM_NAM (str): Formatted prefix for normal messages
        PGM_WRN (str): Formatted prefix for warnings
        PGM_ERR (str): Formatted prefix for errors
        LOGO (str): ASCII art logo for the application
        VERSION (str): Application version number
        CITATION (str): Citation information for the method
        MESSAGE (str): Brief program description
    """

    # Style variables
    BLK = '\033[5;36m'    # Blinking cyan
    TLE = '\033[2;106m'   # Light background title
    HGH = '\033[1;100m'   # Bold highlighted
    WRN = '\033[33m'      # Warning yellow
    ERR = '\033[31m'      # Error red
    EXT = '\033[32m'      # Success green
    STD = '\033[0m'       # Reset styling

    # Program output variables
    PGM_NAM = f"..:{EXT}pyAdMD> {STD}"
    PGM_WRN = f"..+{WRN}pyAdMD-Wrn> {STD}"
    PGM_ERR = f"..%{ERR}pyAdMD-Err> {STD}"

    LOGO = '''
                            █████████       █████ ██████   ██████ ██████████
                           ███░░░░░███     ░░███ ░░██████ ██████ ░░███░░░░███
     ████████  █████ ████ ░███    ░███   ███████  ░███░█████░███  ░███   ░░███
    ░░███░░███░░███ ░███  ░███████████  ███░░███  ░███░░███ ░███  ░███    ░███
     ░███ ░███ ░███ ░███  ░███░░░░░███ ░███ ░███  ░███ ░░░  ░███  ░███    ░███
     ░███ ░███ ░███ ░███  ░███    ░███ ░███ ░███  ░███      ░███  ░███    ███
     ░███████  ░░███████  █████   █████░░████████ █████     █████ ██████████
     ░███░░░    ░░░░░███ ░░░░░   ░░░░░  ░░░░░░░░ ░░░░░     ░░░░░ ░░░░░░░░░░
     ░███       ███ ░███
     █████     ░░██████
    ░░░░░       ░░░░░░
    '''

    VERSION = '2.0'
    CITATION = '''  Please cite:

    \tAdaptive Normal Mode Sampling (aMDeNM) Enhances Exploration of Protein Conformational Space
    \tand Reveals the Functional Role of Frequency Coupling.
    \tP.T. Resende-Lara, M.G.S. Costa, B. Dudas, J. Czigleczki, E. Balog, D. Perahia.
    \tDOI: https://doi.org/10.1021/acs.jctc.6c00398'''

    MESSAGE = "This program can setup and run multi-replica aMDeNM simulations through OpenMM."


class ParameterStorage:
    """
    Handles serialization and deserialization of simulation parameters.

    This class provides functionality to save and load simulation parameters
    to/from JSON files, enabling restart capabilities for the aMDeNM simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
        param_file (str): Default filename for parameter storage.
    """

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize ParameterStorage with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console
        self.param_file = "pyAdMD_params.json"

    def save_parameters(self, args: argparse.Namespace, factors: np.ndarray,
                        nm_parsed: List[int], end_loop: int, cwd: str) -> None:
        """
        Save simulation parameters to a JSON file.

        Serializes the current simulation state including command line arguments,
        mode combination factors, selected modes, and loop information for
        potential restart capabilities.

        Args:
            args (argparse.Namespace): Command line arguments namespace.
            factors (numpy.ndarray): Matrix of combination factors for normal modes (P×N).
            nm_parsed (list): List of mode numbers used in combinations.
            end_loop (int): Final loop iteration count for simulation cycles.
            cwd (str): Current working directory path.

        Note:
            The timestamp is automatically added to track when parameters were saved.
        """
        params = {
            "args": vars(args),
            "factors": factors.tolist() if factors is not None else None,
            "nm_parsed": nm_parsed,
            "end_loop": end_loop,
            "cwd": cwd,
            "timestamp": time.time()
        }

        with open(self.param_file, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"\n{self.console.PGM_NAM}Parameters saved to {self.console.EXT}{self.param_file}{self.console.STD}.")

    def load_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Load simulation parameters from a JSON file.

        Attempts to deserialize previously saved parameters and reconstruct
        the argument namespace object. Handles conversion of factors back to
        numpy array format.

        Returns:
            dict: Dictionary containing loaded parameters with keys:
                - args (argparse.Namespace): Reconstructed argument namespace.
                - factors (numpy.ndarray or None): Combination factors matrix.
                - nm_parsed (list[int]): List of mode numbers used in simulation.
                - end_loop (int): Final loop iteration count.
                - cwd (str): Absolute working directory path.
                - timestamp (float): Unix timestamp of when parameters were saved.
            Returns None if the file is missing or cannot be parsed.

        Raises:
            JSONDecodeError: If the parameter file contains invalid JSON.
            IOError: If the parameter file cannot be accessed.
        """
        if not os.path.exists(self.param_file):
            print(f"{self.console.PGM_ERR}Parameter file {self.console.ERR}{self.param_file}{self.console.STD} not found.")
            return None

        try:
            with open(self.param_file, 'r') as f:
                params = json.load(f)

            # Reconstruct args namespace
            class Args:
                def __init__(self, dict_args: Dict[str, Any]) -> None:
                    for key, value in dict_args.items():
                        setattr(self, key, value)

            params['args'] = Args(params['args'])

            # Convert factors back to numpy array if present
            if params['factors'] is not None:
                params['factors'] = np.array(params['factors'])

            print(f"{self.console.PGM_NAM}Parameters loaded from {self.console.EXT}{self.param_file}{self.console.STD}.")
            return params
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error loading parameters: {self.console.ERR}{e}{self.console.STD}.")
            return None


class ENMCalculator:
    """
    Elastic Network Model calculator for normal mode analysis.

    This class handles the computation of elastic network models, including
    system creation, Hessian matrix computation, and normal mode analysis.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
    """

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize ENMCalculator with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def compute_enm(self, positions_ang: np.ndarray, base_name: str,
                    nm_type: str, nm_parsed: List[int],
                    input_dir: str, psffile: str) -> None:
        """
        Setup and run ENM analysis.

        Args:
            positions_ang (np.ndarray): Atomic positions in Å, shape (N, 3).
                Replaces the former ``coorfile`` parameter; coordinates are
                written to a temporary PDB via a reference Universe so that
                both NAMD and OpenMM input paths feed ENM identically.
            base_name (str): Base filename stem used for ENM output directory
                and file prefixes (e.g. ``"system"`` → ``inputs/system_enm/``).
            nm_type (str): Type of normal mode calculation ('CA' or 'HEAVY').
            nm_parsed (list): List containing mode numbers to analyze.
            input_dir (str): Input directory path.
            psffile (str): PSF topology filename.
        """
        # Create output folder
        output_folder = f"{input_dir}/{base_name}_enm"
        os.makedirs(output_folder, exist_ok=True)
        output_prefix = os.path.join(output_folder, base_name)

        # Prefix to output files
        prefix = "ca" if nm_type.lower() == 'ca' else "heavy"

        # Build a reference Universe and write a temporary PDB for ENM input
        u_ref = make_reference_universe(psffile, positions_ang)
        prot  = u_ref.select_atoms("protein")
        pdb_file = f"{output_prefix}.pdb"
        prot.write(pdb_file, file_format="PDB")

        # Create system
        system, topology, positions = self.create_system(
            pdb_file,
            model_type=nm_type,
            output_prefix=output_prefix,
            spring_constant=1,
        )

        # Compute Hessian
        hessian = self.hessian_enm(
            system,
            positions
        )

        # Mass-weight Hessian
        mw_hessian = self.mass_weight_hessian(
            hessian,
            system
        )

        # Compute Normal Modes
        frequencies, enm, eigenvalues = self.compute_normal_modes(
            mw_hessian,
            n_modes=None,
            use_gpu=True
        )

        # Write modes vectors and trajectories
        print(f"{self.console.PGM_NAM}Writing vectors and trajectories for modes {self.console.EXT}{str(nm_parsed)[1:-1]}{self.console.STD}...")
        mode_vectors_prefix = f"{output_prefix}_{prefix}"

        for mode_idx in nm_parsed:
            self.write_nm_vectors(
                enm,
                frequencies,
                system,
                topology,
                mode_idx,
                pdb_file,
                mode_vectors_prefix,
                model_type=nm_type
            )

            self.write_nm_trajectories(
                enm,
                frequencies,
                system,
                topology,
                mode_idx,
                positions,
                mode_vectors_prefix,
                model_type=nm_type
            )

        np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
        np.save(f"{output_prefix}_{prefix}_modes.npy", enm)
        print(f"{self.console.PGM_NAM}Results saved to {self.console.EXT}{output_prefix}_{prefix}_*.npy{self.console.STD} files.")

    def create_system(self, pdb_file: str, model_type: str = 'ca',
                      cutoff: Optional[float] = None, spring_constant: float = 1.0,
                      output_prefix: str = "input") -> Tuple[mm.System, app.Topology, unit.Quantity]:
        """
        Create an Elastic Network Model system based on the specified model type.

        Args:
            pdb_file (str): Path to the input PDB file
            model_type (str): Type of model to create: 'ca' for Cα-only or 'heavy' for heavy-atom ENM
            cutoff (float): Cutoff distance for interactions in Å.
                            If None, uses default values: 15.0Å for CA model, 12.0Å for heavy-atom model
            spring_constant (float): Spring constant for the ENM bonds in kcal/mol/Å²
            output_prefix (str): Prefix for output files

        Returns:
            system (openmm.System): The created OpenMM system
            topology (openmm.app.Topology): The topology of the system
            positions (openmm.unit.Quantity): The positions of particles in the system

        Raises:
            ValueError: If an invalid model type is specified or no relevant atoms are found
        """
        # Set default cutoffs if not provided
        if cutoff is None:
            cutoff = 15.0 if model_type == 'ca' else 12.0

        if model_type == 'ca':
            return self._create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
        elif model_type == 'heavy':
            return self._create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)
        else:
            raise ValueError(f"Unknown model type: {self.console.ERR}{model_type}{self.console.STD}")

    def _create_ca_system(self, pdb_file: str, cutoff: float,
                          spring_constant: float,
                          output_prefix: str) -> Tuple[mm.System, app.Topology, unit.Quantity]:
        """
        Create a Cα-only ENM system from a PDB file.

        Extracts Cα atoms, assigns uniform carbon masses (12.011 Da), and
        connects pairs within [2.9 Å, cutoff] with a harmonic spring.  The
        reduced structure is saved to {output_prefix}_ca_structure.pdb.

        Args:
            pdb_file (str): Path to the input PDB file.
            cutoff (float): Maximum Cα–Cα distance in Å for ENM bond formation.
            spring_constant (float): Harmonic spring constant in kcal mol⁻¹ Å⁻².
            output_prefix (str): Prefix used when writing the Cα PDB output file.

        Returns:
            system (openmm.System): System with one particle per Cα atom and a CustomBondForce encoding the ENM potential.
            topology (openmm.app.Topology): Reduced topology containing only Cα atoms.
            positions (openmm.unit.Quantity): Cα positions in nanometres.

        Raises:
            ValueError: If no Cα atoms are found in the PDB file.
        """
        pdb = app.PDBFile(pdb_file)

        # Extract Cα atoms and their positions
        ca_info = []
        positions_list = []
        for atom in pdb.topology.atoms():
            if atom.name == 'CA':
                pos = pdb.positions[atom.index]
                ca_info.append((atom.index, atom.residue))
                positions_list.append([pos.x, pos.y, pos.z])

        if not ca_info:
            print(f"{self.console.PGM_ERR}No Cα atoms found in the structure.")

        n_atoms = len(ca_info)
        print(f"{self.console.PGM_NAM}Selected {self.console.EXT}{n_atoms}{self.console.STD} Cα atoms.")

        # Create a simplified topology with only Cα atoms
        new_topology = app.Topology()
        new_chain = new_topology.addChain()
        residue_map = {}

        for i, (orig_idx, residue) in enumerate(ca_info):
            if residue not in residue_map:
                new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
                residue_map[residue] = new_res
            new_topology.addAtom("CA", app.element.carbon, residue_map[residue])

        # Create the system and add particles
        system = mm.System()
        positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
        positions_quantity = unit.Quantity(positions)

        carbon_mass = 12.011 * unit.daltons
        for _ in range(n_atoms):
            system.addParticle(carbon_mass)

        # Create ENM force field
        enm_force = mm.CustomBondForce("0.5 * k * (r - r0)**2")
        enm_force.addGlobalParameter("k", spring_constant)
        enm_force.addPerBondParameter("r0")

        # Calculate distance matrix and create bonds
        pos_np = np.array(positions_list, dtype=np.float32)
        dist_matrix = squareform(pdist(pos_np))
        cutoff_nm = cutoff * 0.1    # Convert Å to nm
        min_distance_nm = 2.9 * 0.1 # Minimum distance in nm (2.9 Å)

        bonds = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = dist_matrix[i, j]
                # Apply both minimum distance and cutoff
                if min_distance_nm <= dist <= cutoff_nm:
                    bonds.append((i, j, dist))

        for i, j, dist in bonds:
            enm_force.addBond(i, j, [dist])

        system.addForce(enm_force)
        print(f"{self.console.PGM_NAM}Added {self.console.EXT}{len(bonds)}{self.console.STD} ENM bonds with cutoff={self.console.EXT}{cutoff}{self.console.STD} Å, "
              f"min_distance={self.console.EXT}{min_distance_nm}{self.console.STD} Å, k={self.console.EXT}{spring_constant}{self.console.STD} kcal/mol/Å².")
        system.addForce(mm.CMMotionRemover())

        # Save the Cα structure
        ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
        with open(ca_pdb_file, 'w') as f:
            app.PDBFile.writeFile(new_topology, positions_quantity, f)
        print(f"{self.console.PGM_NAM}Cα structure saved to {self.console.EXT}{ca_pdb_file}{self.console.STD}.")

        # Convert HETATM to ATOM
        self.convert_hetatm_to_atom(ca_pdb_file)

        return system, new_topology, positions_quantity

    def _create_heavy_system(self, pdb_file: str, cutoff: float,
                             spring_constant: float,
                             output_prefix: str) -> Tuple[mm.System, app.Topology, unit.Quantity]:
        """
        Create a heavy-atom ENM system from a PDB file.

        Extracts all non-hydrogen atoms, assigns element-specific masses, and
        connects pairs within [2.0 Å, cutoff] with a harmonic spring.  The
        reduced structure is saved to {output_prefix}_heavy.pdb.

        Args:
            pdb_file (str): Path to the input PDB file.
            cutoff (float): Maximum inter-atom distance in Å for ENM bond formation.
            spring_constant (float): Harmonic spring constant in kcal mol⁻¹ Å⁻².
            output_prefix (str): Prefix used when writing the heavy-atom PDB output file.

        Returns:
            system (openmm.System): System with one particle per heavy atom and a CustomBondForce encoding the ENM potential.
            topology (openmm.app.Topology): Reduced topology containing only heavy atoms.
            positions (openmm.unit.Quantity): Heavy-atom positions in nanometres.

        Raises:
            ValueError: If no heavy atoms are found in the PDB file.
        """
        pdb = app.PDBFile(pdb_file)

        # Identify heavy atoms (non-hydrogen)
        heavy_atoms = []
        positions_list = []
        for atom in pdb.topology.atoms():
            if atom.element != app.element.hydrogen:
                pos = pdb.positions[atom.index]
                heavy_atoms.append((atom.index, atom.residue, atom.name, atom.element))
                positions_list.append([pos.x, pos.y, pos.z])

        if not heavy_atoms:
            print(f"{self.console.PGM_ERR}No heavy atoms found in the structure.")

        n_atoms = len(heavy_atoms)
        print(f"{self.console.PGM_NAM}Selected {self.console.EXT}{n_atoms}{self.console.STD} heavy atoms.")

        # Create new topology with only heavy atoms
        new_topology = app.Topology()
        new_chain = new_topology.addChain()
        residue_map = {}

        for i, (orig_idx, residue, name, element) in enumerate(heavy_atoms):
            if residue not in residue_map:
                new_res = new_topology.addResidue(f"{residue.name}{residue.id}", new_chain)
                residue_map[residue] = new_res
            new_topology.addAtom(name, element, residue_map[residue])

        # Create the system and add particles with appropriate masses
        system = mm.System()
        positions = [mm.Vec3(*pos) * unit.nanometer for pos in positions_list]
        positions_quantity = unit.Quantity(positions)

        for _, _, _, element in heavy_atoms:
            system.addParticle(element.mass)

        # Create ENM force field
        enm_force = mm.CustomBondForce("0.5 * k * (r - r0)**2")
        enm_force.addGlobalParameter("k", spring_constant)
        enm_force.addPerBondParameter("r0")

        # Calculate distance matrix and create bonds
        pos_np = np.array(positions_list, dtype=np.float32)
        dist_matrix = squareform(pdist(pos_np))
        cutoff_nm = cutoff * 0.1    # Convert Å to nm
        min_distance_nm = 2.0 * 0.1 # Minimum distance in nm (2.0 Å)

        # Add bonds within cutoff range
        bonds = []
        for i in range(n_atoms):
            for j in range(i+1, n_atoms):
                dist = dist_matrix[i, j]
                # Apply both minimum distance and cutoff
                if min_distance_nm <= dist <= cutoff_nm:
                    bonds.append((i, j, dist))

        for i, j, dist in bonds:
            enm_force.addBond(i, j, [dist])

        system.addForce(enm_force)
        print(f"{self.console.PGM_NAM}Added {self.console.EXT}{len(bonds)}{self.console.STD} ENM bonds with cutoff={self.console.EXT}{cutoff}{self.console.STD} Å, "
              f"min_distance={self.console.EXT}{min_distance_nm}{self.console.STD} Å, k={self.console.EXT}{spring_constant}{self.console.STD} kcal/mol/Å².")
        system.addForce(mm.CMMotionRemover())

        # Save heavy atom structure
        heavy_pdb_file = f"{output_prefix}_heavy.pdb"
        with open(heavy_pdb_file, 'w') as f:
            app.PDBFile.writeFile(new_topology, positions_quantity, f)
        print(f"{self.console.PGM_NAM}Heavy-atom structure saved to {self.console.EXT}{heavy_pdb_file}{self.console.STD}.")

        return system, new_topology, positions_quantity

    def hessian_enm(self, system: mm.System, positions: unit.Quantity) -> coo_array:
        """
        Build and regularize the sparse ENM Hessian.

        Extracts bond parameters from the OpenMM CustomBondForce, assembles the
        (3N × 3N) Hessian as a CSR sparse matrix, and applies a small diagonal
        regularization (1e-8) for numerical stability.

        Args:
            system (openmm.System): System containing the ENM CustomBondForce.
            positions (openmm.unit.Quantity): Particle positions.

        Returns:
            hessian (scipy.sparse.csr_array): Regularized sparse Hessian (3N × 3N).

        Raises:
            ValueError: If no CustomBondForce is found in the system.
        """
        n_particles = system.getNumParticles()
        n_dof = 3 * n_particles

        # Locate the CustomBondForce that encodes the ENM harmonic springs
        enm_force = next((f for f in system.getForces() if isinstance(f, mm.CustomBondForce)), None)
        if enm_force is None:
            raise ValueError("No ENM force found in system")

        # Retrieve the global spring constant k and convert positions to a plain numpy array (nm)
        k_val = enm_force.getGlobalParameterDefaultValue(0)
        num_bonds = enm_force.getNumBonds()
        pos_array = np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)], dtype=np.float64)

        print(f"{self.console.PGM_NAM}Computing sparse Hessian for {self.console.EXT}{n_particles}{self.console.STD} "
              f"particles ({self.console.EXT}{num_bonds}{self.console.STD} bonds)...")
        t0 = time.time()

        # Extract bond parameters from the OpenMM force into a compact (num_bonds × 3) array
        # Columns: [atom_i, atom_j, r0] where r0 is the equilibrium bond length in nm
        bonds_list = np.empty((num_bonds, 3), dtype=np.float64)
        for bond_idx in range(num_bonds):
            i, j, [r0] = enm_force.getBondParameters(bond_idx)
            bonds_list[bond_idx] = (i, j, r0)

        # Assemble the sparse Hessian from bond geometry
        hessian = self._compute_hessian_sparse(pos_array, bonds_list, k_val, n_particles)

        # Ensure CSR format for efficient arithmetic operations
        hessian = hessian.tocsr()

        # Apply a small Tikhonov regularisation (λI, λ = 1e-8) to the diagonal to prevent
        # exact singularity; the true zero eigenvalues (rigid-body modes) remain near-zero
        # and are filtered out during diagonalisation
        reg = diags([1e-8] * n_dof, 0, format='csr')
        hessian = hessian + reg

        nnz = hessian.nnz
        dense_elements = n_dof ** 2
        print(f"{self.console.PGM_NAM}Sparse Hessian: {self.console.EXT}{nnz:,}{self.console.STD} non-zeros "
                    f"({self.console.EXT}{100*nnz/dense_elements:.3f}%{self.console.STD} density).")
        print(f"{self.console.PGM_NAM}ENM Hessian computed in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")

        return hessian

    def _compute_hessian_sparse(self, pos_array: np.ndarray, bonds: np.ndarray,
                                k_val: float, n_particles: int) -> coo_array:
        """
        Assemble the ENM Hessian as a sparse CSR matrix.

        Each bonded pair (i, j) contributes four 3×3 blocks: +block on diagonals
        (i,i) and (j,j), -block on off-diagonals (i,j) and (j,i). Contributions
        are collected in flat COO arrays and summed on conversion to CSR.

        Args:
            pos_array (ndarray): Particle positions (N×3), float64, in nm.
            bonds (ndarray): Bond table with columns [i, j, r0], float64.
            k_val (float): Spring constant in kcal/mol/Å².
            n_particles (int): Number of particles.

        Returns:
            hessian (scipy.sparse.csr_array): Sparse Hessian (3N × 3N). Duplicate diagonal entries
                                              from overlapping bonds have been summed.

        Raises:
            ValueError: If no entries are assembled (empty bond list).
        """
        n_bonds = bonds.shape[0]
        # Each bond contributes 4 blocks of 3×3 scalars = 36 entries; pre-allocate at maximum
        max_entries = n_bonds * 36

        row_idx = np.empty(max_entries, dtype=np.int32)
        col_idx = np.empty(max_entries, dtype=np.int32)
        values  = np.empty(max_entries, dtype=np.float64)
        ptr = 0

        for idx in range(n_bonds):
            i = int(bonds[idx, 0])
            j = int(bonds[idx, 1])

            # Displacement vector from atom i to atom j
            r_ij = pos_array[j] - pos_array[i]
            dist = np.sqrt(r_ij[0]**2 + r_ij[1]**2 + r_ij[2]**2)
            if dist < 1e-6:
                # Skip degenerate bonds where atoms are effectively coincident
                continue

            # Unit vector along the bond: e_ij = r_ij / |r_ij|
            e_ij = r_ij / dist

            # 3×3 Kirchhoff block for this bond: B = k * (e_ij ⊗ e_ij)
            # This is the second derivative of V = 0.5*k*(r-r0)² with respect
            # to the Cartesian coordinates of i and j, evaluated at r = r0
            block = k_val * np.outer(e_ij, e_ij)

            # DOF offsets for atoms i and j (3 DOFs each: x, y, z)
            i3, j3 = 3 * i, 3 * j

            for a in range(3):
                for b in range(3):
                    v = block[a, b]
                    # Diagonal blocks: +B at (i,i) and (j,j) — restoring forces on each atom
                    row_idx[ptr] = i3 + a;  col_idx[ptr] = i3 + b;  values[ptr] =  v;  ptr += 1
                    row_idx[ptr] = j3 + a;  col_idx[ptr] = j3 + b;  values[ptr] =  v;  ptr += 1
                    # Off-diagonal blocks: -B at (i,j) and (j,i) — coupling between atoms
                    row_idx[ptr] = i3 + a;  col_idx[ptr] = j3 + b;  values[ptr] = -v;  ptr += 1
                    row_idx[ptr] = j3 + a;  col_idx[ptr] = i3 + b;  values[ptr] = -v;  ptr += 1

        if ptr == 0:
            raise ValueError("No Hessian entries assembled — check bond list.")

        # Assemble the COO matrix and convert to CSR; tocsr() automatically sums duplicate entries
        # on the diagonal (contributions from multiple bonds sharing the same atom pair) without
        # any explicit symmetrisation step
        n_dof = 3 * n_particles
        hessian = coo_array(
            (values[:ptr], (row_idx[:ptr], col_idx[:ptr])),
            shape=(n_dof, n_dof),
            dtype=np.float64
        )
        return hessian.tocsr()

    def mass_weight_hessian(self, hessian: coo_array, system: mm.System) -> coo_array:
        """
        Return the mass-weighted Hessian M^{-1/2} H M^{-1/2}.

        Applies mass-weighting to the Hessian matrix by scaling it with the inverse
        square root of the particle masses. This transformation accounts for atomic mass
        differences in the normal mode analysis.

        Args:
            hessian (scipy.sparse.coo_array): Sparse Hessian matrix (3N × 3N) in COO format.
            system (openmm.System): OpenMM system containing particle mass information.

        Returns:
            mw_hessian (scipy.sparse.csr_array): Mass-weighted Hessian M^{-1/2} H M^{-1/2},
                returned in CSR format for efficient arithmetic operations.

        Note:
            Virtual sites or particles with zero mass are assigned a default mass of 1.0 Da
            to avoid division-by-zero errors during the inverse square root calculation.
        """
        n_particles = system.getNumParticles()

        # Extract per-atom masses in Da; replace any zero-mass virtual sites with 1.0 Da
        # to avoid division-by-zero in the inverse square-root
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                        for i in range(n_particles)])
        masses[masses == 0] = 1.0

        # Expand the per-atom inverse square-root mass to a per-DOF vector (x, y, z repeated)
        # so that a single diagonal matrix D covers all 3N degrees of freedom
        inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

        # Construct the sparse diagonal scaling matrix D = M^{-1/2}
        D = diags(inv_sqrt_m, 0, format='csr')

        # Return the mass-weighted Hessian H_mw = D H D = M^{-1/2} H M^{-1/2}
        # All three matrices are sparse, so no dense allocation occurs
        return D @ hessian @ D

    def gpu_diagonalization(self, hessian: np.ndarray,
                            n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonalize a dense Hessian using GPU acceleration with CuPy.

        Performs full or partial eigendecomposition of a dense Hessian matrix using
        GPU resources (CUDA). Used only for small systems (n_dof < 5000). Uses float32
        for matrices larger than 3000 DOF to reduce VRAM usage. Includes memory pool
        management for efficient GPU memory allocation and deallocation.

        Args:
            hessian (ndarray): Dense Hessian matrix, shape (3N, 3N), float64.
            n_modes (int, optional): Number of lowest eigenvalues/eigenvectors to compute.
                If None, computes all eigenvalues. Default: None.

        Returns:
            eigenvalues (ndarray): Eigenvalues in ascending order, shape (k,).
            eigenvectors (ndarray): Corresponding eigenvectors, shape (3N, k).

        Raises:
            RuntimeError: If GPU resources are unavailable or CUDA operations fail.
            MemoryError: If GPU memory is insufficient for the system size.

        Note:
            Memory is not overloaded: GPU arrays are explicitly freed before
            transferring results back to CPU to avoid holding multiple copies.
        """
        mem_pool = cp.get_default_memory_pool()
        pinned_mem_pool = cp.get_default_pinned_memory_pool()

        with cp.cuda.Device(0):
            # Use float32 for large matrices to halve VRAM usage; float64 otherwise
            dtype = cp.float32 if hessian.shape[0] > 3000 else cp.float64
            hessian_gpu = cp.array(hessian, dtype=dtype)
            # Free the CPU copy immediately to reduce peak memory footprint
            del hessian

            if n_modes is not None:
                # Include 6 extra modes to account for rigid-body modes that will be discarded;
                # subset_by_index requests only the k lowest eigenvalue/eigenvector pairs
                n_modes = min(n_modes + 6, hessian_gpu.shape[0])
                eigenvalues, eigenvectors = cp.linalg.eigh(
                    hessian_gpu, UPLO='L', subset_by_index=[0, n_modes - 1]
                )
            else:
                # Full diagonalisation — only feasible for small systems (n_dof < 5000)
                eigenvalues, eigenvectors = cp.linalg.eigh(hessian_gpu, UPLO='L')

            # Free GPU memory before transferring results to avoid holding two copies simultaneously
            del hessian_gpu
            mem_pool.free_all_blocks()
            pinned_mem_pool.free_all_blocks()

            # Transfer results back to CPU via a synchronised copy using the null stream
            eigenvalues_cpu = cp.asnumpy(eigenvalues, stream=cp.cuda.Stream.null)
            eigenvectors_cpu = cp.asnumpy(eigenvectors, stream=cp.cuda.Stream.null)
            del eigenvalues, eigenvectors

        return eigenvalues_cpu, eigenvectors_cpu

    def compute_normal_modes(self, hessian: Union[np.ndarray, coo_array],
                             n_modes: Optional[int] = None,
                             use_gpu: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute normal modes by partially diagonalizing the mass-weighted Hessian.

        For sparse input, uses the ARPACK shift-invert solver (eigsh with sigma=1e-6)
        to compute only the requested lowest-frequency modes. For dense input with
        n_dof < 5000 and GPU available, uses GPU acceleration; otherwise uses CPU-based
        scipy.linalg.eigh for partial eigendecomposition.

        Args:
            hessian (Union[ndarray, scipy.sparse.coo_array]): Mass-weighted Hessian matrix,
                shape (3N, 3N). Can be dense numpy array or sparse COO/CSR format.
            n_modes (int, optional): Number of non-rigid-body vibrational modes to compute.
                Defaults to 50 if not specified. Automatically increased by 6 to account
                for discarded rigid-body modes.
            use_gpu (bool, optional): Attempt GPU acceleration for small dense systems
                (n_dof < 5000). Falls back to CPU if GPU unavailable. Default: False.

        Returns:
            frequencies (ndarray): Non-zero vibrational frequencies in internal units,
                sorted in ascending order. Shape: (k,) where k <= n_modes.
            modes (ndarray): Normal mode eigenvectors, shape (3N, k), where column i
                corresponds to the i-th frequency.
            eigenvalues (ndarray): Raw eigenvalues from the diagonalization solver before
                filtering rigid-body modes. Shape: (k,).

        Raises:
            ValueError: If the Hessian is singular or contains no valid modes.

        Note:
            Rigid-body modes (eigenvalues < 1e-10 × max_eigenvalue) are automatically
            filtered out during eigenvalue sorting.
        """
        n_dof = hessian.shape[0]
        # Request n_modes + 6 eigenvalues to guarantee that, after discarding the 6 rigid-body
        # modes (near-zero eigenvalues), at least n_modes vibrational modes remain
        n_request = min((n_modes or 50) + 6, n_dof - 1)

        t0 = time.time()

        if issparse(hessian):
            # Shift-invert ARPACK: transforming the problem to (H - σI)⁻¹ maps the smallest
            # eigenvalues of H to the largest of the shifted operator, making convergence fast.
            # σ = 1e-6 sits just above zero to avoid the exact null space of rigid-body modes
            print(f"{self.console.PGM_NAM}Diagonalizing sparse mass-weighted Hessian...")
            eigenvalues, eigenvectors = eigsh(hessian, k=n_request, which='LM', sigma=1e-6)

        else:
            if use_gpu and n_dof < 5000:
                try:
                    print(f"{self.console.PGM_NAM}Diagonalizing dense Hessian using GPU...")
                    eigenvalues, eigenvectors = self.gpu_diagonalization(hessian, n_modes)
                except Exception as e:
                    print(f"{self.console.PGM_WRN}GPU diagonalization failed: {self.console.WRN}{e}{self.console.STD}. Falling back to CPU.")
                    use_gpu = False

            if not use_gpu or n_dof >= 5000:
                # subset_by_index restricts LAPACK to the lowest n_request eigenpairs,
                # avoiding the O(N³) cost of a full diagonalisation
                print(f"{self.console.PGM_NAM}Diagonalizing dense Hessian using CPU...")
                eigenvalues, eigenvectors = eigh(
                    hessian,
                    subset_by_index=[0, n_request - 1],
                    driver='evr',
                    overwrite_a=True,
                    check_finite=False
                )

        print(f"{self.console.PGM_NAM}Diagonalization completed in {self.console.EXT}{time.time() - t0 :.2f}{self.console.STD} seconds.")

        # Identify and discard rigid-body modes: eigenvalues below a relative threshold
        # (1e-10 × max eigenvalue) are treated as numerically zero
        abs_evals = np.abs(eigenvalues)
        threshold = max(np.max(abs_evals) * 1e-10, 1e-10)
        valid_idx = abs_evals > threshold

        # Frequencies are the square root of the eigenvalues of the mass-weighted Hessian
        frequencies = np.sqrt(np.abs(eigenvalues[valid_idx]))
        valid_modes = eigenvectors[:, valid_idx]

        # Sort modes from lowest to highest frequency
        sort_idx = np.argsort(frequencies)
        return frequencies[sort_idx], valid_modes[:, sort_idx], eigenvalues

    def convert_hetatm_to_atom(self, pdb_file: str) -> None:
        """
        Replace HETATM records with ATOM records in a PDB file in place.

        Args:
            pdb_file (str): Path to the PDB file to modify. Overwritten in place.
        """
        with open(pdb_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            if line.startswith('HETATM'):
                # Replace HETATM with ATOM while preserving spacing
                new_line = 'ATOM  ' + line[6:]
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(pdb_file, 'w') as f:
            f.writelines(new_lines)

    def write_nm_vectors(self, modes: np.ndarray, frequencies: np.ndarray,
                         system: mm.System, topology: app.Topology, nm: int,
                         pdb_file: str, output_prefix: str, model_type: str = 'ca') -> None:
        """
        Write normal mode eigenvectors to XYZ files for the complete protein structure.

        Extracts mode displacement vectors for a single normal mode and writes them to
        an XYZ format file. Maps ENM-reduced structure atoms back to the full PDB structure
        for visualization. Computes and reports the vibrational frequency in cm⁻¹.

        Args:
            modes (numpy.ndarray): Normal mode eigenvectors matrix, shape (3N, M),
                where N is number of ENM particles and M is total number of modes.
            frequencies (numpy.ndarray): Vibrational frequencies of all modes.
                Shape: (M,), in internal angular frequency units.
            system (openmm.System): OpenMM system containing particle information.
            topology (openmm.app.Topology): Topology of the reduced ENM structure.
            nm (int): Mode index (0-based) to write to file.
            pdb_file (str): Path to the original full PDB file (used for atom mapping).
            output_prefix (str): Directory and filename prefix for output file.
                File will be named {output_prefix}_mode_{nm}.xyz
            model_type (str, optional): ENM model type ('ca' for Cα-only or 'heavy'
                for heavy-atom). Used for atom mapping in the lookup. Default: 'ca'.

        Returns:
            None

        Note:
            Output frequency is converted from internal units to cm⁻¹ using a factor
            of 108.58. The method performs O(1) atom mapping using a residue-name
            dictionary lookup for efficiency.
        """
        # Read the original PDB file to get complete atom information
        pdb = app.PDBFile(pdb_file)
        n_original_atoms = pdb.topology.getNumAtoms()

        # Build element list and (residue.index, atom.name) -> orig_idx lookup in one pass
        elements = []
        orig_lookup = {}
        for orig_idx, atom in enumerate(pdb.topology.atoms()):
            elements.append(atom.element.symbol)
            key = (atom.residue.index, atom.name)
            orig_lookup[key] = orig_idx

        n_particles = system.getNumParticles()

        # Map ENM atoms to original indices via O(1) lookup
        enm_to_original_map = []
        for atom in topology.atoms():
            key = (atom.residue.index, 'CA' if model_type == 'ca' else atom.name)
            enm_to_original_map.append(orig_lookup[key])

        # Write each mode to a separate XYZ file
        freq = frequencies[nm] * 108.58  # Convert to cm⁻¹
        output_file = f"{output_prefix}_mode_{nm}.xyz"

        with open(output_file, 'w') as f:
            # Write header
            f.write(f"{n_original_atoms}\n")
            f.write(f"Normal Mode {nm}, Frequency: {freq:.2f} cm⁻¹\n")

            # Extract and reshape the mode vector for ENM atoms
            mode_vector = modes[:, nm].reshape(n_particles, 3)

            # Create full vector and map ENM mode vectors via advanced indexing
            full_vector = np.zeros((n_original_atoms, 3))
            full_vector[enm_to_original_map] = mode_vector

            # Write coordinates for each atom
            for i in range(n_original_atoms):
                x, y, z = full_vector[i]
                f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")

    def write_nm_trajectories(self, modes: np.ndarray, frequencies: np.ndarray,
                              system: mm.System, topology: app.Topology, nm: int,
                              positions: unit.Quantity, output_prefix: str,
                              model_type: str, amplitude: float = 4,
                              num_frames: int = 34) -> None:
        """
        Write multi-frame PDB trajectories visualizing atomic motion along normal modes.

        Generates a multi-frame PDB file showing smooth oscillatory motion of the protein
        along a specified normal mode direction. Displacements are mass-weighted and
        scaled to a user-specified amplitude. The motion is approximated with 4 linear
        segments: equilibrium > negative peak > equilibrium > positive peak > equilibrium.

        Args:
            modes (ndarray, shape (3N, M), float64): Normal mode eigenvectors;
                column i contains the eigenvector for mode i (0-based indexing).
            frequencies (ndarray, shape (M,), float64): Vibrational frequencies in
                internal angular units. Used for logging only (converted to cm⁻¹).
            system (openmm.System): OpenMM system containing particle masses.
                Used for mass-weighting the mode displacement vector.
            topology (openmm.app.Topology): Topology of the ENM-reduced structure,
                used for writing PDB records in output trajectory.
            nm (int): Mode index (0-based) to visualize as a trajectory.
            positions (openmm.unit.Quantity): Equilibrium atomic positions, any OpenMM
                length unit. Automatically converted to nm for internal computation.
            output_prefix (str): Directory and filename prefix for output file.
                File will be named {output_prefix}_mode_{nm}_traj.pdb
            model_type (str): ENM model type ('ca' or 'heavy'). Used for logging only.
            amplitude (float, optional): Peak displacement amplitude in Ångströms.
                Controls maximum deviation from equilibrium. Default: 4.0 Å.
            num_frames (int, optional): Total number of MODEL frames in the trajectory.
                Should be even for symmetric back-and-forth motion. Default: 34.

        Returns:
            None

        Raises:
            ValueError: If num_frames is less than 4 (insufficient for 4-segment motion).

        Note:
            Mass-weighting is applied as: u_mw = u / sqrt(m_i) to account for atomic mass
            differences. RMS normalization ensures consistent scaling across different modes
            regardless of eigenvector magnitude.
        """
        n_particles = system.getNumParticles()

        freq = frequencies[nm] * 108.58  # Convert to cm⁻¹
        output_file = f"{output_prefix}_mode_{nm}_traj.pdb"

        # Mass-weight the mode vector
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1 / np.sqrt(masses), 3)
        u = modes[:, nm] * inv_sqrt_m

        rms = np.linalg.norm(u) / np.sqrt(n_particles)

        # Scale displacement to the desired amplitude
        scaled_disp = u.reshape(n_particles, 3) * (amplitude / rms * 0.1)
        orig_pos = positions.value_in_unit(unit.nanometer)
        orig_pos_np = np.array([[p.x, p.y, p.z] for p in orig_pos])

        # Create a smooth oscillation trajectory
        # Build a smooth back-and-forth oscillation trajectory (like a sinusoidal wave
        # approximated with 4 linear segments): 0 to -A > 0 > +A > 0.
        # Each segment spans 25% of the total frames.
        seg1 = int(num_frames * 0.25)
        seg2 = int(num_frames * 0.25)
        seg3 = int(num_frames * 0.25)
        seg4 = num_frames - seg1 - seg2 - seg3

        displacements = np.zeros((num_frames, n_particles, 3))

        for frame in range(num_frames):
            if frame < seg1:
                factor = -frame / seg1
            elif frame < seg1 + seg2:
                factor = -1 + (frame - seg1) / seg2
            elif frame < seg1 + seg2 + seg3:
                factor = (frame - seg1 - seg2) / seg3
            else:
                factor = 1 - (frame - seg1 - seg2 - seg3) / seg4

            displacements[frame] = scaled_disp * factor

        # Write the trajectory to a PDB file
        with open(output_file, 'w') as f:
            for frame in range(num_frames):
                new_pos_np = orig_pos_np + displacements[frame]

                new_positions = []
                for i in range(n_particles):
                    x, y, z = new_pos_np[i]
                    new_positions.append(mm.Vec3(x, y, z))

                positions_quantity = unit.Quantity(new_positions, unit.nanometer)

                f.write(f"MODEL     {frame+1:5d}\n")
                app.PDBFile.writeFile(topology, positions_quantity, f, keepIds=True)
                f.write("ENDMDL\n")

        # Convert HETATM to ATOM in trajectory
        self.convert_hetatm_to_atom(output_file)


class ModeExciter:
    """
    Handles the generation and excitation of normal mode combinations.

    This class is responsible for generating linear combinations of normal modes,
    applying excitation energy to these modes, and writing the resulting vectors
    to files for use in molecular dynamics simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
    """

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize the ModeExciter with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def generate_factors(self, P: int, N: int, cwd: str, nm_parsed: List[int],
                         nm_type: str, base_name: str,
                         mda_U: mda.Universe) -> np.ndarray:
        """
        Generate equidistant excitation vectors using a geometry-aware repulsion algorithm.

        The repulsion algorithm distributes P points uniformly on an N-dimensional
        hypersphere, but operates in an orthonormal basis derived from the actual mode
        vectors rather than treating the modes as abstract unit axes. This correctly
        accounts for the geometry of the mode subspace (non-uniform amplitudes and
        non-orthogonality in Cartesian space) while keeping the repulsion computation
        in N dimensions.

        For P == 2N the cross-polytope vertices are used directly, skipping repulsion.

        Args:
            P (int): Number of replicas (points to distribute on the hypersphere).
            N (int): Number of modes (re-computed from nm_parsed; this parameter is ignored).
            cwd (str): Current working directory path.
            nm_parsed (List[int]): List of mode numbers to combine. Defines the subspace dimension.
            nm_type (str): Normal mode type: 'charmm' for CHARMM-computed modes,
                'ca' for Cα-only ENM, or 'heavy' for heavy-atom ENM.
            base_name (str): Base filename stem used to resolve ENM output file paths
                (e.g. ``"system"`` → ``inputs/system_enm/system_ca_mode_7.xyz``).
                Replaces the former ``coorfile`` parameter.
            mda_U (mda.Universe): MDAnalysis Universe for reference structure atom count.

        Returns:
            numpy.ndarray: Combined mode vectors, shape (P, natom, 3), where natom is
                the number of protein atoms. Each row is a normalized linear combination
                of the input mode vectors, suitable for use in replica simulations.

        Raises:
            FileNotFoundError: If mode vector files cannot be located.
            ValueError: If mode vectors have incompatible dimensions.

        Note:
            The algorithm uses QR orthonormalization to create a stable orthonormal basis,
            then distributes points on the N-dimensional unit sphere using iterative
            Coulomb-like repulsion (generalized Thomson problem), finally maps back to
            physical Cartesian space.
        """
        N = len(nm_parsed)
        natom = mda_U.atoms.select_atoms("protein").n_atoms

        # Load mode vectors and flatten to (N, natom*3)
        mode_vectors = []
        if nm_type == 'charmm':
            for mode_num in nm_parsed:
                nm = mda.Universe(f"{cwd}/inputs/mode_nm{mode_num}.crd", format="CRD")
                mode_vectors.append(nm.atoms.positions.copy().ravel())
        else:
            for mode_num in nm_parsed:
                nm = mda.Universe(
                    f"{cwd}/inputs/{base_name}_enm/{base_name}_{nm_type}_mode_{mode_num}.xyz",
                    format="XYZ"
                )
                mode_vectors.append(nm.atoms.positions.copy().ravel())

        # Shape: (N, natom*3)
        mode_matrix = np.array(mode_vectors)

        # Orthonormalise the mode basis via QR decomposition.
        Q, _ = np.linalg.qr(mode_matrix.T)
        Q = Q.T  # (N, natom*3)

        # Initialise coordinates on the N-dimensional unit hypersphere.
        # For the special case P == 2N, the cross-polytope vertices (±e_i) are
        # already maximally separated and require no iterative repulsion.
        if P == 2 * N:
            # Cross-polytope vertices: already maximally separated, skip repulsion
            coords = np.vstack((np.eye(N), -np.eye(N)))
        else:
            coords = np.random.normal(size=(P, N))
            coords /= np.linalg.norm(coords, axis=1, keepdims=True)

            prev_max_force = float('inf')
            stagnation_count = 0
            stagnation_threshold = 5

            for iteration in range(1000000):
                # Pairwise differences between all point pairs: (P, P, N)
                diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                dist += np.eye(P) * 1e-6    # Small diagonal offset avoids self-repulsion singularity

                # Coulomb-like repulsion force: F ∝ 1/r^(N-1) (generalised Thomson problem)
                force = 1.0 / (dist ** (N - 1))
                force_dir = diff / dist[:, :, np.newaxis]
                total_force = np.sum(force[:, :, np.newaxis] * force_dir, axis=1)

                current_max_force = np.max(np.abs(total_force))

                # Stagnation detection: if the maximum force hasn't changed meaningfully
                # for stagnation_threshold consecutive iterations, stop early.
                if abs(current_max_force - prev_max_force) < 1e-6 * 0.1:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                prev_max_force = current_max_force

                # Project force onto the tangent plane of the hypersphere, then
                # take a small gradient step and re-normalise to stay on the sphere.
                tangent_force = total_force - np.sum(total_force * coords, axis=1, keepdims=True) * coords
                coords += 0.001 * tangent_force
                coords /= np.linalg.norm(coords, axis=1, keepdims=True)

                if current_max_force < 1e-6:
                    print(f"{self.console.PGM_NAM}Converged after {self.console.WRN}{iteration+1}{self.console.STD} iterations.")
                    break

                if stagnation_count >= stagnation_threshold:
                    print(f"{self.console.PGM_WRN}Stagnation after {self.console.WRN}{iteration+1}{self.console.STD} iterations.")
                    break

        # Map Q-coordinates back to physical vectors: (P, N) @ (N, natom*3) -> (P, natom*3)
        # This un-does the QR orthonormalisation, recovering unit vectors in Cartesian space
        # that correctly blend the geometry of each mode.
        vecs = coords @ Q
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

        # Log back-projected factors for reference: (P, natom*3) @ (natom*3, N) -> (P, N)
        # These are approximate scalar weights per mode recovered via pseudo-inverse.
        # They are written to CSV for inspection but are not used in downstream computation.
        factors_approx = vecs @ np.linalg.pinv(mode_matrix)
        self._write_factors_csv(factors_approx, nm_parsed, cwd)

        # Reshape to (P, natom, 3) for downstream use
        return vecs.reshape(P, natom, 3)

    def _write_factors_csv(self, factors: np.ndarray, nm_parsed: List[int], cwd: str) -> None:
        """
        Write combination factors to a CSV file for inspection and documentation.

        Saves the back-projected scalar weights (pseudo-inverse reconstruction) of how
        each replica's combined mode vector is weighted in terms of individual mode
        contributions. These weights are approximate and for reference only; the actual
        displacement is computed from the full combined vector.

        Args:
            factors (numpy.ndarray): Matrix of combination factors, shape (P, N),
                where P is the number of replicas and N is the number of modes.
                Each row represents one replica's approximate weighting across modes.
            nm_parsed (list): List of mode numbers included in the combination.
                Used for CSV header generation.
            cwd (str): Current working directory path. Output file will be created
                at {cwd}/rep-struct-list/factors.csv

        Returns:
            None

        Raises:
            IOError: If the output directory cannot be created or file cannot be written.
        """
        header = ['Combination'] + [f'Mode {mode}' for mode in nm_parsed]
        rows = [[str(i + 1)] + [str(v) for v in factors[i]] for i in range(len(factors))]

        # Create output folder
        output_folder = f"{cwd}/rep-struct-list"
        os.makedirs(output_folder, exist_ok=True)

        # Write to CSV
        with open(f"{output_folder}/factors.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{self.console.PGM_NAM}Combination factors written at {self.console.EXT}{output_folder}/factors.csv{self.console.STD}.")

    def combine_modes(self, replicas: int, combined_vectors: np.ndarray,
                      cwd: str, mda_U: mda.Universe) -> None:
        """
        Write pre-computed combined mode vectors to disk in NAMD format.

        Takes the combined mode vectors generated by generate_factors and writes
        each replica's vector to a separate NAMDBIN .vec file for use in MD simulations.

        Args:
            replicas (int): Number of replicas to process.
            combined_vectors (numpy.ndarray): Array of shape (P, natom, 3) containing
                normalized combined mode vectors, as returned by generate_factors.
                P is the number of replicas, natom is the number of protein atoms.
            cwd (str): Current working directory path. Output files will be created
                in {cwd}/rep-struct-list/ directory.
            mda_U (mda.Universe): MDAnalysis Universe containing the system structure,
                used as reference for writing vector files.

        Returns:
            None

        Raises:
            IOError: If the output directory cannot be created or files cannot be written.
            IndexError: If combined_vectors contains fewer replicas than specified.
        """
        output_folder = f"{cwd}/rep-struct-list"
        for rep in range(replicas):
            comb_vec = combined_vectors[rep]
            self._write_vector(comb_vec, f"{output_folder}/rep{rep+1}_vector.vec", mda_U)

        print(f"{self.console.PGM_NAM}Combination vectors written at {self.console.EXT}{output_folder}/rep*_vector.vec{self.console.STD}.")

    def excite(self, q_vector: np.ndarray, user_ek: float, sel_mass: np.ndarray) -> np.ndarray:
        """
        Compute excitation velocity in AKMA units for a given normal mode.

        Calculates the velocity magnitude required to achieve a target kinetic energy
        when moving along a specified mode direction. Uses mass-weighted norm calculation
        to properly account for the masses of the selected atoms.

        Args:
            q_vector (numpy.ndarray): Unit mode vector in Cartesian coordinates,
                shape (n_sel, 3), in Ångströms. Should be normalized to unit length.
            user_ek (float): Target kinetic energy in kcal/mol to inject along the mode.
            sel_mass (numpy.ndarray): Atomic masses of selected atoms, shape (n_sel,),
                in atomic mass units (amu).

        Returns:
            exc_vec (numpy.ndarray): Excitation velocity vector, shape (n_sel, 3),
                in AKMA velocity units (Ångströms/AKMA time units).

        Note:
            AKMA units are the standard unit system used in NAMD/CHARMM. The kinetic
            energy is calculated as: KE = 0.5 * sum(m_i * v_i²), where the velocities
            are mass-weighted projections onto the mode direction.
        """
        # Mass‑weighted squared norm of the mode vector
        mass_weighted_norm = np.sum(sel_mass[:, np.newaxis] * (q_vector ** 2))
        # Scale factor to achieve exactly user_ek
        c = np.sqrt(2 * user_ek / mass_weighted_norm)
        exc_vec = q_vector * c

        return exc_vec

    def _write_vector(self, xyz: np.ndarray, output_file: str, mda_U: mda.Universe) -> None:
        """
        Write coordinate vector to file in NAMD binary format.

        Writes a velocity or displacement vector to a NAMDBIN (.vec) file format
        compatible with NAMD simulations. The vector is embedded within a full-system
        coordinate array (with other atoms zeroed out).

        Args:
            xyz (numpy.ndarray): Coordinate/velocity array to write, shape (n_atoms, 3)
                or (n_sel, 3) for a subset of atoms. Units should match NAMD conventions
                (Ångströms for positions, AKMA for velocities).
            output_file (str): Output file path. File extension typically '.vec'.
            mda_U (mda.Universe): MDAnalysis Universe object used as a reference for
                system size and atom information. Must match the total system size
                if xyz is a subset.

        Returns:
            None

        Raises:
            IOError: If the output file cannot be written.
            ValueError: If xyz dimensions don't match the expected system size.
        """
        # Copy the xyz coordinates into the dataframe
        sys_zeros = mda_U.atoms.select_atoms("all")
        sys_zeros.positions = np.zeros((mda_U.atoms.n_atoms, 3))
        vector = np.append(xyz, sys_zeros.positions, axis=0)
        sys_zeros.positions = vector[:mda_U.atoms.n_atoms]

        # Write the output file
        sys_zeros.write(output_file, file_format="NAMDBIN")


@dataclass
class SystemState:
    """Container for the system state converted from NAMD input files."""
    positions_nm:     np.ndarray        # (N, 3), nm
    velocities_nm_ps: np.ndarray        # (N, 3), nm/ps
    box_vectors_nm:   List[np.ndarray]  # [a, b, c], each (3,) nm


def _kabsch_align(mob: np.ndarray, ref: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Perform mass-weighted Kabsch (rigid-body) alignment of mobile onto reference structure.

    Computes the optimal rotation matrix that minimizes the mass-weighted root-mean-square
    deviation (RMSD) between two sets of coordinates. Uses pure numpy and MDAnalysis
    rotation_matrix function; no Universe objects required.

    Args:
        mob (np.ndarray): Mobile structure coordinates, shape (n, 3), in Ångströms.
        ref (np.ndarray): Reference structure coordinates, shape (n, 3), in Ångströms.
        masses (np.ndarray): Per-atom masses for weighting, shape (n,), in atomic mass units.

    Returns:
        aligned_mob (np.ndarray): Rotated and translated mobile coordinates superposed
            onto the reference, shape (n, 3), in Ångströms.

    Note:
        Both structures are first translated to their respective mass-weighted centers
        of mass before rotation. The final result is then translated to the reference
        center of mass.
    """
    w = masses / masses.sum()
    ref_com = w @ ref
    mob_com = w @ mob
    R, _ = mda_rotation_matrix(mob - mob_com, ref - ref_com, weights=masses)
    return (mob - mob_com) @ R.T + ref_com


class NAMDInputReader:
    """
    Converts NAMD input files to OpenMM-compatible numpy arrays.

    Called once at startup.  After this class has read the initial coordinate,
    velocity, and box files, no NAMD binary format is accessed again.

    The `.vec` combination files written by ModeExciter.combine_modes are also
    read here (once per replica setup).
    """

    @staticmethod
    def read_xsc(xscfile: str) -> List[np.ndarray]:
        """
        Parse a NAMD XSC (extended system configuration) file.

        Reads the box vector information from a NAMD XSC file, which contains
        periodic boundary condition parameters for the simulation cell.

        Args:
            xscfile (str): Path to the XSC file to parse.

        Returns:
            List[np.ndarray]: Three numpy arrays [a_nm, b_nm, c_nm], each of shape (3,),
                representing the box vectors in nanometers. These are the triclinic
                lattice vectors for the simulation cell.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or incomplete.

        Note:
            XSC file format: step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z ...
            Coordinates are converted from Ångströms (NAMD format) to nanometers.
        """
        with open(xscfile) as fh:
            lines = [l for l in fh if not l.startswith('#') and l.strip()]
        vals = list(map(float, lines[0].split()))
        # XSC columns: step a_x a_y a_z  b_x b_y b_z  c_x c_y c_z  o_x o_y o_z
        a = np.array([vals[1], vals[2], vals[3]]) * 0.1   # Å to nm
        b = np.array([vals[4], vals[5], vals[6]]) * 0.1
        c = np.array([vals[7], vals[8], vals[9]]) * 0.1
        return [a, b, c]

    @staticmethod
    def align_box_to_x(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix to align periodic box vectors to OpenMM requirements.

        OpenMM expects box vectors in a specific canonical form: the first vector along
        the x-axis, the second in the xy-plane with positive y-component, and the third
        as a general vector. This method computes the rotation matrix transforming the
        input NAMD box vectors to this standard form.

        Args:
            a (np.ndarray): First box vector, shape (3,), in nm.
            b (np.ndarray): Second box vector, shape (3,), in nm.
            c (np.ndarray): Third box vector, shape (3,), in nm.

        Returns:
            R (np.ndarray): 3×3 orthonormal rotation matrix (numpy array). Satisfies:
                - R @ a = (|a|, 0, 0)
                - R @ b = (bx, by, 0)  with by ≥ 0
                - R @ c = (cx, cy, cz)  (general)

        Note:
            The rotation matrix is computed via Gram-Schmidt orthogonalization.
            All output vectors maintain the same lengths as inputs (rotation only).
        """
        ax = a / np.linalg.norm(a)
        b_proj = b - np.dot(b, ax) * ax
        ay = b_proj / np.linalg.norm(b_proj)
        az = np.cross(ax, ay)
        R = np.vstack([ax, ay, az]).T
        return R

    @staticmethod
    def read_system(psf_file: str, coor_file: str,
                    vel_file: str, xsc_file: str) -> SystemState:
        """
        Read NAMD binary coordinate and velocity files into a SystemState object.

        Loads the initial configuration from NAMD binary files (.coor, .vel)
        and periodic boundary condition information from an XSC file. All coordinates
        are converted to nanometers and velocities to nm/ps for OpenMM compatibility.

        Args:
            psf_file (str): Path to the PSF (topology) file.
            coor_file (str): Path to NAMD binary coordinate file (.coor format).
            vel_file (str): Path to NAMD binary velocity file (.vel format).
            xsc_file (str): Path to XSC file containing box vectors.

        Returns:
            SystemState: Dataclass containing:
                - positions_nm (np.ndarray, shape (N, 3)): Atomic positions in nm.
                - velocities_nm_ps (np.ndarray, shape (N, 3)): Atomic velocities in nm/ps.
                - box_vectors_nm (list of 3 np.ndarray): Triclinic box vectors in nm.

        Raises:
            IOError: If any input file cannot be read.
            ValueError: If file format is invalid or atom counts don't match.

        Note:
            Positions are converted from Å to nm (factor: 0.1).
            Velocities are converted from AKMA units to nm/ps using the constant
            AKMA_VEL_TO_NM_PS defined at module level.
        """
        u_coor = mda.Universe(psf_file, coor_file, format='NAMDBIN')
        positions_nm = u_coor.atoms.positions * 0.1                  # Å to nm

        u_vel = mda.Universe(psf_file, vel_file, format='NAMDBIN')
        velocities_nm_ps = u_vel.atoms.positions * AKMA_VEL_TO_NM_PS # AKMA to nm/ps

        # Read box vectors from XSC file
        box_vectors_nm = NAMDInputReader.read_xsc(xsc_file)

        return SystemState(
            positions_nm=positions_nm,
            velocities_nm_ps=velocities_nm_ps,
            box_vectors_nm=box_vectors_nm,
        )

    @staticmethod
    def read_nm_vector(psf_file: str, vec_file: str) -> np.ndarray:
        """
        Read a NAMD binary .vec file (mode combination vector).

        Loads a mode combination or velocity vector stored in NAMD binary format.
        These vectors are typically generated by ModeExciter.combine_modes and
        represent superpositions of normal mode eigenvectors.

        Args:
            psf_file (str): Path to the PSF (topology) file for system definition.
            vec_file (str): Path to the NAMDBIN .vec file to read.

        Returns:
            positions (np.ndarray): Array of displacement/velocity vectors,
                shape (N, 3), in Ångströms. These are direction vectors, not
                physical positions, and do not require unit conversion.

        Raises:
            IOError: If either input file cannot be read.
            ValueError: If the PSF and VEC files have incompatible atom counts.

        Note:
            The data is returned in Ångströms without conversion, as it represents
            displacement/velocity directions rather than absolute coordinates.
        """
        u = mda.Universe(psf_file, vec_file, format='NAMDBIN')
        return u.atoms.positions.copy()   # Å

    @staticmethod
    def parse_str_box(str_file: str) -> Dict[str, Union[str, float, List[np.ndarray]]]:
        """
        Parse a CHARMM-style .str stream file and extract unit-cell information.

        The file is expected to contain lines of the form::

            set <key>  <value>

        where the relevant keys are ``xtltype``, ``a``, ``b``, ``c``,
        ``alpha``, ``beta``, and ``gamma``.  All other ``set`` directives
        are silently ignored.

        The method also computes the three triclinic basis vectors in nm
        following the standard crystallographic to Cartesian conversion:

        .. code-block:: none

            a_vec = [a, 0, 0]
            b_vec = [b·cos(γ), b·sin(γ), 0]
            c_vec = [c·cos(β),
                     c·(cos(α) − cos(β)·cos(γ)) / sin(γ),
                     c·√(1 − cos²β − ((cos(α) − cos(β)·cos(γ)) / sin(γ))²)]

        All lengths are converted Å to nm before being stored.

        Args:
            str_file (str): Path to the CHARMM .str file.

        Returns:
            dict: Dictionary with the following keys:
                - ``xtltype`` (str): Crystal type label, e.g. ``'orthorhombic'``
                - ``a``, ``b``, ``c`` (float): Cell edge lengths in Ångströms
                - ``alpha``, ``beta``, ``gamma`` (float): Cell angles in degrees
                - ``box_vectors_nm`` (list[np.ndarray]): Three (3,) arrays in nm
                  representing triclinic basis vectors, ready for ``psf.setBox()``

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or required parameters are missing.
        """
        kv: dict = {}
        _required = {'xtltype', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'}

        with open(str_file) as fh:
            for line in fh:
                stripped = line.strip()
                # Accept both "set key value" and "set key  value" (any whitespace)
                if not stripped.lower().startswith('set '):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                key   = parts[1].lower()
                value = parts[2]
                if key in _required:
                    kv[key] = value

        missing = _required - set(kv.keys())
        if missing:
            raise ValueError(
                f"parse_str_box: required key(s) {missing} not found in {str_file}"
            )

        xtltype = kv['xtltype']
        a_ang   = float(kv['a'])
        b_ang   = float(kv['b'])
        c_ang   = float(kv['c'])
        alpha   = float(kv['alpha'])   # degrees
        beta    = float(kv['beta'])
        gamma   = float(kv['gamma'])

        # Crystallographic to Cartesian (lengths stay in nm after the ×0.1 below)
        cos_a = math.cos(math.radians(alpha))
        cos_b = math.cos(math.radians(beta))
        cos_g = math.cos(math.radians(gamma))
        sin_g = math.sin(math.radians(gamma))

        cx = cos_b
        cy = (cos_a - cos_b * cos_g) / sin_g
        cz_sq = 1.0 - cx * cx - cy * cy
        if cz_sq < 0.0:
            raise ValueError(
                f"parse_str_box: non-physical cell angles in {str_file} "
                f"(α={alpha}, β={beta}, γ={gamma})"
            )
        cz = math.sqrt(cz_sq)

        # Build vectors in Å, then convert to nm
        a_vec = np.array([a_ang,            0.0,        0.0]) * 0.1
        b_vec = np.array([b_ang * cos_g, b_ang * sin_g, 0.0]) * 0.1
        c_vec = np.array([c_ang * cx,    c_ang * cy,  c_ang * cz]) * 0.1

        return {
            'xtltype':       xtltype,
            'a':             a_ang,
            'b':             b_ang,
            'c':             c_ang,
            'alpha':         alpha,
            'beta':          beta,
            'gamma':         gamma,
            'box_vectors_nm': [a_vec, b_vec, c_vec],
        }


class OpenMMRestartReader:
    """
    Reads an OpenMM XML restart file (.rst) produced by::

        state = simulation.context.getState(getPositions=True, getVelocities=True)
        with open('output.rst', 'w') as f:
            f.write(XmlSerializer.serialize(state))

    The XML ``<State>`` element carries ``<PeriodicBoxVectors>``,
    ``<Positions>``, and ``<Velocities>`` already in OpenMM-native units
    (nm and nm ps⁻¹), so **no unit conversion is performed**.  The resulting
    ``SystemState`` is structurally identical to the one produced by
    ``NAMDInputReader.read_system``, and all downstream code consumes it
    identically.

    Additionally provides ``box_vectors_to_cell``, which converts three
    Cartesian box vectors (nm) to the ``(a, b, c, α, β, γ)`` representation
    used by ``OpenMMSystemBuilder.build`` when no ``.str`` file is supplied.
    """

    @staticmethod
    def read_state(rst_file: str) -> 'SystemState':
        """
        Parse an OpenMM XML restart file and return a ``SystemState``.

        Reads ``<PeriodicBoxVectors>``, ``<Positions>``, and ``<Velocities>``
        from the ``<State>`` element.  The box-alignment rotation
        (``NAMDInputReader.align_box_to_x``) is applied so that downstream
        OpenMM context initialisation always receives box vectors in the
        canonical form required by OpenMM (a along x, b in xy-plane).

        Args:
            rst_file (str): Path to the OpenMM XML restart file.

        Returns:
            SystemState: positions in nm, velocities in nm/ps, box vectors in nm.

        Raises:
            FileNotFoundError: If ``rst_file`` does not exist.
            ValueError: If the XML is missing required ``<PeriodicBoxVectors>``,
                ``<Positions>``, or ``<Velocities>`` elements.
        """
        if not os.path.exists(rst_file):
            raise FileNotFoundError(f"OpenMM restart file not found: {rst_file}")

        tree = ET.parse(rst_file)
        root = tree.getroot()   # <State>

        # ── Periodic box vectors ──────────────────────────────────────────────
        pbv = root.find('PeriodicBoxVectors')
        if pbv is None:
            raise ValueError(
                f"{rst_file}: <PeriodicBoxVectors> element not found. "
                "Ensure the state was serialised with getPositions=True."
            )

        def _parse_vec3(elem, tag):
            el = elem.find(tag)
            if el is None:
                raise ValueError(f"{rst_file}: <{tag}> element missing inside "
                                  "<PeriodicBoxVectors>.")
            return np.array([float(el.attrib['x']),
                              float(el.attrib['y']),
                              float(el.attrib['z'])], dtype=np.float64)

        a_nm = _parse_vec3(pbv, 'A')
        b_nm = _parse_vec3(pbv, 'B')
        c_nm = _parse_vec3(pbv, 'C')

        # ── Positions ────────────────────────────────────────────────────────
        pos_elem = root.find('Positions')
        if pos_elem is None:
            raise ValueError(f"{rst_file}: <Positions> element not found.")
        positions_nm = np.array(
            [[float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])]
             for p in pos_elem.findall('Position')],
            dtype=np.float64
        )

        # ── Velocities ───────────────────────────────────────────────────────
        vel_elem = root.find('Velocities')
        if vel_elem is None:
            raise ValueError(f"{rst_file}: <Velocities> element not found.")
        velocities_nm_ps = np.array(
            [[float(v.attrib['x']), float(v.attrib['y']), float(v.attrib['z'])]
             for v in vel_elem.findall('Velocity')],
            dtype=np.float64
        )

        # ── Apply the same box-alignment rotation used by NAMDInputReader ────
        # OpenMM itself requires this canonical form; applying it here keeps
        # both input paths consistent so SimulationRunner.initialize_state
        # receives identically shaped data regardless of input engine.
        R = NAMDInputReader.align_box_to_x(a_nm, b_nm, c_nm)
        positions_nm     = (R @ positions_nm.T).T
        velocities_nm_ps = (R @ velocities_nm_ps.T).T
        a_rot = R @ a_nm;  a_rot[1] = 0.0;  a_rot[2] = 0.0
        b_rot = R @ b_nm;  b_rot[2] = 0.0
        c_rot = R @ c_nm

        return SystemState(
            positions_nm=positions_nm,
            velocities_nm_ps=velocities_nm_ps,
            box_vectors_nm=[a_rot, b_rot, c_rot],
        )

    @staticmethod
    def box_vectors_to_cell(a: np.ndarray, b: np.ndarray,
                            c: np.ndarray) -> Dict[str, Any]:
        """
        Convert three Cartesian box vectors (nm) to a ``str_box``-shaped dict.

        Produces the same structure as ``NAMDInputReader.parse_str_box`` so
        that ``OpenMMSystemBuilder.build`` can accept it without modification.
        This is used in OpenMM input mode when no ``.str`` file is provided.

        Args:
            a: First box vector in nm, shape (3,).
            b: Second box vector in nm, shape (3,).
            c: Third box vector in nm, shape (3,).

        Returns:
            dict with keys ``xtltype``, ``a``–``c`` (Å), ``alpha``–``gamma``
            (degrees), and ``box_vectors_nm`` (list of three (3,) arrays).
        """
        a_ang = float(np.linalg.norm(a)) * 10.0
        b_ang = float(np.linalg.norm(b)) * 10.0
        c_ang = float(np.linalg.norm(c)) * 10.0

        # Recover angles from dot products
        a_hat = a / np.linalg.norm(a)
        b_hat = b / np.linalg.norm(b)
        c_hat = c / np.linalg.norm(c)

        cos_alpha = float(np.clip(np.dot(b_hat, c_hat), -1.0, 1.0))
        cos_beta  = float(np.clip(np.dot(a_hat, c_hat), -1.0, 1.0))
        cos_gamma = float(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))

        alpha = math.degrees(math.acos(cos_alpha))
        beta  = math.degrees(math.acos(cos_beta))
        gamma = math.degrees(math.acos(cos_gamma))

        # Classify crystal type by angles (rough heuristic for logging only)
        def _near(v, ref, tol=0.5):
            return abs(v - ref) < tol

        if _near(alpha, 90) and _near(beta, 90) and _near(gamma, 90):
            xtltype = 'orthorhombic'
        elif _near(alpha, 90) and _near(beta, 90) and not _near(gamma, 90):
            xtltype = 'monoclinic'
        elif (_near(alpha, 109.47) and _near(beta, 109.47)
              and _near(gamma, 109.47)):
            xtltype = 'rhdo'
        else:
            xtltype = 'triclinic'

        return {
            'xtltype':        xtltype,
            'a':              a_ang,
            'b':              b_ang,
            'c':              c_ang,
            'alpha':          alpha,
            'beta':           beta,
            'gamma':          gamma,
            'box_vectors_nm': [a.copy(), b.copy(), c.copy()],
        }


class OpenMMSystemBuilder:
    """
    Builds an OpenMM System from CHARMM/NAMD inputs using native OpenMM loaders.

    Called once per run; the resulting System object is shared across all
    replicas (it encodes topology and forces, which are replica-independent).

    Force-field settings are hard-wired:
      PME, 10 Å cutoff, 8 Å switch, HBonds constraints, rigidWater=True.
    """

    # Residue-name sets used for system-type detection
    LIPID_RESIDUES = {
        'POPC', 'POPE', 'DPPC', 'DPPE', 'DLPC', 'DMPC', 'DLPE', 'DMPE',
        'DEPC', 'DSPC', 'DAPE', 'DUPE', 'DHPC', 'LYPC', 'CHL1', 'ERG',
        'POPG', 'DPPG', 'CARD', 'POPS', 'DPPS', 'DOPS', 'SAPI', 'PIPI',
        'PAPE', 'OAPE', 'LAPE', 'LPPC', 'PSM', 'DPCE', 'DXCE', 'POPE',
    }
    NUCLEIC_RESIDUES = {
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        'DA', 'DC', 'DG', 'DT', 'DI',
        'RA', 'RC', 'RG', 'RU',
        'A',  'C',  'G',  'T',  'U',  'I',
    }

    # CHARMM36m parameter files to load from charmm_toppar/
    PARAM_FILES = [
        'par_all36m_prot.prm',
        'par_all36_na.prm',
        'par_all36_carb.prm',
        'par_all36_lipid.prm',
        'par_all36_cgenff.prm',
        'par_interface.prm',
        'toppar_water_ions.str',
    ]

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize the OpenMM system builder.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def detect_system_type(self, topology: app.Topology) -> str:
        """
        Detect the system type based on residue names.

        Args:
            topology (openmm.app.Topology): The topology to analyze.

        Returns:
            str: 'membrane' if any lipid residues found,
                 'nucleic' if any nucleic acid residues found,
                 'globular' otherwise.
        """
        names = {r.name.upper() for r in topology.residues()}
        if names & self.LIPID_RESIDUES:
            return 'membrane'
        if names & self.NUCLEIC_RESIDUES:
            return 'nucleic'
        return 'globular'

    def _collect_params(self, toppar_dir: str) -> List[str]:
        """
        Collect existing parameter file paths from charmm_toppar directory.

        Args:
            toppar_dir (str): Path to the CHARMM parameter directory.

        Returns:
            list: List of existing parameter file paths.
        """
        return [os.path.join(toppar_dir, f)
                for f in self.PARAM_FILES
                if os.path.exists(os.path.join(toppar_dir, f))]

    def build(self, psf_file: str, toppar_dir: str, temperature: float = 303.15,
              pressure: float = 1.01325, str_box: Optional[Dict[str, Any]] = None) -> Tuple[app.CharmmPsfFile, mm.System, str]:
        """
        Build OpenMM system from PSF and CHARMM parameters.

        Args:
            psf_file (str): Path to the PSF topology file.
            toppar_dir (str): Directory containing CHARMM36m parameter files.
            temperature (float): Simulation temperature in K (default: 303.15).
            pressure (float): Target pressure in bar (default: 1.01325).
            str_box (dict, optional): Dictionary returned by NAMDInputReader.parse_str_box().
                When provided, its triclinic basis vectors (in nm) are passed to psf.setBox()
                so that PME grid allocation uses the real cell dimensions at System-build time.
                When None, a 10 Å orthogonal placeholder is used (legacy behaviour, not recommended).

        Returns:
            psf (openmm.app.CharmmPsfFile): Topology object with box set.
            system (openmm.System): OpenMM System ready for Simulation creation.
            system_type (str): One of 'globular', 'membrane', 'nucleic'.

        Raises:
            FileNotFoundError: If no CHARMM parameter files are found in toppar_dir.
        """
        psf = CharmmPsfFile(psf_file)
        system_type = self.detect_system_type(psf.topology)

        # Set the unit-cell box that OpenMM uses for PME grid allocation.
        # Using the real triclinic vectors from the .str file avoids the
        # silent PME mis-sizing that a dummy 10 Ang box causes.
        if str_box is not None:
            psf.setBox(
                str_box['a'] * 0.1 * unit.nanometer,
                str_box['b'] * 0.1 * unit.nanometer,
                str_box['c'] * 0.1 * unit.nanometer,
                str_box['alpha'] * unit.degree,
                str_box['beta']  * unit.degree,
                str_box['gamma'] * unit.degree,
            )
        else:
            psf.setBox(10.0, 10.0, 10.0)   # legacy placeholder (not recommended)

        # Load CHARMM parameters
        param_files = self._collect_params(toppar_dir)
        if not param_files:
            raise FileNotFoundError(
                f"No CHARMM parameter files found in {toppar_dir}. "
                "Ensure charmm_toppar.zip has been extracted."
            )
        params = CharmmParameterSet(*param_files)

        # Build System (matching conf.namd non-bonded settings)
        system = psf.createSystem(
            params,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        # Assign force groups for per-term energy decomposition.
        # Group assignments (NAMD-like naming convention):
        #   0 = HarmonicBondForce        (BOND)
        #   1 = HarmonicAngleForce       (ANGLE)
        #   2 = PeriodicTorsionForce     (DIHED)
        #   3 = CustomTorsionForce       (IMPRP — CHARMM impropers)
        #   4 = NonbondedForce           (ELEC + VDW combined; split at query time)
        #   5 = CMAPTorsionForce         (CMAP)
        #   6 = CustomNonbondedForce     (VDW correction, e.g. NBFIX)
        #   7 = CustomBondForce          (UREY-BRADLEY or extra bond corrections)
        _force_group_map = {
            mm.HarmonicBondForce:     0,
            mm.HarmonicAngleForce:    1,
            mm.PeriodicTorsionForce:  2,
            mm.CustomTorsionForce:    3,
            mm.NonbondedForce:        4,
            mm.CMAPTorsionForce:      5,
            mm.CustomNonbondedForce:  6,
            mm.CustomBondForce:       7,
        }
        for force in system.getForces():
            grp = _force_group_map.get(type(force), 31)
            force.setForceGroup(grp)

        # Add barostat
        if system_type == 'membrane':
            barostat = mm.MonteCarloMembraneBarostat(
                pressure * unit.bar,
                0.0 * unit.bar * unit.nanometer,   # surface tension = 0
                temperature * unit.kelvin,
                mm.MonteCarloMembraneBarostat.XYIsotropic,
                mm.MonteCarloMembraneBarostat.ZFree,
                25,
            )
        else:
            barostat = mm.MonteCarloBarostat(
                pressure * unit.bar,
                temperature * unit.kelvin,
                25,
            )
        system.addForce(barostat)

        print(f"{self.console.PGM_NAM}OpenMM system built: "
              f"{self.console.EXT}{system.getNumParticles()}{self.console.STD} atoms, "
              f"type={self.console.EXT}{system_type}{self.console.STD}.")
        return psf, system, system_type


class OpenMMSimulationEngine:
    """
    Persistent OpenMM simulation wrapper for one aMDeNM replica.

    The (shared) System is passed in; a fresh LangevinMiddleIntegrator and
    Context are created per replica, giving each replica an independent RNG
    seed.  Three reporters are attached once and fire automatically on every
    call to simulation.step():
      - DCDReporter:        1 frame per n_steps-step cycle
      - StateDataReporter:  1 energy/temperature row per cycle
      - CheckpointReporter: exact state every 10 * n_steps steps (10 cycles)
    """

    def __init__(self, console: ConsoleConfig, psf: app.CharmmPsfFile, system: mm.System, temperature: float,
                 platform_name: str = 'auto', n_threads: Optional[int] = None,
                 device_index: int = 0, rep_num: int = 1,
                 is_restart: bool = False, full_ener: bool = False,
                 n_steps: int = 100) -> None:
        """
        Initialize the simulation engine for a single replica.

        Args:
            console (ConsoleConfig): Console configuration object.
            psf (openmm.app.CharmmPsfFile): The PSF topology.
            system (openmm.System): The OpenMM system (shared across replicas).
            temperature (float): Simulation temperature in K.
            platform_name (str): Platform to use: 'auto', 'cuda', 'opencl', 'cpu'.
            n_threads (int, optional): Number of CPU threads for CPU platform.
            device_index (int): GPU device index for CUDA/OpenCL.
            rep_num (int): Replica number (used for output file names).
            is_restart (bool): Whether this is a restart (append to existing output files).
            full_ener (bool): If True, write per-term energy decomposition to
                rep{N}_ener_decomp.log each cycle (--full_ener flag).
            n_steps (int): Number of MD steps per excitation cycle.
        """
        self.console = console
        self.n_atoms = system.getNumParticles()
        self._temperature = temperature
        self._full_ener = full_ener

        platform, properties = self._select_platform(
            platform_name, n_threads, device_index
        )

        # Fresh integrator per replica (independent Langevin RNG state)
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,      # friction
            2.0 * unit.femtoseconds,    # timestep
        )

        self.simulation = app.Simulation(
            psf.topology, system, integrator, platform, properties
        )

        # Attach persistent reporters (append=True on restart)
        dcd_file         = f'rep{rep_num}.dcd'
        log_file         = f'rep{rep_num}.log'
        ener_decomp_file = f'rep{rep_num}_ener_decomp.log'
        self._total_steps = 0   # updated each run_cycle call; needed for 'progress'
        self.simulation.reporters.append(
            app.DCDReporter(dcd_file, n_steps, append=is_restart, enforcePeriodicBox=False)
        )
        # Full StateDataReporter: every available scalar field written to the log file
        self.simulation.reporters.append(
            app.StateDataReporter(
                log_file, n_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                append=is_restart,
            )
        )
        # Checkpoint every 10 cycles
        self.simulation.reporters.append(
            app.CheckpointReporter('checkpoint.chk', 10 * n_steps)
        )

        # If required, print per-term energy decomposition log
        if self._full_ener:
            _ener_mode = 'a' if is_restart else 'w'
            self._ener_decomp_fh = open(ener_decomp_file, _ener_mode)
            if not is_restart:
                self._ener_decomp_fh.write(
                    f"{'REP':>5} {'STEP':>6} {'BOND':>10} {'ANGLE':>10} {'DIHED':>10} "
                    f"{'IMPRP':>10} {'CMAP':>10} {'UBREY':>10} {'NBFIX':>10} "
                    f"{'NONBONDED':>12} {'POTENTIAL':>12} {'KINETIC':>10} "
                    f"{'TOTAL':>12} {'TEMP_K':>8} {'VOL_A3':>12} {'DENS_GCM3':>10}\n"
                )
                self._ener_decomp_fh.flush()

        # Map force group index
        self._force_group_labels = {
            0: 'BOND',
            1: 'ANGLE',
            2: 'DIHED',
            3: 'IMPRP',
            4: 'NONBONDED',   # elec + vdw combined (NonbondedForce)
            5: 'CMAP',
            6: 'NBFIX',
            7: 'UBREY',
        }

        print(f"{console.PGM_NAM}OpenMM platform: "
              f"{console.EXT}{platform.getName()}{console.STD}.")

    def _select_platform(self, prefer: str, n_threads: Optional[int], device_index: int) -> Tuple[Platform, Dict[str, str]]:
        """
        Select the OpenMM platform and return the corresponding properties.

        Args:
            prefer (str): Preferred platform: 'auto', 'cuda', 'opencl', 'cpu'.
            n_threads (int, optional): Number of threads for CPU platform.
            device_index (int): GPU device index for CUDA/OpenCL.

        Returns:
            platform (openmm.Platform): The selected platform.
            properties (dict): Platform-specific properties.

        Raises:
            RuntimeError: If the requested platform is not available.
        """
        gpu_props = {'DeviceIndex': str(device_index), 'Precision': 'mixed'}

        if prefer in ('cuda', 'auto'):
            try:
                return mm.Platform.getPlatformByName('CUDA'), gpu_props
            except Exception:
                if prefer == 'cuda':
                    raise RuntimeError("CUDA platform not available.")

        if prefer in ('opencl', 'auto'):
            try:
                return mm.Platform.getPlatformByName('OpenCL'), gpu_props
            except Exception:
                if prefer == 'opencl':
                    raise RuntimeError("OpenCL platform not available.")

        props = {}
        if n_threads:
            props['Threads'] = str(n_threads)
        return mm.Platform.getPlatformByName('CPU'), props

    def initialize_state(self, state: SystemState) -> None:
        """
        Push positions, velocities, and box vectors from a SystemState into the Context.

        Args:
            state (SystemState): System state object containing positions,
                velocities, and box vectors in nm and nm/ps units.
        """
        a, b, c = state.box_vectors_nm
        a, b, c = np.array(a), np.array(b), np.array(c)
        # Reduce box vectors to OpenMM's required form
        c[0] -= a[0] * np.round(c[0] / a[0])
        c[1] -= b[1] * np.round(c[1] / b[1])
        b[0] -= a[0] * np.round(b[0] / a[0])
        self.simulation.context.setPeriodicBoxVectors(
            mm.Vec3(*a) * unit.nanometer,
            mm.Vec3(*b) * unit.nanometer,
            mm.Vec3(*c) * unit.nanometer,
        )
        pos = [mm.Vec3(*p) for p in state.positions_nm]
        self.simulation.context.setPositions(
            unit.Quantity(pos, unit.nanometer)
        )
        if state.velocities_nm_ps is not None:
            vel = [mm.Vec3(*v) for v in state.velocities_nm_ps]
            self.simulation.context.setVelocities(
                unit.Quantity(vel, unit.nanometer / unit.picosecond)
            )
        else:
            self.simulation.context.setVelocitiesToTemperature(
                self._temperature * unit.kelvin
            )

    def load_checkpoint(self, chk_file: str) -> None:
        """
        Restore exact physical state (pos, vel, box, RNG) from an OpenMM checkpoint.

        Args:
            chk_file (str): Path to the checkpoint file.
        """
        self.simulation.loadCheckpoint(chk_file)

    def save_checkpoint(self, chk_file: str) -> None:
        """
        Save the exact current physical state (pos, vel, box, RNG) to an
        OpenMM checkpoint file, enabling a bit-identical continuation later.

        Args:
            chk_file (str): Output path for the checkpoint file.
        """
        self.simulation.saveCheckpoint(chk_file)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Return current (pos_nm, vel_nm_ps, box_nm) as numpy arrays from the Context.

        Returns:
            positions_nm (np.ndarray): Atomic positions in nm, shape (N,3).
            velocities_nm_ps (np.ndarray): Atomic velocities in nm/ps, shape (N,3).
            box_vectors_nm (list of np.ndarray): Three box vectors in nm, each (3,).
        """
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        return (np.array(pos), np.array(vel), [box[0], box[1], box[2]])

    def set_velocities(self, vel_nm_ps: np.ndarray) -> None:
        """
        Update only the velocity portion of the Context (used for EK rescaling).

        Args:
            vel_nm_ps (np.ndarray): New velocities in nm/ps, shape (N,3).
        """
        vel = [mm.Vec3(*v) for v in vel_nm_ps]
        self.simulation.context.setVelocities(
            unit.Quantity(vel, unit.nanometer / unit.picosecond)
        )

    def get_energy_decomposition(self) -> Dict[str, float]:
        """
        Query each force group individually and return a per-term energy breakdown
        in kcal/mol, equivalent to NAMD's ENERGY: output line.

        The NonbondedForce group (group 4) contains both electrostatics and vdW
        and cannot be split further by OpenMM without separate Force objects; it
        is reported as 'NONBONDED'. All values are in kcal/mol.

        Returns:
            dict mapping label to energy (kcal/mol), plus 'POTENTIAL', 'KINETIC',
            'TOTAL', 'TEMPERATURE', 'VOLUME', 'DENSITY'.
        """
        KJ_TO_KCAL = 1.0 / 4.184
        ctx = self.simulation.context

        energies: Dict[str, float] = {}

        # Per-force-group potential energy terms
        for grp, label in self._force_group_labels.items():
            state = ctx.getState(getEnergy=True, groups={grp})
            e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            energies[label] = e_kj * KJ_TO_KCAL

        # Bulk thermodynamic quantities (full system state)
        full = ctx.getState(getEnergy=True)
        pe = full.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = full.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        energies['POTENTIAL']   = pe * KJ_TO_KCAL
        energies['KINETIC']     = ke * KJ_TO_KCAL
        energies['TOTAL']       = (pe + ke) * KJ_TO_KCAL

        # Temperature: compute exact constrained DOF count, matching StateDataReporter.
        # getNumDegreesOfFreedom() was added in OpenMM 8.1; for older versions we
        # replicate the same formula: 3*N - 3*N_constraints - 3 (CMMotionRemover).
        system = self.simulation.system
        try:
            n_dof = system.getNumDegreesOfFreedom()
        except AttributeError:
            n_dof = 3 * system.getNumParticles()
            for i in range(system.getNumConstraints()):
                n_dof -= 1
            # Each CMMotionRemover removes 3 DOF
            for i in range(system.getNumForces()):
                if isinstance(system.getForce(i), mm.CMMotionRemover):
                    n_dof -= 3
        kB_kj = 0.008314462            # kJ/mol/K
        energies['TEMPERATURE'] = (2.0 * ke) / (n_dof * kB_kj)

        # Box volume (nm³ to Å³) and density
        box = full.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        vol_nm3 = float(np.dot(box[0], np.cross(box[1], box[2])))
        energies['VOLUME'] = vol_nm3 * 1000.0   # Å³

        # Density: sum of masses (amu) / volume (Å³), convert to g/cm³
        # 1 amu/Å³ = 1.66054 g/cm³
        total_mass_amu = sum(
            self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(self.n_atoms)
        )
        vol_cm3 = vol_nm3 * 1e-21          # nm³ to cm³
        energies['DENSITY'] = (total_mass_amu * 1.66054e-24) / vol_cm3  # g/cm³

        return energies

    def write_energy_decomposition(self, step: int, rep: int) -> None:
        """
        Append one row of per-term energies (kcal/mol) to the replica's
        ener_decomp.log file. Fixed-width columns match the header written
        in __init__; no output is sent to stdout.

        Args:
            step (int): Current cycle number.
            rep  (int): Replica number.
        """
        e = self.get_energy_decomposition()
        row = (
            f"{rep:>5d} {step:>6d} "
            f"{e.get('BOND',        0.0):>10.3f} "
            f"{e.get('ANGLE',       0.0):>10.3f} "
            f"{e.get('DIHED',       0.0):>10.3f} "
            f"{e.get('IMPRP',       0.0):>10.3f} "
            f"{e.get('CMAP',        0.0):>10.3f} "
            f"{e.get('UBREY',       0.0):>10.3f} "
            f"{e.get('NBFIX',       0.0):>10.3f} "
            f"{e.get('NONBONDED',   0.0):>12.3f} "
            f"{e.get('POTENTIAL',   0.0):>12.3f} "
            f"{e.get('KINETIC',     0.0):>10.3f} "
            f"{e.get('TOTAL',       0.0):>12.3f} "
            f"{e.get('TEMPERATURE', 0.0):>8.2f} "
            f"{e.get('VOLUME',      0.0):>12.2f} "
            f"{e.get('DENSITY',     0.0):>10.5f}\n"
        )
        self._ener_decomp_fh.write(row)
        self._ener_decomp_fh.flush()

    def close(self) -> None:
        """Close open file handles (call after the simulation loop finishes)."""
        if self._full_ener and hasattr(self, '_ener_decomp_fh') and not self._ener_decomp_fh.closed:
            self._ener_decomp_fh.close()

    def detach_main_dcd(self) -> None:
        """Remove the main replica DCD reporter (index 0) so that the
        de-excitation run does not write extra frames into rep{N}.dcd."""
        self.simulation.reporters.pop(0)

    def run_cycle(self, n_steps: int = 100, rep: int = 0, loop: int = 0) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Advance the simulation by n_steps.

        DCDReporter and StateDataReporter fire automatically.
        CheckpointReporter fires every 10 * n_steps steps (10 cycles).
        If --full_ener was set, per-term energy decomposition is written to
        rep{N}_ener_decomp.log after each cycle.

        Args:
            n_steps (int): Number of integration steps to run.
            rep (int): Replica number, passed through to the energy log.
            loop (int): Current cycle number, passed through to the energy log.

        Returns:
            pos_nm (np.ndarray): Atomic positions in nm, shape (N,3).
            vel_nm_ps (np.ndarray): Atomic velocities in nm/ps, shape (N,3).
            box_nm (list of np.ndarray): Three box vectors in nm, each (3,).
        """
        self.simulation.step(n_steps)
        if self._full_ener:
            self.write_energy_decomposition(step=loop, rep=rep)
        return self.get_state()


def make_reference_universe(psffile: str, positions_ang: np.ndarray) -> mda.Universe:
    """
    Build an MDAnalysis Universe from a PSF topology and a positions array.

    Replaces the pattern ``mda.Universe(psffile, coorfile, format='NAMDBIN')``
    used throughout the codebase solely to obtain topology + real coordinates.
    Works regardless of which input engine (NAMD or OpenMM) produced the run.

    Args:
        psffile (str): Path to the PSF topology file.
        positions_ang (np.ndarray): Atomic positions in Å, shape (N, 3).

    Returns:
        mda.Universe: Universe with ``atoms.positions`` set to ``positions_ang``.
    """
    # positions_ang must be shaped (1, N, 3) for MemoryReader (1 frame).
    # This creates a topology-only Universe with a synthetic single-frame
    # trajectory, which makes u.atoms.positions immediately accessible
    # without requiring any on-disk trajectory file.
    u = mda.Universe(psffile,
                     positions_ang[np.newaxis, :, :].astype(np.float32),
                     format=MemoryReader,
                     n_atoms=positions_ang.shape[0])
    return u


def save_reference_state(input_dir: str, positions_ang: np.ndarray, box_vectors_nm: List[np.ndarray], velocities_nm_ps: Optional[np.ndarray] = None) -> None:
    """
    Persist the initial reference positions, box vectors, and velocities to
    ``inputs/``.

    Written once at ``run`` time so that ``freeenergy`` and any future
    analysis subcommands can recover the reference state without re-reading
    engine-specific input files. Persisting velocities is required so that
    ``restart``/``append`` can correctly initialise replicas that had not yet
    started a single cycle (no OpenMM checkpoint to load velocities from).

    Files written:
        ``{input_dir}/init_reference_positions_ang.npy``
            Shape (N, 3), float64, Å.
        ``{input_dir}/init_reference_box_nm.npy``
            Shape (3, 3), float64, nm (rows = a, b, c vectors).
        ``{input_dir}/init_reference_velocities_nm_ps.npy``
            Shape (N, 3), float64, nm/ps. Only written if ``velocities_nm_ps``
            is provided (kept optional for backward compatibility).

    Args:
        input_dir (str): Path to the ``inputs/`` directory.
        positions_ang (np.ndarray): Atomic positions in Å, shape (N, 3).
        box_vectors_nm (list of np.ndarray): Three box vectors in nm.
        velocities_nm_ps (np.ndarray, optional): Atomic velocities in nm/ps,
            shape (N, 3).
    """
    np.save(os.path.join(input_dir, 'init_reference_positions_ang.npy'),
            positions_ang.astype(np.float64))
    np.save(os.path.join(input_dir, 'init_reference_box_nm.npy'),
            np.array(box_vectors_nm, dtype=np.float64))
    if velocities_nm_ps is not None:
        np.save(os.path.join(input_dir, 'init_reference_velocities_nm_ps.npy'),
                velocities_nm_ps.astype(np.float64))


def load_reference_state(input_dir: str) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
    """
    Load the saved reference positions, box vectors, and velocities from
    ``inputs/``.

    Args:
        input_dir (str): Path to the ``inputs/`` directory.

    Returns:
        positions_ang (np.ndarray): Shape (N, 3), float64, Å.
        box_vectors_nm (list of np.ndarray): Three (3,) arrays in nm.
        velocities_nm_ps (np.ndarray or None): Shape (N, 3), float64, nm/ps.
            ``None`` if the file predates this feature (older pyAdMD run),
            in which case callers must fall back to re-reading engine-specific
            input files before starting any not-yet-begun replica.

    Raises:
        FileNotFoundError: If the position or box .npy file is missing (run
            not yet completed or produced by a very old version of pyAdMD).
    """
    pos_path = os.path.join(input_dir, 'init_reference_positions_ang.npy')
    box_path = os.path.join(input_dir, 'init_reference_box_nm.npy')
    vel_path = os.path.join(input_dir, 'init_reference_velocities_nm_ps.npy')
    for p in (pos_path, box_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Reference state file not found: {p}\n"
                "Re-run 'pyAdMD.py run' with the current version to generate it."
            )
    positions_ang  = np.load(pos_path)
    box_arr        = np.load(box_path)   # (3, 3)
    box_vectors_nm = [box_arr[0], box_arr[1], box_arr[2]]
    velocities_nm_ps = np.load(vel_path) if os.path.exists(vel_path) else None
    return positions_ang, box_vectors_nm, velocities_nm_ps


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

    def __init__(self, console: ConsoleConfig, args: argparse.Namespace, cwd: str, input_dir: str,
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
        if self.args.modeltype.lower() == 'ca':
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
                nm_type=self.args.modeltype.lower(),
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
            if self.args.modeltype.lower() == 'ca':
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
        prefix = "ca" if self.args.modeltype.lower() == 'ca' else "heavy"

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
        if self.args.modeltype.lower() == 'ca':
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
        self.nm_type   = getattr(run_args, 'modeltype', 'CA').lower()
        self.replicas  = getattr(run_args, 'replicas', 10)
        self._temperature = float(getattr(args_fe, 'temp', 303.15))
        self.n_steps      = 100   # steps per cycle, same as run phase

        # Derive input-engine type from saved parameters (default NAMD for
        # backward compatibility with runs produced before -itype was added).
        self._input_engine = getattr(run_args, 'inputtype', 'NAMD').upper()

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
              "skipping expensive recomputation. --cutoff and "
              "--max_centroids are re-applied fresh below.")
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

                print(f"{self.console.PGM_NAM}De-excitation phase {phase_idx + 1}/4: "
                      f"k_bb={k_bb_kcal:.3f}, k_sc={k_sc_kcal:.3f} kcal/mol/Å² "
                      f"({phase_ps:.1f} ps)...")
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
                      f"found for centroid (frame {frame_idx}); falling "
                      f"back to the last frame of {self.console.WRN}"
                      f"prod.dcd{self.console.STD} (velocities re-assigned "
                      "at target temperature — continuation is physically "
                      "valid but not bit-identical).")
                fallback_state = self._load_last_frame_as_state("prod.dcd")
                engine.initialize_state(fallback_state)

            engine.simulation.reporters.append(
                app.DCDReporter("prod.dcd", self.n_steps,
                                append=True, enforcePeriodicBox=False)
            )

            additional_ps = additional_cycles * self.n_steps * 0.002
            print(f"{self.console.PGM_NAM}Extending production for "
                  f"centroid (frame {frame_idx}) by "
                  f"{self.console.EXT}{additional_ps:.1f}{self.console.STD} "
                  f"ps ({additional_cycles} cycles)...")
            engine.simulation.step(additional_cycles * self.n_steps)

            engine.save_checkpoint(self._PROD_CHECKPOINT_FILE)
            engine.close()
            return prod_dcd_path

        except Exception as exc:
            print(f"{self.console.PGM_ERR}Extending centroid (frame "
                  f"{frame_idx}) production failed: "
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
                "Re-run 'pyAdMD.py run' with the current version to generate "
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
                    print(f"{self.console.PGM_WRN}Mode {mode_num} Cα vector is "
                          "near-zero after extraction; skipping.")
                    continue
                mode_vectors_ca[mode_num] = q_ca / norm
            except FileNotFoundError as exc:
                print(f"{self.console.PGM_WRN}Mode file not found for mode "
                      f"{mode_num}: {exc}. Skipping.")

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
            print(f"{self.console.PGM_WRN}2D FEL for modes {m1}×{m2} has no "
                  "populated bins; skipping plot.")
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
                      f"(frame {frame_idx}, "
                      f"cluster size {cluster['size']})...")
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
            print(f"{self.console.PGM_ERR}Run 'pyAdMD.py restart' or "
                  "'pyAdMD.py append' to complete them first, then re-run "
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
            print(f"{self.console.PGM_ERR}Re-run 'pyAdMD.py freeenergy' to "
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


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the pyAdMD analysis tool.

    This function sets up the argument parser with subcommands for different
    operational modes of the pyAdMD tool, including running simulations,
    restarting, appending, analyzing, and cleaning.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments

    Raises:
        SystemExit: If invalid arguments are provided or help is requested
    """
    console = ConsoleConfig()

    parser = argparse.ArgumentParser(description=console.MESSAGE)
    subparsers = parser.add_subparsers(dest='option', help='Available commands')

    # RUN subparser
    opt_run = subparsers.add_parser('run', help="Setup and run simulations")

    # Required arguments for run
    run_required = opt_run.add_argument_group('Required arguments')
    run_required.add_argument('-itype', '--inputtype', action="store", type=str.upper,
                             required=True, choices=["NAMD", "OPENMM"],
                             help="Input engine type: NAMD (binary .coor/.vel/.xsc) or "
                                  "OPENMM (XML restart .rst)")
    run_required.add_argument('-type', '--modeltype', action="store", type=str.upper,
                             default="CA", required=True, choices=["CA", "HEAVY", "CHARMM"],
                             help="Compute ENM or use pre-computed CHARMM normal mode file (default: CA)")
    run_required.add_argument('-psf', '--psffile', action="store", type=str, required=True,
                             help="PSF topology file")
    run_required.add_argument('-pdb', '--pdbfile', action="store", type=str, required=True,
                             help="PDB structure file")

    # NAMD-mode inputs (required when -itype NAMD)
    run_namd = opt_run.add_argument_group('NAMD input files (required when -itype NAMD)')
    run_namd.add_argument('-coor','--coorfile', action="store", type=str, default=None,
                         help="NAMD binary coordinates file (.coor)")
    run_namd.add_argument('-vel', '--velfile', action="store", type=str, default=None,
                         help="NAMD binary velocities file (.vel)")
    run_namd.add_argument('-xsc', '--xscfile', action="store", type=str, default=None,
                         help="NAMD extended system configuration file (.xsc)")
    run_namd.add_argument('-str', '--strfile', action="store", type=str, default=None,
                         help="CHARMM-style stream file for box info and force-field parameters "
                              "(required in NAMD mode; optional in OPENMM mode)")

    # OpenMM-mode inputs (required when -itype OPENMM)
    run_openmm = opt_run.add_argument_group('OpenMM input files (required when -itype OPENMM)')
    run_openmm.add_argument('-rst', '--rstfile', action="store", type=str, default=None,
                            help="OpenMM XML restart file (.rst) written by XmlSerializer.serialize(state) "
                                 "with getPositions=True, getVelocities=True")

    # Optional arguments for run
    run_optional = opt_run.add_argument_group('Optional arguments')
    run_optional.add_argument('-mod', '--modefile', action="store", type=str,
                             help="CHARMM normal mode file (required when type is CHARMM)")
    run_optional.add_argument('-nm', '--modes', action="store", type=str, default="7,8,9",
                             help="Normal modes to excite separated by commas (default: 7,8,9)")
    run_optional.add_argument('-ek', '--energy', action="store", type=float, default=0.125,
                             help="Excitation energy (default: 0.125 kcal/mol)")
    run_optional.add_argument('-t', '--time', action="store", type=int, default=250,
                             help="Total simulation time (default: 250ps)")
    run_optional.add_argument('-sel', '--selection', action="store", type=str, default="protein",
                             help="Atom selection to apply the energy injection (default: protein)")
    run_optional.add_argument('-rep', '--replicas', action="store", type=int, default=10,
                             help="Number of aMDeNM replicas to run (default: 10)")

    # Flags for run
    run_flags = opt_run.add_argument_group('Flags')
    run_flags.add_argument('-n', '--no_correc', action='store_true',
                          help='Compute standard MDeNM calculations')
    run_flags.add_argument('-f', '--fixed', action='store_true',
                          help='Disable excitation vector correction and keep constant excitation energy injections')
    run_flags.add_argument('-r', '--recalc', action='store_true',
                          help='Recompute ENM modes instead of correcting excitation direction when needed')
    run_flags.add_argument('--full_ener', action='store_true',
                          help='Write per-term energy decomposition (BOND, ANGLE, DIHED, etc.) '
                               'to rep{N}_ener_decomp.log each cycle')

    # RESTART subparser
    subparsers.add_parser('restart', help="Restart unfinished simulations")

    # APPEND subparser
    opt_apnd = subparsers.add_parser('append', help="Extend previously computed simulations")
    opt_apnd.add_argument('-t', '--time', action="store", type=int, required=True, default=100,
                         help="Simulation time to append (default: 100ps)")

    # ANALYSIS subparser
    opt_analyze = subparsers.add_parser('analyze', help="Analyze simulation results and generate plots")
    opt_analyze.add_argument('-src', '--source', action="store", type=str.lower,
                            default="pyadmd", choices=["pyadmd", "freeenergy"],
                            help="Trajectory source to analyze: 'pyadmd' for rep{N}.dcd "
                                 "replica trajectories (default), or 'freeenergy' for "
                                 "centroid production trajectories from a completed "
                                 "'freeenergy' run")
    opt_analyze.add_argument('-r', '--rough', action='store_true',
                            help='Perform rough analysis (every 5ps instead of every frame)')

    # Optional skip flags for analysis
    analyze_skip = opt_analyze.add_argument_group('Skip flags (disable individual analyses)')
    analyze_skip.add_argument('--no_rmsd', action='store_true',
                              help='Skip RMSD calculation')
    analyze_skip.add_argument('--no_rg', action='store_true',
                              help='Skip radius of gyration calculation')
    analyze_skip.add_argument('--no_sasa', action='store_true',
                              help='Skip SASA calculation')
    analyze_skip.add_argument('--no_hp', action='store_true',
                              help='Skip hydrophobic exposure calculation')
    analyze_skip.add_argument('--no_rmsf', action='store_true',
                              help='Skip RMSF calculation')
    analyze_skip.add_argument('--no_dssp', action='store_true',
                              help='Skip secondary structure (DSSP) calculation')

    # FREEENERGY subparser
    opt_fe = subparsers.add_parser(
        'freeenergy',
        help="Compute free energy landscapes (Costa et al. JCTC 2015/2023 protocol)"
    )
    fe_params = opt_fe.add_argument_group('Optional parameters')
    fe_params.add_argument(
        '-c', '--cutoff', type=float, default=0.8, metavar='Å',
        help='GROMOS RMSD clustering cutoff in Å (default: 0.8)')
    fe_params.add_argument(
        '-d', '--deexcite', type=int, default=200, metavar='PS',
        help='Total de-excitation MD length per centroid in ps, split over 4 '
             'restraint phases (default: 200)')
    fe_params.add_argument(
        '-p', '--production', type=int, default=800, metavar='PS',
        help='Unrestrained production MD length per centroid in ps (default: 800)')
    fe_params.add_argument(
        '-nm', '--modes', type=str, default=None, metavar='MODES',
        help='Comma-separated mode indices for FEL projection '
             '(default: same as run, e.g. 7,8,9)')
    fe_params.add_argument(
        '--modes_2d', type=str, default=None, metavar='PAIRS',
        help='Mode pairs for 2D FEL plots as space-separated "m1,m2" tokens, '
             'e.g. "7,8 7,9 8,9". Default: all pairwise combinations of --modes.')
    fe_params.add_argument(
        '-b', '--bins', type=int, default=50, metavar='N',
        help='Number of histogram bins for FEL (default: 50)')
    fe_params.add_argument(
        '-T', '--temp', type=float, default=303.15, metavar='K',
        help='Temperature for kBT scaling (default: 303.15 K)')
    fe_params.add_argument(
        '-s', '--sel', type=str, default="protein and name CA",
        metavar='SEL',
        help='MDAnalysis selection string for GROMOS RMSD clustering '
             '(default: "protein and name CA")')
    fe_params.add_argument(
        '--max_centroids', type=int, default=50, metavar='N',
        help='Maximum number of centroids submitted to MD. When the cluster '
             'count exceeds this value, exactly N centroids are selected by '
             'greedy farthest-point (MaxMin) sampling to maximise '
             'conformational diversity (default: 50)')

    # CLEAN subparser
    subparsers.add_parser('clean', help="Erase all previous simulation files")

    # Parse arguments
    args = parser.parse_args()

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Check if a subcommand was provided but no additional arguments
    if hasattr(args, 'option') and len(sys.argv) == 2:
        if args.option == 'run':
            opt_run.print_help()
        elif args.option == 'append':
            opt_apnd.print_help()

    # Validate conditional requirements
    if hasattr(args, 'option') and args.option == 'run':
        if args.modeltype == 'CHARMM' and not args.modefile:
            opt_run.error("The -mod/--modefile argument is required when -type/--type is CHARMM")

    if hasattr(args, 'option') and args.option == 'run':
        if args.modeltype == 'CHARMM' and args.recalc:
            opt_run.error("ENM recalculation is not compatible with CHARMM normal modes")

    # Validate input-engine-specific file requirements
    if hasattr(args, 'option') and args.option == 'run':
        if args.inputtype == 'NAMD':
            missing = [flag for flag, val in [("-coor", args.coorfile),
                                               ("-vel",  args.velfile),
                                               ("-xsc",  args.xscfile),
                                               ("-str",  args.strfile)]
                       if val is None]
            if missing:
                opt_run.error(
                    f"The following arguments are required in NAMD mode: "
                    f"{', '.join(missing)}"
                )
        elif args.inputtype == 'OPENMM':
            if args.rstfile is None:
                opt_run.error(
                    "The -rst/--rstfile argument is required when -itype is OPENMM"
                )

    return args


def unzip_file(filepath: str, dest_dir: str) -> None:
    """
    Extract files from a .zip compressed file.

    Args:
        filepath (str): Path to .zip file to extract.
        dest_dir (str): Path to destination folder for extracted files.
    """
    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(dest_dir)


def write_charmm_nm(nms_to_write: str, psffile: str, modefile: str, cwd: str) -> None:
    """
    Write CHARMM normal mode vectors in NAMD readable format.

    Args:
        nms_to_write (str): Comma-separated list of normal mode numbers to write.
        psffile (str): Path to PSF topology file.
        modefile (str): Path to CHARMM mode file.
        cwd (str): Current working directory path.

    Raises:
        SystemExit: If CHARMM execution fails.
    """
    console = ConsoleConfig()

    # Extract CHARMM topology and parameters files
    unzip_file(f"{cwd}/inputs/charmm_toppar.zip", f"{cwd}/inputs")

    # Create input file listing modes to process
    nms = [f"{t}\n" for t in nms_to_write.split(',')]
    with open(f"{cwd}/inputs/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)

    # Execute CHARMM to generate mode vectors in NAMD format
    os.chdir(f"{cwd}/tools")
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]} modfile={modefile.split('/')[-1]} -o ../wrt-nm.out")

    returned_value = subprocess.call(cmd, shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{console.PGM_ERR}An error occurred while writing the normal mode vectors.")
        print(f"{console.PGM_ERR}Inspect the file {console.ERR}wrt-nm.out{console.STD} for detailed information.")
        sys.exit(1)

    # Return to cwd folder
    os.chdir(f"{cwd}")


def _count_dcd_frames(dcd_path: str) -> int:
    """
    Return the number of frames stored in a DCD file by reading its binary
    header directly — no topology required, immune to atom-count mismatches.

    DCD CORD header layout (CHARMM / OpenMM convention, little-endian):
      bytes  0- 3 : block length (int32)  = 84
      bytes  4- 7 : magic string 'CORD'
      bytes  8-11 : NSET  — total number of frames (int32)

    Args:
        dcd_path (str): Path to the DCD trajectory file.

    Returns:
        int: Number of frames recorded in the DCD header, or 0 on any error.
    """
    try:
        with open(dcd_path, 'rb') as fh:
            fh.read(4)          # block length (84)
            magic = fh.read(4)  # 'CORD'
            if magic not in (b'CORD', b'VELD'):
                return 0
            n_frames = struct.unpack('<i', fh.read(4))[0]
        return max(0, n_frames)
    except Exception:
        return 0


def find_last_completed_cycle(rep_dir: str) -> int:
    """
    Find the last completed cycle in a replica directory.

    In the OpenMM backend no per-cycle step_N.coor files are written; instead
    the authoritative record is the 'cycle' field inside correction_state.json,
    which is updated every 10 cycles by _save_correction_state().  If that file
    is absent (very early crash), the DCD trajectory frame count is used as a
    rough lower bound.

    Args:
        rep_dir (str): Path to replica directory to scan.

    Returns:
        int: Highest completed cycle number, or 0 if none can be determined.
    """
    # Primary: correction_state.json written by _save_correction_state every 10 cycles
    cs_path = os.path.join(rep_dir, "correction_state.json")
    if os.path.exists(cs_path):
        try:
            with open(cs_path) as fh:
                cs = json.load(fh)
            cycle = int(cs.get('cycle', 0))
            if cycle > 0:
                return cycle
        except Exception:
            pass

    # Secondary: count DCD frames by reading the binary header directly.
    dcd_files = glob.glob(f"{rep_dir}/rep*.dcd")
    if dcd_files:
        dcd_files.sort(key=os.path.getmtime, reverse=True)
        n_frames = _count_dcd_frames(dcd_files[0])
        if n_frames > 0:
            return n_frames   # 1 DCD frame per cycle (DCDReporter period = n_steps steps)

    return 0


def check_pyadmd_completion(params: Dict[str, Any]) -> List[Tuple[int, int, int]]:
    """
    Verify that every pyAdMD replica has reached its target cycle count.

    Used by ``analyze -src pyadmd`` as a hard gate before any analysis
    runs: mirrors the assumption the time axis already makes (that
    ``sim_time``/``end_loop`` reflects reality) but makes it an explicit,
    verified precondition instead of a silent one.

    Args:
        params (dict): Parameters dict as loaded from ``pyAdMD_params.json``
            (raw JSON, i.e. ``params['args']`` is a plain dict, as returned
            by ``Analyzer._load_parameters``).

    Returns:
        list[tuple[int, int, int]]: One ``(replica, last_completed_cycle,
            target_cycle)`` tuple per replica that has **not** reached
            ``params['end_loop']``. An empty list means every replica is
            complete.

    Raises:
        ValueError: If ``end_loop`` is missing from ``params`` (e.g. a
            corrupted or pre-existing ``pyAdMD_params.json``), since
            completion cannot be verified without it.
    """
    end_loop = params.get('end_loop')
    if end_loop is None:
        raise ValueError(
            "'end_loop' missing from pyAdMD_params.json; cannot verify "
            "replica completion."
        )

    cwd      = params.get('cwd', os.getcwd())
    args     = params.get('args', {})
    replicas = args.get('replicas', 10)

    incomplete = []
    for rep in range(1, replicas + 1):
        rep_dir    = f"{cwd}/rep{rep}"
        last_cycle = find_last_completed_cycle(rep_dir)
        if last_cycle < end_loop:
            incomplete.append((rep, last_cycle, end_loop))
    return incomplete


def check_freeenergy_completion(cwd: str) -> List[Tuple[int, int, int]]:
    """
    Verify that every centroid's production MD has reached its target
    cycle count.

    Reads ``freeenergy/clustering_summary.csv`` (written by
    ``FreeEnergyCalculator._save_clustering_summary``), which already
    tracks ``production_cycles_done``/``production_cycles_target`` per
    centroid, so no cycle-count recomputation is needed here.

    Args:
        cwd (str): Working directory containing the ``freeenergy/`` output
            folder (same directory a ``run``/``freeenergy`` call was made
            from).

    Returns:
        list[tuple[int, int, int]]: One ``(centroid_frame,
            production_cycles_done, production_cycles_target)`` tuple per
            centroid that has **not** reached its target. An empty list
            means every centroid is complete.

    Raises:
        FileNotFoundError: If ``freeenergy/clustering_summary.csv`` does
            not exist (no ``freeenergy`` run has completed at all).
    """
    csv_path = f"{cwd}/freeenergy/clustering_summary.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'pyAdMD.py freeenergy' first."
        )

    df = pd.read_csv(csv_path)
    incomplete = []
    for _, row in df.iterrows():
        done   = int(row['production_cycles_done'])
        target = int(row['production_cycles_target'])
        if done < target:
            incomplete.append((int(row['centroid_frame']), done, target))
    return incomplete


def main() -> None:
    """Main entry point for the PyAdMD application.

    Handles command-line argument parsing, initialization of components, and
    execution of the requested operation (run, restart, append, or clean).

    The function performs the following primary operations:
    - Parses command-line arguments
    - Initializes console configuration and component classes
    - Handles different operational modes (run, restart, append, analyze, clean)
    - Manages simulation setup, execution, and cleanup
    - Coordinates file operations and parameter storage

    Raises:
        SystemExit: If required files are missing or critical errors occur during execution
    """
    console = ConsoleConfig()

    # Print banner
    banner = (f"{console.BLK}{console.LOGO}{console.STD}\n"
              f"\t\t{console.TLE}Adaptive Molecular Dynamics with Python{console.STD}\n"
              f"\t\t\t     version: {console.VERSION}\n"
              f"\n{console.CITATION}\n")

    print(banner)
    print(__doc__)

    # Parse command line arguments
    args = parse_arguments()

    # Get working directory path
    cwd = os.getcwd()
    input_dir = f"{cwd}/inputs"

    # Initialize component classes
    enm_calculator = ENMCalculator(console)
    mode_exciter = ModeExciter(console)
    param_storage = ParameterStorage(console)

    # Define variables that will be used in multiple blocks
    psffile = pdbfile = coorfile = velfile = xscfile = strfile = modefile = None
    nm_type = energy = selection = replicas = time = None
    nm_parsed = factors = end_loop = None
    sel_mass = init_coor = sys_pdb = None

    ### RUN
    if args.option == 'run':
        print(f"{console.PGM_NAM}{console.TLE}Setup and run aMDeNM simulations{console.STD}\n")

        # Common required files
        psffile = args.psffile
        pdbfile = args.pdbfile
        if args.modefile:
            modefile = args.modefile

        common_files = [psffile, pdbfile]
        if args.modefile:
            common_files.append(modefile)

        # Engine-specific files
        itype = args.inputtype   # 'NAMD' or 'OPENMM'

        if itype == 'NAMD':
            coorfile = args.coorfile
            velfile  = args.velfile
            xscfile  = args.xscfile
            strfile  = args.strfile
            engine_files = [coorfile, velfile, xscfile, strfile]
            rstfile  = None
        else:   # OPENMM
            rstfile  = args.rstfile
            coorfile = velfile = xscfile = None
            strfile  = args.strfile   # optional
            engine_files = [rstfile]
            if strfile:
                engine_files.append(strfile)

        # Existence check + copy to inputs/ folder
        for file in common_files + engine_files:
            if not os.path.isfile(file):
                print(f"{console.PGM_ERR}File {file.split('/')[-1]} not found.")
                sys.exit(1)
            if not os.path.isfile(f"{input_dir}/{file.split('/')[-1]}"):
                shutil.copy(file, input_dir)
                print(f"{console.PGM_WRN}File {console.WRN}{file.split('/')[-1]}{console.STD} "
                      "was copied to inputs folder.")

        # Canonicalise paths to inputs/ folder
        psffile = f"{input_dir}/{psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{pdbfile.split('/')[-1]}"
        if args.modefile:
            modefile = f"{input_dir}/{modefile.split('/')[-1]}"

        if itype == 'NAMD':
            coorfile = f"{input_dir}/{coorfile.split('/')[-1]}"
            velfile  = f"{input_dir}/{velfile.split('/')[-1]}"
            xscfile  = f"{input_dir}/{xscfile.split('/')[-1]}"
            strfile  = f"{input_dir}/{strfile.split('/')[-1]}"
        else:
            rstfile = f"{input_dir}/{rstfile.split('/')[-1]}"
            if strfile:
                strfile = f"{input_dir}/{strfile.split('/')[-1]}"

        # Build initial SystemState (engine-specific)
        if itype == 'NAMD':
            print(f"{console.PGM_NAM}Reading initial NAMD inputs (one-time conversion)...")
            init_state = NAMDInputReader.read_system(psffile, coorfile, velfile, xscfile)
            # Rotate to canonical OpenMM box orientation
            a, b, c = init_state.box_vectors_nm
            R = NAMDInputReader.align_box_to_x(a, b, c)
            init_state.positions_nm     = (R @ init_state.positions_nm.T).T
            init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
            a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
            b_rot = R @ b;  b_rot[2] = 0.0
            c_rot = R @ c
            init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
        else:
            print(f"{console.PGM_NAM}Reading OpenMM restart file: "
                  f"{console.EXT}{rstfile.split('/')[-1]}{console.STD}...")
            init_state = OpenMMRestartReader.read_state(rstfile)
            # (align_box_to_x rotation already applied inside read_state)

        # Reference positions in Å (used for ENM, sys_coor, reference state)
        init_pos_ang = init_state.positions_nm * 10.0   # nm → Å

        # Build reference Universe (topology + real positions, engine-agnostic)
        sys_coor = make_reference_universe(psffile, init_pos_ang)
        sys_mass = sys_coor.atoms.masses          # System atomic mass
        n_atoms  = sys_coor.atoms.n_atoms         # System number of atoms
        sel_mass = sys_coor.atoms.select_atoms(args.selection).masses

        # Save reference positions, box, and velocities for downstream subcommands
        save_reference_state(input_dir, init_pos_ang, init_state.box_vectors_nm,
                              init_state.velocities_nm_ps)
        print(f"{console.PGM_NAM}Reference state saved to "
              f"{console.EXT}inputs/init_reference_*.npy{console.STD}.")

        # Store parameters
        nm_type  = args.modeltype.lower()
        modes    = args.modes
        nm_parsed = [int(s) for s in modes.split(',')]
        energy   = args.energy
        sim_time = args.time
        selection = args.selection
        replicas = args.replicas

        # Derive base_name for ENM output directory naming.
        # NAMD: coorfile prefix; OPENMM: rstfile prefix.
        if itype == 'NAMD':
            base_name = os.path.splitext(os.path.basename(coorfile))[0]
        else:
            base_name = os.path.splitext(os.path.basename(rstfile))[0]

        # Store in args for SimulationRunner access
        n_steps = 100  # MD steps per excitation cycle (2 fs/step → 0.2 ps/cycle)
        args.n_steps = n_steps
        end_loop = int(sim_time / (n_steps * 0.002))

        # Compute / write normal mode vectors
        if nm_type == "ca":
            print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Cα ENM"
                  f"{console.STD}{console.HGH} and writing normal mode vectors "
                  f"{console.EXT}{modes}{console.STD}.")
            enm_calculator.compute_enm(init_pos_ang, base_name, nm_type,
                                       nm_parsed, input_dir, psffile)
        elif nm_type == "heavy":
            print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Heavy atoms ENM"
                  f"{console.STD}{console.HGH} and writing normal mode vectors "
                  f"{console.EXT}{modes}{console.STD}.")
            enm_calculator.compute_enm(init_pos_ang, base_name, nm_type,
                                       nm_parsed, input_dir, psffile)
        elif nm_type == "charmm":
            print(f"\n{console.PGM_NAM}Writing {console.EXT}CHARMM{console.STD} "
                  f"normal mode vectors {console.EXT}{modes}{console.STD}.")
            write_charmm_nm(modes, psffile, modefile, cwd)

        # Extract NAMD topology and parameters files
        unzip_file(f"{input_dir}/charmm_toppar.zip", input_dir)

        # Generate mode combinations
        print(f"\n{console.PGM_NAM}Generating {console.EXT}{replicas}{console.STD} uniformly "
              f"distributed combinations of modes {console.EXT}{modes}{console.STD}.")
        factors = mode_exciter.generate_factors(
            replicas, len(nm_parsed), cwd, nm_parsed, nm_type, base_name, sys_coor
        )
        mode_exciter.combine_modes(replicas, factors, cwd, sys_coor)

        # Save parameters for potential restart/append
        param_storage.save_parameters(args, factors, nm_parsed, end_loop, cwd)

        # Initialize and run SimulationRunner
        sim_runner = SimulationRunner(
            console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
            xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
            mode_exciter, init_state=init_state,
        )
        for rep in range(1, replicas + 1):
            sim_runner.run_simulation(rep, 0, end_loop)

    ### RESTART
    elif args.option == 'restart':
        print(f"{console.PGM_NAM}{console.TLE}Restart unfinished pyAdMD simulation{console.STD}\n")

        # Load parameters
        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        args      = params['args']
        factors   = params['factors']
        nm_parsed = params['nm_parsed']
        end_loop  = params['end_loop']
        cwd       = params['cwd']

        # Reconstruct file paths from stored args
        input_dir = f"{cwd}/inputs"
        itype     = getattr(args, 'inputtype', 'NAMD').upper()

        psffile = f"{input_dir}/{args.psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{args.pdbfile.split('/')[-1]}"

        if itype == 'NAMD':
            coorfile = f"{input_dir}/{args.coorfile.split('/')[-1]}"
            velfile  = f"{input_dir}/{args.velfile.split('/')[-1]}"
            xscfile  = f"{input_dir}/{args.xscfile.split('/')[-1]}"
            strfile  = (f"{input_dir}/{args.strfile.split('/')[-1]}"
                        if getattr(args, 'strfile', None) else None)
            rstfile  = None
        else:
            rstfile  = f"{input_dir}/{args.rstfile.split('/')[-1]}"
            coorfile = velfile = xscfile = None
            strfile  = (f"{input_dir}/{args.strfile.split('/')[-1]}"
                        if getattr(args, 'strfile', None) else None)

        # Get parameters from loaded args
        nm_type   = args.modeltype.lower()
        energy    = args.energy
        selection = args.selection
        replicas  = args.replicas

        # Rebuild initial SystemState from saved reference state if
        # available (avoids re-reading engine-specific files on restart).
        try:
            ref_positions_ang, ref_box_nm, ref_vel_nm_ps = load_reference_state(input_dir)
            init_state = SystemState(
                positions_nm     = ref_positions_ang * 0.1,
                velocities_nm_ps = ref_vel_nm_ps,   # may still be None (older run); reassigned per-replica from checkpoint for already-started replicas
                box_vectors_nm   = ref_box_nm,
            )
            print(f"{console.PGM_NAM}Reference state loaded from saved .npy files.")
        except FileNotFoundError:
            # Legacy fallback: re-read the original engine-specific files.
            if itype == 'NAMD':
                print(f"{console.PGM_WRN}Reference state not found; "
                      "reading NAMD binary inputs.")
                init_state = NAMDInputReader.read_system(
                    psffile, coorfile, velfile, xscfile)
                a, b, c = init_state.box_vectors_nm
                R = NAMDInputReader.align_box_to_x(a, b, c)
                init_state.positions_nm     = (R @ init_state.positions_nm.T).T
                init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
                a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
                b_rot = R @ b;  b_rot[2] = 0.0
                c_rot = R @ c
                init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
            else:
                print(f"{console.PGM_WRN}Reference state not found; "
                      "reading OpenMM restart file.")
                init_state = OpenMMRestartReader.read_state(rstfile)

        # Fallback to re-reading the original engine-specific input files so that
        # velocities_nm_ps is populated before any fresh-start replica needs it.
        if init_state.velocities_nm_ps is None:
            print(f"{console.PGM_WRN}Reference state has no velocities; "
                  "reading original input files to recover them.")
            if itype == 'NAMD':
                _vel_state = NAMDInputReader.read_system(
                    psffile, coorfile, velfile, xscfile)
                a, b, c = _vel_state.box_vectors_nm
                R = NAMDInputReader.align_box_to_x(a, b, c)
                init_state.velocities_nm_ps = (R @ _vel_state.velocities_nm_ps.T).T
            else:
                _vel_state = OpenMMRestartReader.read_state(rstfile)
                init_state.velocities_nm_ps = _vel_state.velocities_nm_ps
            del _vel_state

        init_pos_ang = init_state.positions_nm * 10.0
        sys_coor = make_reference_universe(psffile, init_pos_ang)
        sys_mass = sys_coor.atoms.masses
        n_atoms  = sys_coor.atoms.n_atoms
        sel_mass = sys_coor.atoms.select_atoms(selection).masses

        # Initialize the simulation runner
        sim_runner = SimulationRunner(
            console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
            xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
            mode_exciter, init_state=init_state,
        )

        # Check if any replicas need to be processed
        replicas_to_process = []
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                replicas_to_process.append(rep)
            else:
                last_cycle = find_last_completed_cycle(rep_dir)
                if last_cycle < end_loop:
                    replicas_to_process.append(rep)

        if not replicas_to_process:
            print(f"\n{console.PGM_WRN}All replicas are already completed. No need to restart.")
            return

        # Process replicas that need to be restarted
        for rep in replicas_to_process:
            rep_dir    = f"{cwd}/rep{rep}"
            last_cycle = find_last_completed_cycle(rep_dir)

            if os.path.exists(rep_dir):
                mode_label = ("Standard" if args.no_correc else
                              "Constant" if args.fixed else "Adaptive")
                print(f"\n{console.PGM_NAM}{console.HGH}Restarting {mode_label} MDeNM "
                      f"calculations for {console.EXT}Replica {rep}{console.STD}"
                      f"{console.HGH} from {console.EXT}step {last_cycle}{console.STD}")

            correction_state = {}
            if os.path.exists(f"{rep_dir}/correction_state.json"):
                with open(f"{rep_dir}/correction_state.json", 'r') as f:
                    correction_state = json.load(f)

            sim_runner.run_simulation(rep, last_cycle, end_loop, correction_state)

    ### APPEND
    elif args.option == 'append':
        print(f"{console.PGM_NAM}{console.TLE}Append previous pyAdMD simulation{console.STD}\n")

        additional_time = args.time

        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        args_dict          = params['args']
        factors            = params['factors']
        nm_parsed          = params['nm_parsed']
        original_end_loop  = params['end_loop']
        cwd                = params['cwd']

        # Calculate new end loop and update total time
        n_steps           = args_dict.n_steps
        additional_steps  = int(additional_time / (n_steps * 0.002))
        new_end_loop      = original_end_loop + additional_steps
        args_dict.time    = args_dict.time + additional_time
        params['end_loop'] = new_end_loop
        param_storage.save_parameters(args_dict, factors, nm_parsed, new_end_loop, cwd)

        # Reconstruct file paths
        input_dir = f"{cwd}/inputs"
        itype     = getattr(args_dict, 'inputtype', 'NAMD').upper()

        psffile = f"{input_dir}/{args_dict.psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{args_dict.pdbfile.split('/')[-1]}"

        if itype == 'NAMD':
            coorfile = f"{input_dir}/{args_dict.coorfile.split('/')[-1]}"
            velfile  = f"{input_dir}/{args_dict.velfile.split('/')[-1]}"
            xscfile  = f"{input_dir}/{args_dict.xscfile.split('/')[-1]}"
            strfile  = (f"{input_dir}/{args_dict.strfile.split('/')[-1]}"
                        if getattr(args_dict, 'strfile', None) else None)
            rstfile  = None
        else:
            rstfile  = f"{input_dir}/{args_dict.rstfile.split('/')[-1]}"
            coorfile = velfile = xscfile = None
            strfile  = (f"{input_dir}/{args_dict.strfile.split('/')[-1]}"
                        if getattr(args_dict, 'strfile', None) else None)

        nm_type   = args_dict.modeltype.lower()
        energy    = args_dict.energy
        selection = args_dict.selection
        replicas  = args_dict.replicas

        # Rebuild initial SystemState from reference state
        try:
            ref_positions_ang, ref_box_nm, ref_vel_nm_ps = load_reference_state(input_dir)
            init_state = SystemState(
                positions_nm     = ref_positions_ang * 0.1,
                velocities_nm_ps = ref_vel_nm_ps,
                box_vectors_nm   = ref_box_nm,
            )
            print(f"{console.PGM_NAM}Reference state loaded from .npy files.")
        except FileNotFoundError:
            if itype == 'NAMD':
                print(f"{console.PGM_WRN}Reference state not found; "
                      "re-reading NAMD binary inputs.")
                init_state = NAMDInputReader.read_system(
                    psffile, coorfile, velfile, xscfile)
                a, b, c = init_state.box_vectors_nm
                R = NAMDInputReader.align_box_to_x(a, b, c)
                init_state.positions_nm     = (R @ init_state.positions_nm.T).T
                init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
                a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
                b_rot = R @ b;  b_rot[2] = 0.0
                c_rot = R @ c
                init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
            else:
                print(f"{console.PGM_WRN}Reference state not found; "
                      "reading OpenMM restart file.")
                init_state = OpenMMRestartReader.read_state(rstfile)

        # Same guard as 'restart': recover velocities from original input
        # files if the saved reference state consume velocity values.
        if init_state.velocities_nm_ps is None:
            print(f"{console.PGM_WRN}Reference state has no velocities; "
                  "reading original input files to recover them.")
            if itype == 'NAMD':
                _vel_state = NAMDInputReader.read_system(
                    psffile, coorfile, velfile, xscfile)
                a, b, c = _vel_state.box_vectors_nm
                R = NAMDInputReader.align_box_to_x(a, b, c)
                init_state.velocities_nm_ps = (R @ _vel_state.velocities_nm_ps.T).T
            else:
                _vel_state = OpenMMRestartReader.read_state(rstfile)
                init_state.velocities_nm_ps = _vel_state.velocities_nm_ps
            del _vel_state

        init_pos_ang = init_state.positions_nm * 10.0
        sys_coor = make_reference_universe(psffile, init_pos_ang)
        sys_mass = sys_coor.atoms.masses
        n_atoms  = sys_coor.atoms.n_atoms
        sel_mass = sys_coor.atoms.select_atoms(selection).masses

        # Initialize the simulation runner
        sim_runner = SimulationRunner(
            console, args_dict, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
            xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
            mode_exciter, init_state=init_state,
        )

        # Check if any replicas need to be extended
        replicas_to_extend = []
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} "
                      "directory not found, skipping.")
                continue
            last_cycle = find_last_completed_cycle(rep_dir)
            if last_cycle < original_end_loop:
                print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} "
                      "hasn't completed the original simulation, skipping.")
                continue
            replicas_to_extend.append(rep)

        if not replicas_to_extend:
            print(f"{console.PGM_WRN}No replicas to extend. All replicas either "
                  "don't exist or haven't completed the original simulation.")
            return

        for rep in replicas_to_extend:
            rep_dir = f"{cwd}/rep{rep}"
            print(f"\n{console.PGM_NAM}{console.HGH}Extending {console.EXT}Replica {rep}"
                  f"{console.STD}{console.HGH} for {console.EXT}{additional_time}"
                  f"{console.STD}{console.HGH} picoseconds{console.STD}")

            correction_state = {}
            if os.path.exists(f"{rep_dir}/correction_state.json"):
                with open(f"{rep_dir}/correction_state.json", 'r') as f:
                    correction_state = json.load(f)

            sim_runner.run_simulation(rep, original_end_loop, new_end_loop,
                                      correction_state)

    ### ANALYZE
    elif args.option == 'analyze':
        print(f"{console.PGM_NAM}{console.TLE}Analyze pyAdMD results{console.STD}\n")
        analyzer = Analyzer(
            console,
            rough=args.rough,
            no_rmsd=args.no_rmsd,
            no_rg=args.no_rg,
            no_sasa=args.no_sasa,
            no_hp=args.no_hp,
            no_rmsf=args.no_rmsf,
            no_dssp=args.no_dssp,
            source=args.source,
        )
        if args.source == 'freeenergy':
            analyzer.analyze_all_centroids()
        else:
            analyzer.analyze_all_replicas()

    ### FREE ENERGY
    elif args.option == 'freeenergy':
        print(f"{console.PGM_NAM}{console.TLE}Free Energy Landscape Calculation"
              f"{console.STD}\n")

        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        fe_calc = FreeEnergyCalculator(console, params, args)
        fe_calc.run()

    ### CLEAN
    elif args.option == 'clean':
        print(f"{console.PGM_NAM}{console.TLE}Clean previous pyAdMD setup files{console.STD}\n")

        # Removing previous replicas folders
        files = os.listdir(cwd)
        for item in files:
            if item.endswith((".json", "summary.txt")):
                 os.remove(os.path.join(cwd, item))
            if item.startswith(("rep", "analysis", "freeenergy")):
                shutil.rmtree(os.path.join(cwd, item), ignore_errors=True)

        # Removing previous configuration files
        files = os.listdir(input_dir)
        for item in files:
            if item.endswith((".txt", ".out", ".crd", ".psf", ".pdb", ".coor",
                              ".vel", ".xsc", ".str", ".mod", ".rst", ".npy")):
                os.remove(os.path.join(input_dir, item))
            # Removing previous ENM calculations
            if item.endswith("_enm"):
                shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)
        for item in ("charmm_toppar"):
                shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)

        print(f"{console.PGM_NAM}Erasing is done.\n")


if __name__ == "__main__":
    main()
