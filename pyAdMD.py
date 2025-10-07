"""
The MDeNM (Molecular Dynamics with excited Normal Modes) mmethod uses
multiple short MD simulations where specific low-frequency normal modes
are kinetically excited. This approach injects additional atomic velocities
along randomly determined linear combinations of NM vectors, creating an
effective coupling between slow and fast molecular motions.

Our enhanced approach, aMDeNM, automatically manages energy injection while
respecting the natural constraints imposed by the protein structure and its
environment during conformational sampling. This prevents structural distortions
throughout the simulation. Since thermal motions are inherently stochastic,
normal mode eigenvectors naturally evolve as the structure moves between
potential energy wells. While small displacements along modes work well,
larger displacements could potentially deform the protein structure without
proper safeguards. The key advantage of our method is its ability to adaptively
adjust displacement directions based on structural and energetic constraints
from both the system and its environment, enabling exploration of new pathways.

This Python implementation brings the aMDeNM method to the community in an
accessible, user-friendly package.
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
from pathlib import Path
import seaborn as sns
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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
    from scipy.sparse import diags
    import cupy as cp
    import numba
    from numba import njit, prange, float64, int32

    # Bio/chemistry-specific imports
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    from Bio.PDB import PDBParser, ShrakeRupley
    import openmm as mm
    from openmm import app, unit, Platform
except ImportError as e:
    print(f"Required libraries not found: {e}")
    sys.exit(1)


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

    VERSION = '1.4'
    CITATION = '''  Please cite:

    \tAdaptive collective motions: a hybrid method to improve
    \tconformational sampling with molecular dynamics and normal modes.
    \tPT Resende-Lara, MGS Costa, B Dudas, D Perahia.
    \tDOI: https://doi.org/10.1101/2022.11.29.517349'''

    MESSAGE = "This program can setup and run multi-replica aMDeNM simulations through NAMD."


class ParameterStorage:
    """Handles serialization and deserialization of simulation parameters.

    This class provides functionality to save and load simulation parameters
    to/from JSON files, enabling restart capabilities for the aMDeNM simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
        param_file (str): Default filename for parameter storage.
    """

    def __init__(self, console):
        """Initialize ParameterStorage with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console
        self.param_file = "pyAdMD_params.json"

    def save_parameters(self, args, factors, nm_parsed, end_loop, cwd):
        """Save simulation parameters to a JSON file.

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

        print(f"\n{self.console.PGM_NAM}Parameters saved to {self.console.EXT}{self.param_file}{self.console.STD}")

    def load_parameters(self):
        """Load simulation parameters from a JSON file.

        Attempts to deserialize previously saved parameters and reconstruct
        the argument namespace object. Handles conversion of factors back to
        numpy array format.

        Returns:
            Dictionary containing loaded parameters with keys:
            - args: Reconstructed argument namespace
            - factors: Combination factors as numpy array
            - nm_parsed: List of mode numbers
            - end_loop: Final loop iteration
            - cwd: Working directory path
            - timestamp: Save timestamp

            Returns None if loading fails.

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
                def __init__(self, dict_args):
                    for key, value in dict_args.items():
                        setattr(self, key, value)

            params['args'] = Args(params['args'])

            # Convert factors back to numpy array if present
            if params['factors'] is not None:
                params['factors'] = np.array(params['factors'])

            print(f"{self.console.PGM_NAM}Parameters loaded from {self.console.EXT}{self.param_file}{self.console.STD}")
            return params
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error loading parameters: {self.console.ERR}{e}{self.console.STD}")
            return None


class ENMCalculator:
    """Elastic Network Model calculator for normal mode analysis.

    This class handles the computation of elastic network models, including
    system creation, Hessian matrix computation, and normal mode analysis.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
    """

    def __init__(self, console):
        """Initialize ENMCalculator with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def compute_enm(self, coorfile, nm_type, nm_parsed, input_dir, psffile):
        """Setup and run ENM analysis.

        Args:
            coorfile (str): Input coordinate file name.
            nm_type (str): Type of normal mode calculation ('CA' or 'HEAVY').
            nm_parsed (list): List containing mode numbers to analyze.
            input_dir (str): Input directory path.
            psffile (str): PSF topology filename.
        """
        # Create output folder
        base_name = os.path.splitext(os.path.basename(coorfile))[0]
        output_folder = f"{input_dir}/{base_name}_enm"
        os.makedirs(output_folder, exist_ok=True)
        output_prefix = os.path.join(output_folder, base_name)

        # Prefix to output files
        prefix = "ca" if nm_type.lower() == 'ca' else "heavy"

        # Create the PDB input to compute the ENM
        pdb_input = mda.Universe(psffile, coorfile, format="NAMDBIN")
        pdb_input = pdb_input.atoms.select_atoms("protein")
        pdb_file = f"{output_prefix}.pdb"
        pdb_input.write(pdb_file, file_format="PDB")

        # Create system
        system, topology, positions = self.create_system(
            pdb_file,
            model_type=nm_type,
            output_prefix=output_prefix,
            spring_constant=1,
        )

        # Compute Hessian
        hessian = self.hessian_enm(system, positions)

        # Mass-weight Hessian
        mw_hessian = self.mass_weight_hessian(hessian, system)

        # Compute Normal Modes
        frequencies, enm, eigenvalues = self.compute_normal_modes(
            mw_hessian,
            n_modes=None,
            use_gpu=True
        )

        # Write mode vectors
        print(f"{self.console.PGM_NAM}Writing vectors for modes {self.console.EXT}{str(nm_parsed)[1:-1]}{self.console.STD}...")
        mode_vectors_prefix = f"{output_prefix}_{prefix}"

        for mode_idx in nm_parsed:
            self.write_nm_vectors(
                enm, frequencies, system, topology,
                mode_idx,
                mode_vectors_prefix,
                pdb_file,
                model_type=nm_type
            )

        np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
        np.save(f"{output_prefix}_{prefix}_modes.npy", enm)
        print(f"{self.console.PGM_NAM}Results saved to {self.console.EXT}{output_prefix}_{prefix}_*.npy{self.console.STD} files")

    def create_system(self, pdb_file, model_type='ca', cutoff=None, spring_constant=1.0, output_prefix="input"):
        """Create an Elastic Network Model system based on specified model type.

        Args:
            pdb_file (str): Path to the input PDB file.
            model_type (str): Type of model to create: 'ca' for Cα-only or 'heavy' for heavy-atom ENM.
            cutoff (float): Cutoff distance for interactions in Å.
            spring_constant (float): Spring constant for the ENM bonds in kcal/mol/Å².
            output_prefix (str): Prefix for output files.

        Returns:
            Tuple containing:
                - system (openmm.System): The created OpenMM system
                - topology (openmm.app.Topology): The topology of the system
                - positions (list): The positions of particles in the system

        Raises:
            ValueError: If an unknown model type is specified.
        """
        # Set default cutoffs if not provided: 15.0Å for CA model, 12.0Å for heavy-atom model
        if cutoff is None:
            cutoff = 15.0 if model_type == 'ca' else 12.0

        if model_type.lower() == 'ca':
            return self._create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
        elif model_type.lower() == 'heavy':
            return self._create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)
        else:
            raise ValueError(f"Unknown model type: {self.console.ERR}{model_type}{self.console.STD}")

    def _create_ca_system(self, pdb_file, cutoff, spring_constant, output_prefix):
        """Create a Cα-only Elastic Network Model system.

        Args:
            pdb_file (str): Path to the input PDB file.
            cutoff (float): Cutoff distance for interactions in Å.
            spring_constant (float): Spring constant for the ENM bonds in kcal/mol/Å².
            output_prefix (str): Prefix for output files.

        Returns:
            Tuple containing:
                - system (openmm.System): The created OpenMM system
                - topology (openmm.app.Topology): The topology of the system
                - positions (list): The positions of particles in the system
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
            sys.exit(1)

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
        enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
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
              f"min_distance={self.console.EXT}2.9{self.console.STD} Å, k={self.console.EXT}{spring_constant}{self.console.STD} kcal/mol/Å².")
        system.addForce(mm.CMMotionRemover())

        # Save the Cα structure
        ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
        with open(ca_pdb_file, 'w') as f:
            app.PDBFile.writeFile(new_topology, positions_quantity, f)
        print(f"{self.console.PGM_NAM}Cα structure saved to {self.console.EXT}{ca_pdb_file}{self.console.STD}.")

        # Convert HETATM to ATOM
        self.convert_hetatm_to_atom(ca_pdb_file)

        return system, new_topology, positions_quantity

    def _create_heavy_system(self, pdb_file, cutoff, spring_constant, output_prefix):
        """Create a heavy-atom Elastic Network Model system.

        Args:
            pdb_file (str): Path to the input PDB file.
            cutoff (float): Cutoff distance for interactions in Å.
            spring_constant (float): Spring constant for the ENM bonds in kcal/mol/Å².
            output_prefix (str): Prefix for output files.

        Returns:
            Tuple containing:
                - system (openmm.System): The created OpenMM system
                - topology (openmm.app.Topology): The topology of the system
                - positions (list): The positions of particles in the system
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
            sys.exit(1)

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
        enm_force = mm.CustomBondForce("0.5 * k * (r - r0)^2")
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
              f"min_distance={self.console.EXT}2.0{self.console.STD} Å, k={self.console.EXT}{spring_constant}{self.console.STD} kcal/mol/Å².")
        system.addForce(mm.CMMotionRemover())

        # Save heavy atom structure
        heavy_pdb_file = f"{output_prefix}_heavy_structure.pdb"
        with open(heavy_pdb_file, 'w') as f:
            app.PDBFile.writeFile(new_topology, positions_quantity, f)
        print(f"{self.console.PGM_NAM}Heavy-atom structure saved to {self.console.EXT}{heavy_pdb_file}{self.console.STD}.")

        return system, new_topology, positions_quantity

    @staticmethod
    def convert_hetatm_to_atom(pdb_file):
        """Convert HETATM records to ATOM in PDB files for compatibility.

        Args:
            pdb_file (str): Path to the PDB file to convert.
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

    @staticmethod
    @njit(float64[:,:](float64[:,:], float64[:,:], float64, int32), parallel=True, fastmath=True)
    def compute_hessian(pos_array, bonds, k_val, n_particles):
        """Compute the Hessian matrix for an Elastic Network Model using CPU parallelization.

        Args:
            pos_array (numpy.ndarray): Array of particle positions (N×3).
            bonds (numpy.ndarray): Array of bonds with format [i, j, r0] for each bond.
            k_val (float): Spring constant.
            n_particles (int): Number of particles in the system.

        Returns:
            hessian (numpy.ndarray): The computed Hessian matrix (3N×3N).
        """
        n_dof = 3 * n_particles
        hessian = np.zeros((n_dof, n_dof), dtype=np.float64)

        for idx in prange(bonds.shape[0]):
            i = int(bonds[idx, 0])
            j = int(bonds[idx, 1])
            r0 = bonds[idx, 2]

            r_ij = pos_array[j] - pos_array[i]
            dist = np.sqrt(r_ij[0]**2 + r_ij[1]**2 + r_ij[2]**2)

            if dist > 1e-6:
                e_ij = r_ij / dist
                block = k_val * np.outer(e_ij, e_ij)

                i3 = 3 * i
                j3 = 3 * j

                # Update diagonal blocks
                for a in range(3):
                    for b in range(3):
                        hessian[i3 + a, i3 + b] += block[a, b]
                        hessian[j3 + a, j3 + b] += block[a, b]

                        # Update off-diagonal blocks
                        hessian[i3 + a, j3 + b] -= block[a, b]
                        hessian[j3 + a, i3 + b] -= block[a, b]

        return hessian

    def hessian_enm(self, system, positions):
        """Build, compute and regularize Hessian matrix for an Elastic Network Model.

        Args:
            system (openmm.System): The system containing the ENM force.
            positions (list): The positions of particles in the system.

        Returns:
            hessian (numpy.ndarray): The computed Hessian matrix (3N×3N).

        Raises:
            ValueError: If no ENM force is found in the system.
        """
        n_particles = system.getNumParticles()
        n_dof = 3 * n_particles

        enm_force = next((f for f in system.getForces() if isinstance(f, mm.CustomBondForce)), None)
        if enm_force is None:
            print(f"{self.console.PGM_ERR}No ENM force found in system.")
            raise ValueError("No ENM force found in system")

        k_val = enm_force.getGlobalParameterDefaultValue(0)
        num_bonds = enm_force.getNumBonds()
        pos_array = np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)], dtype=np.float64)

        start_time = time.time()

        # Precompute bonds array with fixed memory layout
        bonds_list = np.empty((num_bonds, 3), dtype=np.float64)
        for bond_idx in range(num_bonds):
            i, j, [r0] = enm_force.getBondParameters(bond_idx)
            bonds_list[bond_idx] = (i, j, r0)

        # Compute Hessian
        hessian = self.compute_hessian(pos_array, bonds_list, k_val, n_particles)

        # Symmetrize and regularize
        hessian = 0.5 * (hessian + hessian.T)
        hessian.flat[::n_dof+1] += 1e-8  # Add regularization directly to diagonal

        duration = time.time() - start_time
        print(f"{self.console.PGM_NAM}Computed ENM Hessian for {self.console.EXT}{n_particles}{self.console.STD} particles in {self.console.EXT}{duration:.2f}{self.console.STD} seconds")

        return hessian

    @staticmethod
    def mass_weight_hessian(hessian, system):
        """Apply mass-weighting to the Hessian matrix.

        Args:
            hessian (numpy.ndarray): The Hessian matrix to mass-weight.
            system (openmm.System): The system containing particle masses.

        Returns:
            mw_hessian (numpy.ndarray): The mass-weighted Hessian matrix.
        """
        n_particles = system.getNumParticles()
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
        masses[masses == 0] = 1.0  # Avoid division by zero

        inv_sqrt_m = 1 / np.sqrt(masses)
        m_vector = np.repeat(inv_sqrt_m, 3)

        # Diagonal multiplication using sparse matrices
        D = diags(m_vector, 0)
        return D @ hessian @ D

    @staticmethod
    def gpu_diagonalization(hessian, n_modes=None):
        """Diagonalize the Hessian matrix using GPU acceleration.

        Args:
            hessian (numpy.ndarray): The Hessian matrix to diagonalize.
            n_modes (int): Number of modes to compute. If None, computes all modes.

        Returns:
            Tuple containing:
                - eigenvalues (numpy.ndarray): The eigenvalues of the Hessian matrix
                - eigenvectors (numpy.ndarray): The eigenvectors of the Hessian matrix
        """
        # Use memory pool for efficient GPU memory management
        mem_pool = cp.get_default_memory_pool()
        pinned_mem_pool = cp.get_default_pinned_memory_pool()

        # Transfer Hessian to GPU using pinned memory for faster transfer
        with cp.cuda.Device(0):
            # Use float32 for faster computation if precision allows
            if hessian.shape[0] > 3000:
                hessian_gpu = cp.array(hessian, dtype=cp.float32)
            else:
                hessian_gpu = cp.array(hessian, dtype=cp.float64)

            # Free CPU memory immediately
            del hessian

            if n_modes is not None:
                # Use partial diagonalization for specified number of modes
                n_modes = min(n_modes + 6, hessian_gpu.shape[0])
                eigenvalues, eigenvectors = cp.linalg.eigh(
                    hessian_gpu,
                    UPLO='L',
                    subset_by_index=[0, n_modes-1]
                )
            else:
                # Full diagonalization
                eigenvalues, eigenvectors = cp.linalg.eigh(hessian_gpu, UPLO='L')

            # Free GPU memory immediately after computation
            del hessian_gpu
            mem_pool.free_all_blocks()
            pinned_mem_pool.free_all_blocks()

            # Transfer results back to CPU using pinned memory
            eigenvalues_cpu = cp.asnumpy(eigenvalues, stream=cp.cuda.Stream.null)
            eigenvectors_cpu = cp.asnumpy(eigenvectors, stream=cp.cuda.Stream.null)

            # Free GPU memory
            del eigenvalues, eigenvectors

        return eigenvalues_cpu, eigenvectors_cpu

    def compute_normal_modes(self, hessian, n_modes=None, use_gpu=False):
        """Compute normal modes by diagonalizing the Hessian matrix.

        Args:
            hessian (numpy.ndarray): The Hessian matrix to diagonalize.
            n_modes (int): Number of modes to compute. If None, computes all modes.
            use_gpu (bool): Whether to use GPU acceleration for diagonalization.

        Returns:
            Tuple containing:
                - frequencies (numpy.ndarray): The frequencies of the normal modes
                - modes (numpy.ndarray): The normal mode vectors
                - eigenvalues (numpy.ndarray): The eigenvalues of the Hessian matrix
        """
        start_time = time.time()

        if use_gpu and cp.is_available():
            print(f"{self.console.PGM_NAM}Diagonalizing mass-weighted Hessian using GPU acceleration...")
            try:
                eigenvalues, eigenvectors = self.gpu_diagonalization(hessian, n_modes)
            except Exception as e:
                print(f"{self.console.PGM_WRN}GPU diagonalization failed: {self.console.WRN}{e}{self.console.STD}. Falling back to CPU.")
                use_gpu = False

        if not use_gpu or not cp.is_available():
            # CPU diagonalization with optimized parameters
            if n_modes is not None:
                print(f"{self.console.PGM_NAM}Diagonalizing mass-weighted Hessian using CPU optimization...")
                n_modes = min(n_modes + 6, hessian.shape[0])
                eigenvalues, eigenvectors = eigh(
                    hessian,
                    subset_by_index=[0, n_modes-1],
                    driver='evr',       # Fastest driver for symmetric matrices
                    overwrite_a=True,
                    check_finite=False  # Skip finite check for performance
                )
            else:
                print(f"{self.console.PGM_NAM}Diagonalizing mass-weighted Hessian using CPU...")
                eigenvalues, eigenvectors = eigh(
                    hessian,
                    driver='evr',       # Fastest driver for symmetric matrices
                    overwrite_a=True,
                    check_finite=False  # Skip finite check for performance
                )

        duration = time.time() - start_time
        print(f"{self.console.PGM_NAM}Diagonalization completed in {self.console.EXT}{duration:.2f}{self.console.STD} seconds")

        # Calculate frequencies from eigenvalues
        abs_evals = np.abs(eigenvalues)
        threshold = np.max(abs_evals) * 1e-10
        valid_idx = abs_evals > threshold

        frequencies = np.sqrt(np.abs(eigenvalues[valid_idx]))
        valid_modes = eigenvectors[:, valid_idx]

        sort_idx = np.argsort(frequencies)
        sorted_frequencies = frequencies[sort_idx]
        sorted_modes = valid_modes[:, sort_idx]

        return sorted_frequencies, sorted_modes, eigenvalues

    def write_nm_vectors(self, modes, frequencies, system, topology, nm, output_prefix, pdb_file, model_type='ca'):
        """Write normal mode eigenvectors to XYZ files for complete protein structure.

        Args:
            modes (numpy.ndarray): The normal mode vectors (3N×M).
            frequencies (numpy.ndarray): The frequencies of the normal modes.
            system (openmm.System): The system containing particle information.
            topology (openmm.app.Topology): The topology of the system.
            nm (int): Mode number to write.
            output_prefix (str): Prefix for output XYZ files.
            pdb_file (str): Path to the original PDB file to get complete atom information.
            model_type (str): Type of model ('ca' or 'heavy').
        """
        # Read the original PDB file to get complete atom information
        pdb = app.PDBFile(pdb_file)
        n_original_atoms = pdb.topology.getNumAtoms()

        # Get element symbols and atom names from original topology
        elements = []
        atom_names = []
        for atom in pdb.topology.atoms():
            elements.append(atom.element.symbol)
            atom_names.append(atom.name)

        n_particles = system.getNumParticles()

        # Create a mapping from ENM atoms to original atoms
        enm_to_original_map = []

        if model_type == 'ca':
            # For Cα model, map Cα atoms to their positions in the original structure
            for atom_idx, atom in enumerate(topology.atoms()):
                # Find the corresponding atom in the original structure
                for orig_idx, orig_atom in enumerate(pdb.topology.atoms()):
                    if (orig_atom.name == 'CA' and
                        orig_atom.residue.index == atom.residue.index):
                        enm_to_original_map.append(orig_idx)
                        break
        else:
            # For heavy atom model, map heavy atoms to their positions in the original structure
            for atom_idx, atom in enumerate(topology.atoms()):
                # Find the corresponding atom in the original structure
                for orig_idx, orig_atom in enumerate(pdb.topology.atoms()):
                    if (orig_atom.name == atom.name and
                        orig_atom.residue.index == atom.residue.index):
                        enm_to_original_map.append(orig_idx)
                        break

        # Write each mode to a separate XYZ file
        freq = frequencies[nm] * 108.58  # Convert to cm⁻¹
        output_file = f"{output_prefix}_mode_{nm}.xyz"

        with open(output_file, 'w') as f:
            # Write header
            f.write(f"{n_original_atoms}\n")
            f.write(f"Normal Mode {nm}, Frequency: {freq:.2f} cm⁻¹\n")

            # Extract and reshape the mode vector for ENM atoms
            mode_vector = modes[:, nm].reshape(n_particles, 3)

            # Create a full vector for all atoms, initialized to zero
            full_vector = np.zeros((n_original_atoms, 3))

            # Map ENM mode vectors to the correct positions in the full vector
            for enm_idx, orig_idx in enumerate(enm_to_original_map):
                full_vector[orig_idx] = mode_vector[enm_idx]

            # Write coordinates for each atom
            for i in range(n_original_atoms):
                x, y, z = full_vector[i]
                f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")


class ModeExciter:
    """Handles the generation and excitation of normal mode combinations.

    This class is responsible for generating linear combinations of normal modes,
    applying excitation energy to these modes, and writing the resulting vectors
    to files for use in molecular dynamics simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
    """

    def __init__(self, console):
        """Initialize the ModeExciter with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def generate_factors(self, P, N, cwd, nm_parsed):
        """Generate factors for linear combinations of normal modes.

        Generates P points uniformly distributed on an N-dimensional hypersphere
        surface using a repulsion algorithm. For the special case where P=2N,
        generates vertices of a cross-polytope.

        Args:
            P (int): Number of points to generate (number of replicas).
            N (int): Dimensionality of the space (number of modes).
            cwd (str): Current working directory path.
            nm_parsed (list): List of mode numbers being combined.

        Returns:
            numpy.ndarray: P×N matrix of combination factors.

        Notes:
            Uses a repulsion algorithm to spread points evenly on the hypersphere.
            For P=2N, generates cross-polytope vertices for optimal distribution.
        """
        # Store the modes indexes
        N = len(nm_parsed)

        if P == 2 * N:
            # Special case: vertices of a cross-polytope
            positive_units = np.eye(N)
            factors = np.vstack((positive_units, -positive_units))

        else:
            # Generate uniformly distributed points
            # on a N-dimensional hypersphere using a repulsion algorithm.

            # Initialize with random points on the hypersphere
            factors = np.random.normal(size=(P, N))
            norms = np.linalg.norm(factors, axis=1, keepdims=True)
            factors = factors / norms

            # Variables for stagnation detection
            prev_max_force = float('inf')
            stagnation_count = 0
            stagnation_threshold = 5  # Number of iterations with no significant change to trigger break

            # Use a repulsion algorithm to spread points evenly
            for iteration in range(1000000):
                # Calculate all pairwise distances
                diff = factors[:, np.newaxis, :] - factors[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)

                # Avoid division by zero
                dist += np.eye(P) * 1e-6

                # Calculate repulsion forces taking into account the geometry of the space
                force = 1 / (dist ** (N-1))

                # Calculate direction of forces
                force_dir = diff / dist[:, :, np.newaxis]

                # Sum forces acting on each point
                total_force = np.sum(force[:, :, np.newaxis] * force_dir, axis=1)

                # Calculate current maximum force
                current_max_force = np.max(np.abs(total_force))

                # Check for stagnation (force not changing significantly)
                if abs(current_max_force - prev_max_force) < 1e-6 * 0.1:
                    stagnation_count += 1
                else:
                    stagnation_count = 0

                prev_max_force = current_max_force

                # Move points according to forces (in the tangent plane)
                for i in range(P):
                    # Project force onto tangent plane
                    tangent_force = total_force[i] - np.dot(total_force[i], factors[i]) * factors[i]

                    # Move point in tangent direction
                    factors[i] += 0.001 * tangent_force

                    # Project back to hypersphere
                    factors[i] /= np.linalg.norm(factors[i])

                # Check for convergence or stagnation
                if current_max_force < 1e-6:
                    print(f"{self.console.PGM_NAM}Converged to tolerance after {self.console.WRN}{iteration+1}{self.console.STD} iterations.")
                    break

                if stagnation_count >= stagnation_threshold:
                    print(f"{self.console.PGM_WRN}Breaking due to stagnation after {self.console.WRN}{iteration+1}{self.console.STD} iterations.")
                    break

        # Write a csv file containing the combination factors
        self._write_factors_csv(factors, nm_parsed, cwd)

        return factors

    def _write_factors_csv(self, factors, nm_parsed, cwd):
        """Write combination factors to a CSV file.

        Args:
            factors (numpy.ndarray): Matrix of combination factors (P×N).
            nm_parsed (list): List of mode numbers used in combinations.
            cwd (str): Current working directory path.
        """
        # Create the header
        header = ['Combination'] + [f'Mode {mode}' for mode in nm_parsed]
        # Create the data rows
        rows = []
        for i in range(len(factors)):
            row = [str(i + 1)]
            for val in factors[i]:
                row.append(str(val))
            rows.append(row)

        # Create output folder
        output_folder = f"{cwd}/rep-struct-list"
        os.makedirs(output_folder, exist_ok=True)

        # Write to CSV
        with open(f"{output_folder}/factors.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"{self.console.PGM_NAM}Combination factors written at {self.console.EXT}{output_folder}/factors.csv{self.console.STD}.")

    def combine_modes(self, replicas, modes, factors, nm_type, coorfile, cwd, mda_U):
        """Combine normal modes and write resulting vectors.

        Creates linear combinations of specified normal modes using precomputed
        factors, normalizes the resulting vectors, and writes them to files.

        Args:
            replicas (int): Number of replicas to process.
            modes (list): List of mode numbers to combine.
            factors (numpy.ndarray): Matrix of combination factors (P×N).
            nm_type (str): Type of normal mode calculation ('CHARMM', 'CA', or 'HEAVY').
            coorfile (str): Coordinate filename for reference.
            cwd (str): Current working directory path.
            mda_U (mda.Universe): MDAnalysis Universe containing system structure.
        """
        # Set output folder
        output_folder = f"{cwd}/rep-struct-list"

        for rep in range(replicas):
            # Create an empty vector to store the combination
            natom = mda_U.atoms.select_atoms("protein").n_atoms
            comb_vec = np.zeros((natom, 3))

            for nm_idx in range(len(modes)):
                if nm_type == 'charmm':
                    nm = mda.Universe(f"{cwd}/inputs/mode_nm{modes[nm_idx]}.crd", format="CRD")
                else:
                    base_name = os.path.splitext(os.path.basename(coorfile))[0]
                    nm = mda.Universe(f"{cwd}/inputs/{base_name}_enm/{base_name}_{nm_type}_mode_{modes[nm_idx]}.xyz", format="XYZ")
                nm = nm.atoms.positions

                # Apply the factors to the modes
                comb = (nm.T * factors[rep, nm_idx]).T

                # Accumulate modes to obtain a new combined vector
                comb_vec = np.add(comb, comb_vec)

            # Normalize and write the combined vector
            comb_vec /= np.linalg.norm(comb_vec)
            self._write_vector(comb_vec, f"{output_folder}/rep{rep+1}_vector.vec", mda_U)

        print(f"{self.console.PGM_NAM}Combination vectors written at {self.console.EXT}{output_folder}/rep*_vector.vec{self.console.STD}.")

    def excite(self, q_vector, user_ek, sel_mass):
        """Scale combined modes to apply excitation energy.

        Args:
            q_vector (numpy.ndarray): Combined normal mode vector to excite.
            user_ek (float): Excitation energy in kcal/mol.
            sel_mass (numpy.ndarray): Atomic masses of selected atoms.

        Returns:
            numpy.ndarray: Scaled excitation vector.

        Notes:
            The excitation vector is scaled according to: v = q * sqrt(2E/m)
        """
        # Excite
        fscale = np.sqrt((2 * user_ek) / sel_mass)
        exc_vec = (q_vector.T * fscale).T

        return exc_vec

    def _write_vector(self, xyz, output_file, mda_U):
        """Write coordinate vector to file in NAMDBIN format.

        Args:
            xyz (numpy.ndarray): Coordinate array to write.
            output_file (str): Output filename.
            mda_U (mda.Universe): MDAnalysis Universe for reference structure.
        """
        # Copy the xyz coordinates into the dataframe
        sys_zeros = mda_U.atoms.select_atoms("all")
        sys_zeros.positions = np.zeros((mda_U.atoms.n_atoms, 3))
        vector = np.append(xyz, sys_zeros.positions, axis=0)
        sys_zeros.positions = vector[:mda_U.atoms.n_atoms]

        # Write the output file
        sys_zeros.write(output_file, file_format="NAMDBIN")


class SimulationRunner:
    """Handles running, restarting, and appending aMDeNM simulations.

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
        sel_mass (np.ndarray): Atomic masses of selected atoms.
        init_coor (np.ndarray): Initial coordinates of selected atoms.
        energy (float): Excitation energy value.
        mode_exciter (ModeExciter): Mode exciter instance.
        sys_pdb (mda.Universe): System structure universe.
    """
    def __init__(self, console, args, cwd, input_dir, psffile, pdbfile, coorfile,
                 velfile, xscfile, strfile, sel_mass, init_coor, energy, mode_exciter, sys_pdb):
        """Initialize SimulationRunner with all required components.

        Args:
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
            sel_mass (numpy.ndarray): Atomic masses of selected atoms.
            init_coor (numpy.ndarray): Initial coordinates of selected atoms.
            energy (float): Excitation energy value.
            mode_exciter (ModeExciter): Mode exciter instance.
            sys_pdb (mda.Universe): System structure universe.
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
        self.sel_mass = sel_mass
        self.init_coor = init_coor
        self.energy = energy
        self.mode_exciter = mode_exciter
        self.sys_pdb = sys_pdb

        # Define energy correction thresholds
        self.top = energy * 1.25
        self.bottom = energy * 0.75

        # Define correction variables
        self.globfreq = self.cos_alpha = self.qrms_correc = 0.5

    def run_simulation(self, rep, start_loop, end_loop, correction_state=None):
        """Run simulation for a specific replica.

        Handles both new simulations and restarts by managing simulation state,
        energy correction, and trajectory analysis.

        Args:
            rep (int): Replica number to simulate.
            start_loop (int): Starting loop index (0 for new simulations).
            end_loop (int): Ending loop index.
            correction_state (dict): Dictionary containing correction state for restart/append.

        Returns:
            dict: Dictionary containing final correction state after simulation completion.

        Raises:
            SystemExit: If NAMD simulation fails to execute properly.
        """
        rep_dir = f"{self.cwd}/rep{rep}"

        if start_loop == 0:
            # New simulation
            if self.args.no_correc:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Standard MDeNM calculations for {self.console.EXT}Replica {rep}{self.console.STD}")
            elif self.args.fixed:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Constant MDeNM calculations for {self.console.EXT}Replica {rep}{self.console.STD}")
            else:
                print(f"\n{self.console.PGM_NAM}{self.console.HGH}Starting Adaptive MDeNM calculations for {self.console.EXT}Replica {rep}{self.console.STD}")

            os.makedirs(rep_dir, exist_ok=True)
            os.chdir(rep_dir)

            # Copy the NM combination vector to the replica folder
            shutil.copy(f"{self.cwd}/rep-struct-list/rep{rep}_vector.vec", "pff_vector.vec")

            # Excite the combined vector according to user-defined energy increment
            print(f"{self.console.PGM_NAM}Writing the excitation vector with a Ek injection of {self.console.EXT}{self.energy}{self.console.STD} kcal/mol.")
            q_vec = mda.Universe(self.psffile, "pff_vector.vec", format="NAMDBIN")
            q_vec = q_vec.atoms.select_atoms(self.args.selection).positions
            exc_vec = self.mode_exciter.excite(q_vec, self.energy, self.sel_mass)

            # Write the combination and the excited vector
            self.mode_exciter._write_vector(q_vec, "cntrl_vector.vec", self.sys_pdb)
            self.mode_exciter._write_vector(exc_vec, "excitation.vel", self.sys_pdb)

            # Initialize variables for this new replica
            loop = 0
            cnt = 1
            vp, ek, qp, rmsp = [[], [], [], []]
            ref_str = f"step_{loop}.coor"
            shutil.copy(self.coorfile, "correc_ref.coor")
            shutil.copy(self.coorfile, f"step_{loop}.coor")
            shutil.copy(self.velfile, f"step_{loop}.vel")
            shutil.copy(self.xscfile, f"step_{loop}.xsc")

            # Save initial correction state
            correction_state = {
                'cnt': cnt,
                'ref_str': ref_str,
                'qrms_correc': self.qrms_correc
            }
        else:
            # Restart/append existing simulation
            os.chdir(rep_dir)

            # Load the correction state if it exists
            if correction_state is None and os.path.exists("correction_state.json"):
                with open("correction_state.json", 'r') as f:
                    correction_state = json.load(f)

            # Extract correction variables
            cnt = correction_state.get('cnt', 1)
            ref_str = correction_state.get('ref_str', f"step_{start_loop}.coor")
            self.qrms_correc = correction_state.get('qrms_correc', 0.5)

            # Continue the simulation from the last cycle
            loop = start_loop

            # Read the projections if they exist
            vp, ek, qp, rmsp = [[], [], [], []]
            for proj_file in ["vp-proj.out", "ek-proj.out", "coor-proj.out", "rms-proj.out"]:
                if os.path.exists(proj_file):
                    with open(proj_file, 'r') as f:
                        lines = f.readlines()
                    if proj_file == "vp-proj.out":
                        vp = lines
                    elif proj_file == "ek-proj.out":
                        ek = lines
                    elif proj_file == "coor-proj.out":
                        qp = lines
                    elif proj_file == "rms-proj.out":
                        rmsp = lines

        # Continue the simulation loop
        while loop < end_loop:
            # Loop update
            loop += 1

            # Run NAMD
            shutil.copy(f"{self.cwd}/inputs/conf.namd", 'conf.namd')
            namd_conf = Path('conf.namd')
            now = time.strftime("%H:%M:%S")
            print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: running {self.console.EXT}step {loop}{self.console.STD} of {self.console.WRN}{end_loop}{self.console.STD}...")
            run_namd(namd_conf, self.psffile, self.pdbfile, self.strfile, loop)

            # EVALUATE IF IT IS NECESSARY TO CHANGE THE EXCITATION DIRECTION
            if [self.args.no_correc == False] and [self.args.fixed == False]:
                coor_ref = mda.Universe(self.psffile, "correc_ref.coor", format="NAMDBIN")
                coor_ref = coor_ref.atoms.select_atoms(self.args.selection).positions

                coor_curr = mda.Universe(self.psffile, f"step_{loop}.coor", format="NAMDBIN")
                coor_curr = coor_curr.atoms.select_atoms(self.args.selection).positions

                # Compute the difference and mass-weight the difference
                # (qcurr - qref) * sqrt(m)
                diff = ((coor_curr - coor_ref).T * np.sqrt(self.sel_mass)).T

                # Read the excitation vector
                cntrl_vec = mda.Universe(self.psffile, "cntrl_vector.vec", format="NAMDBIN")
                cntrl_vec = cntrl_vec.atoms.select_atoms(self.args.selection).positions

                # Project the current coordinates onto Q
                q_proj = np.sum(diff * cntrl_vec)
                rms_check = np.sqrt((q_proj ** 2) / np.sum(self.sel_mass))

                # Correct the excitation direction or recompute ENM modes
                if rms_check >= self.qrms_correc:
                    # If --recalc flag is set, recompute ENM
                    if hasattr(self.args, 'recalc') and self.args.recalc:
                        nm_parsed = [int(s) for s in self.args.modes.split(',')]
                        self._recompute_enm_modes(rep, loop, nm_parsed, cnt)
                        cnt += 1
                    else:
                        # Otherwise, correct the Q vector
                        self._correct_excitation_direction(loop, ref_str, cntrl_vec, cnt)
                        cnt += 1

                    # Update the rms correction variable value
                    self.qrms_correc += self.globfreq

            # OBTAIN THE VELOCITIES AND KINETIC ENERGY PROJECTED ONTO Q
            # Open the current velocities file and mass-weight
            curr_vel = mda.Universe(self.psffile, f"step_{loop}.vel", format="NAMDBIN")
            curr_vel = ((curr_vel.atoms.select_atoms(self.args.selection).positions).T * np.sqrt(self.sel_mass)).T

            # Read the excitation vector
            cntrl_vec = mda.Universe(self.psffile, "cntrl_vector.vec", format="NAMDBIN")
            cntrl_vec = cntrl_vec.atoms.select_atoms(self.args.selection).positions

            # Compute the scalar projection of velocity
            velo = np.sum(curr_vel * cntrl_vec) / np.sum(cntrl_vec * cntrl_vec)
            vp.append(f"{str(round(velo, 5))}\n")

            # Compute the vectorial projection of velocity
            v_proj = np.sum(curr_vel * cntrl_vec) / np.sum(cntrl_vec * cntrl_vec) * cntrl_vec

            # Calculate the kinetic energy from projected velocities
            ek_vel = 0.5 * np.sum(v_proj ** 2)
            ek.append(f"{str(round(ek_vel, 5))}\n")

            # Write the unmass-weighted velocity projection
            v_proj = (v_proj.T / np.sqrt(self.sel_mass)).T
            self.mode_exciter._write_vector(v_proj, "velo_proj.vel", self.sys_pdb)

            # PROJECT THE COORDINATES ONTO Q
            # Open the current coordinates file
            curr_coor = mda.Universe(self.psffile, f"step_{loop}.coor", format="NAMDBIN")
            curr_coor = curr_coor.atoms.select_atoms(self.args.selection).positions

            # Compare the current with the initial coordinates
            diff = curr_coor - self.init_coor
            diff = (diff.T * np.sqrt(self.sel_mass)).T

            # Calculate the dot product between qcurr and Q
            q_proj = np.sum(diff * cntrl_vec)
            qp.append(f"{str(round(q_proj, 5))}\n")

            # Compute the rms displacement along the vector Q
            mrms = np.sqrt((q_proj ** 2) / sum(self.sel_mass))
            rmsp.append(f"{str(round(mrms, 5))}\n")

            # Skip excitation vector rescaling (Original MDeNM)
            if self.args.no_correc:
                continue

            # RESCALE KINETIC ENERGY ACCORDING TO VALUES PROJECTED ONTO VECTOR Q
            '''Re-excite the NM vector when ek is below inferior limit
            or relax the energy when ek is above superior limit'''
            if (ek_vel < self.bottom) or (ek_vel > self.top):
                # Read current and excitation velocities
                curr_vel = mda.Universe(self.psffile, f"step_{loop}.vel", format="NAMDBIN")
                curr_vel = curr_vel.coord.positions
                exc_vec = mda.Universe(self.psffile, "excitation.vel", format="NAMDBIN")
                exc_vec = exc_vec.coord.positions
                v_proj = mda.Universe(self.psffile, "velo_proj.vel", format="NAMDBIN")
                v_proj = v_proj.coord.positions

                # Compute the difference between the projected and the excitation velocities
                # and then sum to the current velocities: Vnew = Vdyna + (VQ - Vp)
                new_vel = curr_vel + (exc_vec - v_proj)
                self.mode_exciter._write_vector(new_vel, f"step_{loop}.vel", self.sys_pdb)

            # Save correction state periodically
            if loop % 10 == 0:
                correction_state = {
                    'cnt': cnt,
                    'ref_str': ref_str,
                    'qrms_correc': self.qrms_correc
                }
                with open("correction_state.json", 'w') as f:
                    json.dump(correction_state, f)

        # Write the projections into files
        for i,j in zip((vp, ek, qp, rmsp), ("vp", "ek", "coor", "rms")):
            with open(f"{j}-proj.out", 'w') as write:
                write.writelines(i)

        # De-excite the system
        shutil.copy(f"{self.input_dir}/deexcitation.namd", 'deexcitation.namd')
        deexc_conf = Path('deexcitation.namd')
        now = time.strftime("%H:%M:%S")
        print(f"{self.console.PGM_NAM}{now} {self.console.EXT}Replica {rep}{self.console.STD}: running the {self.console.EXT}de-excitation step{self.console.STD}...")
        run_namd(deexc_conf, self.psffile, self.pdbfile, self.strfile, loop, deexcitation=True)

        # Return the final correction state
        return {
            'cnt': cnt,
            'ref_str': ref_str,
            'qrms_correc': self.qrms_correc
        }

    def _recompute_enm_modes(self, rep, loop, nm_parsed, cnt):
        """Recompute ENM modes using the current structure with random combination factor."""
        console = ConsoleConfig()
        print(f"\n{console.PGM_NAM}Recomputing ENM modes for {console.EXT}Replica {rep}{console.STD} at step {console.EXT}{loop}{console.STD}")

        # Get the Replica directory path for this step
        rep_dir = os.getcwd()

        # Use current coordinate file as input
        current_coor = f"step_{loop}.coor"

        # Initialize ENM calculator
        enm_calculator = ENMCalculator(self.console)

        try:
            # Compute ENM using current structure
            enm_calculator.compute_enm(
                coorfile=current_coor,
                nm_type=self.args.modeltype.lower(),
                nm_parsed=nm_parsed,
                input_dir=rep_dir,  # Output to replica's ENM directory
                psffile=self.psffile
            )

            # Generate new combination using RANDOM factors
            self._generate_new_excitation_vector(rep, loop, nm_parsed, rep_dir, cnt)

            print(f"{console.PGM_NAM}ENM recomputation completed for {console.EXT}Replica {rep}{console.STD}\n")

        except Exception as e:
            print(f"{console.PGM_ERR}ENM recomputation failed: {console.ERR}{e}{console.STD}")
            # Fall back to standard correction
            self._correct_excitation_direction(loop, f"step_{loop-1}.coor", None, cnt)

    def _generate_new_excitation_vector(self, rep, loop, nm_parsed, rep_dir, cnt):
        """Generate new excitation vector from recomputed ENM modes using random factors."""
        console = ConsoleConfig()

        # Generate new random factors for this recombination
        print(f"{console.PGM_NAM}Generating new random factors for ENM recombination")
        num_modes = len(nm_parsed)

        # Generate random points on N-dimensional hypersphere
        factors = np.random.normal(size=num_modes)
        factors = factors / np.linalg.norm(factors)

        # Get base name for coordinate file
        base_name = os.path.splitext(os.path.basename(f"step_{loop}.coor"))[0]

        # Combine the new modes
        prefix = "ca" if self.args.modeltype.lower() == 'ca' else "heavy"

        comb_vec = None
        natom = self.sys_pdb.atoms.select_atoms("protein").n_atoms

        for i, mode_num in enumerate(nm_parsed):
            # Load the new mode vector
            mode_file = f"{rep_dir}/{base_name}_enm/{base_name}_{prefix}_mode_{mode_num}.xyz"
            if os.path.exists(mode_file):
                mode_universe = mda.Universe(mode_file, format="XYZ")
                mode_vec = mode_universe.atoms.positions

                if comb_vec is None:
                    comb_vec = np.zeros((natom, 3))

                # Apply random factor and accumulate
                comb_vec += mode_vec * factors[i]
            else:
                print(f"{console.PGM_WRN}Mode file {mode_file} not found")

        if comb_vec is not None:
            # Normalize the combined vector
            comb_vec /= np.linalg.norm(comb_vec)

            # Rename previous excitation vector files
            if os.path.exists("excitation.vel"):
                shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
            if os.path.exists("cntrl_vector.vec"):
                shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

            # Write the new control vector
            self.mode_exciter._write_vector(comb_vec, "cntrl_vector.vec", self.sys_pdb)

            # Excite and write the new excited vector
            exc_vec = self.mode_exciter.excite(comb_vec, self.energy, self.sel_mass)
            self.mode_exciter._write_vector(exc_vec, "excitation.vel", self.sys_pdb)

            print(f"{console.PGM_NAM}New excitation vector generated from recomputed ENM modes with random factors")

            # Save the factors for reference
            factors_file = f"{rep_dir}/{base_name}_enm/factors.csv"
            with open(factors_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Mode', 'Factor'])
                for mode_num, factor in zip(nm_parsed, factors):
                    writer.writerow([mode_num, factor])
            print(f"{console.PGM_NAM}Combination factors saved to {console.EXT}{factors_file}{console.STD}")
        else:
            raise ValueError("Could not generate new excitation vector")

    def _correct_excitation_direction(self, loop, ref_str, cntrl_vec, cnt):
        """Corrects the excitation vector direction."""
        # Compute the average structure of the last excitation
        ts = mda.Universe(self.psffile, f"step_{loop - 1}.coor", format="NAMDBIN")
        avg_positions = ts.atoms.select_atoms(self.args.selection).positions
        ts = mda.Universe(self.psffile, f"step_{loop}.coor", format="NAMDBIN")
        avg_positions += ts.atoms.select_atoms(self.args.selection).positions
        avg_positions = avg_positions / 2
        self.mode_exciter._write_vector(avg_positions, f"average_{loop}.coor", self.sys_pdb)

        # Open the reference and mobile structures
        ref = mda.Universe(self.psffile, ref_str, format="NAMDBIN")
        ref = ref.atoms.select_atoms(self.args.selection)
        mob = mda.Universe(self.psffile, f"average_{loop}.coor", format="NAMDBIN")
        mob = mob.atoms.select_atoms(self.args.selection)

        # Align the structures and compute the mass-weighted difference
        align.alignto(mob, ref, select="protein", weights="mass")
        diff = ((mob.positions - ref.positions).T * np.sqrt(self.sel_mass)).T

        # Normalize the mass-weighted difference vector
        diff /= np.linalg.norm(diff)

        # Project the current coordinates onto Q
        dotp = np.sum(diff * cntrl_vec)

        # Set the average structure as the new reference for the next steps
        ref_str = f"average_{loop}.coor"

        if dotp <= self.cos_alpha:
            shutil.copy(f"step_{loop}.coor", f"correc_ref.coor")
            self.qrms_correc = 0

        # Rename the previous excitation vector files
        shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
        shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")

        # Write the corrected excitation vector
        print(f"{self.console.PGM_NAM}Writing the corrected excitation vector.")
        self.mode_exciter._write_vector(diff, "cntrl_vector.vec", self.sys_pdb)

        # Excite and write the new excited vector
        exc_vec = self.mode_exciter.excite(diff, self.energy, self.sel_mass)
        self.mode_exciter._write_vector(exc_vec, "excitation.vel", self.sys_pdb)


class Analyzer:
    """Analyzes simulation results and generates plots.

    This class handles computation and visualization of various structural
    properties from simulation trajectories including RMSD, radius of gyration,
    SASA, hydrophobic exposure, secondary structure, and RMSF.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output
        param_file (str): Path to parameter JSON file
        rough (bool): If True, analyze every 5ps instead of every frame
    """
    def __init__(self, console, param_file="pyAdMD_params.json", rough=False):
        """Initializes Analyzer with configuration and parameters.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output
            param_file (str): Path to parameter JSON file
            rough (bool): If True, analyze every 5ps instead of every frame
        """
        self.console = console
        self.param_file = param_file
        self.rough = rough
        self.params = self._load_parameters()

        # Create analysis directory
        self.analysis_dir = "analysis"
        os.makedirs(self.analysis_dir, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [6, 6]
        plt.rcParams['figure.dpi'] = 100

        # Determine number of CPU cores to use
        self.num_cores = mp.cpu_count()
        print(f"{self.console.PGM_NAM}Using {self.console.EXT}{self.num_cores}{self.console.STD} CPU cores for parallel processing")

    def _load_parameters(self):
        """Loads simulation parameters from JSON file.

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

    def analyze_all_replicas(self):
        """
        Analyze all replicas and generate plots.

        This method processes all replica directories, computes structural
        properties, generates visualizations, and creates summary reports.
        """
        start_time = time.time()

        if self.params is None:
            return

        cwd = self.params.get('cwd', os.getcwd())
        args = self.params['args']
        replicas = args.get('replicas', 10)
        sim_time = args.get('time', 250)  # Total simulation time in ps

        all_data = []
        all_rmsf_data = []  # Store RMSF data per residue

        # Prepare arguments for parallel processing
        replica_args = []
        replica_dirs = []
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                print(f"{self.console.PGM_WRN}{self.console.WRN}Replica {rep}{self.console.STD} directory not found, skipping.")
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
                # Estimate frame step from first replica
                try:
                    coord_files = sorted(glob.glob(f"{replica_dirs[0]}/step_*.coor"))
                    if coord_files:
                        u = mda.Universe(f"{replica_dirs[0]}/../inputs/{self.params['args']['psffile'].split('/')[-1]}",
                                        coord_files, format="NAMDBIN")
                        n_frames = len(u.trajectory)
                        frame_step = max(1, int(5 / (sim_time / n_frames)))
                        print(f"{self.console.PGM_NAM}Using rough analysis: analyzing every {self.console.EXT}{frame_step}{self.console.STD}"
                              f" frames ({frame_step * (sim_time/n_frames):.1f} ps)\n")
                except:
                    pass

        # Process replicas in parallel using CPU cores
        if replica_args:
            # Use multiprocessing for CPU-bound tasks
            with mp.Pool(processes=min(self.num_cores, len(replica_args))) as pool:
                # Create a progress tracking function
                def update_progress(result):
                    nonlocal completed
                    completed += 1
                    rep_num, _, _ = result  # Unpack the result tuple
                    print(f"{self.console.PGM_NAM}Completed analysis of {self.console.EXT}Replica {rep_num}{self.console.STD}"
                          f" [{self.console.EXT}{completed}{self.console.STD}/{self.console.WRN}{len(replica_args)}{self.console.STD}]")

                completed = 0
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

        # Save RMSF data to separate CSV
        csv_file = f"{self.analysis_dir}/rmsf.csv"
        self._save_to_csv(all_rmsf_data, csv_file)
        print(f"\n{self.console.PGM_NAM}Average RMSF results saved to {self.console.EXT}{csv_file}{self.console.STD}")

        # Save all data to CSV
        csv_file = f"{self.analysis_dir}/analysis_results.csv"
        self._save_to_csv(all_data, csv_file)
        print(f"{self.console.PGM_NAM}Analysis results saved to {self.console.EXT}{csv_file}{self.console.STD}")

        # Generate plots
        self._generate_plots(all_data, sim_time)

        # Generate RMSF plots
        self._generate_rmsf_avg_plot(all_rmsf_data)

        # Generate HTML summary
        self._generate_html_summary(all_data, sim_time)

        duration = time.time() - start_time
        print(f"\n{self.console.PGM_NAM}Analysis complete in {self.console.EXT}{duration:.2f}{self.console.STD} seconds.")
        print(f"{self.console.PGM_NAM}Results saved into {self.console.EXT}{self.analysis_dir}{self.console.STD} folder.")

    def _analyze_replica_parallel(self, rep_dir, rep_num, sim_time, rep_analysis_dir):
        """Analyzes a single simulation replica in parallel.

        Args:
            rep_dir (str): Path to replica directory
            rep_num (int): Replica number identifier
            sim_time (int): Total simulation time in picoseconds
            rep_analysis_dir (str): Output directory for replica-specific analysis

        Returns:
            Tuple containing:
                - data (list): List of analysis data dictionaries
                - data_rmsf (list): List of RMSF data dictionaries
        """
        try:
            print(f"{self.console.PGM_NAM}Starting analysis of {self.console.EXT}Replica {rep_num}{self.console.STD}...")
            result = self.analyze_replica(rep_dir, rep_num, sim_time, rep_analysis_dir)
            return (rep_num, result[0], result[1])  # Return rep_num along with data
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}replica {rep_num}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}")
            return (rep_num, [], [])  # Return rep_num even on error

    def analyze_replica(self, rep_dir, rep_num, sim_time, rep_analysis_dir):
        """Analyzes a single simulation replica.

        Args:
            rep_dir (str): Path to replica directory
            rep_num (int): Replica number identifier
            sim_time (int): Total simulation time in picoseconds
            rep_analysis_dir (str): Output directory for replica-specific analysis

        Returns:
            Tuple containing:
                - data (list): List of analysis data dictionaries
                - data_rmsf (list): List of RMSF data dictionaries
        """
        # Find all coordinate files
        coord_files = sorted(glob.glob(f"{rep_dir}/step_*.coor"))
        if not coord_files:
            print(f"{self.console.PGM_WRN}No coordinate files found for {self.console.WRN}replica {rep_num}{self.console.STD}")
            return [], []

        # Load PSF file
        psf_file = f"{rep_dir}/../inputs/{self.params['args']['psffile'].split('/')[-1]}"
        if not os.path.exists(psf_file):
            print(f"{self.console.PGM_ERR}PSF file not found for replica {self.console.ERR}{rep_num}{self.console.STD}")
            return [], []

        try:
            # Create universe
            u = mda.Universe(psf_file, coord_files, format="NAMDBIN")

            # Get the actual number of frames
            n_frames = len(u.trajectory)

            # Calculate time points (assuming each step is 0.2 ps)
            time_points = np.linspace(0, sim_time, n_frames)

            # Determine frame step for rough analysis
            frame_step = 1
            if self.rough:
                # Calculate step to get approximately 5ps intervals
                frame_step = max(1, int(5 / (sim_time / n_frames)))
                # Don't print this for each replica - already printed once

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
            # Don't print individual computation messages for each replica

            for i, ts in enumerate(u.trajectory):
                # Skip frames if rough analysis is enabled
                if self.rough and i % frame_step != 0:
                    continue

                # Ensure we don't exceed the time_points array bounds
                if i >= len(time_points):
                    break

                frame_data = {
                    'replica': rep_num,
                    'time': time_points[i],
                    'rmsd': self._calc_rmsd(selection, ref_positions),
                    'radius_gyration': self._calc_rog(selection),
                    'sasa': self._calc_sasa(u, rep_num, i),
                    'hydrophobic_exposure': self._calculate_hp(u)
                }

                # Calculate secondary structure for selected frames only
                if not self.rough or i % (frame_step * 5) == 0:  # Less frequent for SS to save time
                    ss_data = self._calc_ss(u, rep_num, i)
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

            # Calculate RMSF per residue (Cα atoms)
            rmsf_data = self._calc_rmsf(u, rep_num)

            # Generate replica-specific plots
            self._generate_replica_plots(data, rmsf_data, sim_time, rep_analysis_dir, rep_num)

            return data, rmsf_data
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error analyzing {self.console.ERR}replica {rep_num}{self.console.STD}: {self.console.ERR}{e}{self.console.STD}")
            traceback.print_exc()
            return [], []

    def _process_frame(self, u, selection, ref_positions, frame_idx, time_val, rep_num, frame_step):
        """Process a single frame and compute all properties.

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
            'replica': rep_num,
            'time': time_val,
            'rmsd': self._calc_rmsd(selection, ref_positions),
            'radius_gyration': self._calc_rog(selection),
            'sasa': self._calc_sasa(u, rep_num, frame_idx),
            'hydrophobic_exposure': self._calculate_hp(u)
        }

        # Calculate secondary structure for selected frames only
        if not self.rough or frame_idx % (frame_step * 5) == 0:  # Less frequent for SS to save time
            ss_data = self._calc_ss(u, rep_num, frame_idx)
            frame_data.update(ss_data)
        else:
            # Default values if no SS calculation
            frame_data.update({'helix': 0, 'sheet': 0, 'coil': 0, 'turn': 0, 'other': 0})

        return frame_data

    def _calc_rmsd(self, selection, ref_positions):
        """Calculates RMSD against reference positions.

        Args:
            selection (mda.AtomGroup): Atom selection to calculate RMSD for
            ref_positions (numpy.ndarray): Reference positions for comparison

        Returns:
            float: Calculated RMSD value in Angstroms
        """
        try:
            if len(selection) == 0:
                return 0

            # Ensure we're comparing the same number of atoms
            if len(selection.positions) != len(ref_positions):
                return 0

            # Calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((selection.positions - ref_positions) ** 2, axis=1)))
            return rmsd
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not calculate RMSD: {e}")
            return 0

    def _calc_rog(self, selection):
        """Calculates radius of gyration.

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

    def _calc_sasa(self, u, rep_num, frame_idx):
        """Calculates solvent accessible surface area using Bio.PDB.SASA.

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
            # Fallback to simple estimation
            try:
                selection = u.select_atoms("protein")
                if len(selection) == 0:
                    selection = u.select_atoms("all")
                return len(selection) * 15  # Approximate 15 Å² per atom
            except:
                return 0

    def _calculate_hp(self, universe):
        """Calculates percentage of hydrophobic residues in the protein.

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

    def _calc_rmsf(self, universe, rep_num):
        """Calculates RMSF per residue for Cα atoms using CPU only.

        Args:
            universe (mda.Universe): MDAnalysis Universe object
            rep_num (int): Replica number identifier

        Returns:
            list: List of dictionaries containing RMSF data per residue
        """
        try:
            # Select Cα atoms
            calphas = universe.select_atoms("protein and name CA")
            if len(calphas) == 0:
                print(f"{self.console.PGM_WRN}No Cα atoms found for RMSF calculation")
                return []

            # Use CPU implementation
            return self._calc_rmsf_cpu(universe, calphas, rep_num)

        except Exception as e:
            print(f"{self.console.PGM_ERR}Error calculating RMSF: {e}")
            return []

    def _calc_rmsf_cpu(self, universe, calphas, rep_num):
        """Calculate RMSF using CPU.

        Args:
            universe (mda.Universe): MDAnalysis Universe object
            calphas (mda.AtomGroup): Cα atoms selection
            rep_num (int): Replica number identifier

        Returns:
            list: List of dictionaries containing RMSF data per residue
        """
        # Align trajectory to first frame using Cα atoms
        ref_coords = calphas.positions.copy()
        rmsf_values = np.zeros(len(calphas))

        # Calculate RMSF manually
        for ts in universe.trajectory:
            # Align to reference
            mobile_coords = calphas.positions
            R, rmsd = align.rotation_matrix(mobile_coords, ref_coords)
            calphas.positions = np.dot(mobile_coords, R.T)

            # Accumulate squared deviations
            rmsf_values += np.sum((calphas.positions - ref_coords) ** 2, axis=1)

        # Calculate RMSF
        rmsf_values = np.sqrt(rmsf_values / len(universe.trajectory))

        # Prepare RMSF data
        rmsf_data = []
        for i, atom in enumerate(calphas):
            rmsf_data.append({
                'replica': rep_num,
                'residue_index': atom.residue.resid,
                'residue_name': atom.residue.resname,
                'rmsf': rmsf_values[i]
            })

        return rmsf_data

    def _calc_ss(self, u, rep_num, frame_idx):
        """Calculates secondary structure content using DSSP.

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
                print(f"{self.console.PGM_WRN}No protein atoms found for secondary structure analysis")
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

    def _parse_dssp_output(self, dssp_path):
        """Parses DSSP output file and counts secondary structure types.

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

    def _plot_ss_rep(self, df, sim_time, rep_analysis_dir, rep_num):
        """Creates secondary structure plot for a single replica.

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
            plt.title(f'Secondary Structure Evolution - Replica {rep_num}')
            plt.xlim(0, sim_time)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f"{rep_analysis_dir}/secondary_structure.png", bbox_inches='tight', dpi=300)
            plt.close()

        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create secondary structure plot for replica {rep_num}: {e}")

    def _save_to_csv(self, data, csv_file):
        """Saves analysis data to CSV file.

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

    def _plot_rmsf_rep(self, rmsf_data, rep_analysis_dir, rep_num):
        """Creates RMSF plot for a single replica.

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
            plt.title(f'RMSF per Residue (Cα) - Replica {rep_num}')
            plt.grid(True, alpha=0.3)

            # Save plot
            plt.savefig(f"{rep_analysis_dir}/rmsf_plot.png", bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            print(f"{self.console.PGM_WRN}Could not create RMSF plot for replica {rep_num}: {e}")

    def _generate_html_summary(self, data, sim_time):
        """Generates an HTML summary of the analysis results.

        Args:
            data (list): List of analysis data dictionaries
            sim_time (int): Total simulation time in picoseconds
        """
        if not data:
            print(f"{self.console.PGM_WRN}No data to generate HTML summary.")
            return

        df = pd.DataFrame(data)

        # Calculate statistics for each replica
        summary_data = {}
        for replica in df['replica'].unique():
            rep_data = df[df['replica'] == replica]
            replica_summary = {
                'Final Helix (residues)': rep_data['helix'].iloc[-1] if not rep_data.empty else 0,
                'Final Sheet (residues)': rep_data['sheet'].iloc[-1] if not rep_data.empty else 0,
                'Final Coil (residues)': rep_data['coil'].iloc[-1] if not rep_data.empty else 0,
                'Final Turn (residues)': rep_data['turn'].iloc[-1] if not rep_data.empty else 0,
                'Final Other (residues)': rep_data['other'].iloc[-1] if not rep_data.empty else 0,
                'Max RMSD (Å)': rep_data['rmsd'].max() if not rep_data.empty else 0,
                'Final Radius of Gyration (Å)': rep_data['radius_gyration'].iloc[-1] if not rep_data.empty else 0,
                'Final SASA (Å²)': rep_data['sasa'].iloc[-1] if not rep_data.empty else 0,
                'Max Hydrophobic Exposure (%)': rep_data['hydrophobic_exposure'].max() if not rep_data.empty else 0
            }
            summary_data[f'Replica {replica}'] = replica_summary

        # Calculate averages across all replicas
        avg_summary = {}
        for stat in ['helix', 'sheet', 'coil', 'turn', 'other', 'rmsd', 'radius_gyration', 'sasa', 'hydrophobic_exposure']:
            if stat == 'rmsd' or stat == 'hydrophobic_exposure':
                # For these, we want the max value for each replica, then average those
                max_values = df.groupby('replica')[stat].max()
                avg_summary[f'Average Max {stat.upper()}'] = max_values.mean()
            else:
                # For others, we want the final value for each replica, then average those
                final_values = df.groupby('replica')[stat].last()
                avg_summary[f'Average Final {stat.upper()}'] = final_values.mean()

        # Generate HTML file with escaped curly braces in CSS
        html_file = f"{self.analysis_dir}/analysis_summary.html"
        with open(html_file, 'w') as f:
            # Use double curly braces to escape them in the CSS part
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

                <h2>Summary Statistics</h2>
                <h3>Individual Replica Results</h3>
                {replica_tables}

                <h3>Average Across All Replicas</h3>
                <table>
                    {avg_table_rows}
                </table>

                <h2>Analysis Plots</h2>
                <div class="plot-grid">
                    {plot_items}
                </div>

                <h2>Notes</h2>
                <ul>
                    <li>Secondary structure content is calculated using DSSP</li>
                    <li>Values represent the number of residues in each secondary structure type</li>
                    <li>SASA is calculated using Bio.PDB.SASA (Shrake-Rupley algorithm)</li>
                    <li>For detailed analysis, see the files in the replica-specific subdirectories</li>
                </ul>
            </body>
            </html>
            """
            f.write(html_template.format(
                date=time.strftime("%Y-%m-%d %H:%M:%S"),
                replica_tables=self._html_rep_tables(summary_data),
                avg_table_rows=self._html_summary_avg_table(avg_summary),
                plot_items=self._html_summary_plots()
            ))

        print(f"{self.console.PGM_NAM}HTML summary saved to {self.console.EXT}{html_file}{self.console.STD}")

    def _generate_plots(self, data, sim_time):
        """Generates plots from analysis data.

        Args:
            data (list): List of analysis data dictionaries
            sim_time (int): Total simulation time in picoseconds
        """
        if not data:
            print(f"{self.console.PGM_WRN}No data to generate plots.")
            return

        df = pd.DataFrame(data)

        # Create individual plots for each property
        properties = [
            ('rmsd', 'RMSD (Å)', 'RMSD'),
            ('radius_gyration', 'Radius of Gyration (Å)', 'Radius of Gyration'),
            ('sasa', 'SASA (Å²)', 'SASA'),
            ('hydrophobic_exposure', 'Hydrophobic Exposure (%)', 'Hydrophobic Exposure')
        ]

        for prop, ylabel, title in properties:
            plt.figure(figsize=(8, 6))

            for replica in df['replica'].unique():
                rep_data = df[df['replica'] == replica]
                plt.plot(rep_data['time'], rep_data[prop], label=f'Replica {replica}', alpha=0.7, linewidth=1.5)

            plt.xlabel('Time (ps)')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xlim(0, sim_time)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save individual plot
            plt.savefig(f"{self.analysis_dir}/{prop}_plot.png", bbox_inches='tight', dpi=300)
            plt.close()

        # Create average secondary structure plot
        self._generate_ss_avg_plot(df, sim_time)

    def _generate_rmsf_avg_plot(self, rmsf_data):
        """Generates average RMSF plot across all replicas.

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

    def _generate_ss_avg_plot(self, df, sim_time):
        """Creates stacked area plot for average secondary structure.

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

    def _html_rep_tables(self, summary_data):
        """Generates HTML tables for replica summaries.

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
                    html_tables += f"<tr><td>{stat}</td><td>{value:.0f}</td></tr>"
                else:
                    html_tables += f"<tr><td>{stat}</td><td>{value:.2f}</td></tr>"
            html_tables += "</table>"
        return html_tables

    def _html_summary_avg_table(self, avg_summary):
        """Generates HTML table for average summary.

        Args:
            avg_summary (dict): Dictionary containing average summary data

        Returns:
            str: HTML string containing average summary table
        """
        html_rows = ""
        for stat, value in avg_summary.items():
            if 'HELIX' in stat or 'SHEET' in stat or 'COIL' in stat or 'TURN' in stat or 'OTHER' in stat:
                html_rows += f"<tr><td>{stat}</td><td>{value:.0f}</td></tr>"
            else:
                html_rows += f"<tr><td>{stat}</td><td>{value:.2f}</td></tr>"
        return html_rows

    def _html_summary_plots(self):
        """Generates HTML img elements for plots.

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

    def _generate_replica_plots(self, data, rmsf_data, sim_time, rep_analysis_dir, rep_num):
        """Generates plots for a single replica analysis.

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

        # Create individual plots for each property
        properties = [
            ('rmsd', 'RMSD (Å)', 'RMSD'),
            ('radius_gyration', 'Radius of Gyration (Å)', 'Radius of Gyration'),
            ('sasa', 'SASA (Å²)', 'SASA'),
            ('hydrophobic_exposure', 'Hydrophobic Exposure (%)', 'Hydrophobic Exposure'),
        ]

        for prop, ylabel, title in properties:
            plt.figure(figsize=(6, 6))
            plt.plot(df['time'], df[prop], label=f'Replica {rep_num}', color='blue', linewidth=2)
            plt.xlabel('Time (ps)')
            plt.ylabel(ylabel)
            plt.title(f'{title} - Replica {rep_num}')
            plt.xlim(0, sim_time)
            plt.grid(True, alpha=0.3)

            # Save individual plot
            plt.savefig(f"{rep_analysis_dir}/{prop}_plot.png", bbox_inches='tight', dpi=300)
            plt.close()

        # Create RMSF plot for this replica
        if rmsf_data:
            self._plot_rmsf_rep(rmsf_data, rep_analysis_dir, rep_num)

        # Create secondary structure plot for this replica
        self._plot_ss_rep(df, sim_time, rep_analysis_dir, rep_num)

        # Save replica data to CSV
        csv_file = f"{rep_analysis_dir}/analysis_results.csv"
        df.to_csv(csv_file, index=False)

        # Save RMSF data to CSV
        if rmsf_data:
            rmsf_df = pd.DataFrame(rmsf_data)
            rmsf_csv_file = f"{rep_analysis_dir}/rmsf.csv"
            rmsf_df.to_csv(rmsf_csv_file, index=False)


def parse_arguments():
    """Parse command-line arguments for the pyAdMD analysis tool.

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
    run_required.add_argument('-type', '--modeltype', action="store", type=str.upper,
                             default="CA", required=True, choices=["CA", "HEAVY", "CHARMM"],
                             help="Compute ENM or use pre-computed CHARMM normal mode file (default: CA)")
    run_required.add_argument('-psf', '--psffile', action="store", type=str, required=True,
                             help="PSF topology file")
    run_required.add_argument('-pdb', '--pdbfile', action="store", type=str, required=True,
                             help="PDB structure file")
    run_required.add_argument('-coor','--coorfile', action="store", type=str, required=True,
                             help="NAMD coordinates file")
    run_required.add_argument('-vel', '--velfile', action="store", type=str, required=True,
                             help="NAMD velocities file")
    run_required.add_argument('-xsc', '--xscfile', action="store", type=str, required=True,
                             help="NAMD PBC file")
    run_required.add_argument('-str', '--strfile', action="store", type=str, required=True,
                             help="NAMD additional box info file")

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

    # RESTART subparser
    subparsers.add_parser('restart', help="Restart unfinished simulations")

    # APPEND subparser
    opt_apnd = subparsers.add_parser('append', help="Extend previously computed simulations")
    opt_apnd.add_argument('-t', '--time', action="store", type=int, required=True, default=100,
                         help="Simulation time to append (default: 100ps)")

    # ANALYSIS subparser
    opt_analyze = subparsers.add_parser('analyze', help="Analyze simulation results and generate plots")
    opt_analyze.add_argument('-r', '--rough', action='store_true',
                            help='Perform rough analysis (every 5ps instead of every frame)')

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

    return args


def unzip_file(filepath, dest_dir):
    """Extract files from a .zip compressed file.

    Args:
        filepath (str): Path to .zip file to extract.
        dest_dir (str): Path to destination folder for extracted files.
    """
    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(dest_dir)


def write_charmm_nm(nms_to_write, psffile, modefile, cwd):
    """Write CHARMM normal mode vectors in NAMD readable format.

    This function:
    1. Extracts CHARMM topology and parameter files
    2. Creates input file listing modes to process
    3. Executes CHARMM to generate mode vectors in NAMD format

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

    nms = [f"{t}\n" for t in nms_to_write.split(',')]
    with open(f"{cwd}/inputs/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)

    os.chdir(f"{cwd}/tools")
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]} modfile={modefile.split('/')[-1]} -o ../wrt-nm.out")

    returned_value = subprocess.call(cmd, shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{console.PGM_ERR}An error occurred while writing the normal mode vectors.")
        print(f"{console.PGM_ERR}Inspect the file {console.ERR}wrt-nm.out{console.STD} for detailed information.")
        sys.exit(1)


def run_namd(conf_file, psffile, pdbfile, strfile, loop_step, deexcitation=False):
    """Configure and run NAMD simulations.

    Replaces placeholder variables in NAMD configuration file and executes
    NAMD3 for either excitation or deexcitation steps.

    Args:
        conf_file (pathlib.Path): Path to NAMD configuration file.
        psffile (str): Path to PSF topology file.
        pdbfile (str): Path to PDB structure file.
        strfile (str): Path to STR structure file with box information.
        loop_step (int): Current simulation step number.
        deexcitation (bool): Whether to run deexcitation step (default: False).

    Raises:
        SystemExit: If NAMD execution fails.
    """
    console = ConsoleConfig()

    # Edit NAMD configuration file
    conf_file.write_text(conf_file.read_text().replace('$PSF', psffile))
    conf_file.write_text(conf_file.read_text().replace('$PDB', pdbfile))
    conf_file.write_text(conf_file.read_text().replace('$STR', strfile))

    if not deexcitation:
        conf_file.write_text(conf_file.read_text().replace('$COOR', str(loop_step - 1)))
        conf_file.write_text(conf_file.read_text().replace('$VEL', str(loop_step - 1)))
        conf_file.write_text(conf_file.read_text().replace('$XSC', str(loop_step - 1)))
        conf_file.write_text(conf_file.read_text().replace('$OUTPUT', str(loop_step)))

        # Run NAMD
        cmd = f"namd3 conf.namd > step_{loop_step}.log"
        returned_value = subprocess.call(cmd, shell=True,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if returned_value != 0:
            print(f"{console.PGM_ERR}An error occurred while running NAMD.\n"
                f"{console.PGM_ERR}Inspect the file {console.ERR}step_{loop_step}.log{console.STD} for detailed information.\n")
            sys.exit()
    else:
        conf_file.write_text(conf_file.read_text().replace('$COOR', str(loop_step)))
        conf_file.write_text(conf_file.read_text().replace('$VEL', str(loop_step)))
        conf_file.write_text(conf_file.read_text().replace('$XSC', str(loop_step)))
        conf_file.write_text(conf_file.read_text().replace('$TS', str(int(loop_step * 100))))

        # Run NAMD
        cmd = f"namd3 deexcitation.namd > deexcitation.log"
        returned_value = subprocess.call(cmd, shell=True,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if returned_value != 0:
            print(f"{console.PGM_ERR}An error occurred while running NAMD.\n"
                f"{console.PGM_ERR}Inspect the file {console.ERR}deexcitation.log{console.STD} for detailed information.\n")
            sys.exit()


def find_last_completed_cycle(rep_dir):
    """Find the last completed cycle in a replica directory.

    Scans for coordinate files and extracts the highest cycle number.

    Args:
        rep_dir (str): Path to replica directory to scan.

    Returns:
        int: Highest completed cycle number, or 0 if no cycles found.
    """
    coord_files = glob.glob(f"{rep_dir}/step_*.coor")
    if not coord_files:
        return 0

    # Extract cycle numbers from filenames
    cycles = []
    for file in coord_files:
        try:
            cycle_num = int(os.path.basename(file).split('_')[1].split('.')[0])
            cycles.append(cycle_num)
        except (ValueError, IndexError):
            continue

    if not cycles:
        return 0

    return max(cycles)


def main():
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

        # Store file paths
        psffile  = args.psffile
        pdbfile  = args.pdbfile
        coorfile = args.coorfile
        velfile  = args.velfile
        xscfile  = args.xscfile
        strfile  = args.strfile
        if args.modefile:
            modefile = args.modefile

        # Test if the provided files exist
        file_list = [psffile, pdbfile, coorfile, velfile, xscfile, strfile]
        if args.modefile:
            file_list.append(modefile)

        for file in file_list:
            if not os.path.isfile(file):
                print(f"{console.PGM_ERR}File {file.split('/')[-1]} not found.")
                sys.exit(1)
            # Test if the provided files are at the input folder and copy them if not
            if not os.path.isfile(f"{input_dir}/{file.split('/')[-1]}"):
                shutil.copy(file, input_dir)
                print(f"{console.PGM_WRN}File {console.WRN}{file.split('/')[-1]}{console.STD} was copied to inputs folder.")

        # Process file paths
        psffile = f"{input_dir}/{psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{pdbfile.split('/')[-1]}"
        coorfile = f"{input_dir}/{coorfile.split('/')[-1]}"
        velfile = f"{input_dir}/{velfile.split('/')[-1]}"
        xscfile = f"{input_dir}/{xscfile.split('/')[-1]}"
        strfile = f"{input_dir}/{strfile.split('/')[-1]}"
        if args.modefile:
            modefile = f"{input_dir}/{modefile.split('/')[-1]}"

        # Store parameters
        nm_type = args.modeltype.lower()
        modes = args.modes
        nm_parsed = [int(s) for s in modes.split(',')]
        energy = args.energy
        sim_time = args.time
        selection = args.selection
        replicas = args.replicas

        # Get some information from the system
        sys_pdb = mda.Universe(psffile, coorfile, format="NAMDBIN")
        sys_mass = sys_pdb.atoms.masses  # System atomic mass
        init_coor = sys_pdb.atoms.select_atoms(selection).positions
        sel_atom = sys_pdb.atoms.select_atoms(selection).n_atoms  # Number of selected atoms
        sel_mass = sys_pdb.atoms.select_atoms(selection).masses  # Selection atomic mass

        # Define the number of excitation cycles
        # sim_time / (total_steps * timestep)
        end_loop = int(sim_time / (100 * 0.002))

        # Compute/Write the normal mode vectors
        if nm_type == "ca":
            print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Cα ENM{console.STD}{console.HGH} "
                  f"and writing normal mode vectors {console.EXT}{modes}{console.STD}.")
            enm_calculator.compute_enm(coorfile, nm_type, nm_parsed, input_dir, psffile)
        elif nm_type == "heavy":
            print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Heavy atoms ENM{console.STD}{console.HGH} "
                  f"and writing normal mode vectors {console.EXT}{modes}{console.STD}.")
            enm_calculator.compute_enm(coorfile, nm_type, nm_parsed, input_dir, psffile)
        elif nm_type == "charmm":
            print(f"\n{console.PGM_NAM}Writing {console.EXT}CHARMM{console.STD} normal mode vectors {console.EXT}{modes}{console.STD}.")
            write_charmm_nm(modes, psffile, modefile, cwd)

        # Extract NAMD topology and parameters files
        unzip_file(f"{input_dir}/namd_toppar.zip", input_dir)

        # Generate factors to uniformly combine the modes
        print(f"\n{console.PGM_NAM}Generating {console.EXT}{replicas}{console.STD} uniformly distributed "
              f"combinations of modes {console.EXT}{modes}{console.STD}.")
        factors = mode_exciter.generate_factors(replicas, len(nm_parsed), cwd, nm_parsed)

        # Continue with existing code
        mode_exciter.combine_modes(replicas, nm_parsed, factors, nm_type, coorfile, cwd, sys_pdb)

        # Save parameters for potential restart/append
        param_storage.save_parameters(args, factors, nm_parsed, end_loop, cwd)

        # Initialize the simulation runner
        sim_runner = SimulationRunner(
            console, args, cwd, input_dir, psffile, pdbfile, coorfile,
            velfile, xscfile, strfile, sel_mass, init_coor, energy,
            mode_exciter, sys_pdb
        )

        # Run simulations for all replicas
        for rep in range(1, replicas + 1):
            sim_runner.run_simulation(rep, 0, end_loop)

    ### RESTART
    elif args.option == 'restart':
        print(f"{console.PGM_NAM}{console.TLE}Restart unfinished pyAdMD simulation{console.STD}\n")

        # Load parameters
        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        args = params['args']
        factors = params['factors']
        nm_parsed = params['nm_parsed']
        end_loop = params['end_loop']
        cwd = params['cwd']

        # Reconstruct file paths from stored args
        input_dir = f"{cwd}/inputs"
        psffile = f"{input_dir}/{args.psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{args.pdbfile.split('/')[-1]}"
        coorfile = f"{input_dir}/{args.coorfile.split('/')[-1]}"
        velfile = f"{input_dir}/{args.velfile.split('/')[-1]}"
        xscfile = f"{input_dir}/{args.xscfile.split('/')[-1]}"
        strfile = f"{input_dir}/{args.strfile.split('/')[-1]}"

        # Get parameters from loaded args
        nm_type = args.modeltype.lower()
        energy = args.energy
        selection = args.selection
        replicas = args.replicas

        # Get some information from the system
        sys_pdb = mda.Universe(psffile, coorfile, format="NAMDBIN")
        sel_mass = sys_pdb.atoms.select_atoms(selection).masses
        init_coor = sys_pdb.atoms.select_atoms(selection).positions

        # Initialize the simulation runner
        sim_runner = SimulationRunner(
            console, args, cwd, input_dir, psffile, pdbfile, coorfile,
            velfile, xscfile, strfile, sel_mass, init_coor, energy,
            mode_exciter, sys_pdb
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
            rep_dir = f"{cwd}/rep{rep}"
            last_cycle = find_last_completed_cycle(rep_dir)

            # Unfinished replicas
            if os.path.exists(rep_dir):
                if args.no_correc:
                    print(f"\n{console.PGM_NAM}{console.HGH}Restarting Standard MDeNM calculations for {console.EXT}Replica "
                            f"{rep}{console.STD}{console.HGH} from {console.EXT}step {last_cycle}{console.STD}")
                elif args.fixed:
                    print(f"\n{console.PGM_NAM}{console.HGH}Restarting Constant MDeNM calculations for {console.EXT}Replica "
                            f"{rep}{console.STD}{console.HGH} from {console.EXT}step {last_cycle}{console.STD}")
                else:
                    print(f"\n{console.PGM_NAM}{console.HGH}Restarting Adaptive MDeNM calculations for {console.EXT}Replica "
                            f"{rep}{console.STD}{console.HGH} from {console.EXT}step {last_cycle}{console.STD}")

            # Load correction state if it exists
            correction_state = {}
            if os.path.exists(f"{rep_dir}/correction_state.json"):
                with open(f"{rep_dir}/correction_state.json", 'r') as f:
                    correction_state = json.load(f)

            sim_runner.run_simulation(rep, last_cycle, end_loop, correction_state)

    ### APPEND
    elif args.option == 'append':
        print(f"{console.PGM_NAM}{console.TLE}Append previous pyAdMD simulation{console.STD}\n")

        # Store parameters
        additional_time = args.time

        # Update parameters with new end loop
        # Load parameters
        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        args_dict = params['args']
        factors = params['factors']
        nm_parsed = params['nm_parsed']
        original_end_loop = params['end_loop']
        cwd = params['cwd']

        # Calculate new end loop and update total time
        additional_steps = int(additional_time / (100 * 0.002))
        new_end_loop = original_end_loop + additional_steps

        # Update the total simulation time in the parameters
        original_time = args_dict.time
        new_total_time = original_time + additional_time
        args_dict.time = new_total_time

        params['end_loop'] = new_end_loop
        # Save parameters with updated time
        param_storage.save_parameters(args_dict, factors, nm_parsed, new_end_loop, cwd)

        # Reconstruct file paths from stored args
        input_dir = f"{cwd}/inputs"
        psffile = f"{input_dir}/{args_dict.psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{args_dict.pdbfile.split('/')[-1]}"
        coorfile = f"{input_dir}/{args_dict.coorfile.split('/')[-1]}"
        velfile = f"{input_dir}/{args_dict.velfile.split('/')[-1]}"
        xscfile = f"{input_dir}/{args_dict.xscfile.split('/')[-1]}"
        strfile = f"{input_dir}/{args_dict.strfile.split('/')[-1]}"

        # Get parameters from loaded args
        nm_type = args_dict.modeltype.lower()
        energy = args_dict.energy
        selection = args_dict.selection
        replicas = args_dict.replicas

        # Get some information from the system
        sys_pdb = mda.Universe(psffile, coorfile, format="NAMDBIN")
        sel_mass = sys_pdb.atoms.select_atoms(selection).masses
        init_coor = sys_pdb.atoms.select_atoms(selection).positions

        # Initialize the simulation runner
        sim_runner = SimulationRunner(
            console, args_dict, cwd, input_dir, psffile, pdbfile, coorfile,
            velfile, xscfile, strfile, sel_mass, init_coor, energy,
            mode_exciter, sys_pdb
        )

        # Check if any replicas need to be extended
        replicas_to_extend = []
        for rep in range(1, replicas + 1):
            rep_dir = f"{cwd}/rep{rep}"
            if not os.path.exists(rep_dir):
                print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} directory not found, skipping.")
                continue

            last_cycle = find_last_completed_cycle(rep_dir)
            if last_cycle < original_end_loop:
                print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} hasn't completed the original simulation, skipping.")
                continue

            replicas_to_extend.append(rep)

        if not replicas_to_extend:
            print(f"{console.PGM_WRN}No replicas to be extended. All replicas either don't exist or haven't completed the original simulation.")
            return

        # Extend each replica that has completed the original simulation
        for rep in replicas_to_extend:
            rep_dir = f"{cwd}/rep{rep}"
            print(f"\n{console.PGM_NAM}{console.HGH}Extending {console.EXT}Replica {rep}{console.STD}{console.HGH} for "
                  f"{console.EXT}{additional_time}{console.STD}{console.HGH} picosseconds{console.STD}")

            # Load correction state if it exists
            correction_state = {}
            if os.path.exists(f"{rep_dir}/correction_state.json"):
                with open(f"{rep_dir}/correction_state.json", 'r') as f:
                    correction_state = json.load(f)

            sim_runner.run_simulation(rep, original_end_loop, new_end_loop, correction_state)

    ### ANALYZE
    elif args.option == 'analyze':
        print(f"{console.PGM_NAM}{console.TLE}Analyze pyAdMD results{console.STD}\n")
        analyzer = Analyzer(console, rough=args.rough)
        analyzer.analyze_all_replicas()

    ### CLEAN
    elif args.option == 'clean':
        print(f"{console.PGM_NAM}{console.TLE}Clean previous pyAdMD setup files{console.STD}\n")

        # Removing previous replicas folders
        files = os.listdir(cwd)
        for item in files:
            if item.endswith((".json", "summary.txt")):
                 os.remove(os.path.join(cwd, item))
            if item.startswith(("rep", "analysis")):
                shutil.rmtree(os.path.join(cwd, item), ignore_errors=True)

        # Removing previous configuration files
        files = os.listdir(input_dir)
        for item in files:
            if item.endswith((".txt", ".out", ".crd", ".psf", ".pdb", ".coor", ".vel", ".xsc", ".str", ".mod")):
                os.remove(os.path.join(input_dir, item))
            # Removing previous ENM calculations
            if item.endswith("_enm"):
                shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)
        for item in ("charmm_toppar", "namd_toppar"):
                shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)

        print(f"{console.PGM_NAM}Erasing is done.\n")


if __name__ == "__main__":
    main()
