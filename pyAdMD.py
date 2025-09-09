"""
The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of
multiple-replica short MD simulations in which motions described by a given
subset of low-frequency NMs are kinetically excited. This is achieved by
injecting additional atomic velocities along several randomly determined
linear combinations of NM vectors, thus allowing an efficient coupling
between slow and fast motions.

This new approach, aMDeNM, automatically controls the energy injection and
take the natural constraints imposed by the structure and the environment
into account during protein conformational sampling, which prevent structural
distortions all along the simulation. Due to the stochasticity of thermal
motions, NM eigenvectors move away from the original directions when used to
displace the protein, since the structure evolves into other potential energy
wells. Therefore, the displacement along the modes is valid for small
distances, but the displacement along greater distances may deform the
structure of the protein if no care is taken. The advantage of this method is
to adaptively change the direction used to displace the system, taking into
account the structural and energetic constraints imposed by the system itself
and the medium, which allows the system to explore new pathways.

This program provides a Python implementation of the aMDeNM method.
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
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import warnings

# Ignore MDAnalysis attribute warnings raised during ENM computation
warnings.filterwarnings("ignore", module='MDAnalysis')
# Ignore Bio deprecation warnings raised by MDAnalysis calling
warnings.filterwarnings("ignore", module='Bio')

try:
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh
    from scipy.sparse import diags
    import numba
    from numba import njit, prange, float64, int32
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import cupy as cp
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    import openmm as mm
    from openmm import app, unit, Platform
except ImportError as e:
    print(f"Required libraries not found: {e}")
    sys.exit(1)


class ConsoleConfig:
    """
    Configuration class for PyAdMD application providing console styling and messages.

    Attributes
    ----------
    BLK, TLE, HGH, WRN, ERR, EXT, STD : str
        ANSI escape codes for console text styling
    PGM_NAM, PGM_WRN, PGM_ERR : str
        Formatted program output prefixes
    LOGO : str
        ASCII art logo for the application
    VERSION : str
        Application version number
    CITATION : str
        Citation information for the method
    MESSAGE : str
        Brief description of the program
    """

    # Style variables
    BLK = '\033[5;36m'
    TLE = '\033[2;106m'
    HGH = '\033[1;100m'
    WRN = '\033[33m'
    ERR = '\033[31m'
    EXT = '\033[32m'
    STD = '\033[0m'

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

    VERSION = '1.2'
    CITATION = '''  Please cite:

    \tAdaptive collective motions: a hybrid method to improve
    \tconformational sampling with molecular dynamics and normal modes.
    \tPT Resende-Lara, MGS Costa, B Dudas, D Perahia.
    \tDOI: https://doi.org/10.1101/2022.11.29.517349'''

    MESSAGE = "This program can setup and run multi-replica aMDeNM simulations through NAMD."


class ENMCalculator:
    """
    Class for Elastic Network Model calculations.

    This class handles the computation of elastic network models, including
    system creation, Hessian matrix computation, and normal mode analysis.

    Parameters
    ----------
    console : ConsoleConfig
        Console configuration object for formatted output
    """

    def __init__(self, console):
        self.console = console

    def compute_enm(self, coorfile, nm_type, nm_parsed, input_dir, psffile):
        """
        Setup and run ENM analysis.

        Parameters
        ----------
        coorfile : str
            Input coordinate file name
        nm_type : str
            Type of normal mode calculation ('CA' or 'HEAVY')
        nm_parsed : list
            List containing mode numbers to analyze
        input_dir : str
            Input directory path
        psffile : str
            PSF topology filename
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
        """
        Create an Elastic Network Model system based on the specified model type.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file
        model_type : str, optional
            Type of model to create: 'ca' for Cα-only or 'heavy' for heavy-atom ENM
        cutoff : float, optional
            Cutoff distance for interactions in Å. If None, uses default values:
            15.0Å for CA model, 12.0Å for heavy-atom model
        spring_constant : float, optional
            Spring constant for the ENM bonds in kcal/mol/Å²
        output_prefix : str, optional
            Prefix for output files

        Returns
        -------
        system : openmm.System
            The created OpenMM system
        topology : openmm.app.Topology
            The topology of the system
        positions : openmm.unit.Quantity
            The positions of particles in the system

        Raises
        ------
        ValueError
            If an unknown model type is specified
        """
        # Set default cutoffs if not provided
        if cutoff is None:
            cutoff = 15.0 if model_type == 'ca' else 12.0

        if model_type.lower() == 'ca':
            return self._create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
        elif model_type.lower() == 'heavy':
            return self._create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)
        else:
            raise ValueError(f"Unknown model type: {self.console.ERR}{model_type}{self.console.STD}")

    def _create_ca_system(self, pdb_file, cutoff, spring_constant, output_prefix):
        """
        Create a Cα-only Elastic Network Model system.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file
        cutoff : float
            Cutoff distance for interactions in Å
        spring_constant : float
            Spring constant for the ENM bonds in kcal/mol/Å²
        output_prefix : str
            Prefix for output files

        Returns
        -------
        system : openmm.System
            The created OpenMM system
        topology : openmm.app.Topology
            The topology of the system
        positions : openmm.unit.Quantity
            The positions of particles in the system
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
        print(f"{self.console.PGM_NAM}C-alpha structure saved to {self.console.EXT}{ca_pdb_file}{self.console.STD}.")

        # Convert HETATM to ATOM
        self.convert_hetatm_to_atom(ca_pdb_file)

        return system, new_topology, positions_quantity

    def _create_heavy_system(self, pdb_file, cutoff, spring_constant, output_prefix):
        """
        Create a heavy-atom Elastic Network Model system.

        Parameters
        ----------
        pdb_file : str
            Path to the input PDB file
        cutoff : float
            Cutoff distance for interactions in Å
        spring_constant : float
            Spring constant for the ENM bonds in kcal/mol/Å²
        output_prefix : str
            Prefix for output files

        Returns
        -------
        system : openmm.System
            The created OpenMM system
        topology : openmm.app.Topology
            The topology of the system
        positions : openmm.unit.Quantity
            The positions of particles in the system
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
                new_res = new_topology.addResidue(f"{residure.name}{residue.id}", new_chain)
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
        """
        Convert HETATM records to ATOM in PDB files for compatibility with visualization tools.

        Parameters
        ----------
        pdb_file : str
            Path to the PDB file to convert
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
        """
        Compute the Hessian matrix for an Elastic Network Model using CPU parallelization.

        Parameters
        ----------
        pos_array : ndarray
            Array of particle positions (N×3)
        bonds : ndarray
            Array of bonds with format [i, j, r0] for each bond
        k_val : float
            Spring constant
        n_particles : int
            Number of particles in the system

        Returns
        -------
        hessian : ndarray
            The computed Hessian matrix (3N×3N)
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
        """
        Build, compute and regularize Hessian matrix for an Elastic Network Model.

        Parameters
        ----------
        system : openmm.System
            The system containing the ENM force
        positions : openmm.unit.Quantity
            The positions of particles in the system

        Returns
        -------
        hessian : ndarray
            The computed Hessian matrix (3N×3N)

        Raises
        ------
        ValueError
            If no ENM force is found in the system
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
        """
        Apply mass-weighting to the Hessian matrix.

        Parameters
        ----------
        hessian : ndarray
            The Hessian matrix to mass-weight
        system : openmm.System
            The system containing particle masses

        Returns
        -------
        mw_hessian : ndarray
            The mass-weighted Hessian matrix
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
        """
        Diagonalize the Hessian matrix using GPU acceleration.

        Parameters
        ----------
        hessian : ndarray
            The Hessian matrix to diagonalize
        n_modes : int, optional
            Number of modes to compute. If None, computes all modes.

        Returns
        -------
        eigenvalues : ndarray
            The eigenvalues of the Hessian matrix
        eigenvectors : ndarray
            The eigenvectors of the Hessian matrix
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
        """
        Compute normal modes by diagonalizing the Hessian matrix.

        Parameters
        ----------
        hessian : ndarray
            The Hessian matrix to diagonalize
        n_modes : int, optional
            Number of modes to compute. If None, computes all modes.
        use_gpu : bool, optional
            Whether to use GPU acceleration for diagonalization

        Returns
        -------
        frequencies : ndarray
            The frequencies of the normal modes
        modes : ndarray
            The normal mode vectors
        eigenvalues : ndarray
            The eigenvalues of the Hessian matrix
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
        """
        Write normal mode eigenvectors to separate XYZ files for the complete protein structure.

        Parameters
        ----------
        modes : ndarray
            The normal mode vectors (3N×M)
        frequencies : ndarray
            The frequencies of the normal modes
        system : openmm.System
            The system containing particle information
        topology : openmm.app.Topology
            The topology of the system
        nm : int
            Mode to write
        output_prefix : str
            Prefix for output XYZ files
        pdb_file : str
            Path to the original PDB file to get complete atom information
        model_type : str
            Type of model ('charmm', 'ca' or 'heavy')
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
    """
    Class for combining and exciting normal modes.

    This class handles the generation of linear combinations of normal modes
    and the application of excitation energy to these modes.

    Parameters
    ----------
    console : ConsoleConfig
        Console configuration object for formatted output
    """

    def __init__(self, console):
        self.console = console

    def generate_factors(self, P, N, cwd, nm_parsed):
        """
        Generate factors for linear combinations of N orthonormal vectors
        that produce P points uniformly distributed on a N-dimensional hypersphere surface.

        Parameters
        ----------
        P : int
            Number of points to generate
        N : int
            Dimensionality of the space
        cwd : str
            Current working directory
        nm_parsed : list
            List of mode numbers

        Returns
        -------
        numpy.ndarray
            P×N matrix of factors for linear combinations
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
        """
        Write combination factors to a CSV file.

        Parameters
        ----------
        factors : numpy.ndarray
            Matrix of factors for linear combinations
        nm_parsed : list
            List of mode numbers
        cwd : str
            Current working directory
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
        """
        Combine, normalize and write normal modes.

        Parameters
        ----------
        replicas : int
            Number of replicas to compute
        modes : list
            List of integers containing the normal mode numbers
        factors : numpy.ndarray
            Matrix of factors for linear combinations
        nm_type : str
            Type of normal mode calculation ('CHARMM', 'CA' or 'HEAVY')
        coorfile : str
            Coordinate filename
        cwd : str
            Current working directory
        mda_U : MDAnalysis.Universe
            System structure
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
        """
        Scale the combined normal modes to be used as additional
        velocities during aMDeNM simulations.

        Parameters
        ----------
        q_vector : matrix
            Combined vector to excite
        user_ek : float
            User defined excitation energy
        sel_mass : array
            Selection atomic mass

        Returns
        -------
        matrix
            Excitation vector
        """
        # Excite
        fscale = np.sqrt((2 * user_ek) / sel_mass)
        exc_vec = (q_vector.T * fscale).T

        return exc_vec

    def _write_vector(self, xyz, output_file, mda_U):
        """
        Write a set of coordinates in a new file.

        Parameters
        ----------
        xyz : array
            Vector containing the xyz coordinates
        output_file : str
            Output file name
        mda_U : MDAnalysis.Universe
            System structure
        """
        # Copy the xyz coordinates into the dataframe
        sys_zeros = mda_U.atoms.select_atoms("all")
        sys_zeros.positions = np.zeros((mda_U.atoms.n_atoms, 3))
        vector = np.append(xyz, sys_zeros.positions, axis=0)
        sys_zeros.positions = vector[:mda_U.atoms.n_atoms]

        # Write the output file
        sys_zeros.write(output_file, file_format="NAMDBIN")


class ParameterStorage:
    """
    Class to store and retrieve simulation parameters.

    This class handles the serialization and deserialization of simulation
    parameters to/from JSON files for restart capabilities.

    Parameters
    ----------
    console : ConsoleConfig
        Console configuration object for formatted output
    """

    def __init__(self, console):
        self.console = console
        self.param_file = "pyAdMD_params.json"

    def save_parameters(self, args, factors, nm_parsed, end_loop, cwd):
        """
        Save simulation parameters to a JSON file.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments
        factors : numpy.ndarray
            Matrix of factors for linear combinations
        nm_parsed : list
            List of mode numbers
        end_loop : int
            Final loop iteration count
        cwd : str
            Current working directory
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
        """
        Load simulation parameters from a JSON file.

        Returns
        -------
        dict or None
            Loaded parameters dictionary or None if loading failed
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


class SimulationRunner:
    """
    Class to handle running, restarting, and appending simulations.

    This class provides a unified interface for managing simulation runs,
    including initialization, execution, and cleanup of replica simulations.

    Parameters
    ----------
    console : ConsoleConfig
        Console configuration object for formatted output
    args : argparse.Namespace
        Command line arguments
    cwd : str
        Current working directory
    input_dir : str
        Input directory path
    psffile : str
        PSF topology file path
    pdbfile : str
        PDB structure file path
    coorfile : str
        Coordinate file path
    velfile : str
        Velocity file path
    xscfile : str
        Extended system configuration file path
    strfile : str
        Structure file path
    sel_mass : array
        Atomic masses of selected atoms
    init_coor : array
        Initial coordinates of selected atoms
    energy : float
        Excitation energy value
    mode_exciter : ModeExciter
        Mode exciter instance
    sys_pdb : MDAnalysis.Universe
        System structure universe
    """

    def __init__(self, console, args, cwd, input_dir, psffile, pdbfile, coorfile,
                 velfile, xscfile, strfile, sel_mass, init_coor, energy, mode_exciter, sys_pdb):
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
        """
        Run simulation for a specific replica.

        Parameters
        ----------
        rep : int
            Replica number
        start_loop : int
            Starting loop index
        end_loop : int
            Ending loop index
        correction_state : dict, optional
            Correction state for restart/append

        Returns
        -------
        dict
            Final correction state after simulation completion
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

                # Correct the excitation direction if necessary
                if rms_check >= self.qrms_correc:
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
                    cnt += 1

                    # Write the corrected excitation vector
                    print(f"{self.console.PGM_NAM}Writing the corrected excitation vector.")
                    self.mode_exciter._write_vector(diff, "cntrl_vector.vec", self.sys_pdb)

                    # Excite and write the new excited vector
                    exc_vec = self.mode_exciter.excite(diff, self.energy, self.sel_mass)
                    self.mode_exciter._write_vector(exc_vec, "excitation.vel", self.sys_pdb)

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


def parse_arguments():
    """
    Parse command-line arguments for the pyAdMD analysis.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    console = ConsoleConfig()

    parser = argparse.ArgumentParser(description=console.MESSAGE)
    subparsers = parser.add_subparsers(dest='option', help='Available commands')

    # RUN subparser
    opt_run = subparsers.add_parser('run', help="Setup and run simulations.")

    # Required arguments for run
    run_required = opt_run.add_argument_group('Required arguments')
    run_required.add_argument('-type', '--type', action="store", type=str.upper,
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
    run_flags.add_argument('--no_correc', action='store_true',
                          help='Compute standard MDeNM calculations')
    run_flags.add_argument('--fixed', action='store_true',
                          help='Disable excitation vector correction and keep constant excitation energy injections')

    # RESTART subparser
    subparsers.add_parser('restart', help="Restart unfinished simulations.")

    # APPEND subparser
    opt_apnd = subparsers.add_parser('append', help="Extend previously computed simulations.")
    opt_apnd.add_argument('-t', '--time', action="store", type=int, required=True, default=100,
                         help="Simulation time to append (default: 100ps)")

    # CLEAN subparser
    subparsers.add_parser('clean', help="Erase all previous simulation files.")

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
        if args.type == 'CHARMM' and not args.modefile:
            opt_run.error("The -mod/--modefile argument is required when -type/--type is CHARMM")

    return args


def unzip_file(filepath, dest_dir):
    """
    Extract files from a .zip compressed file.

    Parameters
    ----------
    filepath : str
        Path to .zip file
    dest_dir : str
        Path to destination folder
    """
    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(dest_dir)


def write_charmm_nm(nms_to_write, psffile, modefile, cwd):
    """
    Write CHARMM normal mode vectors in NAMD readable format.

    Parameters
    ----------
    nms_to_write : str
        Comma-separated list of normal modes to write
    psffile : str
        PSF filename
    modefile : str
        Mode filename
    cwd : str
        Current working directory
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
    """
    Configure and run NAMD simulations.

    Parameters
    ----------
    conf_file : str
        NAMD configuration file
    psffile : str
        PSF filename
    pdbfile : str
        PDB filename
    strfile : str
        STR filename
    loop_step : int
        Current loop_step step
    deexcitation : boolean, optional
        Choose whether NAMD run is excitation or deexcitation
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
    """
    Find the last completed cycle in a replica directory.

    Parameters
    ----------
    rep_dir : str
        Path to replica folder

    Returns
    -------
    int
        Last completed cycle number, or 0 if no cycles found
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
    """
    Main function to run the PyAdMD application.

    This function serves as the entry point for the application, handling
    command-line argument parsing, initialization of components, and
    execution of the requested operation (run, restart, append, or clean).
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

    # RUNNING OPTIONS
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
        nm_type = args.type.lower()
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
        nm_type = args.type.lower()
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

    elif args.option == 'append':
        print(f"{console.PGM_NAM}{console.TLE}Append previous pyAdMD simulation{console.STD}\n")

        # Store parameters
        additional_time = args.time

        # Update parameters with new end loop
        # Load parameters
        params = param_storage.load_parameters()
        if params is None:
            sys.exit(1)

        args = params['args']
        factors = params['factors']
        nm_parsed = params['nm_parsed']
        original_end_loop = params['end_loop']
        cwd = params['cwd']

        # Calculate new end loop
        additional_steps = int(additional_time / (100 * 0.002))
        new_end_loop = original_end_loop + additional_steps

        params['end_loop'] = new_end_loop
        param_storage.save_parameters(args, factors, nm_parsed, new_end_loop, cwd)

        # Reconstruct file paths from stored args
        input_dir = f"{cwd}/inputs"
        psffile = f"{input_dir}/{args.psffile.split('/')[-1]}"
        pdbfile = f"{input_dir}/{args.pdbfile.split('/')[-1]}"
        coorfile = f"{input_dir}/{args.coorfile.split('/')[-1]}"
        velfile = f"{input_dir}/{args.velfile.split('/')[-1]}"
        xscfile = f"{input_dir}/{args.xscfile.split('/')[-1]}"
        strfile = f"{input_dir}/{args.strfile.split('/')[-1]}"

        # Get parameters from loaded args
        nm_type = args.type.lower()
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

    elif args.option == 'clean':
        print(f"{console.PGM_NAM}{console.TLE}Clean previous pyAdMD setup files{console.STD}\n")

        # Removing previous replicas folders
        files = os.listdir(cwd)
        for item in files:
            if item.endswith(".json"):
                 os.remove(os.path.join(cwd, item))
            if item.startswith("rep"):
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
