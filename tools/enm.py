import numpy as np
import openmm as mm
from openmm import app, unit, Platform
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import diags, identity
import numba
from numba import njit, prange, float64, int32
import matplotlib.pyplot as plt
from matplotlib import colors
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cupy as cp
import argparse
import sys
import os
import time
import csv
import logging

class AnsiColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m',            # Grey
        'INFO': '\033[0m',              # Reset
        'WARNING': '\033[33m',          # Yellow
        'ERROR': '\033[31m',            # Red
        'CRITICAL': '\033[91m\033[1m',  # Bright Red + Bold
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET_COLOR)
        return f"{color}{log_message}{self.RESET_COLOR}"

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler with color formatting
console_handler = logging.StreamHandler()
console_formatter = AnsiColorFormatter("..:ENM> {levelname}: {message}", style="{")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler for enm.out
file_handler = logging.FileHandler('enm.out')
file_formatter = logging.Formatter("{asctime} ..:ENM> {levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

def parse_arguments():
    """Parse command-line arguments for the ENM analysis."""
    parser = argparse.ArgumentParser(description='Elastic Network Model Normal Mode Analysis')

    # Required arguments
    parser.add_argument('-i', '--input', required=True, help='Input PDB file')

    # Optional arguments with defaults
    parser.add_argument('-o', '--output', default='output', help='Output folder name')

    parser.add_argument('-t', '--type', choices=['ca', 'heavy'], default='ca',
                       help='Model type: ca (Cα-only) or heavy (heavy atoms). Default: ca')
    parser.add_argument('-c', '--cutoff', type=float, default=None,
                       help='Cutoff distance for interactions in Å. Default: 15.0 for CA, 12.0 for heavy atoms')
    parser.add_argument('-k', '--spring_constant', type=float, default=1.0,
                       help='Spring constant for ENM bonds in kcal/mol/Å². Default: 1.0')
    parser.add_argument('-m', '--max_modes', type=int, default=None,
                       help='Number of non-rigid modes to compute. Default: all modes')
    parser.add_argument('-n', '--output_modes', type=int, default=10,
                       help='Number of modes to save and analyze. Default: 10 modes')

    # Boolean flags to enable/disable features
    parser.add_argument('--no_nm_vec', action='store_false',
                       help='Disable writing mode vectors')
    parser.add_argument('--no_nm_trj', action='store_false',
                       help='Disable writing mode trajectories')
    parser.add_argument('--no_collectivity', action='store_false',
                       help='Disable collectivity calculation')
    parser.add_argument('--no_contributions', action='store_false',
                       help='Disable mode contributions plot')
    parser.add_argument('--no_rmsf', action='store_false',
                       help='Disable RMSF plot')
    parser.add_argument('--no_dccm', action='store_false',
                       help='Disable DCCM plot')
    parser.add_argument('--no_gpu', action='store_false',
                       help='Disable GPU acceleration')

    return parser.parse_args()

def create_system(pdb_file, model_type='ca', cutoff=None, spring_constant=1.0, output_prefix="input"):
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
        If an invalid model type is specified or no relevant atoms are found
    """
    # Set default cutoffs if not provided
    if cutoff is None:
        cutoff = 15.0 if model_type == 'ca' else 12.0

    if model_type == 'ca':
        return _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
    elif model_type == 'heavy':
        return _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)
    else:
        raise ValueError("Invalid model type. Choose 'ca' or 'heavy'")

def _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix):
    """Create a Cα-only Elastic Network Model system."""
    logger.info("Creating Cα-only system using Elastic Network Model...")
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
        raise ValueError("No Cα atoms found in the structure")

    n_atoms = len(ca_info)
    logger.info(f"Selected {n_atoms} Cα atoms")

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
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.9Å, k={spring_constant} kcal/mol/Å²")
    system.addForce(mm.CMMotionRemover())

    # Save the Cα structure
    ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
    with open(ca_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"C-alpha structure saved to {ca_pdb_file}\n")

    # Convert HETATM to ATOM
    convert_hetatm_to_atom(ca_pdb_file)

    return system, new_topology, positions_quantity

def _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix):
    """Create a heavy-atom Elastic Network Model system."""
    logger.info("Creating heavy-atom system using Elastic Network Model...")
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
        raise ValueError("No heavy atoms found in the structure")

    n_atoms = len(heavy_atoms)
    logger.info(f"Selected {n_atoms} heavy atoms")

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
    logger.info(f"Added {len(bonds)} ENM bonds with cutoff={cutoff}Å, min_distance=2.0Å, k={spring_constant} kcal/mol/Å²")
    system.addForce(mm.CMMotionRemover())

    # Save heavy atom structure
    heavy_pdb_file = f"{output_prefix}_aa_structure.pdb"
    with open(heavy_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    logger.info(f"Heavy-atom structure saved to {heavy_pdb_file}\n")

    return system, new_topology, positions_quantity

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

def hessian_enm(system, positions):
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
        raise ValueError("No ENM force found in system")

    k_val = enm_force.getGlobalParameterDefaultValue(0)
    num_bonds = enm_force.getNumBonds()
    pos_array = np.array([[p.x, p.y, p.z] for p in positions.value_in_unit(unit.nanometer)], dtype=np.float64)

    logger.info(f"Computing Hessian for {n_particles} particles...")
    start_time = time.time()

    # Precompute bonds array with fixed memory layout
    bonds_list = np.empty((num_bonds, 3), dtype=np.float64)
    for bond_idx in range(num_bonds):
        i, j, [r0] = enm_force.getBondParameters(bond_idx)
        bonds_list[bond_idx] = (i, j, r0)

    # Compute Hessian
    hessian = compute_hessian(pos_array, bonds_list, k_val, n_particles)

    # Symmetrize and regularize
    hessian = 0.5 * (hessian + hessian.T)
    hessian.flat[::n_dof+1] += 1e-8  # Add regularization directly to diagonal

    duration = time.time() - start_time
    logger.info(f"ENM Hessian computed in {duration:.2f} seconds\n")

    return hessian

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

def compute_normal_modes(hessian, n_modes=None, use_gpu=False):
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
        logger.info("Diagonalizing mass-weighted Hessian using GPU acceleration...")
        try:
            eigenvalues, eigenvectors = gpu_diagonalization(hessian, n_modes)
        except Exception as e:
            logger.warning(f"GPU diagonalization failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu or not cp.is_available():
        # CPU diagonalization with optimized parameters
        if n_modes is not None:
            logger.info("Diagonalizing mass-weighted Hessian using CPU optimization...")
            n_modes = min(n_modes + 6, hessian.shape[0])
            eigenvalues, eigenvectors = eigh(
                hessian,
                subset_by_index=[0, n_modes-1],
                driver='evr',       # Fastest driver for symmetric matrices
                overwrite_a=True,
                check_finite=False  # Skip finite check for performance
            )
        else:
            logger.info("Diagonalizing mass-weighted Hessian using CPU...")
            eigenvalues, eigenvectors = eigh(
                hessian,
                driver='evr',       # Fastest driver for symmetric matrices
                overwrite_a=True,
                check_finite=False  # Skip finite check for performance
            )

    duration = time.time() - start_time
    logger.info(f"Diagonalization completed in {duration:.2f} seconds\n")

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

def write_nm_vectors(modes, frequencies, system, topology, output_prefix, n_modes=10, start_mode=7):
    """
    Write normal mode eigenvectors to separate XYZ files.

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
    output_prefix : str
        Prefix for output XYZ files
    n_modes : int, optional
        Number of modes to write (default: 10)
    start_mode : int, optional
        First mode to write (default: 7, skipping first 6 rigid-body modes)
    """
    n_particles = system.getNumParticles()

    # Ensure we don't exceed available modes
    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write.")
        return

    # Get element symbols from topology
    elements = []
    for atom in topology.atoms():
        elements.append(atom.element.symbol)

    # Write each mode to a separate XYZ file
    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # Convert to cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}.xyz"

        with open(output_file, 'w') as f:
            # Write header
            f.write(f"{n_particles}\n")
            f.write(f"Normal Mode {mode_number}, Frequency: {freq:.2f} cm⁻¹\n")

            # Extract and reshape the mode vector
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)

            # Write coordinates for each atom
            for i in range(n_particles):
                x, y, z = mode_vector[i]
                f.write(f"{elements[i]:2s} {x:14.10f} {y:14.10f} {z:14.10f}\n")

def write_nm_trajectories(topology, positions, modes, frequencies, output_prefix, system, model_type, n_modes=10, start_mode=7, amplitude=4, num_frames=34):
    """
    Write PDB trajectories showing the motion along multiple normal modes.

    Parameters
    ----------
    topology : openmm.app.Topology
        The topology of the system
    positions : openmm.unit.Quantity
        The equilibrium positions of particles
    modes : ndarray
        The normal mode vectors (3N×M)
    frequencies : ndarray
        The frequencies of the normal modes
    output_prefix : str
        Prefix for output PDB files
    system : openmm.System
        The system containing particle masses
    model_type : str
        Type of model ('ca' or 'heavy')
    n_modes : int, optional
        Number of modes to write (default: 10)
    start_mode : int, optional
        First mode to write (default: 7, skipping first 6 rigid-body modes)
    amplitude : float
        The amplitude of motion to visualize
    num_frames : int, optional
        Number of frames in the trajectory
    """
    n_particles = system.getNumParticles()

    # Ensure we don't exceed available modes
    n_modes = min(n_modes, modes.shape[1] - start_mode)
    if n_modes <= 0:
        logger.warning("No modes available to write trajectories.")
        return

    for mode_idx in range(start_mode, start_mode + n_modes):
        freq = frequencies[mode_idx] * 108.58  # Convert to cm⁻¹
        mode_number = mode_idx + 1
        output_file = f"{output_prefix}_mode_{mode_number}_traj.pdb"

        # Mass-weight the mode vector
        masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])
        masses[masses == 0] = 1.0
        inv_sqrt_m = np.repeat(1 / np.sqrt(masses), 3)
        u = modes[:, mode_idx] * inv_sqrt_m

        rms = np.linalg.norm(u) / np.sqrt(n_particles)
        if rms < 1e-10:
            logger.warning(f"Skipping near-zero mode {mode_number}")
            continue

        # Scale displacement to the desired amplitude
        scaled_disp = u.reshape(n_particles, 3) * (amplitude / rms * 0.1)
        orig_pos = positions.value_in_unit(unit.nanometer)
        orig_pos_np = np.array([[p.x, p.y, p.z] for p in orig_pos])

        # Create a smooth oscillation trajectory
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
        convert_hetatm_to_atom(output_file)

def compute_collectivity(mode_vector, n_atoms):
    """
    Compute the collectivity of a normal mode (Tama & Sanejouand, 2001).

    Parameters
    ----------
    mode_vector : ndarray
        The normal mode vector
    n_atoms : int
        Number of atoms in the system

    Returns
    -------
    collectivity : float
        The collectivity value (0 to 1)
    """
    u = mode_vector.reshape(n_atoms, 3)
    norms = np.linalg.norm(u, axis=1)
    p = norms**2
    p += 1e-12  # Avoid log(0)
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy) / n_atoms

def write_collectivity(frequencies, modes, system, output_file, n_modes=20):
    """
    Write mode collectivities to a CSV file.

    Parameters
    ----------
    frequencies : ndarray
        The frequencies of the normal modes
    modes : ndarray
        The normal mode vectors
    system : openmm.System
        The system containing particle masses
    output_file : str
        Path to the output CSV file
    n_modes : int, optional
        Number of modes to include in the output
    """
    n_particles = system.getNumParticles()
    masses = [system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)]
    inv_sqrt_m = 1 / np.sqrt(masses)
    m_vector = np.repeat(inv_sqrt_m, 3)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Mode', 'Frequency (cm⁻¹)', 'Collectivity'])

        for i in range(6, min(6+n_modes, modes.shape[1])):
            # Mass-weight the mode
            mw_mode = modes[:, i] * m_vector
            mw_mode /= np.linalg.norm(mw_mode)

            # Convert to wavenumber
            freq_cm = frequencies[i] * 108.58  # Conversion factor

            # Calculate collectivity
            kappa = compute_collectivity(mw_mode, n_particles)

            writer.writerow([i+1, f"{freq_cm:.2f}", f"{kappa:.4f}"])

    logger.info(f"Saved collectivity data to {output_file}\n")

def plot_mode_contributions(eigenvalues, output_file=None, n_modes=10):
    """
    Plot the cumulative proportion of variance explained by the first N non-rigid modes.

    Parameters
    ----------
    eigenvalues : ndarray
        The eigenvalues of the Hessian matrix
    n_modes : int, optional
        Number of modes to include in the plot
    output_file : str, optional
        Path to save the plot. If None, displays the plot.
    """
    # Exclude rigid-body modes (first 6 near-zero eigenvalues)
    non_rigid_evals = eigenvalues[6:]

    # Calculate the variance explained by each mode
    # In NMA, the variance is proportional to 1/λ (fluctuation magnitude)
    epsilon = 1e-10
    variances = 1 / (np.abs(non_rigid_evals) + epsilon)

    # Calculate the total variance
    total_variance = np.sum(variances)

    # Calculate proportion of variance for each mode
    proportion_variance = variances / total_variance

    # Calculate cumulative proportion of variance
    cumulative_variance = np.cumsum(proportion_variance[:n_modes]) * 100

    # Create plot
    plt.figure(figsize=(12, 6))

    # Create subplot for proportion of variance
    plt.subplot(1, 2, 1)
    modes_indices = np.arange(1, n_modes+1)
    plt.bar(modes_indices, proportion_variance[:n_modes] * 100, alpha=0.7, color='skyblue')
    plt.title('Proportion of Variance by Mode')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Proportion of Variance (%)')
    plt.xticks(modes_indices)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Create subplot for cumulative proportion of variance
    plt.subplot(1, 2, 2)
    plt.plot(modes_indices, cumulative_variance, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    plt.title('Cumulative Proportion of Variance')
    plt.xlabel('Mode Index (excluding rigid-body modes)')
    plt.ylabel('Cumulative Variance (%)')
    plt.xticks(modes_indices)
    plt.ylim(0, 100)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add percentage labels to points
    for i, val in enumerate(cumulative_variance):
        plt.annotate(f'{val:.1f}%', (modes_indices[i], val),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontsize=9)

    plt.tight_layout()

    # Add explanatory text
    plt.figtext(0.5, 0.01,
                f"First {n_modes} non-rigid modes account for {cumulative_variance[-1]:.1f}% of total variance",
                ha="center", fontsize=10, style='italic')

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved mode contribution plot to {output_file}\n")
    else:
        plt.show()

def plot_atomic_fluctuations(system, eigenvalues, modes, topology, output_file=None, temperature=300, n_modes=None, start_mode=6):
    """
    Plot residue fluctuations (RMSF) calculated from normal modes.

    Parameters
    ----------
    system : openmm.System
        The system containing particle masses
    eigenvalues : ndarray
        The eigenvalues of the Hessian matrix
    modes : ndarray
        The normal mode vectors
    topology : openmm.app.Topology
        The topology of the system
    output_file : str, optional
        Path to save the plot. If None, displays the plot.
    temperature : float, optional
        Temperature in Kelvin for fluctuation calculation
    n_modes : int, optional
        Number of modes to include in calculation. If None, uses all non-rigid modes.
    start_mode : int, optional
        First mode to include (default: 6, skipping first 6 rigid-body modes)
    """
    n_particles = system.getNumParticles()

    # If n_modes not specified, use all non-rigid modes
    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    # Calculate RMSF for each atom
    rmsf_atom = np.zeros(n_particles)

    # Boltzmann constant in kcal/(mol·K)
    k_B = 0.0019872041

    # Get masses
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])

    # Calculate contribution from each mode
    for mode_idx in range(start_mode, start_mode + n_modes):
        # Skip near-zero eigenvalues to avoid division by zero
        if abs(eigenvalues[mode_idx]) < 1e-10:
            continue

        # Get the mode vector and reshape to (n_particles, 3)
        mode_vector = modes[:, mode_idx].reshape(n_particles, 3)

        # Calculate the mean square fluctuation for this mode
        # MSF = (k_B * T / ω²) * |u_i|² / m_i
        # where u_i is the displacement vector for atom i in this mode
        omega_sq = eigenvalues[mode_idx]
        msf_contribution = (k_B * temperature / omega_sq) * np.sum(mode_vector**2, axis=1) / masses

        # Add to total RMSF
        rmsf_atom += msf_contribution

    # Take square root to get RMSF in nm
    rmsf_atom = np.sqrt(rmsf_atom)

    # Convert to Angstrom (1 nm = 10 Å)
    rmsf_atom *= 10

    # Group atoms by residue
    residue_rmsf = {}
    residue_indices = {}
    for atom in topology.atoms():
        residue = atom.residue
        residue_id = residue.id
        if residue_id not in residue_rmsf:
            residue_rmsf[residue_id] = []
            residue_indices[residue_id] = len(residue_rmsf) - 1
        residue_rmsf[residue_id].append(rmsf_atom[atom.index])

    # Calculate average RMSF per residue
    residue_ids = sorted(residue_rmsf.keys())
    residue_means = [np.mean(residue_rmsf[resid]) for resid in residue_ids]
    residue_nums = list(range(len(residue_ids)))

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot RMSF
    plt.plot(residue_nums, residue_means, 'b-', linewidth=1, alpha=0.7)
    plt.fill_between(residue_nums, 0, residue_means, alpha=0.3)

    plt.xlabel('Residue Index')
    plt.ylabel('RMS Fluctuation (Å)')
    plt.title(f'Residue Fluctuations from Normal Modes\n(T={temperature}K, {n_modes} modes)')
    plt.grid(True, alpha=0.3)

    # Add statistics to the plot
    avg_rmsf = np.mean(residue_means)
    max_rmsf = np.max(residue_means)
    plt.axhline(y=avg_rmsf, color='r', linestyle='--', alpha=0.7,
                label=f'Average: {avg_rmsf:.2f} Å')
    plt.legend()

    # Set x-axis ticks to show residue indices
    if len(residue_nums) > 0:
        tick_step = max(1, len(residue_nums) // 10)
        x_ticks = np.arange(0, len(residue_nums), tick_step)
        plt.xticks(x_ticks, x_ticks)

    # Adjust layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue RMSF plot saved to {output_file}\n")
    else:
        plt.show()

def plot_residue_cross_correlation(system, eigenvalues, modes, topology, output_file=None, temperature=300, n_modes=None, start_mode=6, use_gpu=True, use_multithreading=True):
    """
    Compute and plot residue dynamical cross-correlation matrix calculated from normal modes.

    Parameters
    ----------
    system : openmm.System
        The system containing particle masses
    eigenvalues : ndarray
        The eigenvalues of the Hessian matrix
    modes : ndarray
        The normal mode vectors
    topology : openmm.app.Topology
        The topology of the system
    output_file : str, optional
        Path to save the plot. If None, displays the plot.
    temperature : float, optional
        Temperature in Kelvin for correlation calculation
    n_modes : int, optional
        Number of modes to include in calculation
    start_mode : int, optional
        First mode to include (default: 6, skipping first 6 rigid-body modes)
    use_gpu : bool, optional
        Whether to use GPU acceleration for calculation
    use_multithreading : bool, optional
        Whether to use multithreading for CPU calculation
    """
    start_time = time.time()

    n_particles = system.getNumParticles()

    # If n_modes not specified, use all non-rigid modes
    if n_modes is None:
        n_modes = modes.shape[1] - start_mode

    # Get residue information
    residues = list(topology.residues())
    n_residues = len(residues)

    # Map atoms to residues
    atom_to_residue = np.zeros(n_particles, dtype=int)
    for atom in topology.atoms():
        residue = atom.residue
        atom_to_residue[atom.index] = residues.index(residue)

    # Create residue assignment matrix (n_particles x n_residues)
    R = np.zeros((n_particles, n_residues))
    for i in range(n_residues):
        R[np.where(atom_to_residue == i)[0], i] = 1.0

    # Boltzmann constant in kcal/(mol·K)
    k_B = 0.0019872041

    # Get masses
    masses = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(n_particles)])

    # Precompute mass factors
    mass_factor = 1 / np.sqrt(masses)

    # Initialize correlation matrix
    correlation_matrix = np.zeros((n_residues, n_residues))

    if use_gpu and cp.is_available():
        logger.info("Using GPU acceleration for DCCM calculation...")
        try:
            # Transfer data to GPU
            masses_gpu = cp.array(masses)
            eigenvalues_gpu = cp.array(eigenvalues[start_mode:start_mode+n_modes])
            modes_gpu = cp.array(modes[:, start_mode:start_mode+n_modes])
            R_gpu = cp.array(R)

            # Precompute mass factors on GPU
            mass_factor_gpu = 1 / cp.sqrt(masses_gpu)

            # Initialize GPU correlation matrix
            correlation_matrix_gpu = cp.zeros((n_residues, n_residues))

            # Calculate contribution from each mode on GPU
            for mode_idx in range(n_modes):
                # Skip near-zero eigenvalues to avoid division by zero
                if cp.abs(eigenvalues_gpu[mode_idx]) < 1e-10:
                    continue

                # Get the mode vector and reshape to (n_particles, 3)
                mode_vector = modes_gpu[:, mode_idx].reshape(n_particles, 3)

                # Calculate mass-weighted vectors
                weighted_vectors = mode_vector * mass_factor_gpu[:, cp.newaxis]

                # Calculate the correlation matrix for this mode
                # C = (k_B * T / ω²) * (U @ U.T) where U is mass-weighted
                factor = k_B * temperature / eigenvalues_gpu[mode_idx]
                atom_corr_matrix = factor * cp.dot(weighted_vectors, weighted_vectors.T)

                # Aggregate to residue level using matrix multiplication
                # R.T @ atom_corr_matrix @ R
                residue_corr_matrix = cp.dot(cp.dot(R_gpu.T, atom_corr_matrix), R_gpu)

                # Add to total correlation matrix
                correlation_matrix_gpu += residue_corr_matrix

            # Transfer result back to CPU
            correlation_matrix = cp.asnumpy(correlation_matrix_gpu)

        except Exception as e:
            logger.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu or not cp.is_available():
        logger.info("Using CPU for DCCM calculation...")

        # Use the same matrix-based approach for CPU
        for mode_idx in range(start_mode, start_mode + n_modes):
            # Skip near-zero eigenvalues to avoid division by zero
            if abs(eigenvalues[mode_idx]) < 1e-10:
                continue

            # Get the mode vector and reshape to (n_particles, 3)
            mode_vector = modes[:, mode_idx].reshape(n_particles, 3)

            # Calculate mass-weighted vectors
            weighted_vectors = mode_vector * mass_factor[:, np.newaxis]

            # Calculate the correlation matrix for this mode
            # C = (k_B * T / ω²) * (U @ U.T) where U is mass-weighted
            factor = k_B * temperature / eigenvalues[mode_idx]
            atom_corr_matrix = factor * np.dot(weighted_vectors, weighted_vectors.T)

            # Aggregate to residue level using matrix multiplication
            # R.T @ atom_corr_matrix @ R
            residue_corr_matrix = np.dot(np.dot(R.T, atom_corr_matrix), R)

            # Add to total correlation matrix
            correlation_matrix += residue_corr_matrix

    # Normalize to get correlation coefficients between -1 and 1
    diag = np.diag(correlation_matrix)
    norm_matrix = np.sqrt(np.outer(diag, diag))
    correlation_matrix = correlation_matrix / (norm_matrix + 1e-10)  # Avoid division by zero

    duration = time.time() - start_time
    logger.info(f"DCCM calculation completed in {duration:.2f} seconds")

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create a diverging colormap
    cmap = plt.cm.RdBu_r
    norm = colors.Normalize(vmin=-1, vmax=1)

    # Plot the correlation matrix with inverted y-axis
    im = plt.imshow(correlation_matrix, cmap=cmap, norm=norm, aspect='auto', origin='lower')

    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=12)

    # Set labels
    plt.xlabel('Residue Index', fontsize=12)
    plt.ylabel('Residue Index', fontsize=12)
    plt.title(f'Residue Cross-Correlation Matrix\n(T={temperature}K, {n_modes} modes)', fontsize=14)

    # Set ticks to show approximately 10 ticks per axis
    tick_step = max(1, n_residues // 10)
    residue_ticks = np.arange(0, n_residues, tick_step)
    residue_labels = [f'{residues[i].id}' for i in residue_ticks]

    plt.xticks(residue_ticks, residue_labels, rotation=90)
    plt.yticks(residue_ticks, residue_labels)

    # Adjust layout
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Residue dynamical cross-correlation plot saved to {output_file}\n")

        # Also save the correlation matrix as a numpy file
        np.save(output_file.replace('.png', '.npy'), correlation_matrix)
    else:
        plt.show()

def main():
    """
    Main function to perform Normal Mode Analysis.

    Configuration is set via the CONFIG dictionary, which includes:
    - PDB_FILE: Input PDB file
    - MODEL_TYPE: 'ca' for Cα-only or 'heavy' for heavy-atom ENM
    - CUTOFF: Cutoff distance for interactions
    - SPRING_CONSTANT: Spring constant for ENM bonds
    - MAX_MODES: Number of non-rigid modes to compute
    - OUTPUT_FOLDER: Folder name where to write the output files
    - OUTPUT_MODES: Number of modes to save
    - WRITE_NM_VEC: Whether to write mode vectors to text files
    - WRITE_NM_TRJ: Whether to write mode trajectory files
    - COLLECTIVITY: Whether to compute modes collectivity
    - PLOT_CONTRIBUTIONS: Whether to plot modes cumulative contribution to internal dynamics
    - PLOT_RMSF: Whether to plot modes RMSF
    - PLOT_DCCM: Whether to build and plot residue dynamical cross correlation matrix
    - USE_GPU: Whether to use GPU acceleration
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Map arguments to CONFIG dictionary
    CONFIG = {
        "PDB_FILE": args.input,
        "MODEL_TYPE": args.type,
        "CUTOFF": args.cutoff,
        "SPRING_CONSTANT": args.spring_constant,
        "MAX_MODES": args.max_modes,
        "OUTPUT_FOLDER": args.output,
        "OUTPUT_MODES": args.output_modes,
        "WRITE_NM_VEC": args.no_nm_vec,
        "WRITE_NM_TRJ": args.no_nm_trj,
        "COLLECTIVITY": args.no_collectivity,
        "PLOT_CONTRIBUTIONS": args.no_contributions,
        "PLOT_RMSF": args.no_rmsf,
        "PLOT_DCCM": args.no_dccm,
        "USE_GPU": args.no_gpu
    }

    # Create output folder
    output_folder = CONFIG["OUTPUT_FOLDER"]
    os.makedirs(output_folder, exist_ok=True)

    # Get input filename without extension
    input_file = CONFIG["PDB_FILE"]
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_prefix = os.path.join(output_folder, base_name)

    logger.info("Starting Normal Mode Analysis...\n")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Mode: {'Cα-only ENM' if CONFIG['MODEL_TYPE'] == 'ca' else 'Heavy-atom ENM'}")
    logger.info(f"Cutoff: {CONFIG['CUTOFF'] or ('15.0Å' if CONFIG['MODEL_TYPE'] == 'ca' else '12.0Å')}")
    logger.info(f"Spring constant: {CONFIG['SPRING_CONSTANT']} kcal/mol/Å²")
    logger.info(f"Number of modes to output: {CONFIG['OUTPUT_MODES']}\n")

    try:
        # Prefix to output files
        if CONFIG["MODEL_TYPE"] == 'ca':
            prefix = "ca"
        else:
            prefix = "aa"

        # Create system
        system, topology, positions = create_system(
            CONFIG["PDB_FILE"],
            model_type=CONFIG["MODEL_TYPE"],
            cutoff=CONFIG["CUTOFF"],
            output_prefix = output_prefix,
            spring_constant=CONFIG["SPRING_CONSTANT"],
        )

        # Compute Hessian
        hessian = hessian_enm(
            system,
            positions
        )

        # Mass-weight Hessian
        mw_hessian = mass_weight_hessian(
            hessian,
            system
        )

        # Compute Normal Modes
        frequencies, modes, eigenvalues = compute_normal_modes(
            mw_hessian,
            n_modes=CONFIG["MAX_MODES"],
            use_gpu=CONFIG["USE_GPU"]
        )

        np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
        np.save(f"{output_prefix}_{prefix}_modes.npy", modes)
        logger.info(f"Results saved to {output_prefix}_{prefix}_*.npy files")

        # Write collectivity data
        if CONFIG["COLLECTIVITY"]:
            collectivity_file = f"{output_prefix}_{prefix}_collectivity.csv"
            write_collectivity(
                frequencies, modes, system,
                collectivity_file,
                n_modes=20  # Use first 20 non-rigid modes
            )

        # Plot internal dynamics contributions
        if CONFIG["PLOT_CONTRIBUTIONS"]:
            output_file = f"{output_prefix}_{prefix}_contributions.png"
            plot_mode_contributions(
                eigenvalues,
                output_file,
                n_modes=20  # Use first 20 non-rigid modes
            )

        # Plot RMSF
        if CONFIG["PLOT_RMSF"]:
            output_file=f"{output_prefix}_{prefix}_rmsf.png"
            rmsf = plot_atomic_fluctuations(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,  # Room temperature
                n_modes=50,       # Use first 50 non-rigid modes
            )

        # Plot Residue Cross Correlation
        if CONFIG["PLOT_DCCM"]:
            output_file=f"{output_prefix}_{prefix}_dccm.png"
            dccm = plot_residue_cross_correlation(
                system, eigenvalues, modes, topology,
                output_file,
                temperature=300,            # Room temperature
                n_modes=50,                 # Use first 50 non-rigid modes
                use_gpu=CONFIG["USE_GPU"],  # Enable GPU usage
                use_multithreading=True     # Enable multithreading for CPU
            )

        # Write mode vectors
        if CONFIG["WRITE_NM_VEC"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Writing vectors for {num_modes} modes...\n")
            mode_vectors_prefix = f"{output_prefix}_{prefix}"
            write_nm_vectors(
                modes, frequencies, system, topology,
                mode_vectors_prefix,
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6  # Start from mode 7 (index 6)
            )

        # Write mode trajectories
        if CONFIG["WRITE_NM_TRJ"]:
            num_modes = min(CONFIG["OUTPUT_MODES"], len(frequencies)-6)
            logger.info(f"Generating trajectories for {num_modes} modes...\n")
            write_nm_trajectories(
                topology, positions, modes, frequencies,
                f"{output_prefix}_{prefix}", system, CONFIG["MODEL_TYPE"],
                n_modes=CONFIG["OUTPUT_MODES"],
                start_mode=6  # Start from mode 7 (index 6)
            )

    except Exception as e:
        logger.error(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
