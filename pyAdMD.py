"""
The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of multiple-replica short MD simulations
in which motions described by a given subset of low-frequency NMs are kinetically excited. This is achieved by injecting
additional atomic velocities along several randomly determined linear combinations of NM vectors, thus allowing an
efficient coupling between slow and fast motions.

This new approach, aMDeNM, automatically controls the energy injection and take the natural constraints imposed by
the structure and the environment into account during protein conformational sampling, which prevent structural
distortions all along the simulation.Due to the stochasticity of thermal motions, NM eigenvectors move away from the
original directions when used to displace the protein, since the structure evolves into other potential energy wells.
Therefore, the displacement along the modes is valid for small distances, but the displacement along greater distances
may deform the structure of the protein if no care is taken. The advantage of this methodology is to adaptively change
the direction used to displace the system, taking into account the structural and energetic constraints imposed by the
system itself and the medium, which allows the system to explore new pathways.
"""

import warnings # ignore deprecated library warnings from MDAnalysis
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import os
import zipfile
import subprocess
import shutil
import sys
import time
import csv
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
import MDAnalysis as mda
from MDAnalysis.analysis import align
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

def parse_arguments():
    """
    Parse command-line arguments for the pyAdMD analysis.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description=message)
    subparsers = parser.add_subparsers()

    # Cleaning
    opt_clean = subparsers.add_parser('clean', help="Erase previous pyAdMD configuration file.")
    opt_clean.set_defaults(clean=True)

    # Run
    opt_run = subparsers.add_parser('run', help="Setup and run pyAdMD simulations.")
    # REQUIRED VIRIABLES
    opt_run.add_argument('-type', '--type',
                        action="store", type=str.upper, default="CA", required=True,
                        choices=["CA", "HEAVY", "CHARMM"],
                        help="REQUIRED: Compute ENM or use pre-computed CHARMM normal mode file (default: CA).")
    opt_run.add_argument('-psf', '--psffile',
                        action="store", type=str, required=True,
                        help="REQUIRED: PSF topology file.")
    opt_run.add_argument('-pdb', '--pdbfile',
                        action="store", type=str, required=True,
                        help="REQUIRED: PDB structure file.")
    opt_run.add_argument('-coor','--coorfile',
                        action="store", type=str, required=True,
                        help="REQUIRED: NAMD coordinates file.")
    opt_run.add_argument('-vel', '--velfile',
                        action="store", type=str, required=True,
                        help="REQUIRED: NAMD velocities file.")
    opt_run.add_argument('-xsc', '--xscfile',
                        action="store", type=str, required=True,
                        help="REQUIRED: NAMD PBC file.")
    # TODO: add additional treatment to the conf.namd file regarding the .str information
    #  including formating it, if necessary. Make it optional.
    opt_run.add_argument('-str', '--strfile',
                        action="store", type=str, required=True,
                        help="REQUIRED: NAMD additional box info file.")
    # OPTIONAL VARIABLES
    opt_run.add_argument('-mod', '--modefile',
                     action="store", type=str,
                     help="REQUIRED: CHARMM normal mode file.")
    opt_run.add_argument('-nm', '--modes',
                        action="store", type=str, default="7,8,9",
                        help="Normal modes to excite separated by commas (default: 7,8,9).")
    opt_run.add_argument('-ek', '--energy',
                        action="store", type=float, default=0.125,
                        help="Excitation energy (default: 0.125 kcal/mol).")
    opt_run.add_argument('-t', '--time',
                        action="store", type=int, default=250,
                        help="Total simulation time (default: 250ps).")
    opt_run.add_argument('-sel', '--selection',
                        action="store", type=str, default="protein",
                        help="Atom selection to apply the energy injection (default: protein).")
    opt_run.add_argument('-rep', '--replicas',
                        action="store", type=int, default=10,
                        help="Number of aMDeNM replicas to run (default: 10).")
    # BOOLEAN VARIABLES
    opt_run.add_argument('--no_correc', action='store_true',
                       help='Compute standard MDeNM calculations.')
    opt_run.add_argument('--fixed', action='store_true',
                       help='Disable excitation vector correction and keep constant excitation energy injections.')

    opt_run.set_defaults(run=True)

    return parser.parse_args()

def unzip(filepath, path_dir):
    """
    Extract files from a .zip compressed file.

    Parameters
    ----------
    filepath : str
        Path to .zip file
    path_dir : str
        Path to destination folder
    """

    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(path_dir)

def write_charmm_nm(nms_to_write):
    """
    Write CHARMM normal mode vectors in NAMD readable format.

    Parameters
    ----------
    nms_to_write : list
        List of normal modes to write
    """

    # Extract CHARMM topology and parameters files
    unzip(f"{input_dir}/charmm_toppar.zip", input_dir)

    nms = [str(t + '\n') for t in nms_to_write.split(',')]
    with open(f"{input_dir}/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)
    os.chdir(f"{cwd}/tools")
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]}"
           f" modfile={modefile.split('/')[-1]} -o ../wrt-nm.out")
    returned_value = subprocess.call(cmd, shell=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{pgmerr}An error occurred while writing the normal mode vectors.\n"
              f"{pgmerr}Inspect the file {err}wrt-nm.out{std} for detailed information.\n")
        sys.exit()

def compute_enm(coorfile, nm_type, nm_parsed):
    """
    Setup and run ENM analysis.

    Parameters
    ----------
    pdb_file : str
        Input file name
    nm_type : str
        Type of normal mode calculation ('CA' or 'HEAVY')
    nm_parsed : list
        List containing mode numbers
    """
    # Create output folder
    base_name = os.path.splitext(os.path.basename(coorfile))[0]
    output_folder = f"{input_dir}/{base_name}_enm"
    os.makedirs(output_folder, exist_ok=True)
    output_prefix = os.path.join(output_folder, base_name)

    # Prefix to output files
    if nm_type.lower() == 'ca':
        prefix = "ca"
    else:
        prefix = "heavy"

    # Create the PDB input to compute the ENM
    pdb_input = mda.Universe(f"{input_dir}/{psffile}", coorfile, format="NAMDBIN")
    pdb_input = pdb_input.atoms.select_atoms("protein")
    pdb_file = f"{output_prefix}.pdb"
    pdb_input.write(pdb_file, file_format="PDB")

    # Create system
    system, topology, positions = create_system(
        pdb_file,
        model_type=nm_type,
        cutoff=None,
        output_prefix = output_prefix,
        spring_constant=1,
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
    frequencies, enm, eigenvalues = compute_normal_modes(
        mw_hessian,
        n_modes=None,
        use_gpu=True
    )

    # Write mode vectors
    print(f"{pgmnam}Writing vectors for {ext}modes {modes}{std}...\n")
    n_modes = int(max(nm_parsed)) - 6
    mode_vectors_prefix = f"{output_prefix}_{prefix}"
    for mode_idx in nm_parsed:
        write_nm_vectors(
            enm, frequencies, system, topology,
            mode_idx,
            mode_vectors_prefix,
            pdb_file,
            model_type=nm_type
        )

    np.save(f"{output_prefix}_{prefix}_frequencies.npy", frequencies)
    np.save(f"{output_prefix}_{prefix}_modes.npy", enm)
    print(f"{pgmnam}Results saved to {ext}{output_prefix}_{prefix}_*.npy{std} files")

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
    """
    # Set default cutoffs if not provided
    if cutoff is None:
        cutoff = 15.0 if model_type == 'ca' else 12.0

    if model_type.lower() == 'ca':
        return _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix)
    elif model_type.lower() == 'heavy':
        return _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix)

def _create_ca_system(pdb_file, cutoff, spring_constant, output_prefix):
    """Create a Cα-only Elastic Network Model system."""
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
        print(f"{pgmerr}No Cα atoms found in the structure.")
        sys.exit()

    n_atoms = len(ca_info)
    print(f"{pgmnam}Selected {ext}{n_atoms}{std} Cα atoms.")

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
    print(f"{pgmnam}Added {ext}{len(bonds)}{std} ENM bonds with cutoff={ext}{cutoff}Å{std}, min_distance={ext}2.9Å{std}, k={ext}{spring_constant} kcal/mol/Å²{std}.")
    system.addForce(mm.CMMotionRemover())

    # Save the Cα structure
    ca_pdb_file = f"{output_prefix}_ca_structure.pdb"
    with open(ca_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    print(f"{pgmnam}C-alpha structure saved to {ext}{ca_pdb_file}{std}.")

    # Convert HETATM to ATOM
    convert_hetatm_to_atom(ca_pdb_file)

    return system, new_topology, positions_quantity

def _create_heavy_system(pdb_file, cutoff, spring_constant, output_prefix):
    """Create a heavy-atom Elastic Network Model system."""
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
        print(f"{pgmerr}No heavy atoms found in the structure.")
        sys.exit()

    n_atoms = len(heavy_atoms)
    print(f"{pgmnam}Selected {ext}{n_atoms}{std} heavy atoms.")

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
    print(f"{pgmnam}Added {ext}{len(bonds)}{std} ENM bonds with cutoff={ext}{cutoff}Å{std}, min_distance={ext}2.0Å{std}, k={ext}{spring_constant} kcal/mol/Å²{std}.")
    system.addForce(mm.CMMotionRemover())

    # Save heavy atom structure
    heavy_pdb_file = f"{output_prefix}_heavy_structure.pdb"
    with open(heavy_pdb_file, 'w') as f:
        app.PDBFile.writeFile(new_topology, positions_quantity, f)
    print(f"{pgmnam}Heavy-atom structure saved to {ext}{heavy_pdb_file}{std}.")

    return system, new_topology, positions_quantity

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
        print(f"{pgmerr}No ENM force found in system.")

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
    hessian = compute_hessian(pos_array, bonds_list, k_val, n_particles)

    # Symmetrize and regularize
    hessian = 0.5 * (hessian + hessian.T)
    hessian.flat[::n_dof+1] += 1e-8  # Add regularization directly to diagonal

    duration = time.time() - start_time
    print(f"{pgmnam}Computed ENM Hessian computed for {ext}{n_particles}{std} particles in {ext}{duration:.2f}{std} seconds")

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
        print(f"{pgmnam}Diagonalizing mass-weighted Hessian using GPU acceleration...")
        try:
            eigenvalues, eigenvectors = gpu_diagonalization(hessian, n_modes)
        except Exception as e:
            print(f"{pgmwrn}GPU diagonalization failed: {e}. Falling back to CPU.")
            use_gpu = False

    if not use_gpu or not cp.is_available():
        # CPU diagonalization with optimized parameters
        if n_modes is not None:
            print(f"{pgmnam}Diagonalizing mass-weighted Hessian using CPU optimization...")
            n_modes = min(n_modes + 6, hessian.shape[0])
            eigenvalues, eigenvectors = eigh(
                hessian,
                subset_by_index=[0, n_modes-1],
                driver='evr',       # Fastest driver for symmetric matrices
                overwrite_a=True,
                check_finite=False  # Skip finite check for performance
            )
        else:
            print(f"{pgmnam}Diagonalizing mass-weighted Hessian using CPU...")
            eigenvalues, eigenvectors = eigh(
                hessian,
                driver='evr',       # Fastest driver for symmetric matrices
                overwrite_a=True,
                check_finite=False  # Skip finite check for performance
            )

    duration = time.time() - start_time
    print(f"{pgmnam}Diagonalization completed in {ext}{duration:.2f}{std} seconds")

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

def write_nm_vectors(modes, frequencies, system, topology, nm, output_prefix, pdb_file, model_type='ca'):
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

def generate_factors(P, N, max_iterations=1000000, tolerance=1e-6):
    """
    Generate factors for linear combinations of N orthonormal vectors
    that produce P points uniformly distributed on a N-dimensional hypersphere surface.

    Parameters
    ----------
    P : int
        Number of points to generate
    N : int
        Dimensionality of the space
    max_iterations : int, optional
        Maximum number of iterations for the repulsion algorithm (default: 1000000)
    tolerance : float, optional
        Convergence tolerance (default: 1e-6)

    Returns
    -------
    numpy.ndarray
        P×N matrix of factors for linear combinations
    """
    # Store the modes indexes to write the csv file
    nm_parsed = N
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
        for iteration in range(max_iterations):
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
            if abs(current_max_force - prev_max_force) < tolerance * 0.1:
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
            if current_max_force < tolerance:
                print(f"{pgmnam}Converged to tolerance after {iteration+1} iterations.")
                break

            if stagnation_count >= stagnation_threshold:
                print(f"{pgmwrn}Breaking due to stagnation after {iteration+1} iterations.")
                break

    # Write a csv file containing the combination factors
    # Create the header
    header = ['Combination'] + [f'Mode {mode}' for mode in nm_parsed]
    # Create the data rows
    rows = []
    for i in range(P):
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
    print(f"{pgmnam}Combination factors written at {ext}{output_folder}/factors.csv{std}.")

    return factors

def wrt_vec(xyz, output_file):
    """
    Write a set of coordinates in a new file.

    Parameters
    ----------
    xyz : array
        Vector containing the xyz coordinates
    output_file : str
        Output file name
    """

    # Copy the xyz coordinates into the dataframe
    sys_zeros.positions = np.zeros((N, 3))
    vector = np.append(xyz, sys_zeros.positions, axis=0)
    sys_zeros.positions = vector[:N]

    # Write the output file
    sys_zeros.write(f"{output_file}", file_format="NAMDBIN")

def combine_modes(replicas, modes, factors, model_type='ca'):
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
    model_type : str
        Type of normal mode calculation ('CHARMM', 'CA' or 'HEAVY')
    """
    # Set output folder
    output_folder = f"{cwd}/rep-struct-list"

    for rep in range(replicas):
        # Create an empty vector to store the combination
        natom = sys_pdb.atoms.select_atoms("protein").n_atoms
        comb_vec = sys_pdb.atoms.select_atoms("protein").positions
        comb_vec = np.zeros((natom, 3))

        for nm_idx in range(len(modes)):
            if nm_type == 'charmm':
                nm = mda.Universe(f"{input_dir}/mode_nm{modes[nm_idx]}.crd", format="CRD")
            else:
                base_name = os.path.splitext(os.path.basename(coorfile))[0]
                nm = mda.Universe(f"{input_dir}/{base_name}_enm/{base_name}_{nm_type}_mode_{modes[nm_idx]}.xyz", format="XYZ")
            nm = nm.atoms.positions

            # Apply the factors to the modes
            comb = (nm.T * factors[rep,nm_idx]).T

            # Acumulate modes to obtain a new combined vector
            comb_vec = np.add(comb, comb_vec)

        # Normalize and write the combined vector
        comb_vec /= np.linalg.norm(comb_vec)
        wrt_vec(comb_vec, f"{output_folder}/rep{rep+1}_vector.vec")

    print(f"{pgmnam}Combination vectors written at {ext}{output_folder}/rep*_vector.vec{std}.")

def excite(q_vector, user_ek):
    """
    Scale the combined normal modes to be used as additional
    velocities during aMDeNM simulations.

    Parameters
    ----------
    q_vector : matrix
        Combined vector to excite
    user_ek : float
        User defined excitation energy

    Returns
    -------
    matrix
        Excitation vector
    """

    # Excite
    fscale = np.sqrt((2 * user_ek) / sel_mass)
    exc_vec = (q_vector.T * fscale).T

    return exc_vec

def clean(folder):
    """
    Delete previous run files.
    """
    # Removing previous replicas folders
    files = os.listdir(folder)
    for item in files:
        if item.startswith("rep"):
            shutil.rmtree(os.path.join(folder, item), ignore_errors=True)
    # Removing previous ENM calculations
    files = os.listdir(folder)
    for item in files:
        if item.endswith("_enm"):
            shutil.rmtree(os.path.join(folder, item), ignore_errors=True)


# Get working directory path
cwd = os.getcwd()
input_dir = f"{cwd}/inputs"

# Style variables
tle = '\033[2;106m'
hgh = '\033[1;100m'
wrn = '\033[33m'
err = '\033[31m'
ext = '\033[32m'
std = '\033[0m'

# Program output variables
pgmnam = f"..:{ext}pyAdMD> {std}"
pgmwrn = f"..+{wrn}pyAdMD-Wrn> {std}"
pgmerr = f"..%{err}pyAdMD-Err> {std}"

# Header variables
# https://manytools.org/hacker-tools/ascii-banner/
logo = '''
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

version = '1.1'
citation = '''  Please cite:

\tAdaptive collective motions: a hybrid method to improve
\tconformational sampling with molecular dynamics and normal modes.
\tPT Resende-Lara, MGS Costa, B Dudas, D Perahia.
\tDOI: https://doi.org/10.1101/2022.11.29.517349'''

banner = (f"{'\033[5;36m'}{logo}{std}\n"
          f"\t\t{tle}Adaptive Molecular Dynamics with Python{std}\n"
          f"\t\t\t     version: {version}\n"
          f"\n{citation}\n")

message="This program can setup and run multi-replica aMDeNM simulations through NAMD."

print(banner)
print(__doc__)

# Create a dictionary containing the user-provided arguments
args = parse_arguments()

# If no argument was provided
if vars(args) == {}:
    print("usage: pyAdMD.py [-h] {clean,run} ...")
    print(f"pyAdMD.py: error: {err}At least one argument must be provided.")
    sys.exit()


# Running options
if 'run' in args:
    # Check if '--modefile' is provided when '--type CHARMM' is used
    if args.type == "CHARMM" and not args.modefile:
        print("usage: pyAdMD.py [-h] {clean,run} ...")
        print(f"pyAdMD.py: error: {err}The --modefile argument is required when --type is CHARMM")
        sys.exit()

    # Store the variable values
    # File names
    if args.modefile: modepath = args.modefile
    psfpath = args.psffile
    pdbpath = args.pdbfile
    coorpath = args.coorfile
    velpath = args.velfile
    xscpath = args.xscfile
    strpath = args.strfile
    # File paths
    if args.modefile: modefile = args.modefile.split('/')[-1]
    psffile = args.psffile.split('/')[-1]
    pdbfile = args.pdbfile.split('/')[-1]
    coorfile = args.coorfile.split('/')[-1]
    velfile = args.velfile.split('/')[-1]
    xscfile = args.xscfile.split('/')[-1]
    strfile = args.strfile.split('/')[-1]
    # Parameters
    nm_type = args.type.lower()
    modes = args.modes
    nm_parsed = [int(s) for s in modes.split(',')]
    energy = args.energy
    sim_time = args.time
    selection = args.selection
    replicas = args.replicas

    print(f"{pgmnam}{tle}Setup and run aMDeNM simulations{std}\n")

    # Test if the provided files exist
    if args.modefile:
        file_list = [modepath, psfpath, pdbpath, coorpath, velpath, xscpath, strpath]
    else:
        file_list = [psfpath, pdbpath, coorpath, velpath, xscpath, strpath]
    for file in file_list:
        if not os.path.isfile(file):
            print(f"{pgmerr}File {err}{file.split('/')[-1]}{std} not found.")
            sys.exit()
        # Test if the provided files are at the input folder and copy them if not
        if not os.path.isfile(f"{input_dir}/{file.split('/')[-1]}"):
            shutil.copy(file, input_dir)
            print(f"{pgmwrn}File {wrn}{file.split('/')[-1]}{std} was copied to inputs folder.")

    # Get some information from the system
    print(f"{pgmnam}Getting system info.")
    sys_pdb = mda.Universe(f"{input_dir}/{psffile}", f"{input_dir}/{coorfile}", format="NAMDBIN")
    N = sys_pdb.atoms.n_atoms                                       # Total atom number
    sys_mass = sys_pdb.atoms.masses                                 # System atomic mass
    sys_zeros = sys_pdb.atoms.select_atoms("all")
    init_coor = sys_pdb.atoms.select_atoms(selection).positions
    sel_atom = sys_pdb.atoms.select_atoms(selection).n_atoms        # Number of selected atoms
    sel_mass = sys_pdb.atoms.select_atoms(selection).masses         # Selection atomic mass

    # Correction variables definition
    globfreq = cos_alpha = 0.5
    qrms_correc = 0.5

    # Define the number of excitation cycles
    # sim_time / (total_steps * timestep)
    end_loop = int(sim_time / (100 * 0.002))

    # Define the top and bottom values for Ek correction
    # 25% window of excitation energy
    top = energy * 1.25
    bottom = energy * 0.75

    # Compute/Write the normal mode vectors
    if nm_type == "ca":
        print(f"\n{pgmnam}{hgh}Computing {ext}Cα ENM{std}{hgh} and writing normal mode vectors {ext}{modes}{std}.")
        compute_enm(f"{input_dir}/{coorfile}", nm_type, nm_parsed) ### TODO: the highest nm in modes variable = MAX_MODES in enm.py
    elif nm_type == "heavy":
        print(f"\n{pgmnam}{hgh}Computing {ext}Heavy atoms ENM{std}{hgh} and writing normal mode vectors {ext}{modes}{std}.")
        compute_enm(f"{input_dir}/{coorfile}", nm_type, nm_parsed) ### TODO: the highest nm in modes variable = MAX_MODES in enm.py
    elif nm_type == "charmm":
        print(f"\n{pgmnam}Writing {ext}CHARMM{std} normal mode vectors {ext}{modes}{std}.")
        write_charmm_nm(modes)

    # Extract NAMD topology and parameters files
    unzip(f"{input_dir}/namd_toppar.zip", input_dir)

    # Generate factors to uniformly combine the modes
    print(f"\n{pgmnam}Generating {ext}{replicas}{std} uniformly distributed combinations of modes {ext}{modes}{std}.")
    factors = generate_factors(replicas, nm_parsed)
    combine_modes(replicas, nm_parsed, factors, nm_type)

    #######################
    ### CALL FOR ACTION ###
    #######################

    # Create the replica folder and enter it
    for rep in range(1, (replicas + 1)):
        if args.no_correc:
            print(f"\n{pgmnam}{hgh}Starting Standard MDeNM calculations for {ext}rep{rep}{std}")
        elif args.fixed:
            print(f"\n{pgmnam}{hgh}Starting Constant MDeNM calculations for {ext}rep{rep}{std}")
        else:
            print(f"\n{pgmnam}{hgh}Starting Adaptive MDeNM calculations for {ext}rep{rep}{std}")
        rep_dir = f"{cwd}/rep{rep}"
        os.makedirs(rep_dir, exist_ok=True)
        os.chdir(rep_dir)

        # Copying the NM combination vector to the replica folder
        shutil.copy(f"{cwd}/rep-struct-list/rep{rep}_vector.vec", "pff_vector.vec")

        # Excite the combined vector according to user-defined energy increment
        print(f"{pgmnam}Writing the excitation vector with a Ek injection of {ext}{energy}{std} kcal/mol.")
        q_vec = mda.Universe(f"{input_dir}/{psffile}", "pff_vector.vec", format="NAMDBIN")
        q_vec = q_vec.atoms.select_atoms(selection).positions
        exc_vec = excite(q_vec, energy)

        # Write the combination and the excited vector
        wrt_vec(q_vec, "cntrl_vector.vec")
        wrt_vec(exc_vec, "excitation.vel")

        # Start the excitation loop, copy initial files and define initial variables
        loop = 0
        cnt = 1
        vp, ek, qp, rmsp = [[], [], [], []]
        ref_str = f"step_{loop}.coor"   # will change eventually during the simulation
        shutil.copy(f"{input_dir}/{coorfile}", f"correc_ref.coor")
        shutil.copy(f"{input_dir}/{coorfile}", f"step_{loop}.coor")
        shutil.copy(f"{input_dir}/{velfile}", f"step_{loop}.vel")
        shutil.copy(f"{input_dir}/{xscfile}", f"step_{loop}.xsc")

        while loop < end_loop:

            if loop == 0:
                # Read the current NAMD velocities file
                vel_curr = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
                vel_curr = vel_curr.coord.positions
                # Read the excitation velocities file
                vel_exc = mda.Universe(f"{input_dir}/{psffile}", "excitation.vel", format="NAMDBIN")
                vel_exc = vel_exc.coord.positions

                # Write the input velocities vel_tot = vel_curr + vel_exc
                vel_tot = vel_curr + vel_exc
                wrt_vec(vel_tot, f"step_{loop}.vel")

            # Loop update
            loop += 1

            # Create NAMD configuration file
            shutil.copy(f"{input_dir}/conf.namd", 'conf.namd')
            namd_conf = Path('conf.namd')
            namd_conf.write_text(namd_conf.read_text().replace('$PSF', f"{input_dir}/{psffile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$PDB', f"{input_dir}/{pdbfile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$STR', f"{input_dir}/{strfile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$COOR', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$VEL', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$XSC', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$OUTPUT', str(loop)))

            # Run NAMD
            now = time.strftime("%H:%M:%S")
            print(f"{pgmnam}{now} {ext}Replica {rep}{std}: running {ext}step {loop}{std} of {end_loop}...")
            run_namd = f"namd3 conf.namd > step_{loop}.log"
            returned_value = subprocess.call(run_namd, shell=True,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if returned_value != 0:
                print(f"{pgmerr}An error occurred while running NAMD.\n"
                      f"{pgmerr}Inspect the file {err}step_{loop}.log{std} for detailed information.\n")
                sys.exit()

            ## EVALUATE IF IT IS NECESSARY TO CHANGE THE EXCITATION DIRECTION ##
            if [args.no_correc == False] and [args.fixed == False]:
                coor_ref = mda.Universe(f"{input_dir}/{psffile}", "correc_ref.coor", format="NAMDBIN")
                coor_ref = coor_ref.atoms.select_atoms(selection).positions

                coor_curr = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
                coor_curr = coor_curr.atoms.select_atoms(selection).positions

                # Compute the difference and mass-weight the difference
                # (qcurr - qref) * sqrt(m)
                diff = ((coor_curr - coor_ref).T * np.sqrt(sel_mass)).T

                # Read the excitation vector
                cntrl_vec = mda.Universe(f"{input_dir}/{psffile}", "cntrl_vector.vec", format="NAMDBIN")
                cntrl_vec = cntrl_vec.atoms.select_atoms(selection).positions

                # Project the current coordinates onto Q
                q_proj = np.sum(diff * cntrl_vec)
                rms_check = np.sqrt((q_proj ** 2) / np.sum(sel_mass))

                # Correct the excitation direction if necessary
                if rms_check >= qrms_correc:
                    # Compute the average structure of the last excitation
                    ts = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop - 1}.coor", format="NAMDBIN")
                    avg_positions = ts.atoms.select_atoms(selection).positions
                    ts = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
                    avg_positions += ts.atoms.select_atoms(selection).positions
                    avg_positions = avg_positions / 2
                    wrt_vec(avg_positions, f"average_{loop}.coor")

                    # Open the reference and mobile structures
                    ref = mda.Universe(f"{input_dir}/{psffile}", ref_str, format="NAMDBIN")
                    ref = ref.atoms.select_atoms(selection)
                    mob = mda.Universe(f"{input_dir}/{psffile}", f"average_{loop}.coor", format="NAMDBIN")
                    mob = mob.atoms.select_atoms(selection)

                    # Align the structures and compute the mass-weighted difference
                    align.alignto(mob, ref, select="protein", weights="mass")
                    diff = ((mob.positions - ref.positions).T * np.sqrt(sel_mass)).T

                    # Normalize the mass-weighted difference vector
                    diff /= np.linalg.norm(diff)

                    # Project the current coordinates onto Q
                    dotp = np.sum(diff * cntrl_vec)

                    # Set the average structure as the new reference for the next steps
                    ref_str = f"average_{loop}.coor"

                    if dotp <= cos_alpha:
                        shutil.copy(f"step_{loop}.coor", f"correc_ref.coor")
                        qrms_correc = 0

                    # Rename the previous excitation vector files
                    shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
                    shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")
                    cnt += 1

                    # Write the corrected excitation vector
                    print(f"{pgmnam}Writing the corrected excitation vector.")
                    wrt_vec(diff, "cntrl_vector.vec")

                    # Excite and write the new excited vector
                    exc_vec = excite(diff, energy)
                    wrt_vec(exc_vec, "excitation.vel")

                    # Update the rms correction variable value
                    qrms_correc += globfreq

            # OBTAIN THE VELOCITIES AND KINETIC ENERGY PROJECTED ONTO Q
            # Open the current velocities file and mass-weight
            curr_vel = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
            curr_vel = ((curr_vel.atoms.select_atoms(selection).positions).T * np.sqrt(sel_mass)).T

            # Read the excitation vector
            cntrl_vec = mda.Universe(f"{input_dir}/{psffile}", "cntrl_vector.vec", format="NAMDBIN")
            cntrl_vec = cntrl_vec.atoms.select_atoms(selection).positions

            # Compute the scalar projection of velocity
            velo = np.sum(curr_vel * cntrl_vec) / np.sum(cntrl_vec * cntrl_vec)
            vp.append(f"{str(round(velo, 5))}\n")

            # Compute the vectorial projection of velocity
            v_proj = np.sum(curr_vel * cntrl_vec) / np.sum(cntrl_vec * cntrl_vec) * cntrl_vec

            # Calculate the kinetic energy from projected velocities
            ek_vel = 0.5 * np.sum(v_proj ** 2)
            ek.append(f"{str(round(ek_vel, 5))}\n")

            # Write the unmass-weighted velocity projection
            v_proj = (v_proj.T / np.sqrt(sel_mass)).T
            wrt_vec(v_proj, "velo_proj.vel")

            # PROJECT THE COORDINATES ONTO Q
            # Open the current coordinates file
            curr_coor = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
            curr_coor = curr_coor.atoms.select_atoms(selection).positions

            # Compare the current with the initial coordinates
            diff = curr_coor - init_coor
            diff = (diff.T * np.sqrt(sel_mass)).T

            # Calculate the dot product between qcurr and Q
            dotp = np.sum(diff * cntrl_vec)
            qp.append(f"{str(round(dotp, 5))}\n")

            # Compute the rms displacement along the vector Q
            mrms = np.sqrt((dotp ** 2) / sum(sel_mass))
            rmsp.append(f"{str(round(mrms, 5))}\n")

            # Skip excitation vector rescaling (Original MDeNM)
            if args.no_correc:
                continue

            # RESCALE KINETIC ENERGY ACCORDING TO VALUES PROJECTED ONTO VECTOR Q
            '''Re-excite the NM vector when ek is below inferior limit
            or relax the energy when ek is above superior limit'''
            if (ek_vel < bottom) or (ek_vel > top):
                # Read current and excitation velocities
                curr_vel = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
                curr_vel = curr_vel.coord.positions
                exc_vec = mda.Universe(f"{input_dir}/{psffile}", "excitation.vel", format="NAMDBIN")
                exc_vec = exc_vec.coord.positions
                v_proj = mda.Universe(f"{input_dir}/{psffile}", "velo_proj.vel", format="NAMDBIN")
                v_proj = v_proj.coord.positions

                # Compute the difference between the projected and the excitation velocities
                # and then sum to the current velocities: Vnew = Vdyna + (VQ - Vp)
                new_vel = curr_vel + (exc_vec - v_proj)
                wrt_vec(new_vel, f"step_{loop}.vel")

        # Write the projections into files
        for i,j in zip((vp, ek, qp, rmsp), ("vp", "ek", "coor", "rms")):
            with open(f"{j}-proj.out", 'w') as write:
                write.writelines(i)

        # De-excite the system
        shutil.copy(f"{input_dir}/deexcitation.namd", 'deexcitation.namd')
        deexc_conf = Path('deexcitation.namd')
        deexc_conf.write_text(deexc_conf.read_text().replace('$PSF', f"{input_dir}/{psffile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$PDB', f"{input_dir}/{pdbfile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$STR', f"{input_dir}/{strfile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$COOR', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$VEL', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$XSC', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$TS', str(int(sim_time / 0.002))))

        # Run NAMD
        now = time.strftime("%H:%M:%S")
        print(f"{pgmnam}{now} {ext}Replica {rep}{std}: running the {ext}de-excitation step{std}...")
        run_namd = f"namd3 deexcitation.namd > deexcitation.log"
        returned_value = subprocess.call(run_namd, shell=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if returned_value != 0:
            print(f"{pgmerr}An error occurred while running NAMD.\n"
                  f"{pgmerr}Inspect the file {err}deexcitation.log{std} for detailed information.\n")
            sys.exit()

elif 'clean' in args:
    print(f"{pgmnam}{tle}Clean previous pyAdMD setup files{std}\n")

    # Removing previous replicas folders
    files = os.listdir(cwd)
    for item in files:
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

    print(f"{pgmnam}Erasing is done.\n")
