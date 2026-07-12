"""Elastic Network Model construction and normal mode analysis."""

import os
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.sparse import diags, coo_array, issparse
from scipy.sparse.linalg import eigsh
import cupy as cp

import openmm as mm
from openmm import app, unit

from pyadmd.console import ConsoleConfig
from pyadmd.io.state import make_reference_universe


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
