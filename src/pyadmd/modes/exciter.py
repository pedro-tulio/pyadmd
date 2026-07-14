"""Normal mode combination generation, excitation velocity, and vector I/O."""

import csv
import os
from typing import List

import numpy as np
import MDAnalysis as mda

from pyadmd.console import ConsoleConfig


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
