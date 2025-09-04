import os
import subprocess
import sys
import csv
import numpy as np
import MDAnalysis as mda
import logging

logger = logging.getLogger("__init__")

def write_charmm_nm(nms_to_write):
    """
    Write CHARMM normal mode vectors in NAMD readable format.

    Parameters
    ----------
    nms_to_write : list
        List of normal modes to write
    """
    nms = [str(t + '\n') for t in nms_to_write.split(',')]
    with open(f"{input_dir}/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)
    os.chdir(f"{cwd}/src")
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]}"
           f" modfile={modefile.split('/')[-1]} -o wrt-nm.out")
    returned_value = subprocess.call(cmd, shell=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        logger.error("An error occurred while writing the normal mode vectors.\n"
                    "Inspect the file wrt-nm.out for detailed information.\n")

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
                logger.info(f"Converged to tolerance after {iteration+1} iterations.")
                break

            if stagnation_count >= stagnation_threshold:
                logger.warning(f"Breaking due to stagnation after {iteration+1} iterations.")
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
    logger.info(f"Combination factors written at {output_folder}/factors.csv.")

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

def combine_modes(replicas, modes, factors):
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
    """
    # Set output folder
    output_folder = f"{cwd}/rep-struct-list"

    for rep in range(replicas):
        # Create an empty vector to store the combination
        natom = sys_pdb.atoms.select_atoms("protein").n_atoms
        comb_vec = sys_pdb.atoms.select_atoms("protein").positions
        comb_vec = np.zeros((natom, 3))

        for nm_idx in range(len(modes)):
            nm = mda.Universe(f"{input_dir}/mode_nm{modes[nm_idx]}.crd", format="CRD")
            nm = nm.atoms.select_atoms("protein").positions

            # Apply the factors to the modes
            comb = (nm.T * factors[rep,nm_idx]).T

            # Acumulate modes to obtain a new combined vector
            comb_vec = np.add(comb, comb_vec)

        # Normalize and write the combined vector
        comb_vec /= np.linalg.norm(comb_vec)
        wrt_vec(comb_vec, f"{output_folder}/rep{rep+1}_vector.vec")

    logger.info(f"Combination vectors written at {output_folder}/rep*_vector.vec.")

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
