import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree
import numpy as np
from scipy.stats import qmc

def generate_initial_points(P, N):
    """
    Generate initial points using quasi-random Halton sequence for better initial distribution
    """
    sampler = qmc.Halton(d=N, scramble=True)
    sample = sampler.random(n=P)
    # Scale to [-1, 1] and project to hypersphere
    points = qmc.scale(sample, -1, 1)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def project_tangent(force, point):
    """
    Project force vector onto tangent space of the hypersphere at point
    """
    return force - np.dot(force, point) * point

def exponential_map(point, tangent_vector, step_size):
    """
    Move point along the geodesic in the direction of tangent_vector
    """
    norm = np.linalg.norm(tangent_vector)
    if norm < 1e-10:
        return point
    return point * np.cos(norm * step_size) + (tangent_vector / norm) * np.sin(norm * step_size)

def generate_hypersphere_points(P, N, max_iterations=10000, tolerance=1e-6):
    """
    Generate factors for linear combinations of N orthonormal vectors
    that produce P points uniformly distributed on a N-dimensional hypersphere surface.

    Parameters:
    P (int): Number of points to generate
    N (int): Dimensionality of the space
    max_iterations (int): Maximum number of iterations for the repulsion algorithm
    tolerance (float): Convergence tolerance

    Returns:
    numpy.ndarray: P×N matrix of factors for linear combinations
    """
    if P == 2 * N:
        # Special case: vertices of a cross-polytope
        positive_units = np.eye(N)
        factors = np.vstack((positive_units, -positive_units))
        return factors

    # Initialize with quasi-random points on the hypersphere
    points = generate_initial_points(P, N)

    # Adaptive parameters
    initial_step_size = 0.01
    cooling_rate = 0.995
    step_size = initial_step_size
    neighbor_count = min(50, P)  # Consider nearest neighbors for force calculation

    # Use a repulsion algorithm to spread points evenly
    for iteration in range(max_iterations):
        # Build k-d tree for efficient nearest neighbor search
        tree = cKDTree(points)

        # Find nearest neighbors for each point
        distances, indices = tree.query(points, k=neighbor_count)

        # Avoid division by zero
        distances = np.clip(distances, 1e-8, None)

        # Calculate repulsion forces (using inverse square law)
        force_matrix = 1.0 / (distances[:, 1:]**2)  # Skip self (first column)

        # Calculate direction of forces
        force_dirs = points[indices[:, 1:]] - points[:, np.newaxis, :]
        force_dirs /= distances[:, 1:, np.newaxis]  # Normalize

        # Sum forces acting on each point
        total_force = np.sum(force_matrix[:, :, np.newaxis] * force_dirs, axis=1)

        # Project forces onto tangent space
        tangent_forces = np.zeros_like(points)
        for i in range(P):
            tangent_forces[i] = project_tangent(total_force[i], points[i])

        # Calculate current maximum force
        max_force = np.max(np.linalg.norm(tangent_forces, axis=1))

        # Move points along the geodesic in the direction of the tangent force
        for i in range(P):
            points[i] = exponential_map(points[i], tangent_forces[i], step_size)
            # Ensure we stay on the unit sphere (numerical stability)
            points[i] /= np.linalg.norm(points[i])

        # Cool step size
        step_size *= cooling_rate

        # Check for convergence
        if max_force < tolerance:
            print(f"Converged to tolerance after {iteration+1} iterations")
            break

        # Print progress occasionally
        if iteration % 100 == 0:
            print(f"Iteration {iteration+1}, max force: {max_force:.6f}, step size: {step_size:.6f}")

    return points

def plot_2d_points(points):
    """
    Generate and plot points on a 2D circle (hypersphere in 2D)
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], s=50, alpha=0.7)

    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='r', linestyle='--', alpha=0.5)
    ax.add_patch(circle)

    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'{len(points)} Points on a 2D Circle')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_3d_points(points):
    """
    Generate and plot points on a 3D sphere (hypersphere in 3D)
    """
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, alpha=0.7)

    # Draw a wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the wireframe with low alpha to make it subtle
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3, linewidth=0.5)

    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_title(f'{len(points)} Points on a 3D Sphere')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def analyze_uniformity(points):
    """
    Analyze the uniformity of points on the hypersphere
    """
    # Calculate pairwise distances
    dists = pdist(points)

    # Calculate statistics
    min_dist = np.min(dists)
    max_dist = np.max(dists)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)

    print(f"Minimum distance: {min_dist:.6f}")
    print(f"Maximum distance: {max_dist:.6f}")
    print(f"Mean distance: {mean_dist:.6f}")
    print(f"Standard deviation: {std_dist:.6f}")

    return min_dist, max_dist, mean_dist, std_dist

# Example usage
if __name__ == "__main__":
    # Test 3D
    P, N = 48, 3

    print(f"\nComputing {P} points in {N} dimensions (improved repulsion algorithm):")
    start_time = time.time()
    points = generate_hypersphere_points(P, N)
    duration = time.time() - start_time
    print(f"Computed points in {duration:.2f} seconds")
    analyze_uniformity(points)
    plot_3d_points(points)

    # Test 2D
    P, N = 30, 2

    print(f"\nComputing {P} points in {N} dimensions (improved repulsion algorithm):")
    start_time = time.time()
    points = generate_hypersphere_points(P, N)
    duration = time.time() - start_time
    print(f"Computed points in {duration:.2f} seconds")
    analyze_uniformity(points)
    plot_2d_points(points)
