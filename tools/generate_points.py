import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist
import numpy as np

def generate_hypersphere_points(P, N, max_iterations=1000000, tolerance=1e-6):
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

    else:
        # Generate uniformly distributed points
        # on a N-dimensional hypersphere using a repulsion algorithm.

        # Initialize with random points on the hypersphere
        points = np.random.normal(size=(P, N))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        # Variables for stagnation detection
        prev_max_force = float('inf')
        stagnation_count = 0
        stagnation_threshold = 5  # Number of iterations with no significant change to trigger break

        # Use a repulsion algorithm to spread points evenly
        for iteration in range(max_iterations):
            # Calculate all pairwise distances
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
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
                tangent_force = total_force[i] - np.dot(total_force[i], points[i]) * points[i]

                # Move point in tangent direction
                points[i] += 0.001 * tangent_force

                # Project back to hypersphere
                points[i] /= np.linalg.norm(points[i])

            # Check for convergence or stagnation
            if current_max_force < tolerance:
                print(f"Converged to tolerance after {iteration+1} iterations")
                break

            if stagnation_count >= stagnation_threshold:
                print(f"Breaking due to stagnation after {iteration+1} iterations")
                break

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
    ax.set_title(f'{P} Points on a 2D Circle')
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
    ax.set_title(f'{P} Points on a 3D Sphere')
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

    print(f"Minimum distance: {min_dist:.4f}")
    print(f"Maximum distance: {max_dist:.4f}")
    print(f"Mean distance: {mean_dist:.4f}")
    print(f"Standard deviation: {std_dist:.4f}")

    return min_dist, max_dist, mean_dist, std_dist

# Example usage
if __name__ == "__main__":
    # Test 3D
    P, N = 48, 3

    print(f"\nComputing {P} points in {N} dimensions (repulsion algorithm):")
    start_time = time.time()
    points = generate_hypersphere_points(P, N)
    duration = time.time() - start_time
    print(f"Computed points in {duration:.2f} seconds")
    points_3d = plot_3d_points(points)
    analyze_uniformity(points)

    # Test 2D
    P, N = 30, 2

    print(f"\nComputing {P} points in {N} dimensions (repulsion algorithm):")
    start_time = time.time()
    points = generate_hypersphere_points(P, N)
    duration = time.time() - start_time
    print(f"Computed points in {duration:.2f} seconds")
    points_2d = plot_2d_points(points)
    analyze_uniformity(points)

    # Test 6D
    P, N = 180, 6

    print(f"\nComputing {P} points in {N} dimensions (repulsion algorithm):")
    start_time = time.time()
    points = generate_hypersphere_points(P, N)
    duration = time.time() - start_time
    print(f"Computed points in {duration:.2f} seconds")
    analyze_uniformity(points)
