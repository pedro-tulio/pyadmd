import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds
from scipy.spatial.distance import pdist, squareform

def uniform_hypersphere_points_optimized(P, N, max_iter=10000, tol=1e-9):
    """
    Uniformly distributed P points on a N-dimensional hypersphere
    using optimization to maximize the minimal distance.

    Parameters:
    P (int): Number of points to generate
    N (int): Dimensionality of the space
    max_iter (int): Maximum number of iterations for optimization
    tol (float): Tolerance for optimization convergence

    Returns:
    numpy.ndarray: P×N matrix of points on the hypersphere
    """
    if P == 2 * N:
        # Special case: vertices of a cross-polytope
        positive_units = np.eye(N)
        points = np.vstack((positive_units, -positive_units))

    else:
        # Generate points on a N-dimensional hypersphere by minimizing an energy function
        # that encourages uniform distribution.

        # Initialize with random points on the hypersphere
        points = np.random.normal(size=(P, N))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        # Flatten the points for optimization
        x0 = points.flatten()

        # Define the energy function (sum of inverse distances)
        def energy(x):
            points = x.reshape(P, N)
            # Ensure points are on the hypersphere
            norms = np.linalg.norm(points, axis=1)
            points = points / norms[:, np.newaxis]

            # Calculate pairwise distances
            dists = pdist(points)

            # Avoid division by zero
            dists = np.maximum(dists, 1e-12)

            # Energy is sum of inverse distances
            return np.sum(1.0 / dists ** (N - 1))

        # Define constraints (each point must lie on the hypersphere)
        constraints = []
        for i in range(P):
            def sphere_constraint(x, i=i):
                point = x[i*N:(i+1)*N]
                return np.linalg.norm(point) - 1.0
            constraints.append({'type': 'eq', 'fun': sphere_constraint})

        # Set bounds to keep points reasonable
        bounds = Bounds(-1.1, 1.1)

        # Run the optimization
        result = minimize(energy, x0, method='SLSQP', constraints=constraints,
                        bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'disp': True})

        # Reshape the result and ensure points are on the hypersphere
        points = result.x.reshape(P, N)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms

        return points

def analyze_distribution(points):
    """
    Analyze the distribution of points on the hypersphere.
    """
    # Calculate pairwise distances
    dists = pdist(points)

    # Calculate statistics
    min_dist = np.min(dists)
    max_dist = np.max(dists)
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)

    print(f"Number of points: {points.shape[0]}")
    print(f"Dimension: {points.shape[1]}")
    print(f"Minimum distance: {min_dist:.6f}")
    print(f"Maximum distance: {max_dist:.6f}")
    print(f"Mean distance: {mean_dist:.6f}")
    print(f"Standard deviation: {std_dist:.6f}")
    print(f"Ratio min/max: {min_dist/max_dist:.6f}")

    return min_dist, max_dist, mean_dist, std_dist

def plot_2d_points(points, title):
    """
    Plot points on a 2D circle.
    """
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
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')

    # Adjust layout
    plt.tight_layout()

    plt.show()

def plot_3d_points(points, title):
    """
    Plot points on a 3D sphere.
    """
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
    ax.plot_wireframe(x, y, z, color='r', alpha=0.5, linewidth=1)

    # Set plot properties
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_title(title)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Adjust layout
    plt.tight_layout()

    plt.show()


# Example usage and comparison
if __name__ == "__main__":
#    # Test with 10 points in 2D
#    P, N = 15, 2
#
#    print("\nGenerating points using energy minimization method:")
#    points_energy = uniform_hypersphere_points_optimized(P, N)
#    analyze_distribution(points_energy)
#    plot_2d_points(points_energy, f'{P} Points on 2D Sphere (Energy Method)')

    # Test with 20 points in 3D
    P, N = 48, 3

    print("\nGenerating points using energy minimization method:")
    start_time = time.time()
    points_energy = uniform_hypersphere_points_optimized(P, N)
    plot_3d_points(points_energy, f'{P} Points on 3D Sphere (Energy Method)')
    duration = time.time() - start_time
    print(f"Computed {P} Points in {duration:.2f} seconds")
    analyze_distribution(points_energy)
