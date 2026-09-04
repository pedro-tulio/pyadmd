# Adaptive Molecular Dynamics with Python

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

The **Adaptive Molecular Dynamics with Excited Normal Modes (aMDeNM)** method applies a kinetic excitation of normal modes (NMs) to enhance molecular dynamics simulations sampling. This technique consists in injecting additional atomic velocities along uniformily distributed combinations of NM vectors, creating an effective coupling between slow and fast molecular motions. The motions described by preselected directions of low-frequency NMs are dynamically adjusted throughout the simulation. By coupling low-frequency NM excitation with adaptive directional adjustments, aMDeNM facilitates extensive exploration of the energy landscape, overcoming the constraints of fixed, rectilinear displacements and alleviating structural stresses and environmental resistance. Importantly, aMDeNM requires only an initial structure without the need to specify predefined target states, distinguishing it from many biased sampling techniques that rely on predefined target conformations.

[![aMDeNM remarkably improves sampling while offering mechanistical insights on biomolecular systems](.github/graphical_abstract.png)](https://pubs.acs.org/doi/10.1021/acs.jctc.6c00398)

**pyAdMD** is a package designed to fully setup, run, and analyze aMDeNM simulations on a dedicated Python environment. This document provides an overview of the method and documents the package functionalities.

* ****

- [Adaptive Molecular Dynamics with Python](#adaptive-molecular-dynamics-with-python)
- [Method Overview](#method-overview)
  - [Equilibration Molecular Dynamics](#equilibration-molecular-dynamics)
  - [CHARMM Normal Modes Analysis (Optional)](#charmm-normal-modes-analysis-optional)
  - [ENM Computation](#enm-computation)
    - [Physical Motivation and Coarse-Graining](#physical-motivation-and-coarse-graining)
    - [Network Construction](#network-construction)
    - [The Hessian Matrix](#the-hessian-matrix)
    - [Normal Mode Analysis and Diagonalization](#normal-mode-analysis-and-diagonalization)
    - [Rigid-Body Modes and the Null Space](#rigid-body-modes-and-the-null-space)
    - [Equipartition and the Physical Meaning of Eigenvalues](#equipartition-and-the-physical-meaning-of-eigenvalues)
  - [Uniform Normal Modes Combination](#uniform-normal-modes-combination)
    - [Problem Definition](#problem-definition)
    - [Mode Subspace Geometry](#mode-subspace-geometry)
    - [Orthonormal Basis Construction](#orthonormal-basis-construction)
    - [Energy Minimization Framework](#energy-minimization-framework)
    - [Dimensionally-Scaled Potential Function](#dimensionally-scaled-potential-function)
    - [Gradient Flow on the Sphere](#gradient-flow-on-the-sphere)
    - [Special Case: Cross-Polytope Initialization](#special-case-cross-polytope-initialization)
    - [Normal Modes Linear Combination](#normal-modes-linear-combination)
  - [Kinetic Energy Control](#kinetic-energy-control)
  - [Excitation Direction Update](#excitation-direction-update)
- [pyAdMD Applications](#pyadmd-applications)
  - [ENM](#enm)
  - [Physical force-field based normal modes](#physical-force-field-based-normal-modes)
- [Configuration](#configuration)
  - [Normal Modes Selection](#normal-modes-selection)
  - [Energy injection](#energy-injection)
  - [Simulation time](#simulation-time)
  - [Excitation direction update](#excitation-direction-update-1)
  - [Number of modes and replicas](#number-of-modes-and-replicas)
  - [Atom selection](#atom-selection)
- [Input Requirements](#input-requirements)
  - [Run](#run)
    - [Parameters](#parameters)
    - [Files](#files)
    - [Feature Flags](#feature-flags)
  - [Append](#append)
    - [Parameters](#parameters-1)
  - [Free Energy](#free-energy)
  - [Analysis](#analysis)
    - [Parameters](#parameters-2)
    - [Feature Flags](#feature-flags-1)
    - [Skip Flags](#skip-flags)
  - [ENM](#enm-1)
    - [Parameters](#parameters-3)
    - [Skip Flags](#skip-flags-1)
    - [Post-hoc Mode Re-writer](#post-hoc-mode-re-writer)
  - [Output Structure](#output-structure)
    - [Directory Organization](#directory-organization)
    - [Output Files Description](#output-files-description)
- [Free Energy Landscape](#free-energy-landscape)
  - [Method Overview](#method-overview-1)
  - [Extending a Previous Free Energy Calculation](#extending-a-previous-free-energy-calculation)
  - [Output Structure](#output-structure-1)
    - [Directory Organization](#directory-organization-1)
  - [Output Files Description](#output-files-description-1)
- [Analysis](#analysis-1)
  - [Basic Structural Properties Calculated](#basic-structural-properties-calculated)
  - [Analysis Modes](#analysis-modes)
    - [Standard Analysis](#standard-analysis)
    - [Rough Analysis](#rough-analysis)
    - [Selective Analysis (Skip Flags)](#selective-analysis-skip-flags)
  - [Trajectory Source](#trajectory-source)
  - [Handling Incomplete Units](#handling-incomplete-units)
  - [Configuration Parameters](#configuration-parameters)
  - [Output Structure](#output-structure-2)
    - [Directory Organization](#directory-organization-2)
  - [Output Files Description](#output-files-description-2)
- [Usage Examples](#usage-examples)
  - [Using OpenMM inputs and heavy atoms NMs](#using-openmm-inputs-and-heavy-atoms-nms)
  - [Using NAMD inputs and Cα NMs with custom parameters](#using-namd-inputs-and-cα-nms-with-custom-parameters)
  - [Using NAMD inputs and CHARMM NMs without direction correction (standard MDeNM)](#using-namd-inputs-and-charmm-nms-without-direction-correction-standard-mdenm)
  - [Restart unfinished pyAdMD simulations](#restart-unfinished-pyadmd-simulations)
  - [Append 100 ps to previously finished pyAdMD simulations](#append-100-ps-to-previously-finished-pyadmd-simulations)
  - [Analyze pyAdMD simulations (all frames)](#analyze-pyadmd-simulations-all-frames)
  - [Analyze every 5 ps skipping DSSP and LMI](#analyze-every-5-ps-skipping-dssp-and-lmi)
  - [Compute a free energy landscape](#compute-a-free-energy-landscape)
  - [Extend a previous free energy calculation with more centroids and production time](#extend-a-previous-free-energy-calculation-with-more-centroids-and-production-time)
  - [Compute a standalone ENM (Cα model, writing modes 7-16)](#compute-a-standalone-enm-cα-model-writing-modes-7-16)
  - [Re-write additional modes from a previous ENM run](#re-write-additional-modes-from-a-previous-enm-run)
  - [Clean previous setup files](#clean-previous-setup-files)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Citing](#citing)
- [License](#license)
- [Contact](#contact)

* ****

# Method Overview
The aMDeNM is a enhanced sampling molecular dynamics method that uses normal modes as collective variables in order do increase the conformational space explored during MD simulation. This is done by injecting an incremental energy to the system, thus assigning additional atomic velocities along the direction of a given NM (or a combination of a NM set). The combination of the velocities from MD and those provided by the NM vectors properly couple slow and fast motions, allowing one to obtain large time scale movements, such as domain transitions in a feasible simulation time. The additional energy  injection and the excitation direction are constantly evaluated and updated to ensure the sampling validity.

The aMDeNM production runs (the excitation cycles themselves) are computed with OpenMM using the CHARMM36m force field. GPU platforms (CUDA, then OpenCL) are preferred automatically, with a CPU fallback if neither is available. We strongly recommend that all system preparation be done with *[CHARMM-GUI](http://charmm-gui.org)*.

## Equilibration Molecular Dynamics
This is a prerequired step to perform aMDeNM simulations. It consists in performing a short equilibration MD to store the final atomic velocities and positions. Two equilibration input paths are supported, selected via `-src`/`--source`:

- **NAMD** (`-src NAMD`): *[NAMD 3.0](http://www.ks.uiuc.edu/Research/namd/)* binary files (`.coor`/`.vel`/`.xsc`) plus a CHARMM-style stream file containing the PBC geometry information (`.str`).
- **OpenMM** (`-src OPENMM`): *[OpenMM](http://openmm.org)* state supplied as an XML restart file (`.rst`) produced with `XmlSerializer.serialize(state)` from a state built with `getPositions=True, getVelocities=True`.

## CHARMM Normal Modes Analysis (Optional)
If using CHARMM-based normal modes, it is also necessary to compute the modes from the last MD coordinates and store the vectors from the low-frequency end of the vibrational spectrum on a binary file.

## ENM Computation
### Physical Motivation and Coarse-Graining

Proteins are not rigid bodies: their function is intimately connected to their internal dynamics, ranging from local side-chain rotations and loop fluctuations to large-scale collective domain motions. Classical molecular dynamics (MD) simulations can capture these phenomena in atomic detail, but they are computationally expensive and often struggle to reach the timescales (microseconds to milliseconds) relevant to biologically important conformational changes.

Elastic Network Models offer a powerful and computationally inexpensive alternative. The central insight behind ENMs is that the low-frequency, large-amplitude collective motions of a protein — those most relevant to function — are predominantly determined by the overall topology of the molecular structure, not by the precise details of atomic interactions. In other words, the shape of the molecule, encoded by which atoms or residues are spatially close to one another, largely dictates the repertoire of accessible motions.

This motivates a **coarse-graining** strategy: instead of representing every atom with a detailed force field, the protein is reduced to a set of representative interaction sites connected by harmonic springs. In the **Cα model** (also called ANM, the Anisotropic Network Model), each residue is represented by a single point placed at its α-carbon. In the **heavy-atom model**, all non-hydrogen atoms are retained, yielding a finer-grained representation at the cost of a larger Hessian matrix. The choice of model involves a tradeoff between computational cost and the resolution of the dynamical description.

### Network Construction

Given a set of $`N`$ interaction sites (Cα atoms or heavy atoms) with equilibrium positions $`\mathbf{r}_i^0`$, the elastic network is constructed by connecting every pair of sites $`i`$ and $`j`$ whose equilibrium distance $`r_{ij}^0 = |\mathbf{r}_i^0 - \mathbf{r}_j^0|`$ falls within a specified **cutoff distance** $`r_c`$:

$$
r_{ij}^0 \leq r_c
$$

Typical cutoff values are $`10–15 Å`$ for the Cα model and $`7–12 Å`$ for the heavy-atom model. The cutoff is a key parameter: too small a value yields a disconnected or sparse network that fails to capture long-range coupling, while too large a value over-densifies the network and can wash out functionally relevant fluctuation patterns.

The total potential energy of the system under the harmonic approximation is:

$$
V = \frac{1}{2} \sum_{i < j} k_{ij} \left(r_{ij} - r_{ij}^0\right)^2
$$

where $`k_{ij}`$ is the spring constant between sites $`i`$ and $`j`$, $`r_{ij}`$ is the instantaneous distance between them, and $`r_{ij}^0`$ is their equilibrium distance. In the simplest ANM formulation, a **uniform spring constant** $`k_{ij} = k`$ is used for all connected pairs. This is a deliberate simplification: the spring constant encodes the stiffness of the local environment, and setting it uniformly to $`k`$ (with default $`k = 1.0 kcal/mol/Å²`$) means the model's predictions are expressed in units relative to $`k`$. More sophisticated variants assign distance-dependent spring constants (e.g., $`k_{ij} \propto (r_{ij}^0)^{-\alpha}`$), but the uniform model already captures the essential topology of collective motions.

### The Hessian Matrix

The dynamical properties of the network are encoded in the **Hessian matrix** $`\mathbf{H}`$, a $`3N \times 3N`$ symmetric matrix of second derivatives of the potential energy with respect to atomic displacements, evaluated at the equilibrium configuration:

$$
H_{i\alpha,\, j\beta} = \frac{\partial^2 V}{\partial u_{i\alpha}\, \partial u_{j\beta}}\Bigg|_{\mathbf{u}=0}
$$

where $`u_{i\alpha}`$ is the displacement of site $`i`$ along Cartesian direction $`\alpha \in \{x, y, z\}`$, and similarly for $`u_{j\beta}`$. The factor of three degrees of freedom per site is what distinguishes this **anisotropic** (ANM) formulation from simpler isotropic models: the Hessian retains the full directional information of each pairwise spring, which is essential for producing oriented mode trajectories and the vector dot products required by the DCCM.

Carrying out the differentiation of the pairwise harmonic potential, the off-diagonal $`3 \times 3`$ super-element connecting sites $`i \neq j`$ is:

$$
\mathbf{H}_{ij} = -\frac{k_{ij}}{(r_{ij}^0)^2} \begin{pmatrix} \Delta x^2 & \Delta x \Delta y & \Delta x \Delta z \\
\Delta y \Delta x & \Delta y^2 & \Delta y \Delta z \\
\Delta z \Delta x & \Delta z \Delta y & \Delta z^2 \end{pmatrix}
$$

where $`\Delta x = x_i^0 - x_j^0`$, $`\Delta y = y_i^0 - y_j^0`$, $`\Delta z = z_i^0 - z_j^0`$ are the components of the equilibrium difference vector $`\mathbf{r}_{ij}^0`$. This super-element is nonzero only when $`r_{ij}^0 \leq r_c`$, so $`\mathbf{H}`$ is sparse for typical cutoff distances. The diagonal blocks are set by the self-consistency condition (Newton's third law):

$$
\mathbf{H}_{ii} = -\sum_{j \neq i} \mathbf{H}_{ij}
$$

which ensures that $`\mathbf{H}`$ is positive semi-definite and that rigid-body motions have zero energy cost (see below).

### Normal Mode Analysis and Diagonalization

The equations of motion for the mass-weighted displacements $`\tilde{\mathbf{u}}_i = \sqrt{m_i}\, \mathbf{u}_i`$ (where $`m_i`$ is the mass of site $`i`$) take the form:

$$
\mathbf{M}^{-1/2} \mathbf{H}\, \mathbf{M}^{-1/2}\, \tilde{\mathbf{u}} = -\lambda\, \tilde{\mathbf{u}}
$$

where $`\mathbf{M}`$ is the $`3N \times 3N`$ diagonal mass matrix. Seeking solutions of the form $`\tilde{\mathbf{u}}(t) = \mathbf{e}^{(k)} e^{i\omega_k t}`$ leads to the standard **eigenvalue problem**:

$$
\tilde{\mathbf{H}}\, \mathbf{e}^{(k)} = \lambda_k\, \mathbf{e}^{(k)}
$$

where $`\tilde{\mathbf{H}} = \mathbf{M}^{-1/2} \mathbf{H}\, \mathbf{M}^{-1/2}`$ is the mass-weighted Hessian. The eigenvalues $`\lambda_k \geq 0`$ are proportional to the squared angular frequencies $`\omega_k^2 = \lambda_k`$ , and the eigenvectors $`\mathbf{e}^{(k)}`$ (also called **normal mode vectors**) define the direction and pattern of collective atomic displacement in mode $`k`$.

Because the Hessian is real, symmetric, and positive semi-definite, it can always be diagonalized by an orthogonal transformation:

$$
\mathbf{H} = \mathbf{U}\, \boldsymbol{\Lambda}\, \mathbf{U}^T
$$

where $`\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_{3N})`$ and $`\mathbf{U}`$ is the matrix of column eigenvectors.

### Rigid-Body Modes and the Null Space

The Hessian of any translationally and rotationally invariant potential has exactly **six zero eigenvalues**, corresponding to three global translations and three global rotations. These are the "trivial" modes: they represent rigid-body motions of the entire molecule that cost no energy. Because the spring network is built around pairwise distances (which are invariant under rigid-body transformations), these six modes are guaranteed to have $`\lambda_k = 0`$ by construction.

In practice, numerical diagonalization yields six eigenvalues very close to — but not exactly — zero, due to floating-point arithmetic. These modes are identified and systematically excluded from all physical analyses. The first **non-trivial** mode is mode 7, corresponding to the lowest-frequency collective internal motion, typically involving the largest-amplitude domain movements. Modes are ordered by increasing frequency: low-frequency modes are large-scale and collective, while high-frequency modes are localized and stiff.

### Equipartition and the Physical Meaning of Eigenvalues

Under the classical harmonic approximation, the **equipartition theorem** states that each normal mode carries an average thermal energy of $`\frac{1}{2}k_B T`$. The mean-square displacement amplitude of mode $`k`$ is therefore:

$$
\langle A_k^2 \rangle = \frac{k_B T}{\lambda_k}
$$

This has a profound implication: **low-frequency modes (small $`\lambda_k`$) contribute large-amplitude fluctuations**, while high-frequency modes (large $`\lambda_k`$) contribute small fluctuations. The total thermal fluctuation of the system is dominated by the handful of lowest-frequency modes, which is why ENM-based analyses of fluctuations and correlations are already quite accurate using only the first 10–20 non-trivial modes.

Furthermore, the **inverse of the Hessian** (its pseudo-inverse, excluding the null space) defines the **covariance matrix** of atomic displacements at thermal equilibrium:

$$
\langle u_{i\alpha}\, u_{j\beta} \rangle = k_B T\, [\mathbf{H}^+]_{i\alpha, j\beta}
$$

where $`\mathbf{H}^+`$ is the Moore-Penrose pseudo-inverse. This relationship is the foundation for the RMSF and DCCM calculations available via `pyadmd enm` (see [ENM (Standalone Normal Mode Analysis)](#enm-standalone-normal-mode-analysis)).

## Uniform Normal Modes Combination

The program generates uniformly distributed excitation vectors through a geometry-aware repulsion-based algorithm. The approach builds on a physics-inspired framework where points behave as charged particles confined to a spherical manifold, interacting through a dimensionally-scaled potential function. Unlike naive implementations that operate in an abstract factor space, the algorithm here accounts for the true geometry of the normal mode subspace before distributing the points, guaranteeing that the resulting excitation vectors are genuinely equidistant in the physical Cartesian space that governs the molecular dynamics.

The *[PDIM algorithm](https://github.com/antonielgomes/dpMDNM/tree/main/PDIM)* was the first implementation built for the same purpose; the design presented here extends that concept with a geometry-corrected basis and a faster, more concise implementation.

### Problem Definition

Let $`\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N \in \mathbb{R}^{3n}`$ be the $`N`$ normal mode vectors selected for excitation, where $`n`$ is the number of selected atoms. Each $`\mathbf{v}_k`$ is a flattened Cartesian displacement vector of length $`3n`$. The set of all normalized linear combinations of these vectors defines an $`N`$-dimensional subspace of $`\mathbb{R}^{3n}`$:

$$
\mathcal{V} = \mathrm{span}\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N\}
$$

Given an integer $`P>0`$, we seek to generate $`P`$ unit vectors $`\{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_P\} \subset \mathcal{V}`$ that maximize the minimal pairwise angular separation:

$$
\max_{{\mathbf{q}_i}} \min_{i \neq j} \|\mathbf{q}_i - \mathbf{q}_j\|
$$

This corresponds to finding an optimal spherical code on the unit hypersphere $`S^{N-1}`$ embedded within $`\mathcal{V}`$, with minimal angular separation between any two excitation directions.

### Mode Subspace Geometry

A naive approach would place the $`N`$ modes on coordinate axes and treat the combination coefficients $`c_k`$ directly as coordinates on $`S^{N-1}`$. This is geometrically valid only if the mode vectors satisfy $`\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \delta_{ij}`$, *i.e.*, they are orthonormal in plain Cartesian space. In practice this condition is not met. ENM mode vectors are orthogonal in the mass-weighted inner product but carry non-uniform amplitudes in Cartesian coordinates, with low-frequency modes typically exhibiting larger displacements than high-frequency ones. CHARMM normal modes may additionally lose orthogonality when projected onto an atomic subset. As a consequence, equal angular spacing of coefficient vectors in abstract factor space does not translate into equal angular spacing of the physical excitation vectors $`\mathbf{q}_i`$.

To correct for this, the algorithm explicitly constructs an orthonormal basis
for $`\mathcal{V}`$ using QR decomposition before running the repulsion algorithm.

### Orthonormal Basis Construction

Assemble the mode matrix $`\mathbf{M}_{\mathrm{nm}} \in \mathbb{R}^{N \times 3n}`$ whose rows are the flattened mode vectors:

$$
\mathbf{M}_{\mathrm{nm}} =
\begin{pmatrix}
— & \mathbf{v}_1^\top & — \\
& \vdots & \\
— & \mathbf{v}_N^\top & —
\end{pmatrix}
$$

Apply QR decomposition to $`\mathbf{M}_{\mathrm{nm}}^\top \in \mathbb{R}^{3n \times N}`$:

$$
\mathbf{M}_{\mathrm{nm}}^\top = \mathbf{Q}_{\mathrm{qr}}\ \mathbf{R}
$$

where $`\mathbf{Q}_{\mathrm{qr}} \in \mathbb{R}^{3n \times N}`$ has orthonormal columns and $`\mathbf{R} \in \mathbb{R}^{N \times N}`$ is upper triangular. Setting $`\mathbf{Q} = \mathbf{Q}_{\mathrm{qr}}^\top \in \mathbb{R}^{N \times 3n}`$ yields an orthonormal row basis for $`\mathcal{V}`$:

$$
\langle \mathbf{Q}_{i,:}, \mathbf{Q}_{j,:} \rangle = \delta_{ij}
$$

A point $`\mathbf{x} \in \mathbb{R}^N`$ on the unit hypersphere $`S^{N-1}`$ maps to a unit physical vector via $`\mathbf{q} = \mathbf{x}\mathbf{Q} \in \mathbb{R}^{3n}`$, and because $`\mathbf{Q}`$ is an isometric embedding, inner products are preserved:

$$
\langle \mathbf{x}_i \mathbf{Q}, \mathbf{x}_j \mathbf{Q} \rangle = \langle \mathbf{x}_i, \mathbf{x}_j \rangle
$$

The repulsion algorithm therefore operates on $`N`$-dimensional coordinates $`\mathbf{x} \in \mathbb{R}^N`$ whose geometry is faithful to the physical mode subspace, at no additional cost relative to working in abstract factor space.

### Energy Minimization Framework

The distribution problem is cast as an energy minimization over $`P`$ points $`\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_P\} \subset S^{N-1}`$:

$$
E = \sum_{i=1}^{P} \sum_{\substack{j=1 \\ j \neq i}}^{P} U \left(\|\mathbf{x}_i - \mathbf{x}_j\|\right)
$$

### Dimensionally-Scaled Potential Function

The repulsive potential uses an inverse power law scaled by the dimensionality of the space:

$$
U(r) = \frac{1}{r^{k}}, \quad k = N - 1
$$

The exponent $`k = N-1`$ is chosen for three reasons. First, the fundamental solution to Laplace's equation in $`N`$ dimensions scales as $`1/r^{N-2}`$, and the gradient of that solution scales as $`1/r^{N-1}`$, making this the natural repulsive force law in $`N`$-dimensional space. Second, the surface area of $`S^{N-1}`$ grows as $`({2\pi e}/{N})^{N/2}`$, so stronger repulsion in higher dimensions is required to counteract the concentration-of-measure effect that causes random points to cluster near the equator. Third, the exponent ensures numerical stability by preventing excessively large or small force values as $`N`$ varies.

### Gradient Flow on the Sphere

The repulsive force on point $`\mathbf{x}_i`$ arising from all other points is:

$$
\mathbf{f}_i = \sum_{\substack{j=1 \\ j \neq i}}^{P}
\frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|^{N+1}}
$$

Since the points must remain on $`S^{N-1}`$, the gradient is projected onto the tangent plane at $`\mathbf{x}_i`$ to eliminate any radial component:

$$
\tilde{\mathbf{f}}_i = \mathbf{f}_i - \left(\mathbf{f}_i \cdot \mathbf{x}_i\right) \mathbf{x}_i
$$

Each iteration updates the points by a fixed step $`\eta = 0.001`$ along their
tangent-plane forces and then renormalizes back onto $`S^{N-1}`$:

$$
\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i + \eta\,\tilde{\mathbf{f}}_i}
{\|\mathbf{x}_i + \eta\,\tilde{\mathbf{f}}_i\|}
$$

Convergence is declared when $`\max_i \|\mathbf{f}_i\|_\infty < 10^{-6}`$, and the iteration is terminated early when the maximum force changes by less than $`10^{-7}`$ over five consecutive iterations (stagnation criterion).

### Special Case: Cross-Polytope Initialization

When $`P=2N`$, the $`2N`$ vertices of the cross-polytope:

$$
\{\pm\mathbf{e}_1, \pm\mathbf{e}_2, \dots, \pm\mathbf{e}_N\}
$$

provide an analytically optimal initialization in $`\mathbf{Q}`$-coordinates and the repulsion loop is skipped. These vertices are already maximally separated on $`S^{N-1}`$ under the symmetry of the cross-polytope, yielding pairwise angular separations of either $`90°`$ or $`180°`$.

### Normal Modes Linear Combination

Once the coordinates $`\{\mathbf{x}_1, \dots, \mathbf{x}_P\}`$ have converged on $`S^{N-1}`$, each is mapped back to a physical excitation vector via the orthonormal basis $`\mathbf{Q}`$:

$$
\mathbf{q}_i = \mathbf{x}_i \mathbf{Q} \in \mathbb{R}^{3n}, \qquad \|\mathbf{q}_i\| = 1
$$

The equivalent scalar combination factors $`\{c_{i,k}\}`$ in terms of the original (non-orthonormal) mode vectors — useful for logging and interpretability — are recovered by least-squares projection:

$$
\mathbf{c}_i = \mathbf{q}_i \, \mathbf{M}_{\mathrm{nm}}^+ \in \mathbb{R}^N
$$

where $`\mathbf{M}_{\mathrm{nm}}^+`$ denotes the Moore–Penrose pseudoinverse of $`\mathbf{M}_{\mathrm{nm}}`$. These approximate coefficients are written to the `factors.csv` output file for reference but do not influence the simulation; the physical vectors $`\mathbf{q}_i`$ are used directly as excitation directions.

**Reproducibility:** the only stochastic step in this procedure is the initial placement of the $`P`$ points on $`S^{N-1}`$ before the repulsion loop runs (skipped entirely when $`P=2N`$, see above). This is controlled by a seeded RNG (`-seed`/`--seed`, default `42`), so `pyadmd run` calls with identical arguments produce identical excitation vectors and `factors.csv` — see [Run Parameters](#parameters).

**Note:** ENM recomputation under `--recalc` (see [Excitation Direction Update](#excitation-direction-update)) draws a *new* random combination each time it fires and is intentionally **not** covered by `--seed`, since its purpose is to re-diversify the excitation direction mid-simulation.

## Kinetic Energy Control
The additional kinetic energy injected in the system has a fast dissipation rate. Therefore, the program constantly checks the injection energy level and rescale the velocities along the excited direction whenever it is necessary. The kinetic energy along the normalized excitation vector $`\mathbf{Q}`$ direction is calculated by projecting first the current velocities to the excitation direction $`\mathbf{Q}`$ as $`\mathbf{V}_p = (\mathbf{V}_{curr} \cdot \mathbf{Q}) \cdot \mathbf{Q}`$, where $`\mathbf{V}_{p}`$ and $`\mathbf{V}_{curr}`$ the $`3N`$-dimensional vectors of the projected and current atomic velocities, respectively. The kinetic energy along the excitation direction is thus given by:

$$
E_k = \frac{1}{2} \mathbf{V}_{p}^T \mathbf{M}\ \mathbf{V}_p
$$

where $`\mathbf{M}`$ is the diagonal mass matrix. At the beginning of each short simulation interval, the remaining excitation energy ($`E_k`$) is adjusted to the desired excitation level ($`E_{exc}`$) by modifying the atomic velocities so $`\mathbf{V}_{new} = \mathbf{V}_{curr} + (\mathbf{V}_{exc} - \mathbf{V}_{p})`$.

With this procedure, the system is kept in a continuous excited state, allowing an effective small, "adiabatic-like" energy injection. The energy injection control is done by projecting the velocities computed during the simulation onto the excited vector, thus obtaining and rescaling the kinetic energy corresponding to it.

## Excitation Direction Update
Since the excitation vector is obtained from the initial conformation, it is dependent of this configuration. As the system is displaced along this direction and change its conformation, the motion loses its directionality due to mainly anharmonic effects. To prevent the structural distortions produced by the displacement along a vector that is no longer valid, the program update the excitation directions based on the trajectory evolution during the previous excitation steps. This procedure allows the system to adaptively find a relaxed path to follow during the next aMDeNM excitations.

If we consider the $`n^{th}`$ simulation, the next excitation vector, $`\mathbf{Q}_{n+1}`$, is determined based on specific parameter values obtained along the trajectory followed in the $`\mathbf{Q}_n`$ direction. A new excitation vector is defined based on two parameters: the first relates to the effective displacement $`\ell`$ along $`\mathbf{Q}_n`$ during the $`n^{th}`$ excited dynamics by projecting the mass-weighted effective displacement vector $`\mathbf{d}_n = \mathbf{M}^{1/2} ({\langle \mathbf{r} \rangle}_n - \mathbf{r}_n^0)`$ onto the normalized mass-weighted excitation vector $`\mathbf{Q}_n`$, where the $`{\langle \mathbf{r} \rangle}_n`$ is the average position of the structures over the last $`0.2~ps`$ obtained in the $`n^{th}`$ excitation, and $`\mathbf{r}_n^0`$ is the starting position for the following simulation.

The second parameter relates to the relative deviation of the vector $`\mathbf{d}_n`$ with respect to vector $`\mathbf{Q}_n`$, evaluated by the angle $`\alpha_n`$ between them. More precisely, we consider the $`{\cos \alpha}_n`$ obtained by taking the scalar product of these vectors after normalizing $`\mathbf{d}_n`$, as following:

$$
{\cos \alpha}_n = \frac {\mathbf{d}_n \mathbf{Q}_n} {\|\mathbf{d}_n\|}
$$

A precise rule is followed to decide whether to modify the excitation vector direction after every short simulation run. The excitation vector is changed as soon as $`\ell_n`$, the displacement along $`\mathbf{Q}_n`$, is larger than a threshold value $`\ell_c`$, and when $`\cos {\alpha}`$ is lower than a threshold value $`{\cos \alpha}_c`$. The conditions for choosing the excitation vector between $`\mathbf{d}_n`$ and $`\mathbf{Q}_n`$ for the next simulation are defined by:

$$
\mathbf{Q}_{n+1} = \begin{cases}
  \mathbf{Q}_n, & \text{if } \ell \leq \ell_c \\
  \mathbf{Q}_n, & \text{if } \ell \gt \ell_c \wedge \cos \alpha \ge  {\cos \alpha}_c \\
  \mathbf{d}_n, & \text{if } \ell \gt \ell_c \wedge \cos \alpha \lt {\cos \alpha}_c
\end{cases}
$$

The default value for $`\ell_c`$ is $`0.5 m^{1/2} Å`$ (with $`m`$ being atomic mass unit), and for $`\alpha`$ is $`60°`$.

**Note on `--recalc` and reproducibility:** when `--recalc` is set, reaching this threshold triggers a full ENM recomputation from the current structure (`SimulationRunner._recompute_enm_modes`) followed by a *brand-new random* linear combination of the recomputed modes, rather than the deterministic displacement-based correction above. This random re-combination is independent of the `-seed`/`--seed` flag described in [Uniform Normal Modes Combination](#normal-modes-linear-combination) and is not currently reproducible run-to-run — by design, since its purpose is to re-diversify the excitation direction after the mode subspace itself has changed.

[Back to top ↩](#)
* ****

# pyAdMD Applications
The Adaptive MDeNM method takes self-computed Cα or heavy atoms ENM modes or CHARMM-computed normal modes as collective variables to improve Molecular Dynamics sampling.

## ENM
Uses simpified force-field based on particles and springs computed automatically by the program. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

## Physical force-field based normal modes
Uses physical force-field based normal modes computed in *[CHARMM](https://www.charmm.org/charmm/)*. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

# Configuration
**pyadmd** is distributed as an installable Python package that computes ENM modes, uniformly distributes linear combinations of modes in the $`N`$-dimensional hypersphere space, manages the OpenMM-based simulations, and computes the projections along the excitation direction, applying corrections whenever necessary. Its bundled data includes the CHARMM driver script used to write down CHARMM-computed normal modes.

One can easily setup and run an Adaptive MDeNM simulation using pyadmd.
The configuration process is straightforward. Some technical aspects will be covered in this section in order to facilitate the method comprehension.

## Normal Modes Selection
Choosing which normal mode to include in the `pyadmd` calculation is crucial for the expected result. We strongly recommend first calculating the ENM modes (`pyadmd enm`), examining them carefully, and then proceeding with the workflow using the `pyadmd run` command. From an exploratory perspective, the collectivity data derived from ENM calculations can provide insights into the importance or dominance of each normal mode within the system's overall dynamics. However, there are phenomena where a specific mode (or set of modes) is better suited to explain the particular conformational transitions involved.

## Energy injection
The excitation time of Adaptive MDeNM is $`0.2~ps`$. This means that every $`0.2~ps`$ the system receives the additional amount of energy defined by the user. Therefore, when studying large scale motions, it is advised to inject small amounts of energy in order to avoid structural distortions caused by an excessive energy injection. Usually, an excitation energy of $`2~kcal/mol`$ is sufficient to achieve a large exploration of the conformational space ($`10~kcal/mol`$ if Cα-only ENM).

## Simulation time
The total simulation time may require a tuning depending on the system size, energy injection and nature of the motion being excited. Considering a large scale global motion, there is a trade-off between the energy injection and the total simulation time. Larger amounts of energy allows a shorter simulation time, however, this may not be advised as discussed above.

## Excitation direction update
As described above, the direction is updated after the system has traveled a distance of $`0.5~Å`$ along the excitation vector and its real displacement has a deviation of $`60°`$ with respect to the theoretical one. The update can also be affected by the amount of energy injected, since higher energy values leads to larger motions. In addition, after each correction the new vector loses directionality due to anharmonic effects. This means that, at a given point, the new vectors are so diffuse that there is no point in proceed the simulation. When this ponit is reached, it is necessary to recompute the normal modes and start again. This is one more reason to not inject high energy values and let the system undergoes the changes slowly.
Alternatively, one can recompute ENM modes instead of change the excitation vector direction (only when the original model type is ENM).

## Number of modes and replicas
The program do a linear combination of the supplied normal modes to compute the excitation direction. This imply that the more modes are provided, the more replicas will be necessary to cover the hyperspace described by these modes.

## Atom selection
Create an atom selection to apply the energy injection using *[MDAnalysis selection language](https://userguide.mdanalysis.org/1.1.1/selections.html)*. Must be written between quotes. This same selection (`-sel`/`--selection` at `run` time) is later reused by `pyadmd analyze` as the default scope for most structural metrics — see [Analysis Selection Scope](#analysis-selection-scope).

# Input Requirements

## Run
### Parameters
- **`-src`/`--source`**: Input engine type, **`NAMD`** (binary `.coor`/`.vel`/`.xsc`) or **`OPENMM`** (XML restart `.rst`) (**required**)

- **`-m`/`--model`**: Normal modes model type, **`CA`** ENM, **`HEAVY`** Atoms ENM or **`CHARMM`** (**required**. Default: **`CA`**)

- **`-nm`/`--modes`**: Normal modes to excite (**optional**. Default: **`7,8,9`**)

- **`-ek`/`--energy`**: Excitation energy injection (**optional**. Default: **`2`** kcal/mol)

- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

- **`-sel`/`--selection`**: Atom selection to apply the energy injection (**optional**. Default: **`"protein"`**). Also becomes the default scope for most `pyadmd analyze` structural metrics — see [Analysis Selection Scope](#analysis-selection-scope).

- **`-rep`/`--replicas`**: Number of replicas to run (**optional**. Default: **`10`**)

- **`-seed`/`--seed`**: Random seed for the uniform mode-combination generation (the repulsion-algorithm initialization described in [Uniform Normal Modes Combination](#uniform-normal-modes-combination)) (**optional**. Default: **`42`**). Fixing this makes `run` reproducible: identical CLI arguments always produce identical excitation vectors and `factors.csv`. Pass a different value to obtain an independent replicate ensemble. Does **not** affect the `--recalc` mid-simulation re-excitation, which remains stochastic by design — see [Excitation Direction Update](#excitation-direction-update).

### Files
`pyadmd run` automatically creates the `inputs/` directory in the current
working directory (if it doesn't already exist) and copies every file listed
below into it.

- **`-psf`/`--psffile`**: PSF structure file containing system molecule-specific information (**required**)

- **`-pdb`/`--pdbfile`**: PDB structure file in Protein Data Bank format (**required**)

- **`-mod`/`--modefile`**: Binary file containg CHARMM normal mode vectors (**optional**. Required only if **`-m = CHARMM`**)

**NAMD input files** (**required when `-src NAMD`**):
- **`-coor`/`--coorfile`**: NAMD binary coordinates file (`.coor`)
- **`-vel`/`--velfile`**: NAMD binary velocities file (`.vel`)
- **`-xsc`/`--xscfile`**: NAMD eXtended System Configuration file (`.xsc`)
- **`-str`/`--strfile`**: CHARMM-style stream file, containing box info and force field parameters/definitions (**required** in NAMD mode; **optional** in OpenMM mode)

**OpenMM input files** (**required when `-src OPENMM`**):
- **`-rst`/`--rstfile`**: OpenMM XML restart file (`.rst`), written via `XmlSerializer.serialize(state)` from a state built with `getPositions=True, getVelocities=True`

### Feature Flags
- **`-n`/`--no-correc`**: Disable excitation vector direction correction and compute standard MDeNM

- **`-f`/`--fixed`**: Disable excitation vector correction and keep constant excitation energy injections

- **`-r`/`--recalc`**: Recompute ENM modes instead of correcting the excitation vector direction. **Note:** the new mode combination generated after each recomputation is drawn from a fresh random unit vector and is not controlled by `-seed`/`--seed` — see [Excitation Direction Update](#excitation-direction-update).

- **`--full_ener`**: Write per-term energy decomposition (BOND, ANGLE, DIHED, IMPRP, CMAP, UBREY, NBFIX, NONBONDED, etc.) to `rep{N}_ener_decomp.log` every cycle

## Append
### Parameters
- **`-t`/`--time`**: Simulation time to append, in ps (**required**)

## Free Energy
All parameters are optional; the `fel` subcommand reads its input trajectories and reference state from files already produced by `run`/`restart`/`append`, so no additional files need to be supplied.

- **`-c`/`--cutoff`**: GROMOS RMSD clustering cutoff, in Å (**optional**. Default: **`1.0`**)
- **`-d`/`--deexcite`**: Total restrained de-excitation MD length per centroid, in ps, split evenly over 4 restraint phases (**optional**. Default: **`200`**)
- **`-p`/`--production`**: Unrestrained production MD length per centroid, in ps (**optional**. Default: **`800`**)
- **`-nm`/`--modes`**: Comma-separated mode indices to project for the FEL (**optional**. Default: same modes used in `run`, *e.g.* `7,8,9`)
- **`--modes-2d`**: Mode pairs for 2D FEL plots, as space-separated `"m1,m2"` tokens (**optional**. Default: all pairwise combinations of --modes, *e.g.*  "7,8 7,9 8,9")
- **`-b`/`--bins`**: Number of histogram bins used for the FEL (**optional**. Default: **`100`**)
- **`-T`/`--temp`**: Temperature for k<sub>B</sub>T scaling and the production ensemble, in K (**optional**. Default: **`303.15`**)
- **`-s`/`--sel`**: MDAnalysis selection string used for GROMOS RMSD clustering (**optional**. Default: **`"protein and name CA"`**)
- **`--max-centroids`**: Maximum number of centroids submitted to MD. When the cluster count exceeds this value, exactly this many centroids are selected by greedy farthest-point (MaxMin) sampling to maximize conformational diversity (**optional**. Default: **`50`**)

**Note:** `-s`/`--sel` and `-T`/`--temp` must stay the same across repeated `fel` calls on the same simulation — see [Extending a Previous Free Energy Calculation](#extending-a-previous-free-energy-calculation). This `-s`/`--sel` selection is independent of `run`'s `-sel`/`--selection`: it controls GROMOS clustering only, while `run`'s selection controls both energy injection and (by default) the scope of most `pyadmd analyze` metrics — see [Analysis Selection Scope](#analysis-selection-scope).

## Analysis
### Parameters
- **`-src`/`--source`**: Trajectory source to analyze, **`pyadmd`** for `rep{N}.dcd` replica trajectories or **`fel`** for `centroid_frame{F}.dcd` production trajectories  (**optional**. Default: **`pyadmd`**)

### Feature Flags
- **`-r`/`--rough`**: Perform rough analysis (**optional**. Analyze every **`5`** ps instead of every frame.)

### Skip Flags
Each analysis step can be independently disabled. When skipped, that metric will not appear in any CSV, plot, or HTML summary output.

- **`--no-rmsd`**: Skip RMSD calculation
- **`--no-rg`**: Skip radius of gyration calculation
- **`--no-sasa`**: Skip SASA and hydrophobic exposure calculation
- **`--no-rmsf`**: Skip  RMSF calculation
- **`--no-dssp`**: Skip secondary structure analysis via DSSP
- **`--no-dccm`**: Skip DCCM (dynamic cross-correlation matrix) calculation
- **`--no-lmi`**: Skip LMI (Linear Mutual Information) calculation

**Note:** Before analysis, the program checks if `pyadmd` or `fel` calls are properly completed. If any unit (pyAdMD replica or free energy centroid) hasn't finished running, `analyze` prints a warning listing the incomplete units and their cycles completed/target, but proceeds anyway — see [Handling Incomplete Units](#handling-incomplete-units) below.

## ENM
### Parameters
- **`-i`/`--input`**: Input PDB file (**required**, unless **`-w`/`--write-modes`** is used)

- **`-o`/`--output`**: Output folder name (**optional**. Default: **`output`**)

- **`-m`/`--model`**: Model type, **`CA`** (Cα-only) or **`HEAVY`** (heavy atoms) (**optional**. Default: **`CA`**). Unlike `run`'s `-m`/`--model`, **`CHARMM`** is not a valid choice here — `enm` only computes ENM normal modes, not CHARMM-derived ones.

- **`-sel`/`--selection`**: Atom selection applied to the input PDB before building the ENM (**optional**. Default: **`"protein"`**). Must be written between quotes if it contains spaces, per [MDAnalysis selection language](https://userguide.mdanalysis.org/1.1.1/selections.html).

- **`-c`/`--cutoff`**: Interaction cutoff distance, in Å (**optional**. Default: **`15.0`** for CA, **`12.0`** for HEAVY)

- **`-k`/`--spring-constant`**: ENM harmonic spring constant, in kcal/mol/Å² (**optional**. Default: **`1.0`**)

- **`--max-modes`**: Number of non-rigid-body vibrational modes to compute (**optional**. Default: **`50`**)

- **`--output-modes`**: Number of modes (file-labeled 1 through N, where label 1 is the first non-rigid mode) to write vectors/trajectories for (**optional**. Default: **`10`**)

### Skip Flags
Collectivity, contributions, RMSF, DCCM, and mode vector/trajectory writing can each be independently disabled:

- **`--no-vec`**: Skip writing mode vectors (`.xyz`)
- **`--no-trj`**: Skip writing mode trajectories (`_traj.pdb`)
- **`--no-collectivity`**: Skip mode collectivity calculation
- **`--no-contributions`**: Skip the variance-contributions plot
- **`--no-rmsf`**: Skip the NMA-predicted RMSF plot (analytical, from the harmonic approximation — not derived from an MD trajectory; see [Analysis](#analysis-1) for the trajectory-based RMSF computed elsewhere in the package)
- **`--no-dccm`**: Skip the NMA-predicted DCCM plot (same analytical distinction as RMSF above)
- **`--no-gpu`**: Disable GPU acceleration

### Post-hoc Mode Re-writer
- **`-w`/`--write-modes`**: Write mode vectors/trajectories from a previously completed `enm` run's saved `*_modes.npy`/`*_frequencies.npy`/structure PDB, without recomputing the ENM. Accepts comma-separated integers and inclusive ranges (`start:end`), *e.g.* `"26,41"`, `"7:10"`, `"42,44:50"`. Requires **`-o`/`--output`** pointing to an existing `enm` output directory.

**Note on mode-file resolution:** `pyadmd enm`'s mode vector/trajectory files (`_mode_{N}.xyz`/`_mode_{N}_traj.pdb`) are written at the ENM's **native reduced resolution** (Cα-only or heavy-atom only, matching whatever the modes were computed on).

## Output Structure
### Directory Organization
```
enm_output/
├── {base_name}_{model}_structure.pdb      # reduced‑resolution structure (Cα or heavy atoms)
├── {base_name}_{model}_frequencies.npy    # vibrational frequencies (filtered, non‑rigid modes)
├── {base_name}_{model}_modes.npy          # eigenvector matrix (filtered, non‑rigid modes)
├── collectivity.csv                       # per‑mode collectivity (κ) and frequency (cm⁻¹) (omitted with --no-collectivity)
├── mode_contributions.png                 # per‑mode and cumulative variance contributions (omitted with --no-contributions)
├── rmsf_plot.png                          # NMA‑predicted residue RMSF (harmonic approximation) (omitted with --no-rmsf)
├── dccm_plot.png                          # NMA‑predicted residue cross‑correlation matrix (omitted with --no-dccm)
├── dccm_matrix.npy                        # raw NMA‑DCCM matrix (omitted with --no-dccm)
├── {base_name}_{model}_mode_{N}.xyz       # displacement vector of mode N (XYZ format) (omitted with --no-vec)
└── {base_name}_{model}_mode_{N}_traj.pdb  # oscillatory PDB trajectory along mode N (multi‑model) (omitted with --no-trj)
```


**Notes:**  
- `{base_name}` is the stem of the input PDB file (e.g., `system`).
- `{model}` is either `ca` (Cα‑only) or `heavy` (heavy atoms).
- The mode vector and trajectory files are written only for the modes specified by `‑‑output-modes` (default: first 10 non‑rigid modes).
- The `-w` / `‑‑write-modes` option re‑uses an existing output directory to write **additional** mode files (vectors/trajectories) without recomputing the ENM.

### Output Files Description

1. **Core Data Files**  
   - **`{base_name}_{model}_structure.pdb`**: PDB file containing only the atoms that participated in the ENM (Cα or heavy atoms). Used as the reference for mode vector/trajectory writing and for RMSF/DCCM plotting.
   - **`{base_name}_{model}_frequencies.npy`**: 1D array of vibrational frequencies (in internal angular units, sorted ascending) after removing the six rigid‑body modes.
   - **`{base_name}_{model}_modes.npy`**: 2D array of shape `(3N, M)`, where column `i` is the mass‑weighted eigenvector for mode `i` (matching the order of `frequencies`). These two NumPy files enable fast post‑hoc re‑writing of vectors/trajectories via `-w`.

2. **Collectivity and Variance Contributions**  
   - **`collectivity.csv`**: CSV with columns `Mode`, `Frequency (cm⁻¹)`, and `Collectivity`. Omitted with `‑‑no-collectivity`.
   - **`mode_contributions.png`**: Two‑panel figure showing (left) the proportion of total mean‑square fluctuation contributed by each of the first `‑‑max-modes` non‑rigid modes (proportional to `1/λ_k` under equipartition), and (right) the cumulative fraction. Omitted with `‑‑no-contributions`.

3. **NMA‑Predicted RMSF and DCCM**  
   - **`rmsf_plot.png`**: Residue‑averaged root‑mean‑square fluctuation (Å) derived from the harmonic approximation. The plot is based on the sum over modes of `(kBT/λ_k) * |u_i^(k)|² / m_i`. Omitted with `‑‑no-rmsf`.
   - **`dccm_plot.png`**: DCCM heatmap, diverging colormap (red = fully correlated, white = uncorrelated, blue = fully anti-correlated).
   - **`dccm_matrix.npy`**: Raw correlation matrix, saved alongside the plot. Both are omitted with `‑‑no-dccm`.

4. **Mode‑Specific Vector and Trajectory Files**  
   - **`{base_name}_{model}_mode_{N}.xyz`**: XYZ‑formatted file listing the displacement vector for mode `N`. The header includes the mode frequency in cm⁻¹. Omitted with `‑‑no-vec`.
   - **`{base_name}_{model}_mode_{N}_traj.pdb`**: Multi‑model PDB showing a smooth oscillation along mode `N`. The trajectory is mass‑weighted and scaled to a peak amplitude (default 4 Å). Omitted with `‑‑no-trj`.

**Note:** When using the post‑hoc mode re‑writer (`pyadmd enm -w "..." -o enm_output`), only the mode‑specific vector and trajectory files are newly written for the requested modes; all other files (core data, collectivity, plots) are left untouched and must already exist from a previous full ENM run.

[Back to top ↩](#)
* ****

# Free Energy Landscape
The **`fel`** subcommand computes a free energy landscape (FEL) from a completed set of aMDeNM replicas, following the two-stage protocol of [Costa *et al.*](https://doi.org/10.1021/acs.jctc.5b00003).

## Method Overview
1. **Merge trajectories**: all `rep*.dcd` replica trajectories are concatenated into a single pseudo-trajectory.
2. **GROMOS clustering**: frames are clustered by Cα RMSD (`-s`/`-c`); when the number of clusters exceeds `--max-centroids`, a maximally diverse subset is selected via greedy farthest-point (MaxMin) sampling on the cluster centroids.
3. **Centroid MD**: each centroid undergoes a 4-phase NVT restrained
   de-excitation (`-d`, progressively decreasing positional restraints on
   backbone and sidechain heavy atoms) followed by unrestrained NPT production
   MD (`-p`). Each de-excitation phase is further split into a
   30%/20%/30%/20% pattern of nominal restraint / brief relief dip, where the dip targets a gentler restraint level rather than the nominal one, periodically releasing local strain due to the force constraints. If a centroid's de-excitation fails, it is automatically retried using an alternative member frame from the same cluster (up to 4 substitutes), and if all of those also fail, retried once more with a reinforced integrator ($`1 fs`$ timestep, $`5 ps^{-1}`$ friction, applied to de-excitation only) before the centroid is finally marked failed and excluded from the FEL.
4. **Mode projection**: every production frame is projected onto each individual normal mode vector as a signed mass-weighted RMS displacement.
5. **FEL computation**: a population histogram (`-b` bins) is converted to $`\Delta G`$ via $`\Delta G = -k_{BT} \cdot ln[P(q)/P_{max}]`$, computed independently per mode (1D) and for user-specified mode pairs (2D, `--modes-2d`).

## Extending a Previous Free Energy Calculation
`fel` can be re-invoked on the same simulation with a larger `--max-centroids` and/or longer `-p`/`--production` to extend an earlier calculation, rather than starting over:

- **Free to change**: `-c`/`--cutoff` and `-d`/`--deexcite`. Changing the cutoff only affects the re-thresholding of the cached pairwise-RMSD matrix. Changing the de-excitation length only affects newly-created centroids going forward; existing centroids keep whatever de-excitation they originally had and are simply extended in production.
- **Must stay the same**: `-s`/`--sel`, `-T`/`--temp`. Mixing clustering selections or temperatures inside one pooled FEL is not physically valid.
- **Never shrinks existing work**: if `--max-centroids` or `-p`/`--production` is *smaller* than the previous call, the program warns and uses the larger of the two values instead. We suggest start with smaller values and append more data, if necessary.

## Output Structure
### Directory Organization
```
fel/
├── run_metadata.json                       # parameters used (gates append behavior)
├── clustering_rmsd_cache.npz               # cached pairwise-RMSD matrix (reused across calls)
├── clustering_rmsd_cache.json              # cache validity metadata (selection, frame count, stride)
├── clustering_summary.csv                  # per-cluster frame index, size, and production status
├── projections_mode[N].npy                 # raw mode projections (Å)
├── fel_mode[N].csv                         # 1D FEL data (coordinate, ΔG)
├── fel_mode[N]_plot.png                    # 1D FEL plot
├── fel_2d_mode[N]_mode[M].png              # 2D FEL plot for a mode pair
├── fel_summary.html                        # HTML summary report
└── centroids/
    └── centroid_frame[F]/                  # one directory per centroid, named by frame index
        ├── centroid_[F].dcd                # production trajectory (appended to on extension)
        ├── centroid_[F].log                # production logfile (appended to on extension)
        ├── prod_checkpoint.chk             # exact final state, for bit-identical extension
        └── checkpoint.chk                  # periodic (every 10 cycles) checkpoint
```

## Output Files Description
1. **Cache Files** 
- **`run_metadata.json`**: the clustering selection, temperature, cutoff, de-excitation length, `max-centroids`, and production length used.
- **`clustering_rmsd_cache.npz`/`.json`**: the pairwise-RMSD matrix over subsampled frames.
- **`clustering_summary.csv`**: summary containing cluster ID, frame index, cluster size, status this run (`fresh`/`extended`/`skipped`/`failed`, annotated with `substitute frame {N}` and/or `reinforced` when a centroid needed those fallbacks), `source_frame_used` (the frame whose coordinates actually produced a successful run — equal to `centroid_frame` unless a substitute member was used), `md_attempts` (total attempts across the standard and reinforced passes), and cycles/ps completed. A `status` of `failed` means every attempt (original frame + substitutes, standard + reinforced settings) failed; that centroid is excluded from the FEL rather than blocking the run.
2. **Plot Files** 
- **`fel_mode[N].csv`/`fel_mode[N]_plot.png`**: 1D free energy landscape per mode, in Å and kcal/mol.
- **`fel_2d_mode[N]_mode[M].png`**: 2D free energy landscape for a mode pair.
3. **HTML Summary**
-  **`fel_summary.html`**: interactive summary with protocol parameters, per-mode FEL statistics, per-centroid production status, and embedded plots.

[Back to top ↩](#)
* ****

# Analysis
The PyAdMD **`analysis`** module provides comprehensive analysis capabilities for molecular dynamics simulations performed using the aMDeNM method. This module processes simulation trajectories and generates detailed structural analysis, visualizations, and summary reports. It can analyze either the aMDeNM replica trajectories from `run`/`restart`/`append` (`-src pyadmd`, default) or the centroid production trajectories from a completed `fel` run (`-src fel`) — see [Trajectory Source](#trajectory-source).

## Basic Structural Properties Calculated

1. **Root Mean Square Deviation (RMSD):**  Measures structural deviation from the initial conformation.

2. **Radius of Gyration (RoG):** Measures the compactness of the protein structure. Useful for identifying folding/unfolding events.

3. **Solvent Accessible Surface Area (SASA):** Calculates the surface area accessible to solvent molecules.

4. **Hydrophobic Exposure:** Solvent-accessible surface area (SASA, Å²) contributed by hydrophobic residues (ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO). Useful for identifying folding/unfolding events: buried hydrophobic patches becoming solvent-exposed (or vice versa) is a hallmark of such transitions.

5. **Root Mean Square Fluctuation (RMSF):** Calculates per-residue flexibility using Cα atoms. Identifies flexible and rigid regions in the protein structure.

6. **Secondary Structure Content:** Calculates secondary structure elements using DSSP. Tracks helix, sheet, coil, turn, and other structural elements over time and reports the number of residues in each secondary structure type.

7. **Dynamic Cross-Correlation Matrix (DCCM):** Measures pairwise linear correlation of Cα residue motions after Kabsch superposition to remove rigid-body rotation/translation. Values range from +1 (fully correlated motion) through 0 (uncorrelated) to −1 (fully anti-correlated motion), useful for identifying coupled domains, allosteric communication paths, and correlated/anti-correlated collective motions.

8. **Linear Mutual Information (LMI):** An alternative, signless measure of residue-residue coupling strength (range [0, 1]) computed via the Gaussian approximation of generalized correlation. Unlike DCCM, LMI reports strongly anti-correlated motion with the same high value as strongly correlated motion, since it measures total coupling rather than its direction.

## Analysis Modes
### Standard Analysis
- Analyzes every frame of the trajectory
- Provides the highest resolution data
- May be computationally intensive

### Rough Analysis
- Analyzes frames at *5ps* intervals
- Significantly reduces computation time
- Suitable for quick overviews or large systems

### Selective Analysis (Skip Flags)
Individual analyses can be disabled at the command line. This is useful when:
- DSSP is not installed (avoid exception error)
- Only a subset of metrics is needed (*e.g.* RMSD + RMSF only)
- Computation time needs to be minimized (SASA, DSSP, and LMI are the most expensive steps; LMI scales as O(n<sub>Cα</sub><sup>2</sup>) with a pairwise covariance computation)

## Trajectory Source
The `-src`/`--source` flag selects which set of trajectories to analyze:

- **`pyadmd`**: analyzes `rep{N}/rep{N}.dcd` replica trajectories, one analysis unit per replica. Output goes to `analysis/` (default).
- **`fel`**: analyzes `fel/centroids/centroid_frame{F}/prod.dcd` production trajectories, one analysis unit per centroid (identified by its merged-trajectory frame index, not a sequential number). Output goes to `analysis/fel/`, kept separate from `pyadmd`-sourced output. All computed metrics (RMSD, RoG, SASA, hydrophobic exposure, RMSF, secondary structure) and skip flags apply identically regardless of source.

In both modes, `analyze` proceeds even if some units haven't finished — see [Handling Incomplete Units](#handling-incomplete-units) below.

## Handling Incomplete Units
Before any analysis runs, the program checks whether every unit (replica or centroid) has finished running. Units that haven't are **not** excluded: they're analyzed using only the cycles they actually completed, with a correctly-scaled time axis reflecting their real elapsed simulation time rather than the run's target time. A console warning lists every incomplete unit (cycles completed/target), and the same information appears in `analysis_summary.html` under an "Incomplete Units" section.

This means `analyze` always produces a result, even against a still-running or partially-crashed simulation. If you want a fully completed dataset instead, finish the incomplete units first — `restart`/`append` for `-src pyadmd`, or a further `fel` call for `-src fel` — then re-run `analyze`.

The analysis pipeline is also resilient to being interrupted itself: each unit's results are written to its own output directory as soon as that unit finishes, and a re-run of `analyze` automatically detects and skips units whose output is already complete, only (re)computing what's missing. Parallel workers are recycled after each unit to keep memory bounded across large batches (*e.g.* hundreds of `fel` centroids).

## Configuration Parameters
The analysis module reads simulation parameters from the `pyAdMD_params.json` file, which includes:

- Number of replicas
- Total simulation time
- Atom selection criteria (used to scope RMSD/RoG/SASA/hydrophobic-exposure/DSSP — see [Analysis Selection Scope](#analysis-selection-scope))
- Input file paths

When `-src fel` is used, the shared production time axis (applied uniformly across all centroids) is instead read from `fel/run_metadata.json`'s `production_ps` value; `pyAdMD_params.json` is still used to locate the shared PSF topology file and the analysis selection.

## Output Structure
### Directory Organization
```
analysis/{fel/}
├── analysis_results.csv                  # Combined analysis data from all units
├── rmsf.csv                              # Combined RMSF data (omitted with --no-rmsf)
├── analysis_summary.html                 # HTML summary report
├── rmsd_plot.png                         # RMSD plot (omitted with --no-rmsd)
├── radius_gyration_plot.png              # Radius of gyration plot (omitted with --no-rg)
├── sasa_plot.png                         # SASA plot (omitted with --no-sasa)
├── hydrophobic_exposure_plot.png         # Hydrophobic exposure plot (omitted with --no-sasa)
├── rmsf_average.png                      # Average RMSF plot (omitted with --no-rmsf)
├── secondary_structure_average.png       # Average secondary structure plot (omitted with --no-dssp)
├── dccm_average.png                      # Average DCCM heatmap (omitted with --no-dccm)
├── dccm_average.npy                      # Average DCCM matrix, raw (omitted with --no-dccm)
├── lmi_average.png                       # Average LMI heatmap (omitted with --no-lmi)
├── lmi_average.npy                       # Average LMI matrix, raw (omitted with --no-lmi)
└── {rep[1-N]}/ or {centroid_frame[F]}/   # Unit-specific directories
    ├── analysis_results.csv              # Unit-specific analysis data
    ├── rmsf.csv                          # Unit-specific RMSF data (omitted with --no-rmsf)
    ├── rmsd_plot.png                     # Unit-specific RMSD plot (omitted with --no-rmsd)
    ├── radius_gyration_plot.png          # Unit-specific RoG plot (omitted with --no-rg)
    ├── sasa_plot.png                     # Unit-specific SASA plot (omitted with --no-sasa)
    ├── hydrophobic_exposure_plot.png     # Unit-specific hydrophobic exposure plot (omitted with --no-sasa)
    ├── rmsf_plot.png                     # Unit-specific RMSF plot (omitted with --no-rmsf)
    ├── secondary_structure.png           # Unit-specific secondary structure plot (omitted with --no-dssp)
    ├── dccm_matrix.npy                   # Unit-specific DCCM matrix, raw (omitted with --no-dccm)
    ├── dccm_plot.png                     # Unit-specific DCCM heatmap (omitted with --no-dccm)
    ├── lmi_matrix.npy                    # Unit-specific LMI matrix, raw (omitted with --no-lmi)
    └── lmi_plot.png                      # Unit-specific LMI heatmap (omitted with --no-lmi)
```
**Note:** With `-src fel`, the same set of files is written under `analysis/fel/` instead, with one subdirectory per centroid (named by frame index, mirroring `fel/centroids/centroid_frame[F]/`) in place of `rep[1-N]/`.

## Output Files Description
1. **CSV Files**
- **`analysis_results.csv`:** Time-series data for RMSD, RoG, SASA, hydrophobic exposure, and secondary structure content
- **`rmsf.csv`:** Per-residue RMSF values for all analyzed units (all replicas, or all centroids with `-src fel`)

2. **Plot Files**
- Individual property plots for each unit (replica, or centroid with `-src fel`)
- Combined plots showing all units
- Average plots across all units

3. **Correlation Matrix Files**
- **`dccm_matrix.npy`** (per-unit) / **`dccm_average.npy`** (cross-unit): raw (n_Cα × n_Cα) DCCM matrix, values in [-1, 1]. Omitted with `--no-dccm`.
- **`dccm_plot.png`** / **`dccm_average.png`**: DCCM heatmap, diverging colormap (red = fully correlated, white = uncorrelated, blue = fully anti-correlated).
- **`lmi_matrix.npy`** / **`lmi_average.npy`**: raw (n_Cα × n_Cα) LMI matrix, values in [0, 1]. Omitted with `--no-lmi`.
- **`lmi_plot.png`** / **`lmi_average.png`**: LMI heatmap, sequential colormap (LMI has no sign).

4. **HTML Summary**
- Interactive summary report with tables and embedded plots
- Statistics for each unit and averages across all units
- An "Incomplete Units" section when any unit hadn't reached its target cycle count at analysis time (see [Handling Incomplete Units](#handling-incomplete-units))
- Easy navigation and visualization of results


Furthermore, some basic analyses are written inside each replica folder at the end of the simulation, they can be found as follows:

- **coor-proj.out:** projection of the MD coordinates onto the normal mode space described by the excitation vector
- **rms-proj.out:** the system RMSD displacement along the excitation vectors
- **vp-proj.out:** projection of the MD velocities onto the normal mode space described by the excitation vector
- **ek-proj.out:** displays the additional kinetic energy at each MD step

[Back to top ↩](#)
* ****

# Usage Examples
Example files are available at the **[tutorial](tutorial)** folder (human calmodulin). We encourage users to test the multiple usages of **pyAdMD** using these files to get familiar with the method.

## Using OpenMM inputs and heavy atoms NMs
```
pyadmd run -src OPENMM \
                     -m HEAVY \
                     -psf tutorial/system.psf \
                     -rst tutorial/system.rst \
                     -pdb tutorial/system.pdb
```
## Using NAMD inputs and Cα NMs with custom parameters
```
pyadmd run -src NAMD \
                     -m CA \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -coor tutorial/system.coor \
                     -vel tutorial/system.vel \
                     -xsc tutorial/system.xsc \
                     -str tutorial/system.str \
                     -nm 7,8 \
                     -ek 0.5 \
                     -t 100 \
                     -sel "protein and resid 4 to 148" \
                     -rep 48
```
## Using NAMD inputs and CHARMM NMs without direction correction (standard MDeNM)
```
pyadmd run -src NAMD \
                     -m CHARMM \
                     -mod tutorial/system.mod \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -coor tutorial/system.coor \
                     -vel tutorial/system.vel \
                     -xsc tutorial/system.xsc \
                     -str tutorial/system.str \
                     --no-correc
```
## Restart unfinished pyAdMD simulations
```
pyadmd restart
```
## Append 100 ps to previously finished pyAdMD simulations
```
pyadmd append -t 100
```
## Analyze pyAdMD simulations (all frames)
```
pyadmd analyze
```
## Analyze every 5 ps skipping DSSP and LMI
```
pyadmd analyze -r --no-dssp --no-lmi
```
## Compute a free energy landscape
```
pyadmd fel -c 2 -p 100
```
## Extend a previous free energy calculation with more centroids and production time
```
pyadmd fel -c 2 -p 500 --max-centroids 100
```
## Compute a standalone ENM (Cα model, writing modes 7-16)
```
pyadmd enm -i tutorial/system.pdb -o enm_output -m CA
```
## Re-write additional modes from a previous ENM run
```
pyadmd enm -w "15,20:22" -o enm_output
```
## Clean previous setup files
```
pyadmd clean
```

[Back to top ↩](#)
* ****

# Installation

`pyadmd` is published on [PyPI](https://pypi.org/project/pyadmd/) and it can be installed by:

```
pip install pyadmd
```


Alternatively, install from source (`pyproject.toml`, `src/` layout). From the
repository root:

```
pip install .
```

`pyadmd` requires a CUDA-enabled GPU. Two dependencies need to be told about
your CUDA toolkit version — both default to CUDA 12.x here:

- `openmm[cuda12]`: the base `openmm` PyPI package only ships the CPU/OpenCL/Reference
  platforms; the `[cuda12]` extra additionally installs the CUDA-compiled
  platform (pip-native — no separate conda install or source build needed).
- `cupy-cuda12x`: imported unconditionally at module load time (`enm/calculator.py`),
  so it's a required dependency, not optional.

If your CUDA toolkit is 12.x but a different minor/major release, or CUDA
13.x, adjust the `openmm[cuda12]` extra to `openmm[cuda13]` accordingly.

**Note:** OpenMM's pip wheels only build CUDA platforms for CUDA 12 and
above — if your CUDA toolkit is 11.x, the `[cuda12]`/`[cuda13]` pip extras
won't work; drop the extra (falling back to the OpenCL platform) or install
via conda instead (`conda install -c conda-forge openmm`, which supports
older CUDA versions), then `pip install` the rest of `pyadmd`'s dependencies
into that environment. Either way, also swap `cupy-cuda12x>=13.6` for
`cupy-cuda11x>=13.6` in `pyproject.toml`.

# Dependencies
Python dependencies are declared in `pyproject.toml` and installed automatically
by `pip install`. Requires Python ≥ 3.12. Core dependencies:

  - `numpy>=2.2.5`
  - `scipy>=1.16.0`
  - `mdanalysis>=2.9.0`
  - `matplotlib>=3.10.0`
  - `seaborn`
  - `pandas`
  - `openmm[cuda12]>=8.3.1` (the `[cuda12]` extra installs the CUDA-compiled
    OpenMM platform; substitute `[cuda13]` for a CUDA 13.x toolkit. See the
    *[OpenMM installation guide](https://docs.openmm.org/latest/userguide/application/01_getting_started.html)*.)
  - `numba>=0.61.2`
  - `biopython`
  - `cupy-cuda12x>=13.6` (GPU-accelerated ENM diagonalization)

External system dependency (must be installed separately):
  - `dssp` (>=4.2.2), required for secondary structure analysis. Refer to the *[DSSP official GitHub repository](https://github.com/PDB-REDO/dssp?tab=readme-ov-file#building)* for building details.

[Back to top ↩](#)
* ****


# Citing
Please cite the following paper if you are using any Adaptive MDeNM application in your work:

Resende-Lara, P. T. *et al.* Adaptive Normal Mode Sampling (aMDeNM) Enhances Exploration of Protein Conformational Space and Reveals the Functional Role of Frequency Coupling. *J. Chem. Theory Comput.* 14 July 2026; **22** (13): 6304–6321. https://doi.org/10.1021/acs.jctc.6c00398

[Back to top ↩](#)
* ****

# License
This project is licensed under the GNU General Public License v3.0 (GPLv3) and is distributed as is, with absolutely no warranty.
See the [LICENSE](LICENSE) file for the full text.

[Back to top ↩](#)
* ****

# Contact
If you experience a bug or have a suggestion, please contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
