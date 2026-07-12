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

The Adaptive Molecular Dynamics with Excited Normal Modes (aMDeNM) method applies a kinetic excitation of normal modes (NMs) to enhance molecular dynamics simulations sampling. This technique consists in injecting additional atomic velocities along a combinations of NM vectors, creating an effective coupling between slow and fast molecular motions. The motions described by preselected directions of low-frequency NMs are dynamically adjusted throughout the simulation. By coupling low-frequency NM excitation with adaptive directional adjustments, aMDeNM facilitates extensive exploration of the energy landscape, overcoming the constraints of fixed, rectilinear displacements and alleviating structural stresses and environmental resistance. Importantly, aMDeNM requires only an initial structure without the need to specify predefined target states, distinguishing it from many biased sampling techniques that rely on predefined target conformations.

This document will give an overview of the aMDeNM method and help to properly setup and run a simulation.

* ****

- [Adaptive Molecular Dynamics with Python](#adaptive-molecular-dynamics-with-python)
- [Method Overview](#method-overview)
  - [Equilibration Molecular Dynamics](#equilibration-molecular-dynamics)
  - [CHARMM Normal Modes Analysis (Optional)](#charmm-normal-modes-analysis-optional)
  - [ENM Computation](#enm-computation)
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
  - [System De-Excitation](#system-de-excitation)
- [pyAdMD Applications](#pyadmd-applications)
  - [ENM](#enm)
  - [Physical force-field based normal modes](#physical-force-field-based-normal-modes)
- [Configuration](#configuration)
  - [Energy injection](#energy-injection)
  - [Simulation time](#simulation-time)
  - [Excitation direction update](#excitation-direction-update-1)
  - [Number of modes and replicas](#number-of-modes-and-replicas)
  - [Atom selection](#atom-selection)
- [Input Requirements](#input-requirements)
  - [Run](#run)
    - [Files](#files)
    - [Parameters](#parameters)
    - [Feature Flags](#feature-flags)
  - [Append](#append)
    - [Parameters](#parameters-1)
  - [Analysis](#analysis)
    - [Parameters](#parameters-2)
    - [Feature Flags](#feature-flags-1)
    - [Skip Flags](#skip-flags)
  - [Free Energy](#free-energy)
- [Analysis](#analysis-1)
  - [Basic Structural Properties Calculated](#basic-structural-properties-calculated)
  - [Analysis Modes](#analysis-modes)
    - [Standard Analysis](#standard-analysis)
    - [Rough Analysis](#rough-analysis)
    - [Selective Analysis (Skip Flags)](#selective-analysis-skip-flags)
  - [Trajectory Source](#trajectory-source)
  - [Configuration Parameters](#configuration-parameters)
  - [Output Structure](#output-structure)
    - [Directory Organization](#directory-organization)
  - [Output Files Description](#output-files-description)
- [Free Energy Landscape](#free-energy-landscape)
  - [Method Overview](#method-overview-1)
  - [Extending a Previous Free Energy Calculation](#extending-a-previous-free-energy-calculation)
  - [Output Structure](#output-structure-1)
    - [Directory Organization](#directory-organization-1)
  - [Output Files Description](#output-files-description-1)
- [Usage Examples](#usage-examples)
  - [Using CHARMM normal modes](#using-charmm-normal-modes)
  - [Using Cα-only ENM with custom parameters](#using-cα-only-enm-with-custom-parameters)
  - [Using heavy atoms without direction correction (standard MDeNM)](#using-heavy-atoms-without-direction-correction-standard-mdenm)
  - [Using an OpenMM XML restart instead of NAMD binaries](#using-an-openmm-xml-restart-instead-of-namd-binaries)
  - [Restart unfinished pyAdMD simulations](#restart-unfinished-pyadmd-simulations)
  - [Append 100 ps to previously finished pyAdMD simulations](#append-100-ps-to-previously-finished-pyadmd-simulations)
  - [Analyze pyAdMD simulations (all frames)](#analyze-pyadmd-simulations-all-frames)
  - [Analyze every 5 ps skipping DSSP and SASA](#analyze-every-5-ps-skipping-dssp-and-sasa)
  - [Analyze freeenergy centroid production trajectories](#analyze-freeenergy-centroid-production-trajectories)
  - [Compute a free energy landscape](#compute-a-free-energy-landscape)
  - [Extend a previous free energy calculation with more centroids and production time](#extend-a-previous-free-energy-calculation-with-more-centroids-and-production-time)
  - [Clean previous setup files](#clean-previous-setup-files)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Citing](#citing)
- [License](#license)
- [Contact](#contact)

* ****

# Method Overview
The aMDeNM is a enhanced sampling molecular dynamics method that uses normal modes as collective variables in order do increase the conformational space explored during MD simulation. This is done by injecting an incremental energy to the system, thus assigning additional atomic velocities along the direction of a given NM (or a combination of a NM set). The combination of the velocities from MD and those provided by the NM vectors properly couple slow and fast motions, allowing one to obtain large time scale movements, such as domain transitions in a feasible simulation time. The additional energy  injection and the excitation direction are constantly evaluated and updated to ensure the sampling validity.

The aMDeNM production runs (the excitation cycles themselves) are computed entirely within OpenMM using the CHARMM36m force field. GPU platforms (CUDA, then OpenCL) are preferred automatically, with a CPU fallback if neither is available. We strongly recommend that all system preparation be done with *[CHARMM-GUI](http://charmm-gui.org)*.

## Equilibration Molecular Dynamics
This is a prerequired step to perform aMDeNM simulations. It consists in performing a short equilibration MD to store the final atomic velocities and positions. Two equilibration input paths are supported, selected via `-src`/`--source`:

- **NAMD** (`-src NAMD`): *[NAMD 3.0](http://www.ks.uiuc.edu/Research/namd/)* binary files (`.coor`/`.vel`/`.xsc`) plus a CHARMM-style stream file (`.str`) are supplied directly.
- **OpenMM** (`-src OPENMM`): *[OpenMM](http://openmm.org)* state supplied as an XML restart file (`.rst`) produced with `XmlSerializer.serialize(state)` from a state built with `getPositions=True, getVelocities=True`.

## CHARMM Normal Modes Analysis (Optional)
If using CHARMM-based normal modes, it is also necessary to compute the modes from the last MD coordinates and store the vectors from the low-frequency end of the vibrational spectrum on a binary file.

## ENM Computation
The program computes Cα or heavy atoms Elastic Network Model using the same algorithms as the software available at our *[ENM github repository](https://github.com/pedro-tulio/enm)*.

## Uniform Normal Modes Combination

The program generates uniformly distributed excitation vectors through a geometry-aware repulsion-based algorithm. The approach builds on a physics-inspired framework where points behave as charged particles confined to a spherical manifold, interacting through a dimensionally-scaled potential function. Unlike naive implementations that operate in an abstract factor space, the algorithm here accounts for the true geometry of the normal mode subspace before distributing the points, guaranteeing that the resulting excitation vectors are genuinely equidistant in the physical Cartesian space that governs the molecular dynamics. The *[PDIM algorithm](https://github.com/antonielgomes/dpMDNM/tree/main/PDIM)* was the first implementation built for the same purpose; the design presented here extends that concept with a geometry-corrected basis and a faster, more concise implementation.

### Problem Definition

Let $`\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N \in \mathbb{R}^{3n}`$ be the $N$ normal mode vectors selected for excitation, where $n$ is the number of selected atoms. Each $`\mathbf{v}_k`$ is a flattened Cartesian displacement vector of length $3n$. The set of all normalized linear combinations of these vectors defines an $N$-dimensional subspace of $`\mathbb{R}^{3n}`$:

$$
\mathcal{V} = \mathrm{span}\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N\}
$$

Given an integer $P>0$, we seek to generate $P$ unit vectors $`\{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_P\} \subset \mathcal{V}`$ that maximize the minimal pairwise angular separation:

$$
\max_{{\mathbf{q}_i}} \min_{i \neq j} \|\mathbf{q}_i - \mathbf{q}_j\|
$$

This corresponds to finding an optimal spherical code on the unit hypersphere $S^{N-1}$ embedded within $\mathcal{V}$, with minimal angular separation between any two excitation directions.

### Mode Subspace Geometry

A naive approach would place the $N$ modes on coordinate axes and treat the combination coefficients $c_k$ directly as coordinates on $S^{N-1}$. This is geometrically valid only if the mode vectors satisfy $`\langle \mathbf{v}_i, \mathbf{v}_j \rangle = \delta_{ij}`$, i.e., they are orthonormal in plain Cartesian space. In practice this condition is not met. ENM mode vectors are orthogonal in the mass-weighted inner product but carry non-uniform amplitudes in Cartesian coordinates, with low-frequency modes typically exhibiting larger displacements than high-frequency ones. CHARMM normal modes may additionally lose orthogonality when projected onto an atomic subset. As a consequence, equal angular spacing of coefficient vectors in abstract factor space does not translate into equal angular spacing of the physical excitation vectors $\mathbf{q}_i$.

To correct for this, the algorithm explicitly constructs an orthonormal basis
for $\mathcal{V}$ using QR decomposition before running the repulsion algorithm.

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

Apply QR decomposition to $\mathbf{M}_{\mathrm{nm}}^\top \in \mathbb{R}^{3n \times N}$:

$$
\mathbf{M}_{\mathrm{nm}}^\top = \mathbf{Q}_{\mathrm{qr}}\ \mathbf{R}
$$

where $`\mathbf{Q}_{\mathrm{qr}} \in \mathbb{R}^{3n \times N}`$ has orthonormal columns and $`\mathbf{R} \in \mathbb{R}^{N \times N}`$ is upper triangular. Setting $`\mathbf{Q} = \mathbf{Q}_{\mathrm{qr}}^\top \in \mathbb{R}^{N \times 3n}`$ yields an orthonormal row basis for $\mathcal{V}$:

$$
\langle \mathbf{Q}_{i,:}, \mathbf{Q}_{j,:} \rangle = \delta_{ij}
$$

A point $`\mathbf{x} \in \mathbb{R}^N`$ on the unit hypersphere $S^{N-1}$ maps to a unit physical vector via $`\mathbf{q} = \mathbf{x}\mathbf{Q} \in \mathbb{R}^{3n}`$, and because $\mathbf{Q}$ is an isometric embedding, inner products are preserved:

$$
\langle \mathbf{x}_i \mathbf{Q}, \mathbf{x}_j \mathbf{Q} \rangle = \langle \mathbf{x}_i, \mathbf{x}_j \rangle
$$

The repulsion algorithm therefore operates on $N$-dimensional coordinates $`\mathbf{x} \in \mathbb{R}^N`$ whose geometry is faithful to the physical mode subspace, at no additional cost relative to working in abstract factor space.

### Energy Minimization Framework

The distribution problem is cast as an energy minimization over $P$ points $`\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_P\} \subset S^{N-1}`$:

$$
E = \sum_{i=1}^{P} \sum_{\substack{j=1 \\ j \neq i}}^{P} U \left(\|\mathbf{x}_i - \mathbf{x}_j\|\right)
$$

### Dimensionally-Scaled Potential Function

The repulsive potential uses an inverse power law scaled by the dimensionality of the space:

$$
U(r) = \frac{1}{r^{k}}, \quad k = N - 1
$$

The exponent $k = N-1$ is chosen for three reasons. First, the fundamental solution to Laplace's equation in $N$ dimensions scales as $1/r^{N-2}$, and the gradient of that solution scales as $1/r^{N-1}$, making this the natural repulsive force law in $N$-dimensional space. Second, the surface area of $S^{N-1}$ grows as $`({2\pi e}/{N})^{N/2}`$, so stronger repulsion in higher dimensions is required to counteract the concentration-of-measure effect that causes random points to cluster near the equator. Third, the exponent ensures numerical stability by preventing excessively large or small force values as $N$ varies.

### Gradient Flow on the Sphere

The repulsive force on point $\mathbf{x}_i$ arising from all other points is:

$$
\mathbf{f}_i = \sum_{\substack{j=1 \\ j \neq i}}^{P}
\frac{\mathbf{x}_i - \mathbf{x}_j}{\|\mathbf{x}_i - \mathbf{x}_j\|^{N+1}}
$$

Since the points must remain on $S^{N-1}$, the gradient is projected onto the tangent plane at $\mathbf{x}_i$ to eliminate any radial component:

$$
\tilde{\mathbf{f}}_i = \mathbf{f}_i - \left(\mathbf{f}_i \cdot \mathbf{x}_i\right) \mathbf{x}_i
$$

Each iteration updates the points by a fixed step $\eta = 0.001$ along their
tangent-plane forces and then renormalizes back onto $S^{N-1}$:

$$
\mathbf{x}_i \leftarrow \frac{\mathbf{x}_i + \eta\,\tilde{\mathbf{f}}_i}
{\|\mathbf{x}_i + \eta\,\tilde{\mathbf{f}}_i\|}
$$

Convergence is declared when $`\max_i \|\mathbf{f}_i\|_\infty < 10^{-6}`$, and the iteration is terminated early when the maximum force changes by less than $10^{-7}$ over five consecutive iterations (stagnation criterion).

### Special Case: Cross-Polytope Initialization

When $P=2N$, the $2N$ vertices of the cross-polytope:

$$
\{\pm\mathbf{e}_1, \pm\mathbf{e}_2, \dots, \pm\mathbf{e}_N\}
$$

provide an analytically optimal initialization in $\mathbf{Q}$-coordinates and the repulsion loop is skipped. These vertices are already maximally separated on $S^{N-1}$ under the symmetry of the cross-polytope, yielding pairwise angular separations of either $90°$ or $180°$.

### Normal Modes Linear Combination

Once the coordinates $`\{\mathbf{x}_1, \dots, \mathbf{x}_P\}`$ have converged on $S^{N-1}$, each is mapped back to a physical excitation vector via the orthonormal basis $\mathbf{Q}$:

$$
\mathbf{q}_i = \mathbf{x}_i \mathbf{Q} \in \mathbb{R}^{3n}, \qquad \|\mathbf{q}_i\| = 1
$$

The equivalent scalar combination factors $\{c_{i,k}\}$ in terms of the original (non-orthonormal) mode vectors — useful for logging and interpretability — are recovered by least-squares projection:

$$
\mathbf{c}_i = \mathbf{q}_i \, \mathbf{M}_{\mathrm{nm}}^+ \in \mathbb{R}^N
$$

where $`\mathbf{M}_{\mathrm{nm}}^+`$ denotes the Moore–Penrose pseudoinverse of $`\mathbf{M}_{\mathrm{nm}}`$. These approximate coefficients are written to the `factors.csv` output file for reference but do not influence the simulation; the physical vectors $\mathbf{q}_i$ are used directly as excitation directions.

## Kinetic Energy Control
The additional kinetic energy injected in the system has a fast dissipation rate. Therefore, the program constantly checks the injection energy level and rescale the velocities along the excited direction whenever it is necessary. The kinetic energy along the normalized excitation vector $\mathbf{Q}$ direction is calculated by projecting first the current velocities to the excitation direction $\mathbf{Q}$ as $`\mathbf{V}_p = (\mathbf{V}_{curr} \cdot \mathbf{Q}) \cdot \mathbf{Q}`$, where $`\mathbf{V}_{p}`$ and $`\mathbf{V}_{curr}`$ the $3N$-dimensional vectors of the projected and current atomic velocities, respectively. The kinetic energy along the excitation direction is thus given by:

$$
E_k = \frac{1}{2} \mathbf{V}_{p}^T \mathbf{M}\ \mathbf{V}_p
$$

where $\mathbf{M}$ is the diagonal mass matrix. At the beginning of each short simulation interval, the remaining excitation energy ($E_k$) is adjusted to the desired excitation level ($E_{exc}$) by modifying the atomic velocities so $`\mathbf{V}_{new} = \mathbf{V}_{curr} + (\mathbf{V}_{exc} - \mathbf{V}_{p})`$.

With this procedure, the system is kept in a continuous excited state, allowing an effective small, "adiabatic-like" energy injection. The energy injection control is done by projecting the velocities computed during the simulation onto the excited vector, thus obtaining and rescaling the kinetic energy corresponding to it.

## Excitation Direction Update
Since the excitation vector is obtained from the initial conformation, it is dependent of this configuration. As the system is displaced along this direction and change its conformation, the motion loses its directionality due to mainly anharmonic effects. To prevent the structural distortions produced by the displacement along a vector that is no longer valid, the program update the excitation directions based on the trajectory evolution during the previous excitation steps. This procedure allows the system to adaptively find a relaxed path to follow during the next aMDeNM excitations.

If we consider the $n^{th}$ simulation, the next excitation vector, $\mathbf{Q}_{n+1}$, is determined based on specific parameter values obtained along the trajectory followed in the $\mathbf{Q}_n$ direction. A new excitation vector is defined based on two parameters: the first relates to the effective displacement $\ell$ along $\mathbf{Q}_n$ during the $n^{th}$ excited dynamics by projecting the mass-weighted effective displacement vector $`\mathbf{d}_n = \mathbf{M}^{1/2} ({\langle \mathbf{r} \rangle}_n - \mathbf{r}_n^0)`$ onto the normalized mass-weighted excitation vector $\mathbf{Q}_n$, where the ${\langle \mathbf{r} \rangle}_n$ is the average position of the structures over the last $0.2 ps$ obtained in the $n^{th}$ excitation, and $\mathbf{r}_n^0$ is the starting position for the following simulation.

The second parameter relates to the relative deviation of the vector $\mathbf{d}_n$ with respect to vector $\mathbf{Q}_n$, evaluated by the angle $\alpha_n$ between them. More precisely, we consider the ${\cos \alpha}_n$ obtained by taking the scalar product of these vectors after normalizing $\mathbf{d}_n$, as following:

$$
{\cos \alpha}_n = \frac {\mathbf{d}_n \mathbf{Q}_n} {\|\mathbf{d}_n\|}
$$

A precise rule is followed to decide whether to modify the excitation vector direction after every short simulation run. The excitation vector is changed as soon as $\ell_n$, the displacement along $\mathbf{Q}_n$, is larger than a threshold value $\ell_c$, and when $\cos {\alpha}$ is lower than a threshold value ${\cos \alpha}_c$. The conditions for choosing the excitation vector between $\mathbf{d}_n$ and $\mathbf{Q}_n$ for the next simulation are defined by:

$$
\mathbf{Q}_{n+1} = \begin{cases}
  \mathbf{Q}_n, & \text{if } \ell \leq \ell_c \\
  \mathbf{Q}_n, & \text{if } \ell \gt \ell_c \wedge \cos \alpha \ge  {\cos \alpha}_c \\
  \mathbf{d}_n, & \text{if } \ell \gt \ell_c \wedge \cos \alpha \lt {\cos \alpha}_c
\end{cases}
$$

The default value for $\ell_c$ is $0.5 m^{1/2} Å$ (with $m$ being atomic mass unit), and for $\alpha$ is $60°$.

## System De-Excitation
At the end of each replica, a de-excitation molecular dynamics is submited in order to recover the equilibrated thermodynamics of the system. This final step aims to remove any residual MDeNM additional energy from the system and provide further sctructural and dynamical exploration.

[Back to top ↩](#)
* ****

# pyAdMD Applications
The Adaptive MDeNM method takes self-computed Cα or heavy atoms ENM modes or CHARMM-computed normal modes as collective variables to improve Molecular Dynamics sampling.

## ENM
Uses simpified force-field based on particles and springs computed automatically by the program. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

## Physical force-field based normal modes
Uses physical force-field based normal modes computed in *[CHARMM](https://www.charmm.org/charmm/)*. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

# Configuration
**pyadmd** is distributed as an installable Python package that computes ENM modes, uniformly distributes linear combinations of modes in the $N$-dimensional hypersphere space, manages the OpenMM-based simulations, and computes the projections along the excitation direction, applying corrections whenever necessary. Its bundled data includes the CHARMM driver script used to write down CHARMM-computed normal modes.

One can easily setup and run an Adaptive MDeNM simulation using pyadmd.
The configuration process is straightforward. Some technical aspects will be covered in this section in order to facilitate the method comprehension.

## Energy injection
The excitation time of Adaptive MDeNM is $0.2 ps$. This means that every $0.2 ps$ the system receives the additional amount of energy defined by the user. Therefore, when studying large scale motions, it is advised to inject small amounts of energy in order to avoid structural distortions caused by an excessive energy injection. Usually, an excitation energy of $0.125 kcal/mol$ is sufficient to achieve a large exploration of the conformational space ($0.5 kcal/mol$ if Cα-only ENM).

## Simulation time
The total simulation time may require a tuning depending on the system size, energy injection and nature of the motion being excited. Considering a large scale global motion, there is a trade-off between the energy injection and the total simulation time. Larger amounts of energy allows a shorter simulation time, however, this may not be advised as discussed above.

## Excitation direction update
As described above, the direction is updated after the system has traveled a distance of $0.5 Å$ along the excitation vector and its real displacement has a deviation of $60°$ with respect to the theoretical one. The update can also be affected by the amount of energy injected, since higher energy values leads to larger motions. In addition, after each correction the new vector loses directionality due to anharmonic effects. This means that, at a given point, the new vectors are so diffuse that there is no point in proceed the simulation. When this ponit is reached, it is necessary to recompute the normal modes and start again. This is one more reason to not inject high energy values and let the system undergoes the changes slowly.
Alternatively, one can recompute ENM modes instead of change the excitation vector direction (only when the original model type is ENM).

## Number of modes and replicas
The program do a linear combination of the supplied normal modes to compute the excitation direction. This imply that the more modes are provided, the more replicas will be necessary to cover the hyperspace described by these modes.

## Atom selection
Create an atom selection to apply the energy injection using *[MDAnalysis selection language](https://userguide.mdanalysis.org/1.1.1/selections.html)*. Must be written between quotes.

[Back to top ↩](#)
* ****

# Input Requirements

## Run
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

### Parameters
- **`-src`/`--source`**: Input engine type, **`NAMD`** (binary `.coor`/`.vel`/`.xsc`) or **`OPENMM`** (XML restart `.rst`) (**required**)

- **`-m`/`--model`**: Normal modes model type, **`CA`** ENM, **`HEAVY`** Atoms ENM or **`CHARMM`** (**required**. Default: **`CA`**)

- **`-nm`/`--modes`**: Normal modes to excite (**optional**. Default: **`7,8,9`**)

- **`-ek`/`--energy`**: Excitation energy injection (**optional**. Default: **`0.125`** kcal/mol)

- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

- **`-sel`/`--selection`**: Atom selection to apply the energy injection (**optional**. Default: **`"protein"`**)

- **`-rep`/`--replicas`**: Number of replicas to run (**optional**. Default: **`10`**)

### Feature Flags
- **`-n`/`--no_correc`**: Disable excitation vector direction correction and compute standard MDeNM

- **`-f`/`--fixed`**: Disable excitation vector correction and keep constant excitation energy injections

- **`-r`/`--recalc`**: Recompute ENM modes instead of correcting the excitation vector direction

- **`--full_ener`**: Write per-term energy decomposition (BOND, ANGLE, DIHED, IMPRP, CMAP, UBREY, NBFIX, NONBONDED, etc.) to `rep{N}_ener_decomp.log` every cycle

## Append
### Parameters
- **`-t`/`--time`**: Simulation time to append, in ps (**required**)

## Analysis
### Parameters
- **`-src`/`--source`**: Trajectory source to analyze, **`pyadmd`** for `rep{N}.dcd` replica trajectories or **`freeenergy`** for centroid production trajectories from a completed `freeenergy` run (**optional**. Default: **`pyadmd`**)

### Feature Flags
- **`-r`/`--rough`**: Perform rough analysis (**optional**. Analyze every **`5`** ps instead of every frame.)

### Skip Flags
Each analysis step can be independently disabled. When skipped, that metric will not appear in any CSV, plot, or HTML summary output.

- **`--no_rmsd`**: Skip RMSD calculation
- **`--no_rg`**: Skip radius of gyration calculation
- **`--no_sasa`**: Skip SASA calculation
- **`--no_hp`**: Skip hydrophobic exposure calculation
- **`--no_rmsf`**: Skip  RMSF calculation
- **`--no_dssp`**: Skip secondary structure analysis via DSSP

**Note:** Before any analysis runs, the program checks if `pyadmd` or `freeenergy` calls are properly completed. If any unit hasn't reached its target, `analyze` aborts and lists the incomplete units rather than analyzing a partial trajectory. Complete them first with `restart`/`append` (pyadmd) or a further `freeenergy` call (freeenergy), then re-run `analyze`.

## Free Energy
All parameters are optional; the `freeenergy` subcommand reads its input trajectories and reference state from files already produced by `run`/`restart`/`append`, so no additional files need to be supplied.

- **`-c`/`--cutoff`**: GROMOS RMSD clustering cutoff, in Å (**optional**. Default: **`0.8`**)
- **`-d`/`--deexcite`**: Total restrained de-excitation MD length per centroid, in ps, split evenly over 4 restraint phases (**optional**. Default: **`200`**)
- **`-p`/`--production`**: Unrestrained production MD length per centroid, in ps (**optional**. Default: **`800`**)
- **`-nm`/`--modes`**: Comma-separated mode indices to project for the FEL (**optional**. Default: same modes used in `run`, e.g. `7,8,9`)
- **`--modes_2d`**: Mode pairs for 2D FEL plots, as space-separated `"m1,m2"` tokens, e.g. `"7,8 7,9 8,9"` (**optional**. Default: all pairwise combinations of `--modes`)
- **`-b`/`--bins`**: Number of histogram bins used for the FEL (**optional**. Default: **`50`**)
- **`-T`/`--temp`**: Temperature for k<sub>B</sub>T scaling and the production ensemble, in K (**optional**. Default: **`303.15`**)
- **`-s`/`--sel`**: MDAnalysis selection string used for GROMOS RMSD clustering (**optional**. Default: **`"protein and name CA"`**)
- **`--max_centroids`**: Maximum number of centroids submitted to MD. When the cluster count exceeds this value, exactly this many centroids are selected by greedy farthest-point (MaxMin) sampling to maximize conformational diversity (**optional**. Default: **`50`**)

**Note:** `-s`/`--sel` and `-T`/`--temp` must stay the same across repeated `freeenergy` calls on the same simulation — see [Extending a Previous Free Energy Calculation](#extending-a-previous-free-energy-calculation).

[Back to top ↩](#)
* ****

# Analysis
The PyAdMD **`analysis`** module provides comprehensive analysis capabilities for molecular dynamics simulations performed using the aMDeNM method. This module processes simulation trajectories and generates detailed structural analysis, visualizations, and summary reports. It can analyze either the aMDeNM replica trajectories from `run`/`restart`/`append` (`-src pyadmd`, default) or the centroid production trajectories from a completed `freeenergy` run (`-src freeenergy`) — see [Trajectory Source](#trajectory-source).

## Basic Structural Properties Calculated

1. **Root Mean Square Deviation (RMSD):**  Measures structural deviation from the initial conformation.

2. **Radius of Gyration (RoG):** Measures the compactness of the protein structure. Useful for identifying folding/unfolding events.

3. **Solvent Accessible Surface Area (SASA):** Calculates the surface area accessible to solvent molecules. Uses the Shrake-Rupley algorithm implemented in Bio.PDB.

4. **Hydrophobic Exposure:** Measures the percentage of hydrophobic residues exposed to solvent. Useful for identifying folding/unfolding events.

5. **Root Mean Square Fluctuation (RMSF):** Calculates per-residue flexibility using Cα atoms. Identifies flexible and rigid regions in the protein structure.

6. **Secondary Structure Content:** Calculates secondary structure elements using DSSP. Tracks helix, sheet, coil, turn, and other structural elements over time and reports the number of residues in each secondary structure type.

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
- DSSP is not installed (`--no_dssp`)
- Only a subset of metrics is needed (e.g. RMSD + RMSF only)
- Computation time needs to be minimized (SASA and DSSP are the most expensive steps)

## Trajectory Source
The `-src`/`--source` flag selects which set of trajectories to analyze:

- **`pyadmd`** (default): analyzes `rep{N}/rep{N}.dcd` replica trajectories, one analysis unit per replica. Output goes to `analysis/`.
- **`freeenergy`**: analyzes `freeenergy/centroids/centroid_frame{F}/prod.dcd` production trajectories, one analysis unit per centroid (identified by its merged-trajectory frame index, not a sequential number). Output goes to `analysis/freeenergy/`, kept separate from `pyadmd`-sourced output. All computed metrics (RMSD, RoG, SASA, hydrophobic exposure, RMSF, secondary structure) and skip flags apply identically regardless of source.

In both modes, `analyze` first verifies every unit has reached its target cycle count and aborts if not — see the [Analysis](#analysis) input requirements note above.

## Configuration Parameters
The analysis module reads simulation parameters from the `pyAdMD_params.json` file, which includes:

- Number of replicas
- Total simulation time
- Atom selection criteria
- Input file paths

When `-src freeenergy` is used, the shared production time axis (applied uniformly across all centroids) is instead read from `freeenergy/run_metadata.json`'s `production_ps` value; `pyAdMD_params.json` is still used to locate the shared PSF topology file.

## Output Structure
### Directory Organization
```
analysis/
├── analysis_results.csv                # Combined analysis data from all replicas
├── rmsf.csv                            # Combined RMSF data (omitted with --no_rmsf)
├── analysis_summary.html               # HTML summary report
├── rmsd_plot.png                       # RMSD plot (omitted with --no_rmsd)
├── radius_gyration_plot.png            # Radius of gyration plot (omitted with --no_rg)
├── sasa_plot.png                       # SASA plot (omitted with --no_sasa)
├── hydrophobic_exposure_plot.png       # Hydrophobic exposure plot (omitted with --no_hp)
├── rmsf_average.png                    # Average RMSF plot (omitted with --no_rmsf)
├── secondary_structure_average.png     # Average secondary structure plot (omitted with --no_dssp)
└── rep[1-N]/                           # Replica-specific directories
    ├── analysis_results.csv            # Replica-specific analysis data
    ├── rmsf.csv                        # Replica-specific RMSF data (omitted with --no_rmsf)
    ├── rmsd_plot.png                   # Replica-specific RMSD plot (omitted with --no_rmsd)
    ├── radius_gyration_plot.png        # Replica-specific RoG plot (omitted with --no_rg)
    ├── sasa_plot.png                   # Replica-specific SASA plot (omitted with --no_sasa)
    ├── hydrophobic_exposure_plot.png   # Replica-specific hydrophobic exposure plot (omitted with --no_hp)
    ├── rmsf_plot.png                   # Replica-specific RMSF plot (omitted with --no_rmsf)
    └── secondary_structure.png         # Replica-specific secondary structure plot (omitted with --no_dssp)
```

With `-src freeenergy`, the same set of files is written under `analysis/freeenergy/` instead, with one subdirectory per centroid (named by frame index, mirroring `freeenergy/centroids/centroid_frame[F]/`) in place of `rep[1-N]/`:
```
analysis/freeenergy/
├── analysis_results.csv                # Combined analysis data from all centroids
├── rmsf.csv                            # Combined RMSF data (omitted with --no_rmsf)
├── analysis_summary.html               # HTML summary report
├── rmsd_plot.png                       # RMSD plot (omitted with --no_rmsd)
├── radius_gyration_plot.png            # Radius of gyration plot (omitted with --no_rg)
├── sasa_plot.png                       # SASA plot (omitted with --no_sasa)
├── hydrophobic_exposure_plot.png       # Hydrophobic exposure plot (omitted with --no_hp)
├── rmsf_average.png                    # Average RMSF plot (omitted with --no_rmsf)
├── secondary_structure_average.png     # Average secondary structure plot (omitted with --no_dssp)
└── centroid_frame[F]/                  # Centroid-specific directories
    ├── analysis_results.csv            # Centroid-specific analysis data
    ├── rmsf.csv                        # Centroid-specific RMSF data (omitted with --no_rmsf)
    ├── rmsd_plot.png                   # Centroid-specific RMSD plot (omitted with --no_rmsd)
    ├── radius_gyration_plot.png        # Centroid-specific RoG plot (omitted with --no_rg)
    ├── sasa_plot.png                   # Centroid-specific SASA plot (omitted with --no_sasa)
    ├── hydrophobic_exposure_plot.png   # Centroid-specific hydrophobic exposure plot (omitted with --no_hp)
    ├── rmsf_plot.png                   # Centroid-specific RMSF plot (omitted with --no_rmsf)
    └── secondary_structure.png         # Centroid-specific secondary structure plot (omitted with --no_dssp)
```

## Output Files Description
1. **CSV Files**
- **`analysis_results.csv`:** Time-series data for RMSD, RoG, SASA, hydrophobic exposure, and secondary structure content
- **`rmsf.csv`:** Per-residue RMSF values for all analyzed units (all replicas, or all centroids with `-src freeenergy`)

2. **Plot Files**
- Individual property plots for each unit (replica, or centroid with `-src freeenergy`)
- Combined plots showing all units
- Average plots across all units

3. **HTML Summary**
- Interactive summary report with tables and embedded plots
- Statistics for each unit and averages across all units
- Easy navigation and visualization of results


Furthermore, some basic analyses are written inside each replica folder at the end of the simulation, they can be found as follows:

- **coor-proj.out:** projection of the MD coordinates onto the normal mode space described by the excitation vector
- **rms-proj.out:** the system RMSD displacement along the excitation vectors
- **vp-proj.out:** projection of the MD velocities onto the normal mode space described by the excitation vector
- **ek-proj.out:** displays the additional kinetic energy at each MD step

[Back to top ↩](#)
* ****

# Free Energy Landscape
The **`freeenergy`** subcommand computes a free energy landscape (FEL) from a completed set of aMDeNM replicas, following the two-stage protocol of Costa *et al.* [DOI: 10.1021/acs.jctc.5b00003](https://doi.org/10.1021/acs.jctc.5b00003).

## Method Overview
1. **Merge trajectories**: all `rep*.dcd` replica trajectories are concatenated into a single pseudo-trajectory.
2. **GROMOS clustering**: frames are clustered by Cα RMSD (`-s`/`-c`); when the number of clusters exceeds `--max_centroids`, a maximally diverse subset is selected via greedy farthest-point (MaxMin) sampling on the cluster centroids.
3. **Centroid MD**: each centroid undergoes a 4-phase restrained de-excitation (`-d`, progressively decreasing positional restraints on backbone and sidechain heavy atoms) followed by unrestrained production MD (`-p`).
4. **Mode projection**: every production frame is projected onto each individual normal mode vector as a signed mass-weighted RMS displacement.
5. **FEL computation**: a population histogram (`-b` bins) is converted to $\Delta G$ via $\Delta G = -k_{BT} \cdot ln[P(q)/P_{max}]$, computed independently per mode (1D) and for user-specified mode pairs (2D, `--modes_2d`).

## Extending a Previous Free Energy Calculation
`freeenergy` can be re-invoked on the same simulation with a larger `--max_centroids` and/or longer `-p`/`--production` to extend an earlier calculation, rather than starting over:

- **Free to change**: `-c`/`--cutoff` and `-d`/`--deexcite`. Changing the cutoff only affects the re-thresholding of the cached pairwise-RMSD matrix. Changing the de-excitation length only affects newly-created centroids going forward; existing centroids keep whatever de-excitation they originally had and are simply extended in production.
- **Must stay the same**: `-s`/`--sel`, `-T`/`--temp`. Mixing clustering selections or temperatures inside one pooled FEL is not physically valid.
- **Never shrinks existing work**: if `--max_centroids` or `-p`/`--production` is *smaller* than the previous call, the program warns and uses the larger of the two values instead. We suggest start with smaller values and append more data, if necessary.


## Output Structure
### Directory Organization
```
freeenergy/
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
        ├── prod.dcd                        # production trajectory (appended to on extension)
        ├── prod_checkpoint.chk             # exact final state, for bit-identical extension
        └── checkpoint.chk                  # periodic (every 10 cycles) checkpoint
```

## Output Files Description
1. **Cache Files** 
- **`run_metadata.json`**: the clustering selection, temperature, cutoff, de-excitation length, `max_centroids`, and production length used.
- **`clustering_rmsd_cache.npz`/`.json`**: the pairwise-RMSD matrix over subsampled frames.
- **`clustering_summary.csv`**: summary containing cluster ID, frame index, cluster size, status this run (fresh/extended/skipped), and cycles/ps completed.
2. **Plot Files** 
- **`fel_mode[N].csv`/`fel_mode[N]_plot.png`**: 1D free energy landscape per mode, in Å and kcal/mol.
- **`fel_2d_mode[N]_mode[M].png`**: 2D free energy landscape for a mode pair.
3. **HTML Summary**
-  **`fel_summary.html`**: interactive summary with protocol parameters, per-mode FEL statistics, per-centroid production status, and embedded plots.

[Back to top ↩](#)
* ****

# Usage Examples
Example files are available at the **`tutorial`** folder (calmodulin). We encourage users to test the multiple usages of **pyAdMD** using these files to get familiar with the method.

## Using CHARMM normal modes
```
pyadmd run -src NAMD \
                     -m CHARMM \
                     -mod tutorial/system.mod \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -coor tutorial/system.coor \
                     -vel tutorial/system.vel \
                     -xsc tutorial/system.xsc \
                     -str tutorial/system.str
```
## Using Cα-only ENM with custom parameters
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
## Using heavy atoms without direction correction (standard MDeNM)
```
pyadmd run -src NAMD \
                     -m HEAVY \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -coor tutorial/system.coor \
                     -vel tutorial/system.vel \
                     -xsc tutorial/system.xsc \
                     -str tutorial/system.str \
                     --no_correc
```
## Using an OpenMM XML restart instead of NAMD binaries
```
pyadmd run -src OPENMM \
                     -m CA \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -rst tutorial/system.rst
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
## Analyze every 5 ps skipping DSSP and SASA
```
pyadmd analyze -r \
                         --no_dssp \
                         --no_sasa
```
## Analyze freeenergy centroid production trajectories
```
pyadmd analyze -src freeenergy
```
## Compute a free energy landscape
```
pyadmd freeenergy -c 2 -p 100
```
## Extend a previous free energy calculation with more centroids and production time
```
pyadmd freeenergy -c 2 -p 500 --max_centroids 100
```
## Clean previous setup files
```
pyadmd clean
```

[Back to top ↩](#)
* ****

# Installation

`pyadmd` is a standard installable Python package (`pyproject.toml`, `src/` layout).
From the repository root:

```
pip install .
```

or, for an editable/development install:

```
pip install -e .
```

This installs the `pyadmd` console-script entry point, so all commands in this
document (`pyadmd run ...`, `pyadmd analyze ...`, etc.) become available directly,
from any working directory. `python -m pyadmd ...` works identically.

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

External system dependency (not pip-installable):
  - `dssp` (4.x), required for secondary structure analysis. Refer to the *[DSSP official GitHub repository](https://github.com/PDB-REDO/dssp?tab=readme-ov-file#building)* for building details.

[Back to top ↩](#)
* ****


# Citing
Please cite the following paper if you are using any Adaptive MDeNM application in your work:

[Resende-Lara, P. T. et al. Adaptive Normal Mode Sampling (aMDeNM) Enhances Exploration of Protein Conformational Space and Reveals the Functional Role of Frequency Coupling. *Journal of Chemical Theory and Computation*. DOI: 10.1021/acs.jctc.6c00398](https://pubs.acs.org/doi/10.1021/acs.jctc.6c00398)

[Back to top ↩](#)
* ****

# License
This project is licensed under the GNU General Public License v3.0 (GPLv3).
See the [LICENSE](LICENSE) file for the full text.

[Back to top ↩](#)
* ****

# Contact
If you experience a bug or have any doubt or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
