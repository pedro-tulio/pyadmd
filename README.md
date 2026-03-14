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

The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of multiple-replica short MD simulations in which motions described by a given subset of low-frequency NMs are kinetically excited. This is achieved by adding additional atomic velocities along several randomly determined linear combinations of NM vectors, thus allowing an efficient coupling between slow and fast motions.

This new approach, Adaptive Molecular Dynamics with excited Normal Modes (aMDeNM), automatically controls the energy injection and take the natural constraints imposed by the structure and the environment into account during protein conformational sampling, which prevent structural distortions all along the simulation.Due to the stochasticity of thermal motions, NM eigenvectors move away from the original directions when used to displace the protein, since the structure evolves into other potential energy wells. Therefore, the displacement along the modes is valid for small distances, but the displacement along greater distances may deform the structure of the protein if no care is taken. The advantage of this methodology is to adaptively change the direction used to displace the system, taking into account the structural and energetic constraints imposed by the system itself and the medium, which allows the system to explore new pathways.

This document will give an overview of the aMDeNM method and help to properly setup and run a simulation.

* ****

- [Adaptive Molecular Dynamics with Python](#adaptive-molecular-dynamics-with-python)
- [Method Overview](#method-overview)
  - [NAMD Equilibration Molecular Dynamics](#namd-equilibration-molecular-dynamics)
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
    - [Feature Flags](#feature-flags-1)
- [Analysis](#analysis-1)
  - [Basic Structural Properties Calculated](#basic-structural-properties-calculated)
  - [Analysis Modes](#analysis-modes)
    - [Standard Analysis](#standard-analysis)
    - [Rough Analysis](#rough-analysis)
  - [Configuration Parameters](#configuration-parameters)
  - [Output Structure](#output-structure)
    - [Directory Organization](#directory-organization)
  - [Output Files Description](#output-files-description)
- [Usage Examples](#usage-examples)
  - [Using CHARMM normal modes](#using-charmm-normal-modes)
  - [Using Cα-only ENM with custom parameters](#using-cα-only-enm-with-custom-parameters)
  - [Using heavy atoms without direction correction (standard MDeNM)](#using-heavy-atoms-without-direction-correction-standard-mdenm)
  - [Restart unfinished pyAdMD simulations](#restart-unfinished-pyadmd-simulations)
  - [Append 100 ps to previously finished pyAdMD simulations](#append-100-ps-to-previously-finished-pyadmd-simulations)
  - [Analyze pyAdMD simulations](#analyze-pyadmd-simulations)
  - [Clean previous setup files](#clean-previous-setup-files)
- [Dependencies](#dependencies)
- [Citing](#citing)
- [Contact](#contact)

* ****

# Method Overview
The aMDeNM is a enhanced sampling molecular dynamics method that uses normal modes as collective variables in order do increase the conformational space explored during MD simulation. This is done by injecting an incremental energy to the system, thus assigning additional atomic velocities along the direction of a given NM (or a combination of a NM set). The combination of the velocities from MD and those provided by the NM vectors properly couple slow and fast motions, allowing one to obtain large time scale movements, such as domain transitions in a feasible simulation time. The additional energy  injection and the excitation direction are constantly evaluated and updated to ensure the sampling validity.

The aMDeNM simulations are done using *[NAMD 3.0](http://www.ks.uiuc.edu/Research/namd/)* and CHARMM36m forcefield. We implemented the code so the molecular dynamics are computed exclusively on GPU. We strongly recommend that all system preparation be done with *[CHARMM-GUI](http://charmm-gui.org)*.

## NAMD Equilibration Molecular Dynamics
This is a prerequired step to perform aMDeNM simulations. It consists in performing a short equilibration MD to store the final atomic velocities and positions.

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
**pyAdMD** is distributed as a Python handler script that compute ENM modes, uniformly distributes linear combinations of modes in the $N$-dimensional hypersphere space, manage NAMD simulations and compute the projections along the excitation direction and apply the correction whenever necessary. An additional <code>tools</code> folder includes scripts not used by NAMD, such as the one to write down the CHARMM normal modes.

One can easily setup and run an Adaptive MDeNM simulation by using this script.
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
- **`-psf`/`--psffile`**: PSF structure file containing system molecule-specific information (**required**)
  
- **`-pdb`/`--pdbfile`**: PDB structure file in Protein Data Bank format (**required**)

- **`-xsc`/`--xscfile`**: NAMD eXtended System Configuration file (**required**)

- **`-coor`/`--coorfile`**: NAMD binary coordinates file (**required**)

- **`-vel`/`--velfile`**: NAMD binary velocities file (**required**)

- **`-str`/`--strfile`**: CHARMM-style stream file, which contains force field parameters and definitions (**required**)
  
- **`-mod`/`--modefile`**: Binary file containg CHARMM normal mode vectors (**optional**. Required only if **`-type = CHARMM`**)

### Parameters
- **`-type`/`--modeltype`**: Normal modes model type, **`CA`** ENM, **`HEAVY`** Atoms ENM or **`CHARMM`** (**required**. Default: **`CA`**)

- **`-nm`/`--modes`**: Normal modes to excite (**optional**. Default: **`7,8,9`**)

- **`-ek`/`--energy`**: Excitation energy injection (**optional**. Default: **`0.125`** kcal/mol)

- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

- **`-sel`/`--selection`**: Atom selection to apply the energy injection (**optional**. Default: **`"protein"`**)

- **`-rep`/`--replicas`**: Number of replicas to run (**optional**. Default: **`10`**)

### Feature Flags
- **`-n`/`--no_correc`**: Disable excitation vector direction correction and compute standard MDeNM

- **`-f`/`--fixed`**: Disable excitation vector correction and keep constant excitation energy injections

- **`-r`/`--recalc`**: Recompute ENM modes instead of correcting the excitation vector direction

## Append
### Parameters
- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

## Analysis
### Feature Flags
- **`-r`/`--rough`**: Perform rough analysis (**optional**. Analyze every **`5`** ps instead of every frame.)

[Back to top ↩](#)
* ****

# Analysis
The PyAdMD **`analysis`** module provides comprehensive analysis capabilities for molecular dynamics simulations performed using the aMDeNM method. This module processes simulation trajectories and generates detailed structural analysis, visualizations, and summary reports.

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

## Configuration Parameters
The analysis module reads simulation parameters from the `pyAdMD_params.json` file, which includes:

- Number of replicas
- Total simulation time
- Atom selection criteria
- Input file paths

## Output Structure
### Directory Organization
```
analysis/
├── analysis_results.csv                # Combined analysis data from all replicas
├── rmsf.csv                            # Combined RMSF data from all replicas
├── analysis_summary.html               # HTML summary report
├── rmsd_plot.png                       # RMSD plot for all replicas
├── radius_gyration_plot.png            # Radius of gyration plot
├── sasa_plot.png                       # SASA plot
├── hydrophobic_exposure_plot.png       # Hydrophobic exposure plot
├── rmsf_average.png                    # Average RMSF plot
├── secondary_structure_average.png     # Average secondary structure plot
└── rep[1-N]/                           # Replica-specific directories
    ├── analysis_results.csv            # Replica-specific analysis data
    ├── rmsf.csv                        # Replica-specific RMSF data
    ├── rmsd_plot.png                   # Replica-specific RMSD plot
    ├── radius_gyration_plot.png        # Replica-specific RoG plot
    ├── sasa_plot.png                   # Replica-specific SASA plot
    ├── hydrophobic_exposure_plot.png   # Replica-specific hydrophobic exposure plot
    ├── rmsf_plot.png                   # Replica-specific RMSF plot
    └── secondary_structure.png         # Replica-specific secondary structure plot
```

## Output Files Description
1. **CSV Files**
- **`analysis_results.csv`:** Time-series data for RMSD, RoG, SASA, hydrophobic exposure, and secondary structure content
- **`rmsf.csv`:** Per-residue RMSF values for all replicas

2. **Plot Files**
- Individual property plots for each replica
- Combined plots showing all replicas
- Average plots across all replicas

3. **HTML Summary**
- Interactive summary report with tables and embedded plots
- Statistics for each replica and averages across all replicas
- Easy navigation and visualization of results


Furthermore, some basic analyses are written inside each replica folder at the end of the simulation, they can be found as follows:

- **coor-proj.out:** projection of the MD coordinates onto the normal mode space described by the excitation vector
- **rms-proj.out:** the system RMSD displacement along the excitation vectors
- **vp-proj.out:** projection of the MD velocities onto the normal mode space described by the excitation vector
- **ek-proj.out:** displays the additional kinetic energy at each MD step

[Back to top ↩](#)
* ****

# Usage Examples
Example files are available at the **`tutorial`** folder (calmodulin). We encourage users to test the multiple usages of **pyAdMD** using these files to get familiar with the method.

## Using CHARMM normal modes
```
python pyAdMD.py run -type CHARMM \
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
python pyAdMD.py run -type CA \
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
python pyAdMD.py run -type HEAVY \
                     -psf tutorial/system.psf \
                     -pdb tutorial/system.pdb \
                     -coor tutorial/system.coor \
                     -vel tutorial/system.vel \
                     -xsc tutorial/system.xsc \
                     -str tutorial/system.str \
                     --no_correc
```
## Restart unfinished pyAdMD simulations
```
python pyAdMD.py restart
```
## Append 100 ps to previously finished pyAdMD simulations
```
python pyAdMD.py append -t 100
```
## Analyze pyAdMD simulations
```
python pyAdMD.py analyze -r
```
## Clean previous setup files
```
python pyAdMD.py clean
```

[Back to top ↩](#)
* ****

# Dependencies
A `conda` enviroment can be easily setup with the provided `pyAdMD.yaml` file containing the necessary Python dependencies, which are:

  - `python=3.12`
  - `numpy=2.2.5`
  - `scipy=1.16.0`
  - `mdanalysis=2.9.0`
  - `matplotlib=3.10.0`
  - `openmm=8.3.1`
  - `numba=0.61.2`
  - `cupy=13.6.` (optional, for GPU-accelerated ENM calculation. **Note:** CuPy requires matching CUDA toolkit.)
  - `dssp=4.x` (for secondary structure analysis, refer to the *[DSSP official GitHub repository](https://github.com/PDB-REDO/dssp?tab=readme-ov-file#building)* for building details.)

Type the following command to create the environment:
```  
conda env create -f pyAdMD.yaml
```

[Back to top ↩](#)
* ****

# Citing
Please cite the following paper if you are using any Adaptive MDeNM application in your work:

[Resende-Lara, P. T. et al. *Adaptive collective motions: a hybrid method to improve conformational sampling with molecular dynamics and normal modes.* bioRxiv. doi: 10.1101/2022.11.29.517349](https://www.biorxiv.org/content/10.1101/2022.11.29.517349)

[Back to top ↩](#)
* ****

# Contact
If you experience a bug or have any doubt or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
