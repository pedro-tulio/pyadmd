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
      - [Energy Minimization Framework](#energy-minimization-framework)
      - [Dimensionally-Scaled Potential Function](#dimensionally-scaled-potential-function)
      - [Normal Modes Linear Combination](#normal-modes-linear-combination)
    - [Kinetic Energy Control](#kinetic-energy-control)
    - [Excitation Direction Update](#excitation-direction-update)
    - [System De-Excitation](#system-de-excitation)
  - [pyAdMD Applications](#pyadmd-applications)
    - [ENM](#enm)
    - [Physical force-field based normal modes](#physical-force-field-based-normal-modes)
    - [pyAdMD files](#pyadmd-files)
    - [Configuration](#configuration)
  - [Input Requirements](#input-requirements)
    - [Run](#run)
      - [Parameters:](#parameters)
      - [Files:](#files)
      - [Feature Flags](#feature-flags)
    - [Append](#append)
      - [Parameters:](#parameters-1)
  - [Usage Examples](#usage-examples)
    - [Using CHARMM normal modes](#using-charmm-normal-modes)
    - [Using Cα-only ENM with custom parameters](#using-cα-only-enm-with-custom-parameters)
    - [Using heavy atoms without direction correction (standard MDeNM)](#using-heavy-atoms-without-direction-correction-standard-mdenm)
    - [Restart unfinished pyAdMD simulations](#restart-unfinished-pyadmd-simulations)
    - [Append 100 ps to previously finished pyAdMD simulations](#append-100-ps-to-previously-finished-pyadmd-simulations)
    - [Clean previous setup files](#clean-previous-setup-files)
  - [Analysis](#analysis)
  - [Citing](#citing)
  - [Dependencies](#dependencies)
  - [Contact](#contact)

* ****

## Method Overview
The aMDeNM is a enhanced sampling molecular dynamics method that uses normal modes as collective variables in order do increase the conformational space explored during MD simulation. This is done by injecting an incremental energy to the system, thus assigning additional atomic velocities along the direction of a given NM (or a combination of a NM set). The combination of the velocities from MD and those provided by the NM vectors properly couple slow and fast motions, allowing one to obtain large time scale movements, such as domain transitions in a feasible simulation time. The additional energy  injection and the excitation direction are constantly evaluated and updated to ensure the sampling validity.

The aMDeNM simulations are done using *[NAMD 3.0](http://www.ks.uiuc.edu/Research/namd/)* and CHARMM36m forcefield. We implemented the code so the molecular dynamics are computed exclusively on GPU. We strongly recommend that all system preparation be done with *[CHARMM-GUI](http://charmm-gui.org)*.

### NAMD Equilibration Molecular Dynamics
This is a prerequired step to perform aMDeNM simulations. It consists in performing a short equilibration MD to store the final atomic velocities and positions.

### CHARMM Normal Modes Analysis (Optional)
If using CHARMM-based normal modes, it is also necessary to compute the modes from the last MD coordinates and store the vectors from the low-frequency end of the vibrational spectrum on a binary file.

### ENM Computation
The program computes Cα or heavy atoms Elastic Network Model using the same algorithms as the software available at our *[ENM github repository](https://github.com/pedro-tulio/enm)*.

### Uniform Normal Modes Combination
The program generates uniformly distributed points on N-dimensional hypersphere through a repulsion-based algorithm which employs a physics-inspired approach where points behave as charged particles confined to the spherical manifold, interacting through a dimensionally-scaled potential function. The generated points are then used as scaling factors to be used in normal modes combinations. The *[PDIM algorithm](https://github.com/antonielgomes/dpMDNM/tree/main/PDIM)* was built to the same purpose. Here we present a different design with a faster and more simplified implementation.

#### Problem Definition
Given an N-dimensional hypersphere $S^{N-1} = \{x \in R^{N}:||x|| = 1\}$ and an integer $P \gt 0$, we seek to generate $P$ points ${x_1 , x_2, \dots, x_P}$ on $S^{N-1}$ that maximize the minimal pairwise distance: $min_{dist} ||xi-xj||$ for $i \neq j$. This corresponds to finding an optimal spherical code with minimal angular separation.

#### Energy Minimization Framework
The algorithm reformulates the geometric optimization as an energy minimization problem with a repulsive potential function $U(r)$ between points:

$$
E = \sum_{i=1}^{P}  \sum_{j=1}^{P} U(||x_i-x_j||) ~~~ \text{for} ~~~ i \neq j
$$

#### Dimensionally-Scaled Potential Function
The implementation uses an inverse power law potential scaled with dimensionality: $U(r) = 1/r^k$ where $k = N-1$. This dimensional scaling accounts for:
1. **Harmonic Properties:** In N dimensions, the fundamental solution to Laplace's equation scales as $1/r^{N-2}$.
2. **Volume Scaling:** The surface area of an N-dimensional hypersphere scales approximately as $({2 \pi e}/N)^{N/2}$, requiring stronger repulsion in higher dimensions to overcome concentration of measure phenomena.
3. **Computational Stability:** The exponent ensures numerical stability by preventing excessively large or small force values in high dimensions

#### Normal Modes Linear Combination 
The generated points are then used as scalar factors that multiply each normal mode vector used in the linear combination that makes up the excitation direction vector $\textbf{Q}$.

### Kinetic Energy Control
The additional kinetic energy injected in the system has a fast dissipation rate. Therefore, the program constantly checks the injection energy level and rescale the velocities along the excited direction whenever it is necessary. The kinetic energy along the normalized excitation vector $\textbf{Q}$ direction is calculated by projecting first the current velocities to the excitation direction $\textbf{Q}$ as $\textbf V_p = (\textbf V_{curr} \textbf{Q}) \textbf{Q}$, where $\textbf V_{p}$ and $\textbf{V}_{curr}$ the 3N-dimensional vectors of the projected and current atomic velocities, respectively. The kinetic energy along the excitation direction is thus given by:

$$
E_k = \frac{1}{2} \textbf{V}_{p}^T \textbf{M} \textbf{V}_p
$$

where $\textbf{M}$ is the diagonal mass matrix. At the beginning of each short simulation interval, the remaining excitation energy ($E_k$) is adjusted to the desired excitation level ($E_{exc}$) by modifying the atomic velocities so $\textbf V_{new} = \textbf V_{curr} + (\textbf V_{exc} - \textbf V_{p})$.

With this procedure, the system is kept in a continuous excited state, allowing an effective small, "adiabatic-like" energy injection. The energy injection control is done by projecting the velocities computed during the simulation onto the excited vector, thus obtaining and rescaling the kinetic energy corresponding to it.

### Excitation Direction Update
Since the excitation vector is obtained from the initial conformation, it is dependent of this configuration. As the system is displaced along this direction and change its conformation, the motion loses its directionality due to mainly anharmonic effects. To prevent the structural distortions produced by the displacement along a vector that is no longer valid, the program update the excitation directions based on the trajectory evolution during the previous excitation steps. This procedure allows the system to adaptively find a relaxed path to follow during the next aMDeNM excitations.

If we consider the *n<sup>th</sup>* simulation, the next excitation vector, $\textbf Q_{n+1}$, is determined based on specific parameter values obtained along the trajectory followed in the $\textbf Q_n$ direction. A new excitation vector is defined based on two parameters: the first relates to the effective displacement $\ell$ along $\textbf Q_n$ during the *n<sup>th</sup>* excited dynamics by projecting the mass-weighted effective displacement vector $\textbf d_n = \textbf M^{1/2} ({\langle \textbf r \rangle}_n - \textbf r_n^0)$ onto the normalized mass-weighted excitation vector $\textbf Q_n$, where the ${\langle \textbf r \rangle}_n$ is the average position of the structures over the last 0.1 ps obtained in the *n<sup>th</sup>* excitation, and $\textbf r_n^0$ is the starting position for the following simulation.

The second parameter relates to the relative deviation of the vector $\textbf d_n$ with respect to vector $\textbf Q_n$, evaluated by the angle $\alpha_n$ between them. More precisely, we consider the $\cos {\alpha}_n$ obtained by taking the scalar product of these vectors after normalizing $\textbf d_n$, as following:

$$
\cos {\alpha}_n = \frac {\textbf d_n \textbf Q_n}{||\textbf d_n||}
$$

A precise rule is followed to decide whether to modify the excitation vector direction after every short simulation run. The excitation vector is changed as soon as $\ell_n$, the displacement along $\textbf Q_n$, is larger than a threshold value $\ell_c$, and when $\cos {\alpha}$ is lower than a threshold value ${\cos \alpha}_c$. The conditions for choosing the excitation vector between $\textbf d_n$ and $\textbf Q_n$ for the next simulation are defined by:

$$
\textbf Q_{n+1} = \begin{cases}
  \textbf Q_n, & \text{if } \ell \leq \ell_c \\
  \textbf Q_n, & \text{if } \ell \gt \ell_c \wedge \cos \alpha \ge  {\cos \alpha}_c \\
  \textbf d_n, & \text{if } \ell \gt \ell_c \wedge \cos\alpha \lt {\cos\alpha}_c
\end{cases}
$$

The default values for $\ell_c$ and $\alpha$ are *0.5 m<sup>1/2</sup>Å* (*m* being atomic mass unit) and *60°*, respectively.

### System De-Excitation
At the end of each replica, a de-excitation molecular dynamics is submited in order to recover the equilibrated thermodynamics of the system. This final step aims to remove any residual MDeNM additional energy from the system and provide further sctructural and dynamical exploration.

[Back to top ↩](#)
* ****

## pyAdMD Applications
The Adaptive MDeNM method takes Cα-only or heavy atoms ENM modes or CHARMM normal modes as collective variables to improve Molecular Dynamics sampling.

### ENM
Uses simpified force-field based on particles and springs computed automatically by the program. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

### Physical force-field based normal modes
Uses physical force-field based normal modes computed in *[CHARMM](https://www.charmm.org/charmm/)*. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.


### pyAdMD files
pyAdMD is distributed as a Python handler script that compute ENM modes, uniformly distributes linear combinations of modes in the N-dimensional hypersphere space, manage NAMD simulations and compute the projections along the excitation direction and apply the correction whenever necessary. An additional <code>tools</code> folder includes scripts not used by NAMD, such as the one to write down the CHARMM normal modes.

### Configuration
One can easily setup and run an Adaptive MDeNM simulation by using this script.
The configuration process is straightforward. Some technical aspects will be covered in this section in order to facilitate the method comprehension.

- **Energy injection:** the excitation time of Adaptive MdeNM is *0.2 ps*. This means that every *0.2 ps* the system receives the additional amount of energy chosen by the user. Therefore, when studying large scale motions, it is advised to inject small amounts of energy in order to avoid structural distortions caused by an excessive energy injection. Usually, an excitation energy of *0.125 kcal/mol* is sufficient to achieve a large exploration of the conformational space (*0.5 kcal/mol* if Cα-only ENM).
- **Simulation time:** the total simulation time may require a tuning depending on the system size, energy injection and nature of the motion being excited. Considering a large scale global motion, there is a trade-off between the energy injection and the total simulation time. Larger amounts of energy allows a shorter simulation time, however, this may not be advised as discussed above.
- **Excitation direction update:** as described above, the direction is updated after the system has traveled a distance of *0.5 Å* along the excitation vector and its real displacement has a deviation of *60°* with respect to the theoretical one. The update can also be affected by the amount of energy injected, since higher energy values leads to larger motions. In addition, after each correction the new vector loses directionality due to anharmonic effects. This means that, at a given point, the new vectors are so diffuse that there is no point in proceed the simulation. When this ponit is reached, it is necessary to recompute the normal modes and start again. This is one more reason to not inject high energy values and let the system undergoes the changes slowly.
- **Number of modes and replicas:** the program do a linear combination of the supplied normal modes to compute the excitation direction. This imply that the more modes are provided, the more replicas will be necessary to cover the hyperspace described by these modes.
- **Atom selection:** create an atom selection to apply the energy injection using *[MDAnalysis selection language](https://userguide.mdanalysis.org/1.1.1/selections.html)*. Must be written between quotes.

[Back to top ↩](#)
* ****

## Input Requirements

### Run
#### Parameters: 
- **`-type`/`--type`**: Normal modes model type, **`CA`** ENM, **`HEAVY`** Atoms ENM or **`CHARMM`** (**required**. Default: **`CA`**)

- **`-nm`/`--modes`**: Normal modes to excite (**optional**. Default: **`7,8,9`**)

- **`-ek`/`--energy`**: Excitation energy injection (**optional**. Default: **`0.125`** kcal/mol)

- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

- **`-sel`/`--selection`**: Atom selection to apply the energy injection (**optional**. Default: **`"protein"`**)

- **`-rep`/`--replicas`**: Number of replicas to run (**optional**. Default: **`10`**)


#### Files:
- **`-psf`/`--psffile`**: PSF structure file containing system molecule-specific information (**required**)
  
- **`-pdb`/`--pdbfile`**: PDB structure file in Protein Data Bank format (**required**)

- **`-xsc`/`--xscfile`**: NAMD eXtended System Configuration file (**required**)

- **`-coor`/`--coorfile`**: NAMD binary coordinates file (**required**)

- **`-vel`/`--velfile`**: NAMD binary velocities file (**required**)

- **`-str`/`--strfile`**: CHARMM-style stream file, which contains force field parameters and definitions (**required**)
  
- **`-mod`/`--modefile`**: Binary file containg CHARMM normal mode vectors (**optional**. Required only if **`-type = CHARMM`**)

#### Feature Flags

- **`--no_correc`**: Disable excitation vector direction correction and compute standard MDeNM

- **`--fixed`**: Disable excitation vector correction and keep constant excitation energy injections

### Append
#### Parameters: 
- **`-t`/`--time`**: Simulation time (**optional**. Default: **`250`** ps)

[Back to top ↩](#)
* ****

## Usage Examples

### Using CHARMM normal modes

```
python pyAdMD.py run -type CHARMM -mod modes.mod -psf setup.psf -pdb system.pdb -coor system.coor -vel system.vel -xsc system.xsc -str system.str
```

### Using Cα-only ENM with custom parameters

```
python pyAdMD.py run -type CA -psf setup.psf -pdb system.pdb -coor system.coor -vel system.vel -xsc system.xsc -str system.str -nm 7,15,20 -ek 0.5 -t 100 -sel "protein and resid 115 to 458" -rep 60
```

### Using heavy atoms without direction correction (standard MDeNM)

```
python pyAdMD.py run -type HEAVY -mod modes.mod -psf setup.psf -pdb system.pdb -coor system.coor -vel system.vel -xsc system.xsc -str system.str --no_correc
```

### Restart unfinished pyAdMD simulations

```
python pyAdMD.py restart
```

### Append 100 ps to previously finished pyAdMD simulations

```
python pyAdMD.py append -t 100
```

### Clean previous setup files

```
python pyAdMD.py clean
```

[Back to top ↩](#)
* ****

  ## Analysis
The MDeNM scripts run some basic analysis at the end of the simulation. Inside each replica folder, they can be found as follows:

- **coor-proj.out:** projection of the MD coordinates onto the normal mode space described by the excitation vector;
- **rms-proj.out:** the system RMSD displacement along the excitation vectors;
- **vp-proj.out:** projection of the MD velocities onto the normal mode space described by the excitation vector;
- **ek-proj.out:** displays the additional kinetic energy at each MD step.

[Back to top ↩](#)
* ****

## Citing
Please cite the following paper if you are using any Adaptive MDeNM application in your work:

[Resende-Lara, P. T. et al. *Adaptive collective motions: a hybrid method to improve conformational sampling with molecular dynamics and normal modes.* bioRxiv. doi: 10.1101/2022.11.29.517349](https://www.biorxiv.org/content/10.1101/2022.11.29.517349)

[Back to top ↩](#)
* ****

## Dependencies

  - `python=3.12`
  - `numpy=2.2.5`
  - `scipy=1.16.0`
  - `mdanalysis=2.9.0`
  - `matplotlib=3.10.0`
  - `openmm=8.3.1`
  - `numba=0.61.2`
  - `cupy=13.6.` (optional, for GPU-accelerated ENM calculation. **Note:** CuPy requires matching CUDA toolkit.)

A `conda` enviroment can be easily setup with the provided `pyAdMD.yaml` file containing the necessary Python dependencies, which are:

```  
conda env create -f pyAdMD.yaml
```

[Back to top ↩](#)
* ****

## Contact
If you experience a bug or have any doubt or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
