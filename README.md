# Adaptive MDeNM User Guide

The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of multiple-replica short MD simulations in which motions described by a given subset of low-frequency NMs are kinetically excited. This is achieved by adding additional atomic velocities along several randomly determined linear combinations of NM vectors, thus allowing an efficient coupling between slow and fast motions.

This new approach, aMDeNM, automatically controls the energy injection and take the natural constraints imposed by the structure and the environment into account during protein conformational sampling, which prevent structural distortions all along the simulation.Due to the stochasticity of thermal motions, NM eigenvectors move away from the original directions when used to displace the protein, since the structure evolves into other potential energy wells. Therefore, the displacement along the modes is valid for small distances, but the displacement along greater distances may deform the structure of the protein if no care is taken. The advantage of this methodology is to adaptively change the direction used to displace the system, taking into account the structural and energetic constraints imposed by the system itself and the medium, which allows the system to explore new pathways.

This document will give an overview of the Adaptive MDeNM method and help to properly setup and run a simulation.
* ****

#### Table of Contents
- [Adaptive MDeNM User Guide](#adaptive-mdenm-user-guide)
      - [Table of Contents](#table-of-contents)
  - [Method Overview](#method-overview)
    - [RMSD Filtering](#rmsd-filtering)
    - [Kinetic Energy Control](#kinetic-energy-control)
    - [Excitation Direction Update](#excitation-direction-update)
  - [aMDeNM Applications](#amdenm-applications)
    - [Physical force-field based normal modes](#physical-force-field-based-normal-modes)
  - [Preparing to run aMDeNM](#preparing-to-run-amdenm)
    - [NAMD input files](#namd-input-files)
    - [CHARMM input files](#charmm-input-files)
    - [Adaptive MDeNM files](#adaptive-mdenm-files)
    - [Configuration](#configuration)
  - [Analysis](#analysis)
  - [Citing](#citing)
  - [Contact](#contact)
* ****

## Method Overview

The Molecular Dynamics with excited Normal Modes (MDeNM) is a enhanced sampling molecular dynamics method that uses normal modes as collective variables in order do increase the conformational space explored during MD simulation. This is done by injecting an incremental energy to the system, thus assigning additional atomic velocities along the direction of a given NM (or a combination of a NM set). The combination of the velocities from MD and those provided by the NM vectors properly couple slow and fast motions, allowing one to obtain large time scale movements, such as domain transitions in a feasible simulation time.

### RMSD Filtering
An optmized MDeNM run is ensured by evaluating the excitation directions through a RMSD threshold (*t*) - that is user defined. During the run, a hypersphere with radius *1 Å* is set up around the reference position. Each point at its surface represents a different possible NM combination. When searching for a new NM combination, the structure is displaced along this direction untill it reaches the hypersphere's surface. Then, its RMSD value is tested against all previously accepted directions. If this value is equal or greater than *t*, this new vector is retained. Otherwise, a new combination must be computed. This step avoids redundant directions to be excited during the simulation, thus improving the computing time.

### Kinetic Energy Control

The additional kinetic energy injected in the system has a fast dissipation rate. Therefore, the program constantly checks the injection energy level and rescale the velocities along the excited direction whenever it is necessary. With this procedure, the system is kept in a continuous excited state, allowing an effective small, "adiabatic-like" energy injection. The energy injection control is done by projecting the velocities computed during the simulation onto the excited vector, thus obtaining and rescaling the kinetic energy corresponding to it.

### Excitation Direction Update

Since the excitation vector is obtained from the initial conformation, it is dependent of this configuration. As the system is displaced along this direction and change its conformation, the motion loses its directionality due to mainly anharmonic effects. To prevent the structural distortions produced by the displacement along a vector that is no longer valid, the program update the excitation directions based on the trajectory evolution during the previous excitation steps. This procedure allows the system to adaptively find a relaxed path to follow during the next MDeNM excitations.

The update depends on two variables: a distance (*r<sub>d</sub>*) by which the system has been displaced along the excitation vector; and a given angle (*α*) by which the real displacement has deviated from the ideal motion described by the excitation vector. Everytime the system reaches a displacement equals to *r<sub>d</sub>* along the excitation direction, the *α* angle is computed. If the deviation is lesser than a threshold value, the current direction is retained and the simulation resumes. Otherwise, a new vector is generated considering the motion presented by the system in the last excitation steps. The default values for *r<sub>d</sub>* and *α* are *0.5 Å* and *60°*, respectively.

[Back to top ↩](#)
* ****

## aMDeNM Applications

The Adaptive MDeNM method takes CHARMM normal modes as collective variables to increase the MD sampling. Other inputs, such as ENM modes will soon be implemented.

### Physical force-field based normal modes

Uses physical force-field based normal modes computed in *[CHARMM](https://www.charmm.org/charmm/)*. A given normal mode (or a linear combination of several modes) is used to excite the system during the molecular dyamics simulation.

[Back to top ↩](#)
* ****

## Preparing to run aMDeNM

The Adaptive MDeNM simulations are done using *[NAMD 3.0](http://www.ks.uiuc.edu/Research/namd/)* and CHARMM36m forcefield. We implemented the code so the molecular dynamics are computed exclusively on GPU. A short equilibration molecular dynamics (NAMD) and the normal modes analysis (CHARMM) must be computed beforehand as prerequisited steps to aMDeNM calculations. We strongly recommend that all system preparation be done with *[CHARMM-GUI](http://charmm-gui.org)*.

### NAMD input files

This is a prerequired step to perform aMDeNM simulations. It consists in performing a short equilibration MD to store the final atomic velocities and positions. At the end of this step, the following files must be provided to the following procedure:
- topology file (.psf);
- structure file (.pdb);
- periodic cell parameters (.xsc);
- final coordinates (.coor);
- final velocities (.vel).
- box and PBC additional parameters (.str);

### CHARMM input files

It is necessary to compute the normal modes from the last MD coordinates and store the vectors from the low-frequency end of the vibrational spectrum on a binary file. Thus, aMDeNM will require the following CHARMM files to properly run:
- system topology (.psf);
- normal modes vectors (.mod).

### Adaptive MDeNM files

Adaptive MDeNM is distributed as a Python handler script that manage the NAMD simulations and compute the projections along the excitation direction and apply the correction whenever necessary. An additional <code>.modules</code> folder includes scripts not used by NAMD, such as the one to write down the CHARMM normal modes.

### Configuration

One can easily setup and run an Adaptive MDeNM simulation by using this script.
The configuration process is straightforward. Some technical aspects will be covered in this section in order to facilitate the method comprehension.

- **Energy injection:** the excitation time of Adaptive MdeNM is *0.2 ps*. This means that every *0.2 ps* the system receives the additional amount of energy chosen by the user. Therefore, when studying large scale motions, it is advised to inject small amounts of energy in order to avoid structural distortions caused by an excessive energy injection. Usually, an excitation energy of *0.125 kcal/mol* is sufficient to achieve a large exploration of the conformational space.
- **Simulation time:** the total simulation time may require a tuning depending on the system size, energy injection and nature of the motion being excited. Considering a large scale global motion, there is a trade-off between the energy injection and the total simulation time. Larger amounts of energy allows a shorter simulation time, however, this may not be advised as discussed above.
- **Excitation direction update:** as described above, the direction is updated after the system has traveled a distance of *0.5 Å* along the excitation vector and its real displacement has a deviation of *60°* with respect to the theoretical one. The update can also be affected by the amount of energy injected, since higher energy values leads to larger motions. In addition, after each correction the new vector loses directionality due to anharmonic effects. This means that, at a given point, the new vectors are so diffuse that there is no point in proceed the simulation. When this ponit is reached, it is necessary to recompute the normal modes and start again. This is one more reason to not inject high energy values and let the system undergoes the changes slowly.
- **Number of modes and replicas:** the program do a linear combination of the supplied normal modes to compute the excitation direction. This imply that the more modes are provided, the more replicas will be necessary to cover the hyperspace described by these modes.

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

## Contact

If you experience a bug or have any doubt or suggestion, feel free to contact:

*[laraptr [at] unicamp.br](mailto:laraptr@unicamp.br)*

[Back to top ↩](#)
