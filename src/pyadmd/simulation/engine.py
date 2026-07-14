"""Persistent per-replica OpenMM simulation wrapper."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import openmm as mm
from openmm import app, unit, Platform

from pyadmd.console import ConsoleConfig
from pyadmd.io.state import SystemState


class OpenMMSimulationEngine:
    """
    Persistent OpenMM simulation wrapper for one aMDeNM replica.

    The (shared) System is passed in; a fresh LangevinMiddleIntegrator and
    Context are created per replica, giving each replica an independent RNG
    seed.  Three reporters are attached once and fire automatically on every
    call to simulation.step():
      - DCDReporter:        1 frame per n_steps-step cycle
      - StateDataReporter:  1 energy/temperature row per cycle
      - CheckpointReporter: exact state every 10 * n_steps steps (10 cycles)

    All MD runs through run_cycle() or run_deexcitation(); no NAMD binary
    files are written.
    """

    def __init__(self, console: ConsoleConfig, psf: app.CharmmPsfFile, system: mm.System, temperature: float,
                 platform_name: str = 'auto', n_threads: Optional[int] = None,
                 device_index: int = 0, rep_num: int = 1,
                 is_restart: bool = False, full_ener: bool = False,
                 n_steps: int = 100) -> None:
        """
        Initialize the simulation engine for a single replica.

        Args:
            console (ConsoleConfig): Console configuration object.
            psf (openmm.app.CharmmPsfFile): The PSF topology.
            system (openmm.System): The OpenMM system (shared across replicas).
            temperature (float): Simulation temperature in K.
            platform_name (str): Platform to use: 'auto', 'cuda', 'opencl', 'cpu'.
            n_threads (int, optional): Number of CPU threads for CPU platform.
            device_index (int): GPU device index for CUDA/OpenCL.
            rep_num (int): Replica number (used for output file names).
            is_restart (bool): Whether this is a restart (append to existing output files).
            full_ener (bool): If True, write per-term energy decomposition to
                rep{N}_ener_decomp.log each cycle (--full_ener flag).
            n_steps (int): Number of MD steps per excitation cycle.
        """
        self.console = console
        self.n_atoms = system.getNumParticles()
        self._temperature = temperature
        self._full_ener = full_ener

        platform, properties = self._select_platform(
            platform_name, n_threads, device_index
        )

        # Fresh integrator per replica (independent Langevin RNG state)
        integrator = mm.LangevinMiddleIntegrator(
            temperature * unit.kelvin,
            1.0 / unit.picosecond,      # friction
            2.0 * unit.femtoseconds,    # timestep
        )

        self.simulation = app.Simulation(
            psf.topology, system, integrator, platform, properties
        )

        # Attach persistent reporters (append=True on restart)
        dcd_file         = f'rep{rep_num}.dcd'
        log_file         = f'rep{rep_num}.log'
        ener_decomp_file = f'rep{rep_num}_ener_decomp.log'
        self._total_steps = 0   # updated each run_cycle call; needed for 'progress'
        self.simulation.reporters.append(
            app.DCDReporter(dcd_file, n_steps, append=is_restart, enforcePeriodicBox=False)
        )
        # Full StateDataReporter: every available scalar field written to the log file
        self.simulation.reporters.append(
            app.StateDataReporter(
                log_file, n_steps,
                step=True,
                time=True,
                potentialEnergy=True,
                kineticEnergy=True,
                totalEnergy=True,
                temperature=True,
                volume=True,
                density=True,
                speed=True,
                append=is_restart,
            )
        )
        # Checkpoint every 10 cycles
        self.simulation.reporters.append(
            app.CheckpointReporter('checkpoint.chk', 10 * n_steps)
        )

        # If required, print per-term energy decomposition log
        if self._full_ener:
            _ener_mode = 'a' if is_restart else 'w'
            self._ener_decomp_fh = open(ener_decomp_file, _ener_mode)
            if not is_restart:
                self._ener_decomp_fh.write(
                    f"{'REP':>5} {'STEP':>6} {'BOND':>10} {'ANGLE':>10} {'DIHED':>10} "
                    f"{'IMPRP':>10} {'CMAP':>10} {'UBREY':>10} {'NBFIX':>10} "
                    f"{'NONBONDED':>12} {'POTENTIAL':>12} {'KINETIC':>10} "
                    f"{'TOTAL':>12} {'TEMP_K':>8} {'VOL_A3':>12} {'DENS_GCM3':>10}\n"
                )
                self._ener_decomp_fh.flush()

        # Map force group index
        self._force_group_labels = {
            0: 'BOND',
            1: 'ANGLE',
            2: 'DIHED',
            3: 'IMPRP',
            4: 'NONBONDED',   # elec + vdw combined (NonbondedForce)
            5: 'CMAP',
            6: 'NBFIX',
            7: 'UBREY',
        }

        print(f"{console.PGM_NAM}OpenMM platform: "
              f"{console.EXT}{platform.getName()}{console.STD}.")

    def _select_platform(self, prefer: str, n_threads: Optional[int], device_index: int) -> Tuple[Platform, Dict[str, str]]:
        """
        Select the OpenMM platform and return the corresponding properties.

        Args:
            prefer (str): Preferred platform: 'auto', 'cuda', 'opencl', 'cpu'.
            n_threads (int, optional): Number of threads for CPU platform.
            device_index (int): GPU device index for CUDA/OpenCL.

        Returns:
            platform (openmm.Platform): The selected platform.
            properties (dict): Platform-specific properties.

        Raises:
            RuntimeError: If the requested platform is not available.
        """
        gpu_props = {'DeviceIndex': str(device_index), 'Precision': 'mixed'}

        if prefer in ('cuda', 'auto'):
            try:
                return mm.Platform.getPlatformByName('CUDA'), gpu_props
            except Exception:
                if prefer == 'cuda':
                    raise RuntimeError("CUDA platform not available.")

        if prefer in ('opencl', 'auto'):
            try:
                return mm.Platform.getPlatformByName('OpenCL'), gpu_props
            except Exception:
                if prefer == 'opencl':
                    raise RuntimeError("OpenCL platform not available.")

        props = {}
        if n_threads:
            props['Threads'] = str(n_threads)
        return mm.Platform.getPlatformByName('CPU'), props

    def initialize_state(self, state: SystemState) -> None:
        """
        Push positions, velocities, and box vectors from a SystemState into the Context.

        Args:
            state (SystemState): System state object containing positions,
                velocities, and box vectors in nm and nm/ps units.
        """
        a, b, c = state.box_vectors_nm
        a, b, c = np.array(a), np.array(b), np.array(c)
        # Reduce box vectors to OpenMM's required form
        c[0] -= a[0] * np.round(c[0] / a[0])
        c[1] -= b[1] * np.round(c[1] / b[1])
        b[0] -= a[0] * np.round(b[0] / a[0])
        self.simulation.context.setPeriodicBoxVectors(
            mm.Vec3(*a) * unit.nanometer,
            mm.Vec3(*b) * unit.nanometer,
            mm.Vec3(*c) * unit.nanometer,
        )
        pos = [mm.Vec3(*p) for p in state.positions_nm]
        self.simulation.context.setPositions(
            unit.Quantity(pos, unit.nanometer)
        )
        if state.velocities_nm_ps is not None:
            vel = [mm.Vec3(*v) for v in state.velocities_nm_ps]
            self.simulation.context.setVelocities(
                unit.Quantity(vel, unit.nanometer / unit.picosecond)
            )
        else:
            self.simulation.context.setVelocitiesToTemperature(
                self._temperature * unit.kelvin
            )

    def load_checkpoint(self, chk_file: str) -> None:
        """
        Restore exact physical state (pos, vel, box, RNG) from an OpenMM checkpoint.

        Args:
            chk_file (str): Path to the checkpoint file.
        """
        self.simulation.loadCheckpoint(chk_file)

    def save_checkpoint(self, chk_file: str) -> None:
        """
        Save the exact current physical state (pos, vel, box, RNG) to an
        OpenMM checkpoint file, enabling a bit-identical continuation later.

        Args:
            chk_file (str): Output path for the checkpoint file.
        """
        self.simulation.saveCheckpoint(chk_file)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Return current (pos_nm, vel_nm_ps, box_nm) as numpy arrays from the Context.

        Returns:
            positions_nm (np.ndarray): Atomic positions in nm, shape (N,3).
            velocities_nm_ps (np.ndarray): Atomic velocities in nm/ps, shape (N,3).
            box_vectors_nm (list of np.ndarray): Three box vectors in nm, each (3,).
        """
        state = self.simulation.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        return (np.array(pos), np.array(vel), [box[0], box[1], box[2]])

    def set_velocities(self, vel_nm_ps: np.ndarray) -> None:
        """
        Update only the velocity portion of the Context (used for EK rescaling).

        Args:
            vel_nm_ps (np.ndarray): New velocities in nm/ps, shape (N,3).
        """
        vel = [mm.Vec3(*v) for v in vel_nm_ps]
        self.simulation.context.setVelocities(
            unit.Quantity(vel, unit.nanometer / unit.picosecond)
        )

    def get_energy_decomposition(self) -> Dict[str, float]:
        """
        Query each force group individually and return a per-term energy breakdown
        in kcal/mol, equivalent to NAMD's ENERGY: output line.

        The NonbondedForce group (group 4) contains both electrostatics and vdW
        and cannot be split further by OpenMM without separate Force objects; it
        is reported as 'NONBONDED'. All values are in kcal/mol.

        Returns:
            dict mapping label to energy (kcal/mol), plus 'POTENTIAL', 'KINETIC',
            'TOTAL', 'TEMPERATURE', 'VOLUME', 'DENSITY'.
        """
        KJ_TO_KCAL = 1.0 / 4.184
        ctx = self.simulation.context

        energies: Dict[str, float] = {}

        # Per-force-group potential energy terms
        for grp, label in self._force_group_labels.items():
            state = ctx.getState(getEnergy=True, groups={grp})
            e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            energies[label] = e_kj * KJ_TO_KCAL

        # Bulk thermodynamic quantities (full system state)
        full = ctx.getState(getEnergy=True)
        pe = full.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        ke = full.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
        energies['POTENTIAL']   = pe * KJ_TO_KCAL
        energies['KINETIC']     = ke * KJ_TO_KCAL
        energies['TOTAL']       = (pe + ke) * KJ_TO_KCAL

        # Temperature: compute exact constrained DOF count, matching StateDataReporter.
        # getNumDegreesOfFreedom() was added in OpenMM 8.1; for older versions we
        # replicate the same formula: 3*N - 3*N_constraints - 3 (CMMotionRemover).
        system = self.simulation.system
        try:
            n_dof = system.getNumDegreesOfFreedom()
        except AttributeError:
            n_dof = 3 * system.getNumParticles()
            for i in range(system.getNumConstraints()):
                n_dof -= 1
            # Each CMMotionRemover removes 3 DOF
            for i in range(system.getNumForces()):
                if isinstance(system.getForce(i), mm.CMMotionRemover):
                    n_dof -= 3
        kB_kj = 0.008314462            # kJ/mol/K
        energies['TEMPERATURE'] = (2.0 * ke) / (n_dof * kB_kj)

        # Box volume (nm³ to Å³) and density
        box = full.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        vol_nm3 = float(np.dot(box[0], np.cross(box[1], box[2])))
        energies['VOLUME'] = vol_nm3 * 1000.0   # Å³

        # Density: sum of masses (amu) / volume (Å³), convert to g/cm³
        # 1 amu/Å³ = 1.66054 g/cm³
        total_mass_amu = sum(
            self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(self.n_atoms)
        )
        vol_cm3 = vol_nm3 * 1e-21          # nm³ to cm³
        energies['DENSITY'] = (total_mass_amu * 1.66054e-24) / vol_cm3  # g/cm³

        return energies

    def write_energy_decomposition(self, step: int, rep: int) -> None:
        """
        Append one row of per-term energies (kcal/mol) to the replica's
        ener_decomp.log file. Fixed-width columns match the header written
        in __init__; no output is sent to stdout.

        Args:
            step (int): Current cycle number.
            rep  (int): Replica number.
        """
        e = self.get_energy_decomposition()
        row = (
            f"{rep:>5d} {step:>6d} "
            f"{e.get('BOND',        0.0):>10.3f} "
            f"{e.get('ANGLE',       0.0):>10.3f} "
            f"{e.get('DIHED',       0.0):>10.3f} "
            f"{e.get('IMPRP',       0.0):>10.3f} "
            f"{e.get('CMAP',        0.0):>10.3f} "
            f"{e.get('UBREY',       0.0):>10.3f} "
            f"{e.get('NBFIX',       0.0):>10.3f} "
            f"{e.get('NONBONDED',   0.0):>12.3f} "
            f"{e.get('POTENTIAL',   0.0):>12.3f} "
            f"{e.get('KINETIC',     0.0):>10.3f} "
            f"{e.get('TOTAL',       0.0):>12.3f} "
            f"{e.get('TEMPERATURE', 0.0):>8.2f} "
            f"{e.get('VOLUME',      0.0):>12.2f} "
            f"{e.get('DENSITY',     0.0):>10.5f}\n"
        )
        self._ener_decomp_fh.write(row)
        self._ener_decomp_fh.flush()

    def close(self) -> None:
        """Close open file handles (call after the simulation loop finishes)."""
        if self._full_ener and hasattr(self, '_ener_decomp_fh') and not self._ener_decomp_fh.closed:
            self._ener_decomp_fh.close()

    def detach_main_dcd(self) -> None:
        """Remove the main replica DCD reporter (index 0) so that the
        de-excitation run does not write extra frames into rep{N}.dcd."""
        self.simulation.reporters.pop(0)

    def run_cycle(self, n_steps: int = 100, rep: int = 0, loop: int = 0) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Advance the simulation by n_steps.

        DCDReporter and StateDataReporter fire automatically.
        CheckpointReporter fires every 10 * n_steps steps (10 cycles).
        If --full_ener was set, per-term energy decomposition is written to
        rep{N}_ener_decomp.log after each cycle.

        Args:
            n_steps (int): Number of integration steps to run.
            rep (int): Replica number, passed through to the energy log.
            loop (int): Current cycle number, passed through to the energy log.

        Returns:
            pos_nm (np.ndarray): Atomic positions in nm, shape (N,3).
            vel_nm_ps (np.ndarray): Atomic velocities in nm/ps, shape (N,3).
            box_nm (list of np.ndarray): Three box vectors in nm, each (3,).
        """
        self.simulation.step(n_steps)
        if self._full_ener:
            self.write_energy_decomposition(step=loop, rep=rep)
        return self.get_state()
