"""Building an OpenMM System from CHARMM/NAMD inputs."""

import os
from typing import Any, Dict, List, Optional, Tuple

import openmm as mm
from openmm import app, unit
from openmm.app import CharmmPsfFile, CharmmParameterSet

from pyadmd.console import ConsoleConfig


class OpenMMSystemBuilder:
    """
    Builds an OpenMM System from CHARMM/NAMD inputs using native OpenMM loaders.

    Called once per run; the resulting System object is shared across all
    replicas (it encodes topology and forces, which are replica-independent).

    Force-field settings are hard-wired:
      PME, 10 Å cutoff, 8 Å switch, HBonds constraints, rigidWater=True.
    """

    # Residue-name sets used for system-type detection
    LIPID_RESIDUES = {
        'POPC', 'POPE', 'DPPC', 'DPPE', 'DLPC', 'DMPC', 'DLPE', 'DMPE',
        'DEPC', 'DSPC', 'DAPE', 'DUPE', 'DHPC', 'LYPC', 'CHL1', 'ERG',
        'POPG', 'DPPG', 'CARD', 'POPS', 'DPPS', 'DOPS', 'SAPI', 'PIPI',
        'PAPE', 'OAPE', 'LAPE', 'LPPC', 'PSM', 'DPCE', 'DXCE', 'POPE',
    }
    NUCLEIC_RESIDUES = {
        'ADE', 'CYT', 'GUA', 'THY', 'URA',
        'DA', 'DC', 'DG', 'DT', 'DI',
        'RA', 'RC', 'RG', 'RU',
        'A',  'C',  'G',  'T',  'U',  'I',
    }

    # CHARMM36m parameter files to load from charmm_toppar/
    PARAM_FILES = [
        'par_all36m_prot.prm',
        'par_all36_na.prm',
        'par_all36_carb.prm',
        'par_all36_lipid.prm',
        'par_all36_cgenff.prm',
        'par_interface.prm',
        'toppar_water_ions.str',
    ]

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize the OpenMM system builder.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console

    def detect_system_type(self, topology: app.Topology) -> str:
        """
        Detect the system type based on residue names.

        Args:
            topology (openmm.app.Topology): The topology to analyze.

        Returns:
            str: 'membrane' if any lipid residues found,
                 'nucleic' if any nucleic acid residues found,
                 'globular' otherwise.
        """
        names = {r.name.upper() for r in topology.residues()}
        if names & self.LIPID_RESIDUES:
            return 'membrane'
        if names & self.NUCLEIC_RESIDUES:
            return 'nucleic'
        return 'globular'

    def _collect_params(self, toppar_dir: str) -> List[str]:
        """
        Collect existing parameter file paths from charmm_toppar directory.

        Args:
            toppar_dir (str): Path to the CHARMM parameter directory.

        Returns:
            list: List of existing parameter file paths.
        """
        return [os.path.join(toppar_dir, f)
                for f in self.PARAM_FILES
                if os.path.exists(os.path.join(toppar_dir, f))]

    def build(self, psf_file: str, toppar_dir: str, temperature: float = 303.15,
              pressure: float = 1.01325, str_box: Optional[Dict[str, Any]] = None) -> Tuple[app.CharmmPsfFile, mm.System, str]:
        """
        Build OpenMM system from PSF and CHARMM parameters.

        Args:
            psf_file (str): Path to the PSF topology file.
            toppar_dir (str): Directory containing CHARMM36m parameter files.
            temperature (float): Simulation temperature in K (default: 303.15).
            pressure (float): Target pressure in bar (default: 1.01325).
            str_box (dict, optional): Dictionary returned by NAMDInputReader.parse_str_box().
                When provided, its triclinic basis vectors (in nm) are passed to psf.setBox()
                so that PME grid allocation uses the real cell dimensions at System-build time.
                When None, a 10 Å orthogonal placeholder is used (legacy behaviour, not recommended).

        Returns:
            psf (openmm.app.CharmmPsfFile): Topology object with box set.
            system (openmm.System): OpenMM System ready for Simulation creation.
            system_type (str): One of 'globular', 'membrane', 'nucleic'.

        Raises:
            FileNotFoundError: If no CHARMM parameter files are found in toppar_dir.
        """
        psf = CharmmPsfFile(psf_file)
        system_type = self.detect_system_type(psf.topology)

        # Set the unit-cell box that OpenMM uses for PME grid allocation.
        # Using the real triclinic vectors from the .str file avoids the
        # silent PME mis-sizing that a dummy 10 Ang box causes.
        if str_box is not None:
            psf.setBox(
                str_box['a'] * 0.1 * unit.nanometer,
                str_box['b'] * 0.1 * unit.nanometer,
                str_box['c'] * 0.1 * unit.nanometer,
                str_box['alpha'] * unit.degree,
                str_box['beta']  * unit.degree,
                str_box['gamma'] * unit.degree,
            )
        else:
            psf.setBox(10.0, 10.0, 10.0)   # legacy placeholder (not recommended)

        # Load CHARMM parameters
        param_files = self._collect_params(toppar_dir)
        if not param_files:
            raise FileNotFoundError(
                f"No CHARMM parameter files found in {toppar_dir}. "
                "Ensure charmm_toppar.zip has been extracted."
            )
        params = CharmmParameterSet(*param_files)

        # Build System (matching conf.namd non-bonded settings)
        system = psf.createSystem(
            params,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            switchDistance=0.8 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        # Assign force groups for per-term energy decomposition.
        # Group assignments (NAMD-like naming convention):
        #   0 = HarmonicBondForce        (BOND)
        #   1 = HarmonicAngleForce       (ANGLE)
        #   2 = PeriodicTorsionForce     (DIHED)
        #   3 = CustomTorsionForce       (IMPRP — CHARMM impropers)
        #   4 = NonbondedForce           (ELEC + VDW combined; split at query time)
        #   5 = CMAPTorsionForce         (CMAP)
        #   6 = CustomNonbondedForce     (VDW correction, e.g. NBFIX)
        #   7 = CustomBondForce          (UREY-BRADLEY or extra bond corrections)
        _force_group_map = {
            mm.HarmonicBondForce:     0,
            mm.HarmonicAngleForce:    1,
            mm.PeriodicTorsionForce:  2,
            mm.CustomTorsionForce:    3,
            mm.NonbondedForce:        4,
            mm.CMAPTorsionForce:      5,
            mm.CustomNonbondedForce:  6,
            mm.CustomBondForce:       7,
        }
        for force in system.getForces():
            grp = _force_group_map.get(type(force), 31)
            force.setForceGroup(grp)

        # Add barostat
        if system_type == 'membrane':
            barostat = mm.MonteCarloMembraneBarostat(
                pressure * unit.bar,
                0.0 * unit.bar * unit.nanometer,   # surface tension = 0
                temperature * unit.kelvin,
                mm.MonteCarloMembraneBarostat.XYIsotropic,
                mm.MonteCarloMembraneBarostat.ZFree,
                25,
            )
        else:
            barostat = mm.MonteCarloBarostat(
                pressure * unit.bar,
                temperature * unit.kelvin,
                25,
            )
        system.addForce(barostat)

        print(f"{self.console.PGM_NAM}OpenMM system built: "
              f"{self.console.EXT}{system.getNumParticles()}{self.console.STD} atoms, "
              f"type={self.console.EXT}{system_type}{self.console.STD}.")
        return psf, system, system_type
