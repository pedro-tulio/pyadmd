"""NAMD binary/stream-file input reading, converted to OpenMM-compatible arrays."""

import math
from typing import Dict, List, Union

import numpy as np
import MDAnalysis as mda

from pyadmd.constants import AKMA_VEL_TO_NM_PS
from pyadmd.io.state import SystemState


class NAMDInputReader:
    """
    Converts NAMD input files to OpenMM-compatible numpy arrays.

    Called once at startup.  After this class has read the initial coordinate,
    velocity, and box files, no NAMD binary format is accessed again.

    The `.vec` combination files written by ModeExciter.combine_modes are also
    read here (once per replica setup).
    """

    @staticmethod
    def read_xsc(xscfile: str) -> List[np.ndarray]:
        """
        Parse a NAMD XSC (extended system configuration) file.

        Reads the box vector information from a NAMD XSC file, which contains
        periodic boundary condition parameters for the simulation cell.

        Args:
            xscfile (str): Path to the XSC file to parse.

        Returns:
            List[np.ndarray]: Three numpy arrays [a_nm, b_nm, c_nm], each of shape (3,),
                representing the box vectors in nanometers. These are the triclinic
                lattice vectors for the simulation cell.

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or incomplete.

        Note:
            XSC file format: step a_x a_y a_z b_x b_y b_z c_x c_y c_z o_x o_y o_z ...
            Coordinates are converted from Ångströms (NAMD format) to nanometers.
        """
        with open(xscfile) as fh:
            lines = [l for l in fh if not l.startswith('#') and l.strip()]
        vals = list(map(float, lines[0].split()))
        # XSC columns: step a_x a_y a_z  b_x b_y b_z  c_x c_y c_z  o_x o_y o_z
        a = np.array([vals[1], vals[2], vals[3]]) * 0.1   # Å to nm
        b = np.array([vals[4], vals[5], vals[6]]) * 0.1
        c = np.array([vals[7], vals[8], vals[9]]) * 0.1
        return [a, b, c]

    @staticmethod
    def align_box_to_x(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """
        Compute rotation matrix to align periodic box vectors to OpenMM requirements.

        OpenMM expects box vectors in a specific canonical form: the first vector along
        the x-axis, the second in the xy-plane with positive y-component, and the third
        as a general vector. This method computes the rotation matrix transforming the
        input NAMD box vectors to this standard form.

        Args:
            a (np.ndarray): First box vector, shape (3,), in nm.
            b (np.ndarray): Second box vector, shape (3,), in nm.
            c (np.ndarray): Third box vector, shape (3,), in nm.

        Returns:
            R (np.ndarray): 3×3 orthonormal rotation matrix (numpy array). Satisfies:
                - R @ a = (|a|, 0, 0)
                - R @ b = (bx, by, 0)  with by ≥ 0
                - R @ c = (cx, cy, cz)  (general)

        Note:
            The rotation matrix is computed via Gram-Schmidt orthogonalization.
            All output vectors maintain the same lengths as inputs (rotation only).
        """
        ax = a / np.linalg.norm(a)
        b_proj = b - np.dot(b, ax) * ax
        ay = b_proj / np.linalg.norm(b_proj)
        az = np.cross(ax, ay)
        R = np.vstack([ax, ay, az]).T
        return R

    @staticmethod
    def read_system(psf_file: str, coor_file: str,
                    vel_file: str, xsc_file: str) -> SystemState:
        """
        Read NAMD binary coordinate and velocity files into a SystemState object.

        Loads the initial configuration from NAMD binary files (.coor, .vel)
        and periodic boundary condition information from an XSC file. All coordinates
        are converted to nanometers and velocities to nm/ps for OpenMM compatibility.

        Args:
            psf_file (str): Path to the PSF (topology) file.
            coor_file (str): Path to NAMD binary coordinate file (.coor format).
            vel_file (str): Path to NAMD binary velocity file (.vel format).
            xsc_file (str): Path to XSC file containing box vectors.

        Returns:
            SystemState: Dataclass containing:
                - positions_nm (np.ndarray, shape (N, 3)): Atomic positions in nm.
                - velocities_nm_ps (np.ndarray, shape (N, 3)): Atomic velocities in nm/ps.
                - box_vectors_nm (list of 3 np.ndarray): Triclinic box vectors in nm.

        Raises:
            IOError: If any input file cannot be read.
            ValueError: If file format is invalid or atom counts don't match.

        Note:
            Positions are converted from Å to nm (factor: 0.1).
            Velocities are converted from AKMA units to nm/ps using the constant
            AKMA_VEL_TO_NM_PS defined at module level.
        """
        u_coor = mda.Universe(psf_file, coor_file, format='NAMDBIN')
        positions_nm = u_coor.atoms.positions * 0.1                  # Å to nm

        u_vel = mda.Universe(psf_file, vel_file, format='NAMDBIN')
        velocities_nm_ps = u_vel.atoms.positions * AKMA_VEL_TO_NM_PS # AKMA to nm/ps

        # Read box vectors from XSC file
        box_vectors_nm = NAMDInputReader.read_xsc(xsc_file)

        return SystemState(
            positions_nm=positions_nm,
            velocities_nm_ps=velocities_nm_ps,
            box_vectors_nm=box_vectors_nm,
        )

    @staticmethod
    def read_nm_vector(psf_file: str, vec_file: str) -> np.ndarray:
        """
        Read a NAMD binary .vec file (mode combination vector).

        Loads a mode combination or velocity vector stored in NAMD binary format.
        These vectors are typically generated by ModeExciter.combine_modes and
        represent superpositions of normal mode eigenvectors.

        Args:
            psf_file (str): Path to the PSF (topology) file for system definition.
            vec_file (str): Path to the NAMDBIN .vec file to read.

        Returns:
            positions (np.ndarray): Array of displacement/velocity vectors,
                shape (N, 3), in Ångströms. These are direction vectors, not
                physical positions, and do not require unit conversion.

        Raises:
            IOError: If either input file cannot be read.
            ValueError: If the PSF and VEC files have incompatible atom counts.

        Note:
            The data is returned in Ångströms without conversion, as it represents
            displacement/velocity directions rather than absolute coordinates.
        """
        u = mda.Universe(psf_file, vec_file, format='NAMDBIN')
        return u.atoms.positions.copy()   # Å

    @staticmethod
    def parse_str_box(str_file: str) -> Dict[str, Union[str, float, List[np.ndarray]]]:
        """
        Parse a CHARMM-style .str stream file and extract unit-cell information.

        The file is expected to contain lines of the form::

            set <key>  <value>

        where the relevant keys are ``xtltype``, ``a``, ``b``, ``c``,
        ``alpha``, ``beta``, and ``gamma``.  All other ``set`` directives
        are silently ignored.

        The method also computes the three triclinic basis vectors in nm
        following the standard crystallographic to Cartesian conversion:

        .. code-block:: none

            a_vec = [a, 0, 0]
            b_vec = [b·cos(γ), b·sin(γ), 0]
            c_vec = [c·cos(β),
                     c·(cos(α) − cos(β)·cos(γ)) / sin(γ),
                     c·√(1 − cos²β − ((cos(α) − cos(β)·cos(γ)) / sin(γ))²)]

        All lengths are converted Å to nm before being stored.

        Args:
            str_file (str): Path to the CHARMM .str file.

        Returns:
            dict: Dictionary with the following keys:
                - ``xtltype`` (str): Crystal type label, e.g. ``'orthorhombic'``
                - ``a``, ``b``, ``c`` (float): Cell edge lengths in Ångströms
                - ``alpha``, ``beta``, ``gamma`` (float): Cell angles in degrees
                - ``box_vectors_nm`` (list[np.ndarray]): Three (3,) arrays in nm
                  representing triclinic basis vectors, ready for ``psf.setBox()``

        Raises:
            IOError: If the file cannot be read.
            ValueError: If the file format is invalid or required parameters are missing.
        """
        kv: dict = {}
        _required = {'xtltype', 'a', 'b', 'c', 'alpha', 'beta', 'gamma'}

        with open(str_file) as fh:
            for line in fh:
                stripped = line.strip()
                # Accept both "set key value" and "set key  value" (any whitespace)
                if not stripped.lower().startswith('set '):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                key   = parts[1].lower()
                value = parts[2]
                if key in _required:
                    kv[key] = value

        missing = _required - set(kv.keys())
        if missing:
            raise ValueError(
                f"parse_str_box: required key(s) {missing} not found in {str_file}"
            )

        xtltype = kv['xtltype']
        a_ang   = float(kv['a'])
        b_ang   = float(kv['b'])
        c_ang   = float(kv['c'])
        alpha   = float(kv['alpha'])   # degrees
        beta    = float(kv['beta'])
        gamma   = float(kv['gamma'])

        # Crystallographic to Cartesian (lengths stay in nm after the ×0.1 below)
        cos_a = math.cos(math.radians(alpha))
        cos_b = math.cos(math.radians(beta))
        cos_g = math.cos(math.radians(gamma))
        sin_g = math.sin(math.radians(gamma))

        cx = cos_b
        cy = (cos_a - cos_b * cos_g) / sin_g
        cz_sq = 1.0 - cx * cx - cy * cy
        if cz_sq < 0.0:
            raise ValueError(
                f"parse_str_box: non-physical cell angles in {str_file} "
                f"(α={alpha}, β={beta}, γ={gamma})"
            )
        cz = math.sqrt(cz_sq)

        # Build vectors in Å, then convert to nm
        a_vec = np.array([a_ang,            0.0,        0.0]) * 0.1
        b_vec = np.array([b_ang * cos_g, b_ang * sin_g, 0.0]) * 0.1
        c_vec = np.array([c_ang * cx,    c_ang * cy,  c_ang * cz]) * 0.1

        return {
            'xtltype':       xtltype,
            'a':             a_ang,
            'b':             b_ang,
            'c':             c_ang,
            'alpha':         alpha,
            'beta':          beta,
            'gamma':         gamma,
            'box_vectors_nm': [a_vec, b_vec, c_vec],
        }
