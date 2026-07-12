"""Reading OpenMM XML restart (.rst) files into engine-agnostic SystemState objects."""

import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict

import numpy as np

from pyadmd.io.namd import NAMDInputReader
from pyadmd.io.state import SystemState


class OpenMMRestartReader:
    """
    Reads an OpenMM XML restart file (.rst) produced by::

        state = simulation.context.getState(getPositions=True, getVelocities=True)
        with open('output.rst', 'w') as f:
            f.write(XmlSerializer.serialize(state))

    The XML ``<State>`` element carries ``<PeriodicBoxVectors>``,
    ``<Positions>``, and ``<Velocities>`` already in OpenMM-native units
    (nm and nm ps⁻¹), so **no unit conversion is performed**.  The resulting
    ``SystemState`` is structurally identical to the one produced by
    ``NAMDInputReader.read_system``, and all downstream code consumes it
    identically.

    Additionally provides ``box_vectors_to_cell``, which converts three
    Cartesian box vectors (nm) to the ``(a, b, c, α, β, γ)`` representation
    used by ``OpenMMSystemBuilder.build`` when no ``.str`` file is supplied.
    """

    @staticmethod
    def read_state(rst_file: str) -> 'SystemState':
        """
        Parse an OpenMM XML restart file and return a ``SystemState``.

        Reads ``<PeriodicBoxVectors>``, ``<Positions>``, and ``<Velocities>``
        from the ``<State>`` element.  The box-alignment rotation
        (``NAMDInputReader.align_box_to_x``) is applied so that downstream
        OpenMM context initialisation always receives box vectors in the
        canonical form required by OpenMM (a along x, b in xy-plane).

        Args:
            rst_file (str): Path to the OpenMM XML restart file.

        Returns:
            SystemState: positions in nm, velocities in nm/ps, box vectors in nm.

        Raises:
            FileNotFoundError: If ``rst_file`` does not exist.
            ValueError: If the XML is missing required ``<PeriodicBoxVectors>``,
                ``<Positions>``, or ``<Velocities>`` elements.
        """
        if not os.path.exists(rst_file):
            raise FileNotFoundError(f"OpenMM restart file not found: {rst_file}")

        tree = ET.parse(rst_file)
        root = tree.getroot()   # <State>

        # ── Periodic box vectors ──────────────────────────────────────────────
        pbv = root.find('PeriodicBoxVectors')
        if pbv is None:
            raise ValueError(
                f"{rst_file}: <PeriodicBoxVectors> element not found. "
                "Ensure the state was serialised with getPositions=True."
            )

        def _parse_vec3(elem, tag):
            el = elem.find(tag)
            if el is None:
                raise ValueError(f"{rst_file}: <{tag}> element missing inside "
                                  "<PeriodicBoxVectors>.")
            return np.array([float(el.attrib['x']),
                              float(el.attrib['y']),
                              float(el.attrib['z'])], dtype=np.float64)

        a_nm = _parse_vec3(pbv, 'A')
        b_nm = _parse_vec3(pbv, 'B')
        c_nm = _parse_vec3(pbv, 'C')

        # ── Positions ────────────────────────────────────────────────────────
        pos_elem = root.find('Positions')
        if pos_elem is None:
            raise ValueError(f"{rst_file}: <Positions> element not found.")
        positions_nm = np.array(
            [[float(p.attrib['x']), float(p.attrib['y']), float(p.attrib['z'])]
             for p in pos_elem.findall('Position')],
            dtype=np.float64
        )

        # ── Velocities ───────────────────────────────────────────────────────
        vel_elem = root.find('Velocities')
        if vel_elem is None:
            raise ValueError(f"{rst_file}: <Velocities> element not found.")
        velocities_nm_ps = np.array(
            [[float(v.attrib['x']), float(v.attrib['y']), float(v.attrib['z'])]
             for v in vel_elem.findall('Velocity')],
            dtype=np.float64
        )

        # ── Apply the same box-alignment rotation used by NAMDInputReader ────
        # OpenMM itself requires this canonical form; applying it here keeps
        # both input paths consistent so SimulationRunner.initialize_state
        # receives identically shaped data regardless of input engine.
        R = NAMDInputReader.align_box_to_x(a_nm, b_nm, c_nm)
        positions_nm     = (R @ positions_nm.T).T
        velocities_nm_ps = (R @ velocities_nm_ps.T).T
        a_rot = R @ a_nm;  a_rot[1] = 0.0;  a_rot[2] = 0.0
        b_rot = R @ b_nm;  b_rot[2] = 0.0
        c_rot = R @ c_nm

        return SystemState(
            positions_nm=positions_nm,
            velocities_nm_ps=velocities_nm_ps,
            box_vectors_nm=[a_rot, b_rot, c_rot],
        )

    @staticmethod
    def box_vectors_to_cell(a: np.ndarray, b: np.ndarray,
                            c: np.ndarray) -> Dict[str, Any]:
        """
        Convert three Cartesian box vectors (nm) to a ``str_box``-shaped dict.

        Produces the same structure as ``NAMDInputReader.parse_str_box`` so
        that ``OpenMMSystemBuilder.build`` can accept it without modification.
        This is used in OpenMM input mode when no ``.str`` file is provided.

        Args:
            a: First box vector in nm, shape (3,).
            b: Second box vector in nm, shape (3,).
            c: Third box vector in nm, shape (3,).

        Returns:
            dict with keys ``xtltype``, ``a``–``c`` (Å), ``alpha``–``gamma``
            (degrees), and ``box_vectors_nm`` (list of three (3,) arrays).
        """
        a_ang = float(np.linalg.norm(a)) * 10.0
        b_ang = float(np.linalg.norm(b)) * 10.0
        c_ang = float(np.linalg.norm(c)) * 10.0

        # Recover angles from dot products
        a_hat = a / np.linalg.norm(a)
        b_hat = b / np.linalg.norm(b)
        c_hat = c / np.linalg.norm(c)

        cos_alpha = float(np.clip(np.dot(b_hat, c_hat), -1.0, 1.0))
        cos_beta  = float(np.clip(np.dot(a_hat, c_hat), -1.0, 1.0))
        cos_gamma = float(np.clip(np.dot(a_hat, b_hat), -1.0, 1.0))

        alpha = math.degrees(math.acos(cos_alpha))
        beta  = math.degrees(math.acos(cos_beta))
        gamma = math.degrees(math.acos(cos_gamma))

        # Classify crystal type by angles (rough heuristic for logging only)
        def _near(v, ref, tol=0.5):
            return abs(v - ref) < tol

        if _near(alpha, 90) and _near(beta, 90) and _near(gamma, 90):
            xtltype = 'orthorhombic'
        elif _near(alpha, 90) and _near(beta, 90) and not _near(gamma, 90):
            xtltype = 'monoclinic'
        elif (_near(alpha, 109.47) and _near(beta, 109.47)
              and _near(gamma, 109.47)):
            xtltype = 'rhdo'
        else:
            xtltype = 'triclinic'

        return {
            'xtltype':        xtltype,
            'a':              a_ang,
            'b':              b_ang,
            'c':              c_ang,
            'alpha':          alpha,
            'beta':           beta,
            'gamma':          gamma,
            'box_vectors_nm': [a.copy(), b.copy(), c.copy()],
        }
