"""DCD-file introspection helpers used for crash recovery / restart bookkeeping."""

import glob
import json
import os
import struct


def _count_dcd_frames(dcd_path: str) -> int:
    """
    Return the number of frames stored in a DCD file by reading its binary
    header directly — no topology required, immune to atom-count mismatches.

    DCD CORD header layout (CHARMM / OpenMM convention, little-endian):
      bytes  0- 3 : block length (int32)  = 84
      bytes  4- 7 : magic string 'CORD'
      bytes  8-11 : NSET  — total number of frames (int32)

    Args:
        dcd_path (str): Path to the DCD trajectory file.

    Returns:
        int: Number of frames recorded in the DCD header, or 0 on any error.
    """
    try:
        with open(dcd_path, 'rb') as fh:
            fh.read(4)          # block length (84)
            magic = fh.read(4)  # 'CORD'
            if magic not in (b'CORD', b'VELD'):
                return 0
            n_frames = struct.unpack('<i', fh.read(4))[0]
        return max(0, n_frames)
    except Exception:
        return 0


def find_last_completed_cycle(rep_dir: str) -> int:
    """
    Find the last completed cycle in a replica directory.

    In the OpenMM backend no per-cycle step_N.coor files are written; instead
    the authoritative record is the 'cycle' field inside correction_state.json,
    which is updated every 10 cycles by _save_correction_state().  If that file
    is absent (very early crash), the DCD trajectory frame count is used as a
    rough lower bound.

    Args:
        rep_dir (str): Path to replica directory to scan.

    Returns:
        int: Highest completed cycle number, or 0 if none can be determined.
    """
    # Primary: correction_state.json written by _save_correction_state every 10 cycles
    cs_path = os.path.join(rep_dir, "correction_state.json")
    if os.path.exists(cs_path):
        try:
            with open(cs_path) as fh:
                cs = json.load(fh)
            cycle = int(cs.get('cycle', 0))
            if cycle > 0:
                return cycle
        except Exception:
            pass

    # Secondary: count DCD frames by reading the binary header directly.
    dcd_files = glob.glob(f"{rep_dir}/rep*.dcd")
    if dcd_files:
        dcd_files.sort(key=os.path.getmtime, reverse=True)
        n_frames = _count_dcd_frames(dcd_files[0])
        if n_frames > 0:
            return n_frames   # 1 DCD frame per cycle (DCDReporter period = n_steps steps)

    return 0
