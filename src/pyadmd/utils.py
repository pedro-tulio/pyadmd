"""Miscellaneous helpers: archive extraction and CHARMM normal mode vector writing."""

import importlib.resources as importlib_resources
import os
import shutil
import subprocess
import sys
import zipfile

from pyadmd.console import ConsoleConfig


def unzip_file(filepath: str, dest_dir: str) -> None:
    """
    Extract files from a .zip compressed file.

    Args:
        filepath (str): Path to .zip file to extract.
        dest_dir (str): Path to destination folder for extracted files.
    """
    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(dest_dir)


def ensure_charmm_toppar(input_dir: str) -> None:
    """
    Ensure ``input_dir`` exists and that a ``charmm_toppar.zip`` archive is
    present inside it, copying the bundled placeholder/real archive from
    package data (``pyadmd/charmm/charmm_toppar.zip``) only if the
    destination file is not already present.

    Called at the start of ``run`` so that both the common-file copy loop
    and ``write_charmm_nm`` (which unzips ``{cwd}/inputs/charmm_toppar.zip``
    internally for the CHARMM model type) always find the directory and
    archive they expect. A user-supplied ``inputs/charmm_toppar.zip`` placed
    there ahead of time is never overwritten.

    Args:
        input_dir (str): Path to the ``inputs/`` directory (created if
            missing).
    """
    os.makedirs(input_dir, exist_ok=True)
    toppar_dest = os.path.join(input_dir, "charmm_toppar.zip")
    if not os.path.exists(toppar_dest):
        packaged_toppar = importlib_resources.files("pyadmd") / "charmm" / "charmm_toppar.zip"
        shutil.copy(str(packaged_toppar), toppar_dest)
        console = ConsoleConfig()
        print(f"{console.PGM_NAM}Bundled {console.EXT}charmm_toppar.zip{console.STD} "
              f"copied to {console.EXT}inputs/{console.STD}.")


def write_charmm_nm(nms_to_write: str, psffile: str, modefile: str, cwd: str) -> None:
    """
    Write CHARMM normal mode vectors in NAMD readable format.

    The CHARMM driver script (``wrt-nm.mdu``) is bundled as package data
    (``pyadmd/charmm/wrt-nm.mdu``) rather than assumed to live in a
    repo-relative ``tools/`` folder next to the invocation directory —
    this is required so ``pyadmd`` works when pip-installed and run from
    any working directory. It is copied into ``{cwd}/tools/`` (created if
    necessary) on first use; all other paths (psf/mod file basenames,
    ``../wrt-nm.out`` output location) are unchanged from the original
    NAMD-era behaviour, since CHARMM is still invoked with ``cwd/tools``
    as its working directory.

    Args:
        nms_to_write (str): Comma-separated list of normal mode numbers to write.
        psffile (str): Path to PSF topology file.
        modefile (str): Path to CHARMM mode file.
        cwd (str): Current working directory path.

    Raises:
        SystemExit: If CHARMM execution fails.
    """
    console = ConsoleConfig()

    # Extract CHARMM topology and parameters files
    unzip_file(f"{cwd}/inputs/charmm_toppar.zip", f"{cwd}/inputs")

    # Create input file listing modes to process
    nms = [f"{t}\n" for t in nms_to_write.split(',')]
    with open(f"{cwd}/inputs/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)

    # Ensure the CHARMM driver script is present in cwd/tools, sourced from
    # the installed package's bundled data rather than a repo-relative folder.
    tools_dir = f"{cwd}/tools"
    os.makedirs(tools_dir, exist_ok=True)
    mdu_dest = os.path.join(tools_dir, "wrt-nm.mdu")
    if not os.path.exists(mdu_dest):
        packaged_mdu = importlib_resources.files("pyadmd") / "charmm" / "wrt-nm.mdu"
        shutil.copy(str(packaged_mdu), mdu_dest)

    # Execute CHARMM to generate mode vectors in NAMD format
    os.chdir(tools_dir)
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]} modfile={modefile.split('/')[-1]} -o ../wrt-nm.out")

    returned_value = subprocess.call(cmd, shell=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{console.PGM_ERR}An error occurred while writing the normal mode vectors.")
        print(f"{console.PGM_ERR}Inspect the file {console.ERR}wrt-nm.out{console.STD} for detailed information.")
        sys.exit(1)

    # Return to cwd folder
    os.chdir(f"{cwd}")
