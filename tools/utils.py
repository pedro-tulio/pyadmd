import zipfile
import os
import shutil
import logging

logger = logging.getLogger("__init__")

def unzip(filepath, path_dir):
    """
    Extract files from a .zip compressed file.

    Parameters
    ----------
    filepath : str
        Path to .zip file
    path_dir : str
        Path to destination folder
    """

    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(path_dir)

def clean(folder):
    """
    Delete previous run files.
    """
    # Removing previous replicas folders
    files = os.listdir(folder)
    for item in files:
        if item.startswith("rep"):
            shutil.rmtree(os.path.join(folder, item), ignore_errors=True)

