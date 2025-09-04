"""
The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of multiple-replica short MD simulations
in which motions described by a given subset of low-frequency NMs are kinetically excited. This is achieved by injecting
additional atomic velocities along several randomly determined linear combinations of NM vectors, thus allowing an
efficient coupling between slow and fast motions.

This new approach, aMDeNM, automatically controls the energy injection and take the natural constraints imposed by
the structure and the environment into account during protein conformational sampling, which prevent structural
distortions all along the simulation. Due to the stochasticity of thermal motions, NM eigenvectors move away from the
original directions when used to displace the protein, since the structure evolves into other potential energy wells.
Therefore, the displacement along the modes is valid for small distances, but the displacement along greater distances
may deform the structure of the protein if no care is taken. The advantage of this methodology is to adaptively change
the direction used to displace the system, taking into account the structural and energetic constraints imposed by the
system itself and the medium, which allows the system to explore new pathways.
"""

import logging

class AnsiColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m',            # Grey
        'INFO': '\033[0m',              # Reset
        'WARNING': '\033[33m',          # Yellow
        'ERROR': '\033[31m',            # Red
        'CRITICAL': '\033[91m\033[1m',  # Bright Red + Bold
    }
    RESET_COLOR = '\033[0m'

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET_COLOR)
        return f"{color}{log_message}{self.RESET_COLOR}"

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Console handler with color formatting
handler = logging.StreamHandler()
formatter = AnsiColorFormatter("..:pyAdMD> {levelname}: {message}", style="{")
handler.setFormatter(formatter)
logger.addHandler(handler)

# File handler for pyAdMD.out
file_handler = logging.FileHandler('enm.out')
file_formatter = logging.Formatter("{asctime} ..:pyAdMD> {levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Style variables
tle = '\033[2;106m'
hgh = '\033[1;100m'
wrn = '\033[33m'
err = '\033[31m'
ext = '\033[32m'
std = '\033[0m'

# Program output variables
pgmnam = f"+{ext}amdenm> {std}"
pgmwrn = f"%{wrn}amdenm-Wrn> {std}"
pgmerr = f"%{err}amdenm-Err> {std}"

# Header variables
# https://manytools.org/hacker-tools/ascii-banner/
logo = '''
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
'''

version = '1.0'
citation = '''  Please cite:

\tAdaptive collective motions: a hybrid method to improve
\tconformational sampling with molecular dynamics and normal modes.
\tPT Resende-Lara, MGS Costa, B Dudas, D Perahia.
\tDOI: https://doi.org/10.1101/2022.11.29.517349'''

banner = (f"{'\033[5;36m'}{logo}{std}\n"
          f"\t\t{tle}Adaptive Molecular Dynamics with Python{std}\n"
          f"\t\t\t     version: {version}\n"
          f"\n{citation}\n")

print(banner)
print(__doc__)

