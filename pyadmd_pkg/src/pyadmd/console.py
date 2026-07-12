"""Console styling and application metadata."""


class ConsoleConfig:
    """
    Configuration class for PyAdMD application providing console styling and messages.

    This class contains ANSI escape codes for console text styling, formatted program
    output prefixes, and application metadata such as version and citation information.

    Attributes:
        BLK (str): ANSI code for blinking cyan text
        TLE (str): ANSI code for light background title style
        HGH (str): ANSI code for bold highlighted text
        WRN (str): ANSI code for warning (yellow) text
        ERR (str): ANSI code for error (red) text
        EXT (str): ANSI code for success/emphasis (green) text
        STD (str): ANSI code to reset text styling
        PGM_NAM (str): Formatted prefix for normal messages
        PGM_WRN (str): Formatted prefix for warnings
        PGM_ERR (str): Formatted prefix for errors
        LOGO (str): ASCII art logo for the application
        VERSION (str): Application version number
        CITATION (str): Citation information for the method
        MESSAGE (str): Brief program description
    """

    # Style variables
    BLK = '\033[5;36m'    # Blinking cyan
    TLE = '\033[2;106m'   # Light background title
    HGH = '\033[1;100m'   # Bold highlighted
    WRN = '\033[33m'      # Warning yellow
    ERR = '\033[31m'      # Error red
    EXT = '\033[32m'      # Success green
    STD = '\033[0m'       # Reset styling

    # Program output variables
    PGM_NAM = f"..:{EXT}pyAdMD> {STD}"
    PGM_WRN = f"..+{WRN}pyAdMD-Wrn> {STD}"
    PGM_ERR = f"..%{ERR}pyAdMD-Err> {STD}"

    LOGO = '''
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

    VERSION = '3.0.0'
    CITATION = '''  Please cite:

    \tAdaptive Normal Mode Sampling (aMDeNM) Enhances Exploration of Protein Conformational Space
    \tand Reveals the Functional Role of Frequency Coupling.
    \tP.T. Resende-Lara, M.G.S. Costa, B. Dudas, J. Czigleczki, E. Balog, D. Perahia.
    \tDOI: https://doi.org/10.1021/acs.jctc.6c00398'''

    MESSAGE = "This program can setup and run multi-replica aMDeNM simulations through OpenMM."
