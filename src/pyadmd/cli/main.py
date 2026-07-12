"""Top-level entry point: banner, argument parsing, and subcommand dispatch."""

import os

import pyadmd
from pyadmd.console import ConsoleConfig
from pyadmd.io.params import ParameterStorage
from pyadmd.enm.calculator import ENMCalculator
from pyadmd.modes.exciter import ModeExciter
from pyadmd.cli.parser import parse_arguments
from pyadmd.cli.commands import (
    cmd_run,
    cmd_restart,
    cmd_append,
    cmd_analyze,
    cmd_freeenergy,
    cmd_clean,
)


def main() -> None:
    """Main entry point for the PyAdMD application.

    Handles command-line argument parsing, initialization of components, and
    execution of the requested operation (run, restart, append, or clean).

    The function performs the following primary operations:
    - Parses command-line arguments
    - Initializes console configuration and component classes
    - Handles different operational modes (run, restart, append, analyze, clean)
    - Manages simulation setup, execution, and cleanup
    - Coordinates file operations and parameter storage

    Raises:
        SystemExit: If required files are missing or critical errors occur during execution
    """
    console = ConsoleConfig()

    # Print banner
    banner = (f"{console.BLK}{console.LOGO}{console.STD}\n"
              f"\t\t{console.TLE}Adaptive Molecular Dynamics with Python{console.STD}\n"
              f"\t\t\t     version: {console.VERSION}\n"
              f"\n{console.CITATION}\n")

    print(banner)
    print(pyadmd.__doc__)

    # Parse command line arguments
    args = parse_arguments()

    # Get working directory path
    cwd = os.getcwd()
    input_dir = f"{cwd}/inputs"

    # Initialize component classes shared across subcommands
    enm_calculator = ENMCalculator(console)
    mode_exciter = ModeExciter(console)
    param_storage = ParameterStorage(console)

    if args.option == 'run':
        cmd_run(args, console, cwd, input_dir, enm_calculator, mode_exciter, param_storage)
    elif args.option == 'restart':
        cmd_restart(console, mode_exciter, param_storage)
    elif args.option == 'append':
        cmd_append(args, console, mode_exciter, param_storage)
    elif args.option == 'analyze':
        cmd_analyze(args, console)
    elif args.option == 'freeenergy':
        cmd_freeenergy(args, console, param_storage)
    elif args.option == 'clean':
        cmd_clean(console, cwd, input_dir)


if __name__ == "__main__":
    main()
