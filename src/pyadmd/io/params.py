"""Serialization/deserialization of run parameters for restart/append."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from pyadmd.console import ConsoleConfig


class ParameterStorage:
    """
    Handles serialization and deserialization of simulation parameters.

    This class provides functionality to save and load simulation parameters
    to/from JSON files, enabling restart capabilities for the aMDeNM simulations.

    Attributes:
        console (ConsoleConfig): Console configuration object for formatted output.
        param_file (str): Default filename for parameter storage.
    """

    def __init__(self, console: ConsoleConfig) -> None:
        """
        Initialize ParameterStorage with console configuration.

        Args:
            console (ConsoleConfig): Console configuration object for formatted output.
        """
        self.console = console
        self.param_file = "pyAdMD_params.json"

    def save_parameters(self, args: argparse.Namespace, factors: np.ndarray,
                        nm_parsed: List[int], end_loop: int, cwd: str) -> None:
        """
        Save simulation parameters to a JSON file.

        Serializes the current simulation state including command line arguments,
        mode combination factors, selected modes, and loop information for
        potential restart capabilities.

        Args:
            args (argparse.Namespace): Command line arguments namespace.
            factors (numpy.ndarray): Matrix of combination factors for normal modes (P×N).
            nm_parsed (list): List of mode numbers used in combinations.
            end_loop (int): Final loop iteration count for simulation cycles.
            cwd (str): Current working directory path.

        Note:
            The timestamp is automatically added to track when parameters were saved.
        """
        params = {
            "args": vars(args),
            "factors": factors.tolist() if factors is not None else None,
            "nm_parsed": nm_parsed,
            "end_loop": end_loop,
            "cwd": cwd,
            "timestamp": time.time()
        }

        with open(self.param_file, 'w') as f:
            json.dump(params, f, indent=4)

        print(f"\n{self.console.PGM_NAM}Parameters saved to {self.console.EXT}{self.param_file}{self.console.STD}.")

    def load_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Load simulation parameters from a JSON file.

        Attempts to deserialize previously saved parameters and reconstruct
        the argument namespace object. Handles conversion of factors back to
        numpy array format.

        Returns:
            dict: Dictionary containing loaded parameters with keys:
                - args (argparse.Namespace): Reconstructed argument namespace.
                - factors (numpy.ndarray or None): Combination factors matrix.
                - nm_parsed (list[int]): List of mode numbers used in simulation.
                - end_loop (int): Final loop iteration count.
                - cwd (str): Absolute working directory path.
                - timestamp (float): Unix timestamp of when parameters were saved.
            Returns None if the file is missing or cannot be parsed.

        Raises:
            JSONDecodeError: If the parameter file contains invalid JSON.
            IOError: If the parameter file cannot be accessed.
        """
        if not os.path.exists(self.param_file):
            print(f"{self.console.PGM_ERR}Parameter file {self.console.ERR}{self.param_file}{self.console.STD} not found.")
            return None

        try:
            with open(self.param_file, 'r') as f:
                params = json.load(f)

            # Reconstruct args namespace
            class Args:
                def __init__(self, dict_args: Dict[str, Any]) -> None:
                    for key, value in dict_args.items():
                        setattr(self, key, value)

            params['args'] = Args(params['args'])

            # Convert factors back to numpy array if present
            if params['factors'] is not None:
                params['factors'] = np.array(params['factors'])

            print(f"{self.console.PGM_NAM}Parameters loaded from {self.console.EXT}{self.param_file}{self.console.STD}.")
            return params
        except Exception as e:
            print(f"{self.console.PGM_ERR}Error loading parameters: {self.console.ERR}{e}{self.console.STD}.")
            return None
