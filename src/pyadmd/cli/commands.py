"""Per-subcommand implementations, split out of the original monolithic main()."""

import json
import os
import shutil
import sys
from typing import Any

from pyadmd.console import ConsoleConfig
from pyadmd.io.dcd import find_last_completed_cycle
from pyadmd.io.namd import NAMDInputReader
from pyadmd.io.openmm_restart import OpenMMRestartReader
from pyadmd.io.params import ParameterStorage
from pyadmd.io.state import (
    SystemState,
    load_reference_state,
    make_reference_universe,
    save_reference_state,
)
from pyadmd.enm.calculator import ENMCalculator
from pyadmd.modes.exciter import ModeExciter
from pyadmd.simulation.runner import SimulationRunner
from pyadmd.freeenergy.calculator import FreeEnergyCalculator
from pyadmd.analysis.analyzer import Analyzer
from pyadmd.utils import ensure_charmm_toppar, unzip_file, write_charmm_nm


def cmd_run(args: Any, console: ConsoleConfig, cwd: str, input_dir: str,
            enm_calculator: ENMCalculator, mode_exciter: ModeExciter,
            param_storage: ParameterStorage) -> None:
    """
    Implements ``pyadmd run``: set up inputs, compute/load normal modes,
    generate mode combinations, and run all replicas.

    Args:
        args: Parsed CLI arguments for the ``run`` subcommand.
        console: Console configuration for formatted output.
        cwd: Current working directory.
        input_dir: ``{cwd}/inputs`` directory path.
        enm_calculator: Shared ENMCalculator instance.
        mode_exciter: Shared ModeExciter instance.
        param_storage: Shared ParameterStorage instance.
    """
    print(f"{console.PGM_NAM}{console.TLE}Setup and run aMDeNM simulations{console.STD}\n")

    # Ensure inputs/ exists and the bundled CHARMM toppar archive is present
    # before any file is copied into it or unzipped from it.
    ensure_charmm_toppar(input_dir)

    # Common required files
    psffile = args.psffile
    pdbfile = args.pdbfile
    modefile = None
    if args.modefile:
        modefile = args.modefile

    common_files = [psffile, pdbfile]
    if args.modefile:
        common_files.append(modefile)

    # Engine-specific files
    itype = args.source   # 'NAMD' or 'OPENMM'

    if itype == 'NAMD':
        coorfile = args.coorfile
        velfile  = args.velfile
        xscfile  = args.xscfile
        strfile  = args.strfile
        engine_files = [coorfile, velfile, xscfile, strfile]
        rstfile  = None
    else:   # OPENMM
        rstfile  = args.rstfile
        coorfile = velfile = xscfile = None
        strfile  = args.strfile   # optional
        engine_files = [rstfile]
        if strfile:
            engine_files.append(strfile)

    # Existence check + copy to inputs/ folder
    for file in common_files + engine_files:
        if not os.path.isfile(file):
            print(f"{console.PGM_ERR}File {file.split('/')[-1]} not found.")
            sys.exit(1)
        if not os.path.isfile(f"{input_dir}/{file.split('/')[-1]}"):
            shutil.copy(file, input_dir)
            print(f"{console.PGM_WRN}File {console.WRN}{file.split('/')[-1]}{console.STD} "
                  "was copied to inputs folder.")

    # Canonicalise paths to inputs/ folder
    psffile = f"{input_dir}/{psffile.split('/')[-1]}"
    pdbfile = f"{input_dir}/{pdbfile.split('/')[-1]}"
    if args.modefile:
        modefile = f"{input_dir}/{modefile.split('/')[-1]}"

    if itype == 'NAMD':
        coorfile = f"{input_dir}/{coorfile.split('/')[-1]}"
        velfile  = f"{input_dir}/{velfile.split('/')[-1]}"
        xscfile  = f"{input_dir}/{xscfile.split('/')[-1]}"
        strfile  = f"{input_dir}/{strfile.split('/')[-1]}"
    else:
        rstfile = f"{input_dir}/{rstfile.split('/')[-1]}"
        if strfile:
            strfile = f"{input_dir}/{strfile.split('/')[-1]}"

    # Build initial SystemState (engine-specific)
    if itype == 'NAMD':
        print(f"{console.PGM_NAM}Reading initial NAMD inputs (one-time conversion)...")
        init_state = NAMDInputReader.read_system(psffile, coorfile, velfile, xscfile)
        # Rotate to canonical OpenMM box orientation
        a, b, c = init_state.box_vectors_nm
        R = NAMDInputReader.align_box_to_x(a, b, c)
        init_state.positions_nm     = (R @ init_state.positions_nm.T).T
        init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
        a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
        b_rot = R @ b;  b_rot[2] = 0.0
        c_rot = R @ c
        init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
    else:
        print(f"{console.PGM_NAM}Reading OpenMM restart file: "
              f"{console.EXT}{rstfile.split('/')[-1]}{console.STD}...")
        init_state = OpenMMRestartReader.read_state(rstfile)
        # (align_box_to_x rotation already applied inside read_state)

    # Reference positions in Å (used for ENM, sys_coor, reference state)
    init_pos_ang = init_state.positions_nm * 10.0   # nm → Å

    # Build reference Universe (topology + real positions, engine-agnostic)
    sys_coor = make_reference_universe(psffile, init_pos_ang)
    sys_mass = sys_coor.atoms.masses          # System atomic mass
    n_atoms  = sys_coor.atoms.n_atoms         # System number of atoms
    sel_mass = sys_coor.atoms.select_atoms(args.selection).masses

    # Save reference positions, box, and velocities for downstream subcommands
    save_reference_state(input_dir, init_pos_ang, init_state.box_vectors_nm,
                          init_state.velocities_nm_ps)
    print(f"{console.PGM_NAM}Reference state saved to "
          f"{console.EXT}inputs/init_reference_*.npy{console.STD}.")

    # Store parameters
    nm_type  = args.model.lower()
    modes    = args.modes
    nm_parsed = [int(s) for s in modes.split(',')]
    energy   = args.energy
    sim_time = args.time
    replicas = args.replicas

    # Derive base_name for ENM output directory naming.
    # NAMD: coorfile prefix; OPENMM: rstfile prefix.
    if itype == 'NAMD':
        base_name = os.path.splitext(os.path.basename(coorfile))[0]
    else:
        base_name = os.path.splitext(os.path.basename(rstfile))[0]

    # Store in args for SimulationRunner access
    n_steps = 100  # MD steps per excitation cycle (2 fs/step → 0.2 ps/cycle)
    args.n_steps = n_steps
    end_loop = int(sim_time / (n_steps * 0.002))

    # Compute / write normal mode vectors
    if nm_type == "ca":
        print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Cα ENM"
              f"{console.STD}{console.HGH} and writing normal mode vectors "
              f"{console.EXT}{modes}{console.STD}.")
        enm_calculator.compute_enm(init_pos_ang, base_name, nm_type,
                                   nm_parsed, input_dir, psffile)
    elif nm_type == "heavy":
        print(f"\n{console.PGM_NAM}{console.HGH}Computing {console.EXT}Heavy atoms ENM"
              f"{console.STD}{console.HGH} and writing normal mode vectors "
              f"{console.EXT}{modes}{console.STD}.")
        enm_calculator.compute_enm(init_pos_ang, base_name, nm_type,
                                   nm_parsed, input_dir, psffile)
    elif nm_type == "charmm":
        print(f"\n{console.PGM_NAM}Writing {console.EXT}CHARMM{console.STD} "
              f"normal mode vectors {console.EXT}{modes}{console.STD}.")
        write_charmm_nm(modes, psffile, modefile, cwd)

    # Extract NAMD topology and parameters files
    unzip_file(f"{input_dir}/charmm_toppar.zip", input_dir)

    # Generate mode combinations
    print(f"\n{console.PGM_NAM}Generating {console.EXT}{replicas}{console.STD} uniformly "
          f"distributed combinations of modes {console.EXT}{modes}{console.STD}.")
    factors = mode_exciter.generate_factors(
        replicas, len(nm_parsed), cwd, nm_parsed, nm_type, base_name, sys_coor
    )
    mode_exciter.combine_modes(replicas, factors, cwd, sys_coor)

    # Save parameters for potential restart/append
    param_storage.save_parameters(args, factors, nm_parsed, end_loop, cwd)

    # Initialize and run SimulationRunner
    sim_runner = SimulationRunner(
        console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
        xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
        mode_exciter, init_state=init_state,
    )
    for rep in range(1, replicas + 1):
        sim_runner.run_simulation(rep, 0, end_loop)


def cmd_restart(console: ConsoleConfig, mode_exciter: ModeExciter,
                 param_storage: ParameterStorage) -> None:
    """
    Implements ``pyadmd restart``: resume any replicas that have not yet
    reached their target cycle count.

    Args:
        console: Console configuration for formatted output.
        mode_exciter: Shared ModeExciter instance.
        param_storage: Shared ParameterStorage instance.
    """
    print(f"{console.PGM_NAM}{console.TLE}Restart unfinished pyAdMD simulation{console.STD}\n")

    # Load parameters
    params = param_storage.load_parameters()
    if params is None:
        sys.exit(1)

    args      = params['args']
    end_loop  = params['end_loop']
    cwd       = params['cwd']

    # Reconstruct file paths from stored args
    input_dir = f"{cwd}/inputs"
    itype     = getattr(args, 'source', 'NAMD').upper()

    psffile = f"{input_dir}/{args.psffile.split('/')[-1]}"
    pdbfile = f"{input_dir}/{args.pdbfile.split('/')[-1]}"

    if itype == 'NAMD':
        coorfile = f"{input_dir}/{args.coorfile.split('/')[-1]}"
        velfile  = f"{input_dir}/{args.velfile.split('/')[-1]}"
        xscfile  = f"{input_dir}/{args.xscfile.split('/')[-1]}"
        strfile  = (f"{input_dir}/{args.strfile.split('/')[-1]}"
                    if getattr(args, 'strfile', None) else None)
        rstfile  = None
    else:
        rstfile  = f"{input_dir}/{args.rstfile.split('/')[-1]}"
        coorfile = velfile = xscfile = None
        strfile  = (f"{input_dir}/{args.strfile.split('/')[-1]}"
                    if getattr(args, 'strfile', None) else None)

    # Get parameters from loaded args
    energy    = args.energy
    selection = args.selection
    replicas  = args.replicas

    # Rebuild initial SystemState from saved reference state if
    # available (avoids re-reading engine-specific files on restart).
    try:
        ref_positions_ang, ref_box_nm, ref_vel_nm_ps = load_reference_state(input_dir)
        init_state = SystemState(
            positions_nm     = ref_positions_ang * 0.1,
            velocities_nm_ps = ref_vel_nm_ps,   # may still be None (older run); reassigned per-replica from checkpoint for already-started replicas
            box_vectors_nm   = ref_box_nm,
        )
        print(f"{console.PGM_NAM}Reference state loaded from saved .npy files.")
    except FileNotFoundError:
        # Legacy fallback: re-read the original engine-specific files.
        if itype == 'NAMD':
            print(f"{console.PGM_WRN}Reference state not found; "
                  "reading NAMD binary inputs.")
            init_state = NAMDInputReader.read_system(
                psffile, coorfile, velfile, xscfile)
            a, b, c = init_state.box_vectors_nm
            R = NAMDInputReader.align_box_to_x(a, b, c)
            init_state.positions_nm     = (R @ init_state.positions_nm.T).T
            init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
            a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
            b_rot = R @ b;  b_rot[2] = 0.0
            c_rot = R @ c
            init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
        else:
            print(f"{console.PGM_WRN}Reference state not found; "
                  "reading OpenMM restart file.")
            init_state = OpenMMRestartReader.read_state(rstfile)

    # Fallback to re-reading the original engine-specific input files so that
    # velocities_nm_ps is populated before any fresh-start replica needs it.
    if init_state.velocities_nm_ps is None:
        print(f"{console.PGM_WRN}Reference state has no velocities; "
              "reading original input files to recover them.")
        if itype == 'NAMD':
            _vel_state = NAMDInputReader.read_system(
                psffile, coorfile, velfile, xscfile)
            a, b, c = _vel_state.box_vectors_nm
            R = NAMDInputReader.align_box_to_x(a, b, c)
            init_state.velocities_nm_ps = (R @ _vel_state.velocities_nm_ps.T).T
        else:
            _vel_state = OpenMMRestartReader.read_state(rstfile)
            init_state.velocities_nm_ps = _vel_state.velocities_nm_ps
        del _vel_state

    init_pos_ang = init_state.positions_nm * 10.0
    sys_coor = make_reference_universe(psffile, init_pos_ang)
    sys_mass = sys_coor.atoms.masses
    n_atoms  = sys_coor.atoms.n_atoms
    sel_mass = sys_coor.atoms.select_atoms(selection).masses

    # Initialize the simulation runner
    sim_runner = SimulationRunner(
        console, args, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
        xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
        mode_exciter, init_state=init_state,
    )

    # Check if any replicas need to be processed
    replicas_to_process = []
    for rep in range(1, replicas + 1):
        rep_dir = f"{cwd}/rep{rep}"
        if not os.path.exists(rep_dir):
            replicas_to_process.append(rep)
        else:
            last_cycle = find_last_completed_cycle(rep_dir)
            if last_cycle < end_loop:
                replicas_to_process.append(rep)

    if not replicas_to_process:
        print(f"\n{console.PGM_WRN}All replicas are already completed. No need to restart.")
        return

    # Process replicas that need to be restarted
    for rep in replicas_to_process:
        rep_dir    = f"{cwd}/rep{rep}"
        last_cycle = find_last_completed_cycle(rep_dir)

        if os.path.exists(rep_dir):
            mode_label = ("Standard" if args.no_correc else
                          "Constant" if args.fixed else "Adaptive")
            print(f"\n{console.PGM_NAM}{console.HGH}Restarting {mode_label} MDeNM "
                  f"calculations for {console.EXT}Replica {rep}{console.STD}"
                  f"{console.HGH} from {console.EXT}step {last_cycle}{console.STD}")

        correction_state = {}
        if os.path.exists(f"{rep_dir}/correction_state.json"):
            with open(f"{rep_dir}/correction_state.json", 'r') as f:
                correction_state = json.load(f)

        sim_runner.run_simulation(rep, last_cycle, end_loop, correction_state)


def cmd_append(args: Any, console: ConsoleConfig, mode_exciter: ModeExciter,
               param_storage: ParameterStorage) -> None:
    """
    Implements ``pyadmd append``: extend previously completed replicas by
    ``args.time`` additional picoseconds.

    Args:
        args: Parsed CLI arguments for the ``append`` subcommand.
        console: Console configuration for formatted output.
        mode_exciter: Shared ModeExciter instance.
        param_storage: Shared ParameterStorage instance.
    """
    print(f"{console.PGM_NAM}{console.TLE}Append previous pyAdMD simulation{console.STD}\n")

    additional_time = args.time

    params = param_storage.load_parameters()
    if params is None:
        sys.exit(1)

    args_dict          = params['args']
    factors            = params['factors']
    nm_parsed          = params['nm_parsed']
    original_end_loop  = params['end_loop']
    cwd                = params['cwd']

    # Calculate new end loop and update total time
    n_steps           = args_dict.n_steps
    additional_steps  = int(additional_time / (n_steps * 0.002))
    new_end_loop      = original_end_loop + additional_steps
    args_dict.time    = args_dict.time + additional_time
    params['end_loop'] = new_end_loop
    param_storage.save_parameters(args_dict, factors, nm_parsed, new_end_loop, cwd)

    # Reconstruct file paths
    input_dir = f"{cwd}/inputs"
    itype     = getattr(args_dict, 'source', 'NAMD').upper()

    psffile = f"{input_dir}/{args_dict.psffile.split('/')[-1]}"
    pdbfile = f"{input_dir}/{args_dict.pdbfile.split('/')[-1]}"

    if itype == 'NAMD':
        coorfile = f"{input_dir}/{args_dict.coorfile.split('/')[-1]}"
        velfile  = f"{input_dir}/{args_dict.velfile.split('/')[-1]}"
        xscfile  = f"{input_dir}/{args_dict.xscfile.split('/')[-1]}"
        strfile  = (f"{input_dir}/{args_dict.strfile.split('/')[-1]}"
                    if getattr(args_dict, 'strfile', None) else None)
        rstfile  = None
    else:
        rstfile  = f"{input_dir}/{args_dict.rstfile.split('/')[-1]}"
        coorfile = velfile = xscfile = None
        strfile  = (f"{input_dir}/{args_dict.strfile.split('/')[-1]}"
                    if getattr(args_dict, 'strfile', None) else None)

    energy    = args_dict.energy
    selection = args_dict.selection
    replicas  = args_dict.replicas

    # Rebuild initial SystemState from reference state
    try:
        ref_positions_ang, ref_box_nm, ref_vel_nm_ps = load_reference_state(input_dir)
        init_state = SystemState(
            positions_nm     = ref_positions_ang * 0.1,
            velocities_nm_ps = ref_vel_nm_ps,
            box_vectors_nm   = ref_box_nm,
        )
        print(f"{console.PGM_NAM}Reference state loaded from .npy files.")
    except FileNotFoundError:
        if itype == 'NAMD':
            print(f"{console.PGM_WRN}Reference state not found; "
                  "re-reading NAMD binary inputs.")
            init_state = NAMDInputReader.read_system(
                psffile, coorfile, velfile, xscfile)
            a, b, c = init_state.box_vectors_nm
            R = NAMDInputReader.align_box_to_x(a, b, c)
            init_state.positions_nm     = (R @ init_state.positions_nm.T).T
            init_state.velocities_nm_ps = (R @ init_state.velocities_nm_ps.T).T
            a_rot = R @ a;  a_rot[1] = 0.0;  a_rot[2] = 0.0
            b_rot = R @ b;  b_rot[2] = 0.0
            c_rot = R @ c
            init_state.box_vectors_nm = [a_rot, b_rot, c_rot]
        else:
            print(f"{console.PGM_WRN}Reference state not found; "
                  "reading OpenMM restart file.")
            init_state = OpenMMRestartReader.read_state(rstfile)

    # Same guard as 'restart': recover velocities from original input
    # files if the saved reference state consume velocity values.
    if init_state.velocities_nm_ps is None:
        print(f"{console.PGM_WRN}Reference state has no velocities; "
              "reading original input files to recover them.")
        if itype == 'NAMD':
            _vel_state = NAMDInputReader.read_system(
                psffile, coorfile, velfile, xscfile)
            a, b, c = _vel_state.box_vectors_nm
            R = NAMDInputReader.align_box_to_x(a, b, c)
            init_state.velocities_nm_ps = (R @ _vel_state.velocities_nm_ps.T).T
        else:
            _vel_state = OpenMMRestartReader.read_state(rstfile)
            init_state.velocities_nm_ps = _vel_state.velocities_nm_ps
        del _vel_state

    init_pos_ang = init_state.positions_nm * 10.0
    sys_coor = make_reference_universe(psffile, init_pos_ang)
    sys_mass = sys_coor.atoms.masses
    n_atoms  = sys_coor.atoms.n_atoms
    sel_mass = sys_coor.atoms.select_atoms(selection).masses

    # Initialize the simulation runner
    sim_runner = SimulationRunner(
        console, args_dict, cwd, input_dir, psffile, pdbfile, coorfile, velfile,
        xscfile, strfile, sys_coor, n_atoms, sys_mass, sel_mass, energy,
        mode_exciter, init_state=init_state,
    )

    # Check if any replicas need to be extended
    replicas_to_extend = []
    for rep in range(1, replicas + 1):
        rep_dir = f"{cwd}/rep{rep}"
        if not os.path.exists(rep_dir):
            print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} "
                  "directory not found, skipping.")
            continue
        last_cycle = find_last_completed_cycle(rep_dir)
        if last_cycle < original_end_loop:
            print(f"{console.PGM_WRN}{console.WRN}Replica {rep}{console.STD} "
                  "hasn't completed the original simulation, skipping.")
            continue
        replicas_to_extend.append(rep)

    if not replicas_to_extend:
        print(f"{console.PGM_WRN}No replicas to extend. All replicas either "
              "don't exist or haven't completed the original simulation.")
        return

    for rep in replicas_to_extend:
        rep_dir = f"{cwd}/rep{rep}"
        print(f"\n{console.PGM_NAM}{console.HGH}Extending {console.EXT}Replica {rep}"
              f"{console.STD}{console.HGH} for {console.EXT}{additional_time}"
              f"{console.STD}{console.HGH} picoseconds{console.STD}")

        correction_state = {}
        if os.path.exists(f"{rep_dir}/correction_state.json"):
            with open(f"{rep_dir}/correction_state.json", 'r') as f:
                correction_state = json.load(f)

        sim_runner.run_simulation(rep, original_end_loop, new_end_loop,
                                  correction_state)


def cmd_analyze(args: Any, console: ConsoleConfig) -> None:
    """
    Implements ``pyadmd analyze``: analyze either pyadmd replica
    trajectories or freeenergy centroid production trajectories.

    Args:
        args: Parsed CLI arguments for the ``analyze`` subcommand.
        console: Console configuration for formatted output.
    """
    print(f"{console.PGM_NAM}{console.TLE}Analyze pyAdMD results{console.STD}\n")
    analyzer = Analyzer(
        console,
        rough=args.rough,
        no_rmsd=args.no_rmsd,
        no_rg=args.no_rg,
        no_sasa=args.no_sasa,
        no_hp=args.no_hp,
        no_rmsf=args.no_rmsf,
        no_dssp=args.no_dssp,
        source=args.source,
    )
    if args.source == 'freeenergy':
        analyzer.analyze_all_centroids()
    else:
        analyzer.analyze_all_replicas()


def cmd_freeenergy(args: Any, console: ConsoleConfig,
                    param_storage: ParameterStorage) -> None:
    """
    Implements ``pyadmd freeenergy``: compute (or extend) the free energy
    landscape from completed aMDeNM replicas.

    Args:
        args: Parsed CLI arguments for the ``freeenergy`` subcommand.
        console: Console configuration for formatted output.
        param_storage: Shared ParameterStorage instance.
    """
    print(f"{console.PGM_NAM}{console.TLE}Free Energy Landscape Calculation"
          f"{console.STD}\n")

    params = param_storage.load_parameters()
    if params is None:
        sys.exit(1)

    fe_calc = FreeEnergyCalculator(console, params, args)
    fe_calc.run()


def cmd_clean(console: ConsoleConfig, cwd: str, input_dir: str) -> None:
    """
    Implements ``pyadmd clean``: erase all previous simulation setup and
    output files.

    Args:
        console: Console configuration for formatted output.
        cwd: Current working directory.
        input_dir: ``{cwd}/inputs`` directory path.
    """
    print(f"{console.PGM_NAM}{console.TLE}Clean previous pyAdMD setup files{console.STD}\n")

    # Removing previous replicas folders
    files = os.listdir(cwd)
    for item in files:
        if item.endswith((".json", "summary.txt")):
             os.remove(os.path.join(cwd, item))
        if item.startswith(("rep", "analysis")):
            shutil.rmtree(os.path.join(cwd, item), ignore_errors=True)

    # Removing previous configuration files
    files = os.listdir(input_dir)
    for item in files:
        if item.endswith((".txt", ".out", ".crd", ".psf", ".pdb", ".coor",
                          ".vel", ".xsc", ".str", ".mod", ".rst", ".npy")):
            os.remove(os.path.join(input_dir, item))
        # Removing previous ENM calculations
        if item.endswith("_enm"):
            shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)
    for item in ("charmm_toppar", "charmm_toppar"):
            shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)

    print(f"{console.PGM_NAM}Erasing is done.\n")
