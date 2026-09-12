"""Command-line argument parsing for the pyadmd CLI."""

import argparse
import sys

from pyadmd.console import ConsoleConfig


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the pyAdMD analysis tool.

    This function sets up the argument parser with subcommands for different
    operational modes of the pyAdMD tool, including running simulations,
    restarting, appending, analyzing, and cleaning.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments

    Raises:
        SystemExit: If invalid arguments are provided or help is requested
    """
    console = ConsoleConfig()

    parser = argparse.ArgumentParser(prog="pyadmd", description=console.MESSAGE)
    subparsers = parser.add_subparsers(dest='option', help='Available commands')

    #################
    # RUN subparser #
    #################

    opt_run = subparsers.add_parser('run', help="Setup and run simulations")

    # Required arguments for run
    run_required = opt_run.add_argument_group('Required arguments')
    run_required.add_argument('-src', '--source', action="store", type=str.upper,
                             required=True, choices=["NAMD", "OPENMM"],
                             help="Input engine type: NAMD (binary .coor/.vel/.xsc) or "
                                  "OPENMM (XML restart .rst)")
    run_required.add_argument('-m', '--model', action="store", type=str.upper,
                             default="CA", required=True, choices=["CA", "HEAVY", "CHARMM"],
                             help="Compute ENM or use pre-computed CHARMM normal mode file (default: CA)")
    run_required.add_argument('-psf', '--psffile', action="store", type=str, required=True,
                             help="PSF topology file")
    run_required.add_argument('-pdb', '--pdbfile', action="store", type=str, required=True,
                             help="PDB structure file")

    # NAMD-mode inputs (required when -src NAMD)
    run_namd = opt_run.add_argument_group('NAMD input files (required when -src NAMD)')
    run_namd.add_argument('-coor','--coorfile', action="store", type=str, default=None,
                         help="NAMD binary coordinates file (.coor)")
    run_namd.add_argument('-vel', '--velfile', action="store", type=str, default=None,
                         help="NAMD binary velocities file (.vel)")
    run_namd.add_argument('-xsc', '--xscfile', action="store", type=str, default=None,
                         help="NAMD extended system configuration file (.xsc)")
    run_namd.add_argument('-str', '--strfile', action="store", type=str, default=None,
                         help="CHARMM-style stream file for box info and force-field parameters "
                              "(required in NAMD mode; optional in OPENMM mode)")

    # OpenMM-mode inputs (required when -src OPENMM)
    run_openmm = opt_run.add_argument_group('OpenMM input files (required when -src OPENMM)')
    run_openmm.add_argument('-rst', '--rstfile', action="store", type=str, default=None,
                            help="OpenMM XML restart file (.rst) written by XmlSerializer.serialize(state) "
                                 "with getPositions=True, getVelocities=True")

    # Optional arguments for run
    run_optional = opt_run.add_argument_group('Optional arguments')
    run_optional.add_argument('-mod', '--modefile', action="store", type=str,
                             help="CHARMM normal mode file (required when type is CHARMM)")
    run_optional.add_argument('-nm', '--modes', action="store", type=str, default="7,8,9", metavar='MODES',
                             help="Normal modes to excite separated by commas (default: 7,8,9)")
    run_optional.add_argument('-ek', '--energy', action="store", type=float, default=2, metavar='KCAL/MOL',
                             help="Excitation energy (default: 2 kcal/mol)")
    run_optional.add_argument('-t', '--time', action="store", type=int, default=250, metavar='PS',
                             help="Total simulation time (default: 250 ps)")
    run_optional.add_argument('-sel', '--selection', action="store", type=str, default="protein",
                             help="Atom selection to apply the energy injection (default: protein)")
    run_optional.add_argument('-rep', '--replicas', action="store", type=int, default=10, metavar='N',
                             help="Number of aMDeNM replicas to run (default: 10)")
    run_optional.add_argument('-seed', '--seed', action="store", type=int, default=42,
                             help="Random seed for the uniform mode-combination generation (default: 42)")

    # Flags for run
    run_flags = opt_run.add_argument_group('Flags')
    run_flags.add_argument('-n', '--no-correc', action='store_true',
                          help='Compute standard MDeNM calculations')
    run_flags.add_argument('-f', '--fixed', action='store_true',
                          help='Disable excitation vector correction and keep constant excitation energy injections')
    run_flags.add_argument('-r', '--recalc', action='store_true',
                          help='Recompute ENM modes instead of correcting excitation direction when needed')
    run_flags.add_argument('--full-ener', action='store_true',
                          help='Write per-term energy decomposition (BOND, ANGLE, DIHED, etc.) '
                               'to rep{N}_ener_decomp.log each cycle')


    #####################
    # RESTART subparser #
    #####################

    subparsers.add_parser('restart', help="Restart unfinished simulations")


    ####################
    # APPEND subparser #
    ####################

    opt_apnd = subparsers.add_parser('append', help="Extend previously computed simulations")
    opt_apnd.add_argument('-t', '--time', action="store", type=int, required=True, default=100, metavar='PS',
                         help="Simulation time to append (default: 100 ps)")


    ###################################
    # FREE ENERGY LANDSCAPE subparser #
    ###################################

    opt_fe = subparsers.add_parser(
        'fel',
        help="Compute free energy landscapes"
    )

    # Optional skip flags for FEL
    fe_params = opt_fe.add_argument_group('Optional parameters')
    fe_params.add_argument(
        '-c', '--cutoff', type=float, default=1.0, metavar='Å',
        help='GROMOS RMSD clustering cutoff in Å (default: 1.0)')
    fe_params.add_argument(
        '-d', '--deexcite', type=int, default=200, metavar='PS',
        help='Total de-excitation MD length per centroid in ps, split over 4 '
             'restraint phases (default: 200)')
    fe_params.add_argument(
        '-p', '--production', type=int, default=800, metavar='PS',
        help='Unrestrained production MD length per centroid in ps (default: 800)')
    fe_params.add_argument(
        '-nm', '--modes', type=str, default=None, metavar='MODES',
        help='Comma-separated mode indices for FEL projection '
             '(default: same as run, e.g. 7,8,9)')
    fe_params.add_argument(
        '--modes-2d', type=str, default=None, metavar='PAIRS',
        help='Mode pairs for 2D FEL plots as space-separated "m1,m2" tokens '
             '(default: all pairwise combinations of --modes, e.g. "7,8 7,9 8,9")')
    fe_params.add_argument(
        '-b', '--bins', type=int, default=100, metavar='N',
        help='Number of histogram bins for FEL (default: 100)')
    fe_params.add_argument(
        '-T', '--temp', type=float, default=303.15, metavar='K',
        help='Temperature for kBT scaling (default: 303.15 K)')
    fe_params.add_argument(
        '-s', '--sel', type=str, default="protein and name CA",
        metavar='SEL',
        help='Atom selection for GROMOS RMSD clustering '
             '(default: "protein and name CA")')
    fe_params.add_argument(
        '--max-centroids', type=int, default=50, metavar='N',
        help='Maximum number of centroids submitted to MD (default: 50)')

    # Exporting flags for FEL
    fe_export = opt_fe.add_argument_group(
        'Cluster inspection (skips the FEL protocol entirely when used)')
    fe_export.add_argument(
        '--export-cluster', type=int, default=None, metavar='FRAME',
        help='Export the member frame list for one cluster, identified by '
             'its centroid_frame value from clustering_summary.csv. '
             'Requires a completed fel run. Writes '
             'fel/exports/cluster_frame{FRAME}/members.csv.')
    fe_export.add_argument(
        '--dump-pdb', action='store_true',
        help='With --export-cluster, also write one PDB file per member '
             'frame to fel/exports/cluster_frame{FRAME}/pdbs/.')


    ######################
    # ANALYSIS subparser #
    ######################

    opt_analyze = subparsers.add_parser('analyze', help="Analyze simulation results and generate plots")
    opt_analyze.add_argument('-src', '--source', action="store", type=str.lower,
                            default="pyadmd", choices=["pyadmd", "fel"],
                            help="Trajectory source to analyze: 'pyadmd' for rep{N}.dcd "
                                 "replica trajectories (default), or 'fel' for "
                                 "centroid production trajectories from a completed "
                                 "'fel' run")
    opt_analyze.add_argument('-r', '--rough', action='store_true',
                            help='Perform rough analysis (every 5ps instead of every frame)')

    # Optional skip flags for analysis
    analyze_skip = opt_analyze.add_argument_group('Skip flags (disable individual analyses)')
    analyze_skip.add_argument('--no-rmsd', action='store_true',
                              help='Skip RMSD calculation')
    analyze_skip.add_argument('--no-rg', action='store_true',
                              help='Skip radius of gyration calculation')
    analyze_skip.add_argument('--no-sasa', action='store_true',
                              help='Skip SASA and hydrophobic exposure calculation')
    analyze_skip.add_argument('--no-rmsf', action='store_true',
                              help='Skip RMSF calculation')
    analyze_skip.add_argument('--no-dssp', action='store_true',
                              help='Skip secondary structure (DSSP) calculation')
    analyze_skip.add_argument('--no-dccm', action='store_true',
                              help='Skip DCCM (dynamic cross-correlation matrix) calculation')
    analyze_skip.add_argument('--no-lmi', action='store_true',
                              help='Skip LMI (Linear Mutual Information) calculation')
    analyze_skip.add_argument('--no-modeproj', action='store_true',
                              help='Skip mode-projection analysis')


    #################
    # ENM subparser #
    #################

    opt_enm = subparsers.add_parser(
        'enm',
        help="Compute an Elastic Network Model / normal mode analysis"
    )

    enm_required = opt_enm.add_argument_group(
        'Required (unless -w/--write-modes is used)')
    enm_required.add_argument(
        '-i', '--input', action="store", type=str, default=None,
        help='Input PDB file')

    # Optional skip flags for ENM
    enm_optional = opt_enm.add_argument_group('Optional parameters')
    enm_optional.add_argument(
        '-o', '--output', action="store", type=str, default='output',
        help='Output folder name (default: output)')
    enm_optional.add_argument(
        '-m', '--model', action="store", type=str.upper, default="CA",
        choices=["CA", "HEAVY"],
        help='Model type: CA (Cα-only) or HEAVY (heavy atoms) (default: CA)')
    enm_optional.add_argument(
        '-sel', '--selection', action="store", type=str, default="protein",
        help='Atom selection to build the ENM (default: "protein")')
    enm_optional.add_argument(
        '-c', '--cutoff', type=float, default=None, metavar='Å',
        help='Interaction cutoff distance in Å '
             '(default: 15.0 for CA, 12.0 for HEAVY)')
    enm_optional.add_argument(
        '-k', '--spring-constant', type=float, default=1.0, metavar='K',
        help='ENM harmonic spring constant in kcal/mol/Å² (default: 1.0)')
    enm_optional.add_argument(
        '--max-modes', type=int, default=None, metavar='N',
        help='Number of non-rigid-body vibrational modes to compute '
             '(default: 50)')
    enm_optional.add_argument(
        '--output-modes', type=int, default=10, metavar='N',
        help='Number of modes to write vectors/trajectories/collectivity/'
             'contributions for (default: 10)')
    enm_optional.add_argument(
        '-w', '--write-modes', metavar='MODES', default=None,
        help='Write mode vectors/trajectories from a previous ENM run. '
             'Accepts comma-separated integers and inclusive ranges '
             '(start:end), e.g. "26,41" "7:10" "42,44:50". Requires '
             '-o/--output pointing to an existing output directory. ')

    # Optional skip flags for ENM
    enm_flags = opt_enm.add_argument_group('Skip flags (disable individual outputs)')
    enm_flags.add_argument('--no-vec', action='store_true',
                          help='Skip writing mode vectors')
    enm_flags.add_argument('--no-trj', action='store_true',
                          help='Skip writing mode trajectories')
    enm_flags.add_argument('--no-collectivity', action='store_true',
                          help='Skip collectivity calculation')
    enm_flags.add_argument('--no-contributions', action='store_true',
                          help='Skip mode contributions plot')
    enm_flags.add_argument('--no-rmsf', action='store_true',
                          help='Skip NMA-predicted RMSF plot')
    enm_flags.add_argument('--no-dccm', action='store_true',
                          help='Skip NMA-predicted DCCM plot')
    enm_flags.add_argument('--no-gpu', action='store_true',
                          help='Disable GPU acceleration (DCCM only; '
                               'diagonalization GPU use is controlled '
                               'separately)')


    ###################
    # CLEAN subparser #
    ###################

    subparsers.add_parser('clean', help="Erase all previous simulation files")

    # Parse arguments
    args = parser.parse_args()

    # Check if no arguments were provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    # Check if a subcommand was provided but no additional arguments
    if hasattr(args, 'option') and len(sys.argv) == 2:
        if args.option == 'run':
            opt_run.print_help()
        elif args.option == 'append':
            opt_apnd.print_help()

    # Validate conditional requirements
    if hasattr(args, 'option') and args.option == 'run':
        if args.model == 'CHARMM' and not args.modefile:
            opt_run.error("The -mod/--modefile argument is required when -m/--model is CHARMM")

    if hasattr(args, 'option') and args.option == 'run':
        if args.model == 'CHARMM' and args.recalc:
            opt_run.error("ENM recalculation is not compatible with CHARMM normal modes")

    # Validate input-engine-specific file requirements
    if hasattr(args, 'option') and args.option == 'run':
        if args.source == 'NAMD':
            missing = [flag for flag, val in [("-coor", args.coorfile),
                                               ("-vel",  args.velfile),
                                               ("-xsc",  args.xscfile),
                                               ("-str",  args.strfile)]
                       if val is None]
            if missing:
                opt_run.error(
                    f"The following arguments are required in NAMD mode: "
                    f"{', '.join(missing)}"
                )
        elif args.source == 'OPENMM':
            if args.rstfile is None:
                opt_run.error(
                    "The -rst/--rstfile argument is required when -src is OPENMM"
                )

    # Validate fel cluster-inspection flags
    if hasattr(args, 'option') and args.option == 'fel':
        if args.dump_pdb and args.export_cluster is None:
            opt_fe.error("--dump-pdb requires --export-cluster FRAME")

    # Validate enm requirements
    if hasattr(args, 'option') and args.option == 'enm':
        if args.write_modes is None and args.input is None:
            opt_enm.error(
                "argument -i/--input is required for normal ENM computation "
                "(omit -i only when using -w/--write-modes)"
            )

    return args
