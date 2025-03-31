'''
Adaptive Molecular Dynamics with excited Normal Modes (aMDeNM)

The MDeNM (Molecular Dynamics with excited Normal Modes) method consists of multiple-replica short MD simulations
in which motions described by a given subset of low-frequency NMs are kinetically excited. This is achieved by injecting
additional atomic velocities along several randomly determined linear combinations of NM vectors, thus allowing an
efficient coupling between slow and fast motions.

This new approach, aMDeNM, automatically controls the energy injection and take the natural constraints imposed by
the structure and the environment into account during protein conformational sampling, which prevent structural
distortions all along the simulation.Due to the stochasticity of thermal motions, NM eigenvectors move away from the
original directions when used to displace the protein, since the structure evolves into other potential energy wells.
Therefore, the displacement along the modes is valid for small distances, but the displacement along greater distances
may deform the structure of the protein if no care is taken. The advantage of this methodology is to adaptively change
the direction used to displace the system, taking into account the structural and energetic constraints imposed by the
system itself and the medium, which allows the system to explore new pathways.
'''

import argparse
import os
import zipfile
import subprocess
import shutil
import sys
import numpy as np # need install
import MDAnalysis as mda # need install
from MDAnalysis.analysis import align
from pathlib import Path
import time as tm
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def unzip(filepath, path_dir):
    '''
    Extract files from a .zip compressed file.
    :param
        (str) filepath: path to .zip file
    :param
        (str) path_dir: path to destination folder
    '''

    with zipfile.ZipFile(filepath, 'r') as compressed:
        compressed.extractall(path_dir)


def write_nm(nms_to_write):
    '''
    Write CHARMM normal mode vectors in NAMD readable format

    :param
        (list) nms_to_write: list or normal modes to write
    '''

    # Extract CHARMM topology and parameters files
    unzip(f"{input_dir}/charmm_toppar.zip", input_dir)

    nms = [str(t + '\n') for t in nms_to_write.split(',')]
    with open(f"{input_dir}/input.txt", 'w') as input_nm:
        input_nm.writelines(nms)
    os.chdir(f"{cwd}/.modules")
    cmd = (f"charmm -i wrt-nm.mdu psffile={psffile.split('/')[-1]}"
           f" modfile={modefile.split('/')[-1]} -o {input_dir}/wrt-nm.out")
    returned_value = subprocess.call(cmd, shell=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{pgmerr}An error occurred while writing the normal mode vectors.\n"
              f"{pgmerr}Inspect the file {err}inputs/wrt-nm.out{std} for detailed information.\n")
        sys.exit()


def combine_modes():
    '''
    Combine and normalize normal modes then apply the RMSD filtering
    '''

    os.chdir(f"{cwd}/.modules")
    cmd = (f"charmm -i rms-filtering.mdu psffile={psffile.split('/')[-1]}"
           f" pdbfile={pdbfile.split('/')[-1]} rmsthreshold={rmsfiltering} rep={replicas} -o {input_dir}/rms-filtering.out")
    returned_value = subprocess.call(cmd, shell=True,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if returned_value != 0:
        print(f"{pgmerr}An error occurred while combining the normal modes.\n"
              f"{pgmerr}Inspect the file {err}inputs/rms-filtering.out{std} for detailed information.\n")
        sys.exit()


def excite(q_vector, user_ek):
    '''
    Scale the combined normal modes to be used as additional
    velocities during aMDeNM simulations

    :param
        (matrix) q_vector: combined vector to excite
    :param
        (float) user_ek: user defined excitation energy
    :return:
        (matrix) exc_vec: excitation vector
    '''

    # Excite
    fscale = np.sqrt((2 * user_ek) / sel_mass)
    exc_vec = (q_vector.T * fscale).T

    return exc_vec

def wrt_vec(xyz, output_file):
    '''
    Write a set of coordinates in a new file

    :param
        (array) xyz: vector containing the xyz coordinates
    :param
        (str) output_file: output file name
    '''

    # Copy the xyz coordinates into the dataframe
    sys_zeros.positions = np.zeros((N, 3))
    vector = np.append(xyz, sys_zeros.positions, axis=0)
    sys_zeros.positions = vector[:N]

    # Write the output file
    sys_zeros.write(f"{output_file}", file_format="NAMDBIN")

def clean():
    '''
    Delete previous run files
    '''
    # Removing previous replicas folders
    files = os.listdir(cwd)
    for item in files:
        if item.startswith("rep"):
            shutil.rmtree(os.path.join(cwd, item), ignore_errors=True)


# Get working directory path
cwd = os.getcwd()
input_dir = f"{cwd}/inputs"

# Style variables
tle = '\033[1;106m'
hgh = '\033[47m'
wrn = '\033[33m'
err = '\033[31m'
ext = '\033[32m'
std = '\033[0m'

# Program output variables
pgmprf = 'amdenm'
pgmnam = f"+{ext}{pgmprf}> {std}"
pgmwrn = f"%{wrn}{pgmprf}-Wrn> {std}"
pgmerr = f"%{err}{pgmprf}-Err> {std}"

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

banner = (f"{logo}\n"
          f"\t\t{tle}Adaptive Molecular Dynamics with Python{std}\n"
          f"\t\t\t     version: {version}\n"
          f"\n{citation}\n")

message="This program can setup and run multi-replica aMDeNM simulations through NAMD."

print(banner)

parser = argparse.ArgumentParser(description=message)
subparsers = parser.add_subparsers()

#######################
### RUNNING OPTIONS ###
#######################

# Cleaning
opt_clean = subparsers.add_parser('clean', help="Erase previous aMDeNM configuration file.")
opt_clean.set_defaults(clean=True)

# Run
opt_run = subparsers.add_parser('run', help="Setup and run aMDeNM simulations.")
# File options (required)
opt_run.add_argument('-mod', '--modefile',
                    action="store", type=str, required=True,
                    help="REQUIRED: CHARMM normal mode file.")
opt_run.add_argument('-psf', '--psffile',
                    action="store", type=str, required=True,
                    help="REQUIRED: PSF file.")
opt_run.add_argument('-pdb', '--pdbfile',
                    action="store", type=str, required=True,
                    help="REQUIRED: PDB file.")
opt_run.add_argument('-coor','--coorfile',
                    action="store", type=str, required=True,
                    help="REQUIRED: NAMD coordinates file.")
opt_run.add_argument('-vel', '--velfile',
                    action="store", type=str, required=True,
                    help="REQUIRED: NAMD velocities file.")
opt_run.add_argument('-xsc', '--xscfile',
                    action="store", type=str, required=True,
                    help="REQUIRED: NAMD PBC file.")
# TODO: add additional treatment to the conf.namd file regarding the .str information
#  including formating it, if necessary. Make it optional.
opt_run.add_argument('-str', '--strfile',
                    action="store", type=str, required=True,
                    help="REQUIRED: NAMD additional box info file.")
# Variables options (optional)
opt_run.add_argument('-nm', '--modes',
                    action="store", type=str, default="7,8,9",
                    help="Normal modes to excite (default: 7,8,9).")
opt_run.add_argument('-ek', '--energy',
                    action="store", type=float, default=0.125,
                    help="Excitation energy (default: 0.125 kcal/mol).")
opt_run.add_argument('-t', '--time',
                    action="store", type=int, default=250,
                    help="Total simulation time (default: 250ps).")
opt_run.add_argument('-sel', '--selection',
                    action="store", type=str, default="protein",
                    help="Atom selection considered in normal modes calculations (default: protein).")
opt_run.add_argument('-rep', '--replicas',
                    action="store", type=int, default=10,
                    help="Number of aMDeNM replicas to run (default: 10).")
opt_run.add_argument('-rms', '--rmsfiltering',
                    action="store", type=float, default=1,
                    help="Value of RMSD filtering (default: 1.0).")
opt_run.set_defaults(run=True)

# Create a dictionary containing the user-provided arguments
args = vars(parser.parse_args())

# If no argument was provided
if args == {}:
    parser.error(f"{err}At least one argument must be provided.{std}")

# Store the values
if 'modefile' in args:
    modepath = args['modefile']
    modefile = args['modefile'].split('/')[-1]
if 'psffile' in args:
    psfpath = args['psffile']
    psffile = args['psffile'].split('/')[-1]
if 'pdbfile' in args:
    pdbpath = args['pdbfile']
    pdbfile = args['pdbfile'].split('/')[-1]
if 'coorfile' in args:
    coorpath = args['coorfile']
    coorfile = args['coorfile'].split('/')[-1]
if 'velfile' in args:
    velpath = args['velfile']
    velfile = args['velfile'].split('/')[-1]
if 'xscfile' in args:
    xscpath = args['xscfile']
    xscfile = args['xscfile'].split('/')[-1]
if 'strfile' in args:
    strpath = args['strfile']
    strfile = args['strfile'].split('/')[-1]
if 'modes' in args: modes = args['modes']
if 'energy' in args: energy = args['energy']
if 'time' in args: time = args['time']
if 'selection' in args: selection = args['selection']
if 'replicas' in args: replicas = args['replicas']
if 'rmsfiltering' in args: rmsfiltering = args['rmsfiltering']

# Running options
if 'run' in args:
    print(f"{pgmnam}{tle}Setup and run aMDeNM simulations{std}\n")

    # Remove previous temporary and replica files
    clean()

    # Test if the provided files exist
    file_list = [modepath, psfpath, pdbpath, coorpath, velpath, xscpath, strpath]
    for file in file_list:
        if not os.path.isfile(file):
            print(f"{pgmerr}File {err}{file.split('/')[-1]}{std} not found.")
            sys.exit()
        # Test if the provided files are at the input folder and copy them if not
        if not os.path.isfile(f"{input_dir}/{file.split('/')[-1]}"):
            shutil.copy(file, input_dir)
            print(f"{pgmwrn}File {wrn}{file.split('/')[-1]}{std} was copied to inputs folder.")

    # Get some information from the system
    print(f"{pgmnam}Getting system info.")
    sys_pdb = mda.Universe(f"{input_dir}/{psffile}", f"{input_dir}/{coorfile}", format="NAMDBIN")
    N = sys_pdb.atoms.n_atoms                                       # Total atom number
    sys_mass = sys_pdb.atoms.masses                                 # System atomic mass
    sys_zeros = sys_pdb.atoms.select_atoms("all")
    init_coor = sys_pdb.atoms.select_atoms(selection).positions
    sel_atom = sys_pdb.atoms.select_atoms(selection).n_atoms        # Number of selected atoms
    sel_mass = sys_pdb.atoms.select_atoms(selection).masses         # Selection atomic mass

    # Correction variables definition
    globfreq = cos_alpha = 0.5
    qrms_correc = 0.5

    # Define the number of excitation cycles
    # time / (total_steps * timestep)
    end_loop = int(time / (100 * 0.002))

    # Define the top and bottom values for Ek correction
    # 25% window of excitation energy
    top = energy * 1.25
    bottom = energy * 0.75

    # Write the normal mode vectors
    print(f"{pgmnam}Writing normal mode vectors {ext}{modes}{std}.")
    write_nm(modes)

    # Extract NAMD topology and parameters files
    unzip(f"{input_dir}/namd_toppar.zip", input_dir)

    # Combine the modes
    print(f"{pgmnam}The RMS filtering threshold is {ext}{rmsfiltering}{std}.")
    print(f"{pgmnam}Generating {ext}{replicas}{std} combinations for modes {ext}{modes}{std}.")
    print(f"{pgmnam}This may take a while.")
    combine_modes()

    #######################
    ### CALL FOR ACTION ###
    #######################

    # Create the replica folder and enter it
    for rep in range(1, (replicas + 1)):
        print(f"{pgmnam}{hgh}Starting aMDeNM calculations for rep{rep}{std}")
        rep_dir = f"{cwd}/rep{rep}"
        os.makedirs(rep_dir, exist_ok=True)
        os.chdir(rep_dir)

        # Copying the NM combination vector and the alphas
        shutil.copy(f"{cwd}/rep-struct-list/rep{rep}-pff-vector.vec", "pff_vector.vec")
        shutil.copy(f"{cwd}/rep-struct-list/rep{rep}-alphas.txt", "alphas.txt")

        # Excite the combined vector according to user-defined energy increment
        print(f"{pgmnam}Writing the excitation vector with a Ek injection of {ext}{energy}{std} kcal/mol.")
        q_vec = mda.Universe(f"{input_dir}/{psffile}", "pff_vector.vec", format="CRD")
        q_vec = q_vec.atoms.select_atoms(selection).positions
        exc_vec = excite(q_vec, energy)

        # Write the combination and the excited vector
        wrt_vec(q_vec, "cntrl_vector.vec")
        wrt_vec(exc_vec, "excitation.vel")

        # Start the excitation loop, copy initial files and define initial variables
        loop = 0
        cnt = 1
        vp, ek, qp, rmsp = [[], [], [], []]
        ref_str = f"step_{loop}.coor"   # will change eventually during the simulation
        shutil.copy(f"{input_dir}/{coorfile}", f"correc_ref.coor")
        shutil.copy(f"{input_dir}/{coorfile}", f"step_{loop}.coor")
        shutil.copy(f"{input_dir}/{velfile}", f"step_{loop}.vel")
        shutil.copy(f"{input_dir}/{xscfile}", f"step_{loop}.xsc")

        while loop < end_loop:

            if loop == 0:
                # Read the current NAMD velocities file
                vel_curr = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
                vel_curr = vel_curr.coord.positions
                # Read the excitation velocities file
                vel_exc = mda.Universe(f"{input_dir}/{psffile}", "excitation.vel", format="NAMDBIN")
                vel_exc = vel_exc.coord.positions

                # Write the input velocities vel_tot = vel_curr + vel_exc
                vel_tot = vel_curr + vel_exc
                wrt_vec(vel_tot, f"step_{loop}.vel")

            # Loop update
            loop += 1

            # Create NAMD configuration file
            shutil.copy(f"{input_dir}/conf.namd", 'conf.namd')
            namd_conf = Path('conf.namd')
            namd_conf.write_text(namd_conf.read_text().replace('$PSF', f"{input_dir}/{psffile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$PDB', f"{input_dir}/{pdbfile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$STR', f"{input_dir}/{strfile}"))
            namd_conf.write_text(namd_conf.read_text().replace('$COOR', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$VEL', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$XSC', str(loop - 1)))
            namd_conf.write_text(namd_conf.read_text().replace('$OUTPUT', str(loop)))

            # Run NAMD
            now = tm.strftime("%H:%M:%S")
            print(f"{pgmnam}{now} {ext}Replica {rep}{std}: running {ext}step {loop}{std} of {end_loop}...")
            run_namd = f"namd3 conf.namd > step_{loop}.log"
            returned_value = subprocess.call(run_namd, shell=True,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if returned_value != 0:
                print(f"{pgmerr}An error occurred while running NAMD.\n"
                      f"{pgmerr}Inspect the file {err}step_{loop}.log{std} for detailed information.\n")
                sys.exit()

            ## EVALUATE IF IT IS NECESSARY TO CHANGE THE EXCITATION DIRECTION ##
            coor_ref = mda.Universe(f"{input_dir}/{psffile}", "correc_ref.coor", format="NAMDBIN")
            coor_ref = coor_ref.atoms.select_atoms(selection).positions

            coor_curr = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
            coor_curr = coor_curr.atoms.select_atoms(selection).positions

            # Compute the difference and mass-weight the difference
            # (qcurr - qref) * sqrt(m)
            diff = ((coor_curr - coor_ref).T * np.sqrt(sel_mass)).T

            # Read the excitation vector
            cntrl_vec = mda.Universe(f"{input_dir}/{psffile}", "cntrl_vector.vec", format="NAMDBIN")
            cntrl_vec = cntrl_vec.atoms.select_atoms(selection).positions

            # Project the current coordinates onto Q
            q_proj = np.sum(diff * cntrl_vec)
            rms_check = np.sqrt((q_proj ** 2) / np.sum(sel_mass))

            # Evaluate the distance displaced along the excitation vector
            ''' First we project the difference between the last and current average structures
            onto the normal modes space to determine the delta value to accept or reject
            the new vector; then this vector difference is normalized to be used as velocities'''
            if rms_check >= qrms_correc:
                # Compute the average structure of the last excitation
                ts = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop - 1}.coor", format="NAMDBIN")
                avg_positions = ts.atoms.select_atoms(selection).positions
                ts = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
                avg_positions += ts.atoms.select_atoms(selection).positions
                avg_positions = avg_positions / 2
                wrt_vec(avg_positions, f"average_{loop}.coor")

                # Open the reference and mobile structures
                ref = mda.Universe(f"{input_dir}/{psffile}", ref_str, format="NAMDBIN")
                ref = ref.atoms.select_atoms(selection)
                mob = mda.Universe(f"{input_dir}/{psffile}", f"average_{loop}.coor", format="NAMDBIN")
                mob = mob.atoms.select_atoms(selection)

                # Align the structures and compute the mass-weighted difference
                align.alignto(mob, ref, select="protein", weights="mass")
                diff = ((mob.positions - ref.positions).T * np.sqrt(sel_mass)).T

                # Normalize the mass-weighted difference vector
                diff = diff / np.sqrt(np.sum(diff * diff))

                # Project the current coordinates onto Q
                dotp = np.sum(diff * cntrl_vec)

                # Set the average structure as the new reference for the next steps
                ref_str = f"average_{loop}.coor"

                if dotp <= cos_alpha:
                    # TODO: add an if to skip this when adding the original MDeNM routine
                    shutil.copy(f"step_{loop}.coor", f"correc_ref.coor")
                    qrms_correc = 0

                # Rename the previous excitation vector files
                shutil.copy("excitation.vel", f"excitation.vel.{cnt}")
                shutil.copy("cntrl_vector.vec", f"cntrl_vector.vec.{cnt}")
                cnt += 1

                # Write the corrected excitation vector
                print(f"{pgmnam}Writing the corrected excitation vector.")
                wrt_vec(diff, "cntrl_vector.vec")

                # Excite and write the new excited vector
                exc_vec = excite(diff, energy)
                wrt_vec(exc_vec, "excitation.vel")

                # Update the rms correction variable value
                qrms_correc += globfreq

            # OBTAIN THE VELOCITIES AND KINETIC ENERGY PROJECTED ONTO Q
            # Open the current velocities file and mass-weight
            curr_vel = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
            curr_vel = ((curr_vel.atoms.select_atoms(selection).positions).T * np.sqrt(sel_mass)).T

            # Read the excitation vector
            cntrl_vec = mda.Universe(f"{input_dir}/{psffile}", "cntrl_vector.vec", format="NAMDBIN")
            cntrl_vec = cntrl_vec.atoms.select_atoms(selection).positions

            # Calculate the dot product between Vcurr and Q
            # Vproj = [ (V·Q)/|Q| ] · Q
            dotp = np.sum(curr_vel * cntrl_vec)
            v_proj = ((cntrl_vec * dotp).T / np.sqrt(sel_mass)).T
            wrt_vec(v_proj, "velo_proj.vel")

            # Compute the scalar projection of velocity
            velo = np.sum(v_proj)
            vp.append(f"{str(round(velo, 5))}\n")

            # Calculate the kinetic energy from projected velocities
            ek_vel = np.sum((v_proj ** 2) / 2)
            ek.append(f"{str(round(ek_vel, 5))}\n")

            # PROJECT THE COORDINATES ONTO Q
            # Open the current coordinates file
            curr_coor = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.coor", format="NAMDBIN")
            curr_coor = curr_coor.atoms.select_atoms(selection).positions

            # Compare the current with the initial coordinates
            diff = curr_coor - init_coor
            diff = (diff.T * np.sqrt(sel_mass)).T

            # Calculate the dot product between qcurr and Q
            dotp = np.sum(diff * cntrl_vec)
            qp.append(f"{str(round(dotp, 5))}\n")

            # Compute the rms displacement along the vector Q
            mrms = np.sqrt((dotp ** 2) / sum(sel_mass))
            rmsp.append(f"{str(round(mrms, 5))}\n")

            # RESCALE KINETIC ENERGY ACCORDING TO VALUES PROJECTED ONTO VECTOR Q
            '''Re-excite the NM vector when ek is below inferior limit
            or relax the energy when ek is above superior limit'''
            if (ek_vel < bottom) or (ek_vel > top):
                # Read current and excitation velocities
                curr_vel = mda.Universe(f"{input_dir}/{psffile}", f"step_{loop}.vel", format="NAMDBIN")
                curr_vel = curr_vel.coord.positions
                exc_vec = mda.Universe(f"{input_dir}/{psffile}", "excitation.vel", format="NAMDBIN")
                exc_vec = exc_vec.coord.positions
                v_proj = mda.Universe(f"{input_dir}/{psffile}", "velo_proj.vel", format="NAMDBIN")
                v_proj = v_proj.coord.positions

                # Compute the difference between the projected and the excitation velocities
                # and then sum to the current velocities: Vnew = Vdyna + (VQ - Vp)
                new_vel = curr_vel + (exc_vec - v_proj)
                wrt_vec(new_vel, f"step_{loop}.vel")

        # Write the projections into files
        for i,j in zip((vp, ek, qp, rmsp), ("vp", "ek", "coor", "rms")):
            with open(f"{j}-proj.out", 'w') as write:
                write.writelines(i)

        # De-excite the system
        shutil.copy(f"{input_dir}/deexcitation.namd", 'deexcitation.namd')
        deexc_conf = Path('deexcitation.namd')
        deexc_conf.write_text(deexc_conf.read_text().replace('$PSF', f"{input_dir}/{psffile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$PDB', f"{input_dir}/{pdbfile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$STR', f"{input_dir}/{strfile}"))
        deexc_conf.write_text(deexc_conf.read_text().replace('$COOR', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$VEL', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$XSC', str(loop)))
        deexc_conf.write_text(deexc_conf.read_text().replace('$TS', str(int(time / 0.002))))

        # Run NAMD
        now = tm.strftime("%H:%M:%S")
        print(f"{pgmnam}{now} {ext}Replica {rep}{std}: running the {ext}de-excitation step{std}...")
        run_namd = f"namd3 deexcitation.namd > deexcitation.log"
        returned_value = subprocess.call(run_namd, shell=True,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if returned_value != 0:
            print(f"{pgmerr}An error occurred while running NAMD.\n"
                  f"{pgmerr}Inspect the file {err}deexcitation.log{std} for detailed information.\n")
            sys.exit()

elif 'clean' in args:
    print(f"{pgmnam}{tle}Clean previous pyAdMD setup files{std}\n")

    # Removing previous configuration files
    files = os.listdir(input_dir)
    for item in files:
        if item.endswith((".txt", ".out", ".crd", ".psf", ".pdb", ".coor", ".vel", ".xsc", ".str", ".mod")):
            os.remove(os.path.join(input_dir, item))
    for item in ("charmm_toppar", "namd_toppar"):
            shutil.rmtree(os.path.join(input_dir, item), ignore_errors=True)

    # Remove previous temporary and replica files
    clean()

    print(f"{pgmnam}Erasing is done.\n")
