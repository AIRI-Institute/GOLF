import argparse
import sys
import traceback

import ase
import ase.io
import time
import psi4
import numpy as np
import pickle

from psi4 import SCFConvergenceError
from psi4.driver.p4util.exceptions import OptimizationConvergenceError

# os.environ['PSI_SCRATCH'] = "/dev/shm/tmp"
# psi4.set_options({ "CACHELEVEL": 0 })

psi4.set_memory("8 GB")
psi4.core.IOManager.shared_object().set_default_path("/dev/shm/")
psi4.core.set_num_threads(4)
psi4.core.set_output_file("/dev/null")

HEADER = "units ang \n nocom \n noreorient \n"
FUNCTIONAL_STRING = "wb97x-d/def2-svp"
PSI4_BOHR2ANGSTROM = 0.52917720859


def read_xyz_file_block(file, look_for_charge=True):
    """ """

    atomic_symbols = []
    xyz_coordinates = []
    charge = 0
    title = ""

    line = file.readline()
    if not line:
        return None
    num_atoms = int(line)
    line = file.readline()
    if "charge=" in line:
        charge = int(line.split("=")[1])

    for _ in range(num_atoms):
        line = file.readline()
        atomic_symbol, x, y, z = line.split()[:4]
        atomic_symbols.append(atomic_symbol)
        xyz_coordinates.append([float(x), float(y), float(z)])

    return atomic_symbols, xyz_coordinates, charge


def read_xyz_file(filename, look_for_charge=True):
    """ """
    mol_data = []
    with open(filename) as xyz_file:
        while xyz_file:
            current_data = read_xyz_file_block(xyz_file, look_for_charge)
            if current_data:
                mol_data.append(current_data)
            else:
                break

    return mol_data


def xyz2psi4mol(atoms, coordinates):
    molecule_string = HEADER + "\n".join(
        [
            " ".join(
                [
                    atom,
                ]
                + list(map(str, x))
            )
            for atom, x in zip(atoms, coordinates)
        ]
    )
    mol = psi4.geometry(molecule_string)
    return mol


def atoms2psi4mol(atoms):
    atomic_numbers = [str(atom) for atom in atoms.get_atomic_numbers().tolist()]
    coordinates = atoms.get_positions().tolist()
    return xyz2psi4mol(atomic_numbers, coordinates)


def get_dft_forces_energy(mol):
    # Energy in Hartrees, force in Hatrees/Angstrom
    try:
        gradient, wfn = psi4.driver.gradient(
            FUNCTIONAL_STRING, **{"molecule": mol, "return_wfn": True}
        )
        energy = wfn.energy()
        forces = -np.array(gradient) / PSI4_BOHR2ANGSTROM
        return energy, forces
    except SCFConvergenceError as e:
        # Set energy to some threshold if SOSCF does not converge
        description = traceback.format_exc()
        print(f"DFT optimization did not converge!\n{description}", file=sys.stderr)
        return None, None
    finally:
        psi4.core.clean()


def update_ase_atoms_positions(atoms, positions):
    atoms.set_positions(positions)


def update_psi4_geometry(molecule, positions):
    psi4matrix = psi4.core.Matrix.from_array(positions / PSI4_BOHR2ANGSTROM)
    molecule.set_geometry(psi4matrix)
    molecule.update_geometry()


def calculate_dft_energy_item(task):
    # Get molecule from the queue
    ase_atoms, _, idx = task

    ase_atoms = ase.Atoms.fromdict(ase_atoms)

    print("task", idx)

    t1 = time.time()

    molecule = atoms2psi4mol(ase_atoms)

    # Perform DFT minimization
    # Energy in Hartree
    # Calculate DFT energy

    energy, gradient = get_dft_forces_energy(molecule)
    not_converged = energy is None

    t = time.time() - t1

    print("time", t)

    return idx, not_converged, energy, gradient


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()

    with open(args.task_path, 'rb') as file_obj:
        task = pickle.load(file_obj)

    result = calculate_dft_energy_item(task)

    with open(args.result_path, 'wb') as file_obj:
        pickle.dump(result, file_obj)
