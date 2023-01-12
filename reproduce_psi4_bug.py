import argparse

from multiprocessing import Manager, Pool
from ase.db import connect

import psi4
from psi4 import SCFConvergenceError
from psi4.driver.p4util.exceptions import OptimizationConvergenceError


psi4.core.set_output_file("/dev/null")


HEADER = "units ang \n nocom \n noreorient \n"
FUNCTIONAL_STRING = "wb97x-d/def2-svp"
psi_bohr2angstroms = 0.52917720859


def xyz2psi4mol(atoms, coordinates):
    molecule_string = HEADER + "\n".join(
        [" ".join([atom, ] + list(map(str, x))) for atom, x in zip(atoms, coordinates)])
    mol = psi4.geometry(molecule_string)
    return mol

def atoms2psi4mol(atoms):
    atomic_numbers = [str(atom) for atom in atoms.get_atomic_numbers().tolist()]
    coordinates = atoms.get_positions().tolist()
    return xyz2psi4mol(atomic_numbers, coordinates)


def get_dft_energy(mol):
    try:
        energy = psi4.driver.energy(FUNCTIONAL_STRING, **{"molecule": mol, "return_wfn": False})
    except SCFConvergenceError as e:
        # Set energy to some threshold if SOSCF does not converge 
        # Multiply by 627.5 to go from Hartree to kcal/mol
        print("DFT optimization did not converge!")
        return -10000.0 * 627.5
    psi4.core.clean()
    return energy * 627.5

def calculate_dft_energy_item(queue, M):
    # Get molecule from the queue
    ase_atoms, _, idx = queue.get()
    molecule = atoms2psi4mol(ase_atoms)

    # Perform DFT minimization
    not_converged = True
    if M > 0:
        psi4.set_options({'geom_maxiter': M})
        try:
            energy = psi4.optimize(FUNCTIONAL_STRING, **{"molecule": molecule, "return_wfn": False})
            not_converged = False
        except OptimizationConvergenceError as e:
            molecule.set_geometry(e.wfn.molecule().geometry())
            energy = e.wfn.energy()
        # Hartree to kcal/mol
        energy *= 627.5
        psi4.core.clean()
    else:
        # Calculate DFT energy
        energy = get_dft_energy(molecule)
    print(energy)
    
    return (idx, not_converged, energy)

def calculate_dft_energy_queue(queue, n_threads, M):
    m = Manager()

    # Create multiprocessing queue
    q = m.Queue()

    # Create a group of parallel writers and start them
    for elem in queue:
        q.put(elem)

    # Create multiprocessing pool
    p = Pool(n_threads)

    # Calculate energy for every item in the queue.
    # To mitigate large variance in dft computation times  
    # for different molecules the size of the queue
    # must be significantly larger then n_threads.
    energy_calculators = []
    for i in range(len(queue)):
        energy_calculators.append(p.apply_async(calculate_dft_energy_item, (q, M)))

    # Wait for the asynchrounous reader threads to finish
    results = [ec.get() for ec in energy_calculators]
    results = sorted(results, key=lambda x:x[0])
    p.terminate()
    p.join()

    return results


def main(num_molecules, num_threads):
    queue = []
    with connect("env/data/train_4k_mff.db") as conn:
        for i, idx in enumerate(range(2343, 2343 + 5 * num_molecules, 5)):
            atoms = conn.get(idx).toatoms()
            queue.append((atoms, len(atoms.get_atomic_numbers()), i))
    result = calculate_dft_energy_queue(queue, n_threads=num_threads, M=0)
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_molecules", default=1, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    args = parser.parse_args()
    main(args.num_molecules, args.num_threads)