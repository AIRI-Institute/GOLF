import os
import psi4
from multiprocessing import Manager, Pool

from psi4 import SCFConvergenceError
from psi4.driver.p4util.exceptions import OptimizationConvergenceError

#os.environ['PSI_SCRATCH'] = "/dev/shm/tmp"

psi4.set_options({
    "CACHELEVEL": 0,
})
psi4.set_memory("8 GB")
#psi4.core.set_output_file("/dev/null")

HEADER = "units ang \n nocom \n noreorient \n"
FUNCTIONAL_STRING = "wb97x-d/def2-svp"
psi_bohr2angstroms = 0.52917720859


def read_xyz_file_block(file, look_for_charge=True):
    """
    """

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
    """
    """
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
        return -260.0 * 627.5
    psi4.core.clean()
    return energy * 627.5


def update_psi4_geometry(molecule, positions):
    psi4matrix = psi4.core.Matrix.from_array(positions / psi_bohr2angstroms)
    molecule.set_geometry(psi4matrix)
    molecule.update_geometry()

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

def calculate_dft_energy_item(queue, M):
    # Get molecule from the queue
    molecule, _, idx = queue.get()

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
        energy = get_dft_energy(molecule[idx])
    
    return (idx, not_converged, energy)
