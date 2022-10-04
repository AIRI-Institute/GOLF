import psi4

from psi4 import SCFConvergenceError

psi4.set_options({
    "CACHELEVEL": 0,
    "SOSCF_MAX_ITER": 30,
})
psi4.set_memory("8 GB")
psi4.core.set_output_file("/dev/null")

HEADER = "units ang \n nocom \n noreorient \n"
FUNCTIONAL_STRING = "wb97x-d/def2-svp"


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


def parse_psi4_molecule(xyz_file):
    mol_data = read_xyz_file(xyz_file)
    atoms, xyz_coordinates, charge = mol_data[0]
    mol_initial = xyz2psi4mol(atoms, xyz_coordinates)
    return mol_initial


def get_dft_energy(mol):
    try:
        energy = psi4.driver.energy(FUNCTIONAL_STRING, **{"molecule": mol, "return_wfn": False})
    except SCFConvergenceError as e:
        # Set energy to some threshold if SOSCF does not converge 
        # Multiply by 627.5 to go from Hartree to kcal/mol
        return -260.0 * 627.5
    psi4.core.clean()
    return energy * 627.5


def update_psi4_geometry(molecule, positions):
    psi4matrix = psi4.core.Matrix.from_array(positions)
    molecule.set_geometry(psi4matrix)
    molecule.update_geometry()
