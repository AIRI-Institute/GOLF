import concurrent.futures
import io
import struct
import ase
import ase.io
import time
import socket
import psi4
import numpy as np
import multiprocessing as mp
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
psi_bohr2angstroms = 0.52917720859
EXECUTOR = None


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
    except SCFConvergenceError as e:
        # Set energy to some threshold if SOSCF does not converge
        print("DFT optimization did not converge!")
        return -10000.0
    psi4.core.clean()
    forces = -np.array(gradient) / psi_bohr2angstroms
    return energy, -np.array(gradient) / psi_bohr2angstroms


def update_ase_atoms_positions(atoms, positions):
    atoms.set_positions(positions)


def update_psi4_geometry(molecule, positions):
    psi4matrix = psi4.core.Matrix.from_array(positions / psi_bohr2angstroms)
    molecule.set_geometry(psi4matrix)
    molecule.update_geometry()


def calculate_dft_energy_queue_old(queue, n_threads):
    global EXECUTOR
    if EXECUTOR is None:
        method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        EXECUTOR = concurrent.futures.ProcessPoolExecutor(
            max_workers=n_threads, mp_context=mp.get_context(method)
        )

    futures = [EXECUTOR.submit(calculate_dft_energy_item, task) for task in queue]
    results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results = sorted(results, key=lambda x: x[0])

    return results


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)


def recv_one_message(sock):
    buf = recvall(sock, 4)
    length, = struct.unpack('!I', buf)
    return recvall(sock, length)


def calculate_dft_energy_queue(queue, n_threads):
    sockets = []

    #for host in [socket.gethostname()]:
    #    for port in [20000, 20001]:

    for host in ["192.168.19.101", "192.168.19.102"]:
        for port in range(20000, 20016):
            print("connect", host, port)

            sock = socket.socket()
            sock.connect((host, port))
            sockets.append(sock)

    results = []
    while len(queue) > 0:
        waitlist = []

        for sock in sockets:
            if len(queue) == 0: break

            task = queue.pop()
            ase_atoms, dummy, idx = task

            ase_atoms = ase_atoms.todict()

            task = (ase_atoms, dummy, idx)
            task = pickle.dumps(task)

            print("job", idx, "send", len(task), "bytes to", sock.getsockname())

            send_one_message(sock, task)
            waitlist.append(sock)

        for sock in waitlist:
            result = recv_one_message(sock)
            print("recv", len(result), "bytes from", sock.getsockname())

            result = pickle.loads(result)

            results.append(result)

    results = sorted(results, key=lambda x: x[0])

    return results


def calculate_dft_energy_item(task):
    # Get molecule from the queue
    ase_atoms, _, idx = task

    ase_atoms = ase.Atoms.fromdict(ase_atoms)

    print("task", idx)

    t1 = time.time()

    molecule = atoms2psi4mol(ase_atoms)

    # Perform DFT minimization
    # Energy in Hartree
    not_converged = True
    # Calculate DFT energy

    energy, gradient = get_dft_forces_energy(molecule)

    t = time.time() - t1

    print("time", t)

    return idx, not_converged, energy, gradient


if __name__ == "__main__":
    import sys

    host = socket.gethostname()
    port = int(sys.argv[1])

    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))

    server_socket.listen(1)

    print(port, "accept")
    conn, address = server_socket.accept()

    while True:
        data = recv_one_message(conn)
        while len(data) == 0:
            print(port, "connection lost, accept")
            conn, address = server_socket.accept()
            data = recv_one_message(conn)

        print(port, "recv", len(data), "bytes from", address)

        task = pickle.loads(data)

        result = calculate_dft_energy_item(task)

        result = pickle.dumps(result)

        print(port, "send", len(result), "bytes to", address)
        send_one_message(conn, result)
