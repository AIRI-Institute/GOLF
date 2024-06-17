import os
import pickle
import socket
import struct
import subprocess
import tempfile
import traceback
from datetime import datetime

import psi4

PORT_RANGE_BEGIN_TRAIN = 20000
PORT_RANGE_BEGIN_EVAL = 30000
HOSTS = [
    "192.168.19.21",
    "192.168.19.22",
    "192.168.19.23",
    "192.168.19.24",
    "192.168.19.25",
    "192.168.19.26",
    "192.168.19.27",
    "192.168.19.28",
    "192.168.19.29",
    "192.168.19.30",
]
# HOSTS = ["192.168.19.103"]


def recvall(sock, count):
    buf = b""
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return newbuf
        buf += newbuf
        count -= len(newbuf)
    return buf


def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack("!I", length))
    sock.sendall(data)


def recv_one_message(sock):
    buf = recvall(sock, 4)
    if not buf:
        return buf
    (length,) = struct.unpack("!I", buf)
    return recvall(sock, length)


def log(conformation_id, message, path, logging):
    if logging:
        with open(path, "a") as file_obj:
            print(
                f"{get_time()} - conformation_id={conformation_id} {message}",
                file=file_obj,
            )


def calculate_dft_energy_tcp_client(task, host, port, logging=False):
    path = f"client_{host}_{port}.out"
    conformation_id, step, ase_atoms = task
    try:
        log(conformation_id, "going to connect", path, logging)
        sock = socket.socket()
        sock.connect((host, port))
        log(conformation_id, "connected", path, logging)

        ase_atoms = ase_atoms.todict()

        task = (ase_atoms, step, conformation_id)
        task = pickle.dumps(task)
        send_one_message(sock, task)
        log(conformation_id, "send one message", path, logging)
        result = recv_one_message(sock)
        log(conformation_id, "received response", path, logging)
        idx, not_converged, energy, force = pickle.loads(result)
        assert conformation_id == idx

        return conformation_id, step, energy, force
    except Exception as e:
        description = traceback.format_exc()
        log(conformation_id, description, path, logging)
        return conformation_id, step, None, None


def get_dft_server_destinations(n_workers, host_file_path=None):
    if host_file_path:
        with open(host_file_path, "r") as f:
            hosts = f.readlines()
    else:
        hosts = HOSTS
    port_range_begin = PORT_RANGE_BEGIN_TRAIN
    destinations = []
    for host in hosts:
        for port in range(port_range_begin, port_range_begin + n_workers):
            destinations.append((host, port))

    return destinations


# Get correct hostname
def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.254.254.254", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


def get_time():
    return datetime.now().strftime("%H:%M:%S")


if __name__ == "__main__":
    import sys

    host = get_ip()
    num_threads = sys.argv[1]
    port = int(sys.argv[2])

    if len(sys.argv) >= 4:
        timeout_seconds = sys.argv[3]
    else:
        timeout_seconds = 600

    if len(sys.argv) >= 5:
        dft_script_path = sys.argv[4]
    else:
        dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        dft_script_path = os.path.join(dir_path, "dft_worker.py")

    server_socket = socket.socket()
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))

    server_socket.listen(1)

    print(port, "accept")
    conn, address = server_socket.accept()
    total_processed = 0

    while True:
        data = recv_one_message(conn)
        while len(data) == 0:
            print(port, "connection lost, accept")
            conn, address = server_socket.accept()
            data = recv_one_message(conn)

        print(
            f"{get_time()} -", port, "recv", len(data), "bytes from", address, end=" "
        )
        task = pickle.loads(data)
        conformation_id = task[2]
        print("conformation_id", conformation_id, flush=True)

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as file_obj:
            pickle.dump(task, file_obj)
            task_path = file_obj.name

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as file_obj:
            result_path = file_obj.name

        result = conformation_id, True, None, None
        try:
            completed_process = subprocess.run(
                [
                    sys.executable,
                    dft_script_path,
                    "--task_path",
                    task_path,
                    "--result_path",
                    result_path,
                    "--num_threads",
                    num_threads,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )

            print(
                f"returncode={completed_process.returncode}\nWorker stdout:\n{completed_process.stdout}",
                flush=True,
            )
            if completed_process.returncode == 0:
                with open(result_path, "rb") as file_obj:
                    result = pickle.load(file_obj)
        except subprocess.TimeoutExpired as e:
            print(e)
            if e.stdout is not None:
                print(f'Worker stdout:\n{e.stdout.decode("utf-8")}', flush=True)
            if e.stderr is not None:
                print(f'Worker stderr:\n{e.stderr.decode("utf-8")}', flush=True)

        os.remove(task_path)
        os.remove(result_path)

        result = pickle.dumps(result)

        total_processed += 1
        print(
            f"{get_time()} -",
            port,
            "going to send",
            len(result),
            "bytes to",
            address,
            flush=True,
        )
        send_one_message(conn, result)
        print(
            f"{get_time()} - data sent, total processed:{total_processed}",
            flush=True,
        )
