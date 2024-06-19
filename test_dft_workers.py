import concurrent.futures
import multiprocessing as mp
import argparse
import numpy as np
import math

from ase.db import connect

from env.dft import calculate_dft_energy_tcp_client, get_dft_server_destinations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostnames",
        type=str,
        required=True,
        help="Path to txt file with ip addresses of CPU-rich machines",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/test_trajectories_initial.db",
        help="Path to database. Defaults to optimization evaluation database",
    )
    parser.add_argument(
        "--num_workers_per_server",
        type=int,
        required=True,
        help="Number of DFT workers per CPU-rich machine",
    )
    args = parser.parse_args()

    dft_server_destinations = get_dft_server_destinations(4, args.hostnames)
    with connect(args.db_path) as conn:
        atoms = conn.get(231).toatoms()

    futures = {}
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    executors = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context(method)
        )
        for _ in range(len(dft_server_destinations))
    ]

    print(f"Going to test {len(dft_server_destinations)} workers: ")
    for i, (host, port) in enumerate(dft_server_destinations):
        print(f"Worker {i}: host {host}, port {port}")
        task = (i, 0, atoms.copy())
        worker_id = i % len(dft_server_destinations)
        future = executors[worker_id].submit(
            calculate_dft_energy_tcp_client,
            task,
            host,
            port,
            False,
        )
        futures[i] = future
    with connect(args.db_path) as conn:
        atoms = conn.get(231).toatoms()

    print("Results:")

    while len(futures) > 0:
        del_future_ids = []
        for future_id, future in futures.items():
            if not future.done():
                continue

            del_future_ids.append(future_id)

            conformation_id, step, energy, force = future.result()
            worker_id = future_id % len(dft_server_destinations)
            host, port = dft_server_destinations[worker_id]
            if energy is None:
                print(
                    f"Worker {worker_id}: (host={host}, port={port}) returned None for conformation_id={conformation_id}.",
                    flush=True,
                )
            else:
                if math.isclose(energy, -899.09231071538):
                    print(f"Worker {worker_id}: (host={host}, port={port}) OK!")
                else:
                    print(
                        f"Worker {worker_id}: (host={host}, port={port}). Returned energy={energy} but energy={-899.09231071538} was expected.",
                        flush=True,
                    )
        for future_id in del_future_ids:
            del futures[future_id]
