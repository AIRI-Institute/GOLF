import argparse
import collections
import concurrent.futures
import datetime
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from pathlib import Path

from ase import Atoms
from ase.db import connect
from schnetpack import properties

from env.dft import calculate_dft_optimization_tcp_client, get_dft_server_destinations

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm

from utils.arguments import str2bool, check_positive


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@dataclass
class StepStats:
    n_iter: int = None
    energy: float = None
    force: np.ndarray = None
    energy_ground_truth: float = None
    force_ground_truth: np.ndarray = None


@dataclass
class ConformationOptimizationStats:
    conformation_id: int = None
    initial_energy_ground_truth: float = None
    optimal_energy_ground_truth: float = None
    non_finite_action: bool = False
    non_finite_action_step: int = -1
    non_finite_energy_ground_truth: float = math.inf
    lbfgs_done: bool = False
    lbfgs_done_step: int = -1
    lbfgs_done_energy_ground_truth: float = math.inf
    step2stats: dict = field(default_factory=dict)

    def pct(self, energy_ground_truth):
        if energy_ground_truth is None:
            return None

        return (self.initial_energy_ground_truth - energy_ground_truth) / (
            self.initial_energy_ground_truth - self.optimal_energy_ground_truth
        )

    def to_dict(self):
        result = {
            "conformation_id": int(self.conformation_id),
            "initial_energy_ground_truth": float(self.initial_energy_ground_truth),
            "optimal_energy_ground_truth": float(self.optimal_energy_ground_truth),
            "non_finite_action": bool(self.non_finite_action),
            "lbfgs_done": bool(self.lbfgs_done),
            "not_finite_action_step": int(self.non_finite_action_step),
            "lbfgs_done_step": int(self.lbfgs_done_step),
        }

        for step in sorted(self.step2stats.keys()):
            step_stats = self.step2stats[step]
            result[f"n_iter@step:{step}"] = float(step_stats.n_iter)
            pct = self.pct(step_stats.energy_ground_truth)
            if pct is not None:
                pct = float(pct)
            result[f"pct_of_minimized_energy@step:{step}"] = pct
            result[f"energy@step:{step}"] = float(step_stats.energy)
            if step_stats.energy_ground_truth is None:
                result[f"energy_gt@step:{step}"] = None
                result[f"energy_mse@step:{step}"] = None
            else:
                result[f"energy_gt@step:{step}"] = float(step_stats.energy_ground_truth)
                result[f"energy_mse@step:{step}"] = float(
                    (step_stats.energy - step_stats.energy_ground_truth) ** 2
                )

            if step_stats.force_ground_truth is None:
                result[f"force_mse@step:{step}"] = None
                result[f"force_norm@step:{step}"] = None
            else:
                result[f"force_mse@step:{step}"] = float(
                    np.mean((step_stats.force - step_stats.force_ground_truth) ** 2)
                )
                result[f"force_norm@step:{step}"] = float(
                    np.mean(np.linalg.norm(step_stats.force_ground_truth, ord="fro"))
                )

        return result


def log_conformation_optimization_stats(
    conformation_optimization_stats, evaluation_metrics_path
):
    with open(evaluation_metrics_path, "a") as file_obj:
        json.dump(conformation_optimization_stats.to_dict(), file_obj)
        file_obj.write("\r\n")


def reserve_db_ids(db_path, db_ids):
    with connect(db_path) as conn:
        for db_id in db_ids:
            n = conn.count(selection=db_id)
            if n == 1:
                conn.update(id=db_id, data={"reserve_id": db_id})
            elif n == 0:
                with conn.managed_connection() as sqlite_conn:
                    cur = sqlite_conn.cursor()
                    cur.execute(f"INSERT INTO systems (id) VALUES ({db_id})")

                conn._write(
                    atoms=Atoms(),
                    key_value_pairs={},
                    data={"reserve_id": db_id},
                    id=db_id,
                )
            else:
                assert False, f"{n} conformations with the same db_id={db_id}!"


def write_to_db(
    db_path, db_ids, atoms_list, smiles_list, initial_energies, optimal_energies
):
    assert len(db_ids) == len(atoms_list)
    with connect(db_path) as conn:
        for db_id, atoms, smiles, initial_energy, optimal_energy in zip(
            db_ids, atoms_list, smiles_list, initial_energies, optimal_energies
        ):
            conn.update(
                id=db_id,
                atoms=atoms,
                delete_keys=["reserve_id"],
                smiles=smiles,
                data={
                    "initial_energy": initial_energy,
                    "optimal_energy": optimal_energy,
                },
            )


def update_data_in_db(db_path, db_id, history):
    if history:
        with connect(db_path) as conn:
            conn.update(id=db_id, data={"history": history})


def aggregate2string(stats, pct_min_threshold, pct_max_threshold, digits=3):
    stats_mean_std = {}
    for key, value in stats.items():
        min_threshold = -math.inf
        max_threshold = math.inf
        if "pct" in key:
            min_threshold = pct_min_threshold
            max_threshold = pct_max_threshold

        value = np.asarray(value)
        mask = (value >= min_threshold) & (value <= max_threshold)
        value = value[mask]
        mean_string = f"{round(value.mean(), digits)}"
        std_string = f"{round(value.std(ddof=0), digits)}"
        stats_mean_std[key] = mean_string, std_string

    return stats_mean_std


def print_stats_table(n_conf, stats2strings):
    name_width = max(map(len, stats2strings.keys()))
    mean_width = max(map(len, (value[0] for value in stats2strings.values())))
    std_width = max(map(len, (value[1] for value in stats2strings.values())))
    dashes = "-" * (name_width + mean_width + std_width + 12)
    lines = [dashes]
    header = f"| n_conf = {n_conf}"
    header += " " * (len(dashes) - len(header) - 1)
    header += "|"
    lines.append(header)
    lines.append(dashes)
    for name, (mean, std) in stats2strings.items():
        name_space = " " * (name_width - len(name))
        mean_space = " " * (mean_width - len(mean))
        std_space = " " * (std_width - len(std))
        lines.append(
            f"| {name}{name_space} | {mean}{mean_space} +/- {std}{std_space} |"
        )
    lines.append(dashes)
    print("\n", "\n".join(lines), sep="", flush=True)


def read_metrics(path):
    metrics = []
    with open(path, "r") as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == "":
                continue

            metrics.append(json.loads(line))

    return metrics


def reconcile_results_db_and_evaluation_metrics(
    results_db_path, evaluation_metrics_path
):
    metrics = sorted(
        read_metrics(evaluation_metrics_path),
        key=lambda metric: metric["conformation_id"],
    )
    metrics_ids = []
    for metric in metrics:
        if len(metrics_ids) == 0 or metric["conformation_id"] == metrics_ids[-1] + 1:
            metrics_ids.append(metric["conformation_id"])
        else:
            break

    all_db_ids = []
    with connect(results_db_path) as conn:
        with conn.managed_connection() as sqlite_conn:
            cur = sqlite_conn.cursor()
            cur.execute("select id from systems order by id asc")
            next_row = cur.fetchone()
            while next_row:
                all_db_ids.append(next_row[0])
                next_row = cur.fetchone()

    db_ids = []
    for idx in all_db_ids:
        if len(db_ids) == 0 or idx == db_ids[-1] + 1:
            db_ids.append(idx)
        else:
            break

    db_ids_to_delete = set(all_db_ids) - set(db_ids)

    assert metrics_ids[0] == db_ids[0]
    if len(metrics_ids) < len(db_ids):
        db_ids_to_delete += set(db_ids[len(metrics_ids) :])
    elif len(metrics_ids) > len(db_ids):
        metrics_ids = metrics_ids[: len(db_ids)]

    with connect(results_db_path) as conn:
        for db_id in db_ids_to_delete:
            del conn[db_id]

    metrics_ids = set(metrics_ids)
    with open(evaluation_metrics_path, "w") as file_obj:
        for metric in metrics:
            if metric["conformation_id"] in metrics_ids:
                json.dump(metric, file_obj)
                file_obj.write("\r\n")


def make_env(args):
    # Env kwargs
    env_kwargs = {
        "db_path": args.db_path,
        "n_parallel": args.n_parallel,
        "timelimit": args.timelimit,
        "sample_initial_conformations": args.sample_initial_conformations,
        "num_initial_conformations": args.num_initial_conformations,
    }

    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        "dft": args.reward == "dft",
        "n_threads": args.n_threads,
        "minimize_on_every_step": args.minimize_on_every_step,
        "minimize_on_done": False,
        "terminate_on_negative_reward": args.terminate_on_negative_reward,
        "max_num_negative_rewards": args.max_num_negative_rewards,
        "evaluation": True,
    }

    eval_env = env_fn(**env_kwargs)
    eval_env = RewardWrapper(eval_env, **reward_wrapper_kwargs)

    return eval_env


def get_not_finished_mask(state, finished):
    n_molecules = state[properties.n_atoms].size(0)
    n_atoms = get_atoms_indices_range(state).cpu().numpy()
    not_finished_mask = np.ones(
        shape=(state[properties.position].size(0),), dtype=np.float32
    )
    for i in range(n_molecules):
        if finished[i]:
            not_finished_mask[n_atoms[i] : n_atoms[i + 1]] = 0

    return np.expand_dims(not_finished_mask, axis=1)


def main(args):

    dft_server_destinations = get_dft_server_destinations(
        args.n_threads, False
    )
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    executors = [
        concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=mp.get_context(method)
        )
        for _ in range(len(dft_server_destinations))
    ]
    futures = {}
    tasks = []
    global_conformation_index = 0

    with connect(args.eval_db_path) as conn:
        for row in conn.select():

            tasks.append(
                (
                    row.id,
                    -1,
                    row.toatoms(),
                )
            )

        for task in tasks:
            conformation_id, step, molecule = task
            worker_id = global_conformation_index % len(dft_server_destinations)
            host, port = dft_server_destinations[worker_id]
            future = executors[worker_id].submit(
                calculate_dft_optimization_tcp_client,
                task,
                host,
                port,
                True,
            )
            futures[global_conformation_index] = future
            global_conformation_index += 1

        pbar_pct = tqdm(total=len(tasks), mininterval=10, desc="Evaluated conformations")
        done_future_ids = set()
        while len(done_future_ids) < len(tasks):
            for future_id, future in futures.items():
                if not future.done() or future_id in done_future_ids:
                    continue

                done_future_ids.add(future_id)
                conformation_id, step, history = future.result()
                if history is None:
                    print(
                        f"DFT did not converged: conformation_id={conformation_id} step={step}",
                        flush=True,
                    )


                update_data_in_db(args.results_db_path, conformation_id, history)

                pbar_pct.update(1)
        pbar_pct.close()

        for future_id in done_future_ids:
            del futures[future_id]

        for executor in executors:
            executor.shutdown(wait=False, cancel_futures=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # Env args
    parser.add_argument(
        "--eval_db_path",
        default="",
        type=str,
        help="Path to molecules database for evaluation",
    )

    parser.add_argument(
        "--results_db_path",
        default="",
        type=str,
        help="Path to molecules database for evaluation",
    )

    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of parallel threads for DFT computations",
    )

    args = parser.parse_args()

    main(args)

