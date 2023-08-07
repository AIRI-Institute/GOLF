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

from AL.make_policies import make_policies
from env.dft import calculate_dft_energy_tcp_client, get_dft_server_destinations

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm

from AL.utils import recollate_batch, get_atoms_indices_range
from AL import DEVICE
from env.moldynamics_env import env_fn
from env.wrappers import RewardWrapper
from utils.arguments import str2bool


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
                self.initial_energy_ground_truth - self.optimal_energy_ground_truth)

    def to_dict(self):
        result = {'conformation_id': int(self.conformation_id),
                  'initial_energy_ground_truth': float(self.initial_energy_ground_truth),
                  'optimal_energy_ground_truth': float(self.optimal_energy_ground_truth),
                  'non_finite_action': bool(self.non_finite_action),
                  'lbfgs_done': bool(self.lbfgs_done),
                  'not_finite_action_step': int(self.non_finite_action_step),
                  'lbfgs_done_step': int(self.lbfgs_done_step),
                  }

        for step in sorted(self.step2stats.keys()):
            step_stats = self.step2stats[step]
            result[f'n_iter@step:{step}'] = float(step_stats.n_iter)
            pct = self.pct(step_stats.energy_ground_truth)
            if pct is not None:
                pct = float(pct)
            result[f'pct_of_minimized_energy@step:{step}'] = pct

        return result


def log_conformation_optimization_stats(conformation_optimization_stats, evaluation_metrics_path):
    with open(evaluation_metrics_path, 'a') as file_obj:
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


def write_to_db(db_path, db_ids, atoms_list, smiles_list, initial_energies, optimal_energies):
    assert len(db_ids) == len(atoms_list)
    with connect(db_path) as conn:
        for db_id, atoms, smiles, energy, optimal_energy in zip(db_ids, atoms_list, smiles_list, initial_energies,
                                                                optimal_energies):
            conn.update(
                id=db_id, atoms=atoms, delete_keys=["reserve_id"], smiles=smiles,
                data={'energy': energy, 'optimal_energy': optimal_energy}
            )


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
        db_ids_to_delete += set(db_ids[len(metrics_ids):])
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
        "molecules_xyz_prefix": args.molecules_xyz_prefix,
        "terminate_on_negative_reward": args.terminate_on_negative_reward,
        "max_num_negative_rewards": args.max_num_negative_rewards,
        "evaluation": True
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
            not_finished_mask[n_atoms[i]: n_atoms[i + 1]] = 0

    return np.expand_dims(not_finished_mask, axis=1)


def main(checkpoint_path, args, config):
    if args.deterministic:
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.use_deterministic_algorithms(True)

    dft_server_destinations = get_dft_server_destinations(args.n_threads, args.port_type == 'eval')
    method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    executors = [concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context(method)) for _ in
                 range(len(dft_server_destinations))]
    futures = {}
    barrier = collections.Counter()
    eval_env = make_env(config)
    eval_policy = make_policies(eval_env, eval_env, config)[0]

    if config.actor != "rdkit":
        agent_path = checkpoint_path / args.agent_path
        eval_policy.actor.load_state_dict(
            torch.load(agent_path, map_location=torch.device(DEVICE))
        )
        eval_policy.actor.to(DEVICE)
        eval_policy.actor.eval()

    state = eval_env.reset()
    if args.resume:
        metrics_ids = set(
            metric["conformation_id"]
            for metric in read_metrics(args.evaluation_metrics_path)
        )
        while set(eval_env.atoms_ids).issubset(metrics_ids):
            state = eval_env.reset()

    reserve_db_ids(args.results_db_path, eval_env.atoms_ids.copy())
    eval_policy.reset(state)

    finished = np.zeros(shape=eval_env.n_parallel, dtype=bool)
    lbfgs_done_steps = np.full(shape=eval_env.n_parallel, fill_value=-1)
    early_stop_steps = list(sorted(args.eval_early_stop_steps))
    early_stop_step_reached = {early_stop_step: np.full(shape=eval_env.n_parallel, fill_value=-1) for early_stop_step in
                               early_stop_steps}

    stats = {}
    for i, conformation_id in enumerate(eval_env.atoms_ids):
        stats[conformation_id] = ConformationOptimizationStats(
            conformation_id=conformation_id,
            initial_energy_ground_truth=eval_env.energy[i][0],
            optimal_energy_ground_truth=eval_env.optimal_energy[i][0]
        )
        barrier[conformation_id] += 1

    n_conf_processed_delta = None
    n_conf_processed_total = 0
    n_conf = args.conf_number if args.conf_number > 0 else eval_env.db_len
    assert n_conf >= args.n_parallel

    start_time = time.perf_counter()
    pbar_optimization = tqdm(total=n_conf, mininterval=10, desc='Optimized conformations')
    pbar_pct = tqdm(total=n_conf, mininterval=10, desc='Evaluated conformations')
    n_iters = np.zeros(shape=eval_env.n_parallel)
    global_conformation_index = 0
    while not np.all(finished).item():
        # Get current timesteps
        episode_timesteps = eval_env.get_env_step()

        # Select next action
        select_action_result = eval_policy.select_action(episode_timesteps)
        actions = select_action_result["action"]
        energies = select_action_result["energy"]
        lbfgs_dones = select_action_result["done"]
        is_finite_action = select_action_result["is_finite_action"]
        n_iters_dones = select_action_result["n_iter"]
        forces = np.asarray(
            np.vsplit(
                select_action_result["anti_gradient"],
                np.cumsum(state[properties.n_atoms].cpu().numpy())[:-1],
            ),
            dtype=object,
        )

        # Handle non-finite actions
        non_finite_action_mask = ~finished & (~is_finite_action)
        for i in np.where(non_finite_action_mask)[0]:
            conformation_id = eval_env.atoms_ids[i]
            stats[conformation_id].non_finite_action = True
            stats[conformation_id].non_finite_action_step = episode_timesteps[i]

        actions *= get_not_finished_mask(state, ~is_finite_action | finished)
        state, rewards, dones, info = eval_env.step(actions)
        dones = ~is_finite_action | np.asarray(dones)
        n_iters += np.asarray(n_iters_dones)
        steps = np.asarray(eval_env.get_env_step(), dtype=np.float32)

        tasks = []

        # Handle different time limits
        for early_stop_step in early_stop_steps:
            if np.all(steps < early_stop_step).item():
                break

            early_stop_step_mask = (
                    ~finished
                    & (steps >= early_stop_step)
                    & (early_stop_step_reached[early_stop_step] == -1)
            )

            for i in np.where(early_stop_step_mask)[0]:
                early_stop_step_reached[early_stop_step][i] = 1
                conformation_id = eval_env.atoms_ids[i]
                step_stats = StepStats(n_iter=n_iters[i], energy=energies[i], force=forces[i])
                stats[conformation_id].step2stats[early_stop_step] = step_stats
                tasks.append((conformation_id, early_stop_step, eval_env.molecule["dft"][i].copy()))

        # Handle lbfgs dones
        lbfgs_dones_mask = ~finished & lbfgs_dones & (lbfgs_done_steps == -1)
        lbfgs_done_steps[lbfgs_dones_mask] = steps[lbfgs_dones_mask]
        for i in np.where(lbfgs_dones_mask)[0]:
            conformation_id = eval_env.atoms_ids[i]
            stats[conformation_id].lbfgs_done = True
            stats[conformation_id].lbfgs_done_step = steps[i]

        for task in tasks:
            conformation_id, step, molecule = task
            worker_id = global_conformation_index % len(dft_server_destinations)
            host, port = dft_server_destinations[worker_id]
            future = executors[worker_id].submit(calculate_dft_energy_tcp_client, task, host, port,
                                                 args.logging_tcp_client)
            futures[global_conformation_index] = future
            barrier[conformation_id] += 1
            global_conformation_index += 1

        # Handle episode termination
        done_envs_ids = []
        for i, done in enumerate(dones):
            if not finished[i] and done:
                n_iters[i] = 0
                for early_stop_step, flags in early_stop_step_reached.items():
                    flags[i] = -1
                lbfgs_done_steps[i] = -1
                done_envs_ids.append(i)

                conformation_id = eval_env.atoms_ids[i]
                barrier[conformation_id] -= 1
                if barrier[conformation_id] == 0:
                    log_conformation_optimization_stats(stats[conformation_id], args.evaluation_metrics_path)
                    del stats[conformation_id]
                    del barrier[conformation_id]
                    pbar_pct.update(1)

        assert n_conf >= n_conf_processed_total

        done_future_ids = set()
        for future_id, future in futures.items():
            if not future.done():
                continue

            done_future_ids.add(future_id)
            conformation_id, step, energy, force = future.result()
            if energy is None:
                print(f'DFT did not converged: conformation_id={conformation_id} step={step}', flush=True)
            step_stats = stats[conformation_id].step2stats[step]
            step_stats.energy_ground_truth = energy
            step_stats.force_ground_truth = force
            barrier[conformation_id] -= 1
            if barrier[conformation_id] == 0:
                log_conformation_optimization_stats(stats[conformation_id], args.evaluation_metrics_path)
                del stats[conformation_id]
                del barrier[conformation_id]
                pbar_pct.update(1)

        for future_id in done_future_ids:
            del futures[future_id]

        if len(done_envs_ids) > 0:
            atoms_ids = [eval_env.atoms_ids[i] for i in done_envs_ids]
            atoms_list = [eval_env.atoms[i].copy() for i in done_envs_ids]
            smiles_list = [eval_env.smiles[i] for i in done_envs_ids]
            initial_energies = [eval_env.energy[i] for i in done_envs_ids]
            optimal_energies = [eval_env.optimal_energy[i] for i in done_envs_ids]
            write_to_db(args.results_db_path, atoms_ids, atoms_list, smiles_list, initial_energies, optimal_energies)

        n_conf_processed_delta = min(
            len(done_envs_ids), n_conf - n_conf_processed_total
        )
        n_conf_processed_total += n_conf_processed_delta
        envs_to_reset = done_envs_ids[: n_conf - n_conf_processed_total]
        if len(envs_to_reset) > 0:
            reset_states = eval_env.reset(indices=envs_to_reset)
            eval_policy.reset(reset_states, indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            atoms_ids = [eval_env.atoms_ids[i] for i in envs_to_reset]
            reserve_db_ids(args.results_db_path, atoms_ids)
            pbar_optimization.update(n_conf_processed_delta)

        for i in envs_to_reset:
            conformation_id = eval_env.atoms_ids[i]
            stats[conformation_id] = ConformationOptimizationStats(
                conformation_id=conformation_id,
                initial_energy_ground_truth=eval_env.energy[i][0],
                optimal_energy_ground_truth=eval_env.optimal_energy[i][0]
            )
            barrier[conformation_id] += 1

        for i in done_envs_ids[n_conf - n_conf_processed_total:]:
            finished[i] = True

    pbar_optimization.update(n_conf_processed_delta)
    pbar_optimization.close()

    for _, future in futures.items():
        conformation_id, step, energy, force = future.result()
        if energy is None:
            print(f'DFT did not converged: conformation_id={conformation_id} step={step}', flush=True)
        step_stats = stats[conformation_id].step2stats[step]
        step_stats.energy_ground_truth = energy
        step_stats.force_ground_truth = force
        barrier[conformation_id] -= 1
        if barrier[conformation_id] == 0:
            log_conformation_optimization_stats(stats[conformation_id], args.evaluation_metrics_path)
            del stats[conformation_id]
            del barrier[conformation_id]
            pbar_pct.update(1)

    pbar_pct.close()
    time_elapsed = time.perf_counter() - start_time
    for executor in executors:
        executor.shutdown(wait=False, cancel_futures=True)

    assert (
            n_conf_processed_total == n_conf
    ), f"Expected processed conformations: {n_conf}. Actual: {n_conf_processed_total}."

    print(
        f"Time elapsed: {datetime.timedelta(seconds=time_elapsed)}. OPS: {round(n_conf / time_elapsed, 3)}"
    )

    if "wandb" in sys.modules and os.environ.get("WANDB_API_KEY"):
        wandb.init(project=args.project, save_code=True, name=args.run_id)
        columns = list(stats.keys())
        table = wandb.Table(columns)
        for values in zip(*stats.values()):
            table.add_data(*values)

        wandb.log({"evaluation_metrics": table})
        wandb.finish()
    else:
        warnings.warn("Could not configure wandb access.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Batch evaluation args
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument(
        "--conf_number",
        default=int(1e5),
        type=int,
        help="Number of conformations to evaluate on",
    )
    parser.add_argument("--pct_max_threshold", type=float, default=2)
    parser.add_argument("--pct_min_threshold", type=float, default=-math.inf)
    parser.add_argument("--summary_log_interval", type=int, default=1000)
    parser.add_argument(
        "--eval_energy_convergence_thresholds",
        nargs="*",
        type=float,
        default=[],
        help="Check evaluation metrics on energy prediction convergence",
    )
    parser.add_argument(
        "--eval_early_stop_steps",
        nargs="*",
        type=int,
        default=[],
        help="Evaluate at multiple time steps during episode",
    )
    parser.add_argument(
        "--results_db",
        type=str,
        default="results.db",
        help="Path to database where results will be stored",
    )
    parser.add_argument(
        "--evaluation_metrics_file", type=str, default="evaluation_metrics.json"
    )
    parser.add_argument(
        "--resume",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Resume evaluation",
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Set deterministic mode for PyTorch",
    )

    # Env args
    parser.add_argument(
        "--eval_db_path",
        default="",
        type=str,
        help="Path to molecules database for evaluation",
    )
    parser.add_argument(
        "--n_parallel",
        default=1,
        type=int,
        help="Number of copies of env to run in parallel",
    )
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of parallel threads for DFT computations",
    )

    policy_arguments = [
        "actor",
        "conformation_optimizer",
        "conf_opt_lr",
        "conf_opt_lr_scheduler",
        "max_iter",
        "lbfgs_device",
        "momentum",
        "lion_beta1",
        "lion_beta2",
        "action_norm_limit",
        "grad_threshold"
    ]
    # AL args
    parser.add_argument(
        "--actor",
        type=str,
        choices=["AL", "rdkit"],
        help="Actor type. Rdkit can be used for evaluation only",
    )
    parser.add_argument(
        "--conformation_optimizer",
        type=str,
        choices=["GD", "Lion", "LBFGS"],
        help="Conformation optimizer type",
    )
    parser.add_argument(
        "--conf_opt_lr",
        type=float,
        help="Initial learning rate for conformation optimizer.",
    )
    parser.add_argument(
        "--conf_opt_lr_scheduler",
        choices=["Constant", "CosineAnnealing"],
        help="Conformation optimizer learning rate scheduler type",
    )
    parser.add_argument(
        "--action_norm_limit",
        type=float,
        help="Upper limit for action norm. Action norms larger get scaled down",
    )

    # LBFGS args
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Number of iterations in the inner cycle LBFGS",
    )
    parser.add_argument(
        "--lbfgs_device",
        type=str,
        choices=["cuda", "cpu"],
        help="LBFGS device type",
    )
    parser.add_argument(
        "--grad_threshold",
        type=float,
        help="Terminates optimization when norm of the gradient is smaller than the threshold",
    )

    # GD args
    parser.add_argument(
        "--momentum",
        type=float,
        help="Momentum argument for gradient descent confromation optimizer",
    )

    # Lion args
    parser.add_argument(
        "--lion_beta1",
        type=float,
        help="Beta_1 for Lion conformation optimizer",
    )
    parser.add_argument(
        "--lion_beta2",
        type=float,
        help="Beta_2 for Lion conformation optimizer",
    )

    # Timelimit args
    parser.add_argument(
        "--timelimit", default=100, type=int, help="Timelimit for MD env"
    )
    parser.add_argument(
        "--terminate_on_negative_reward",
        default=True,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Terminate the episode when enough negative rewards are encountered",
    )
    parser.add_argument(
        "--max_num_negative_rewards",
        default=1,
        type=int,
        help="Max number of negative rewards to terminate the episode",
    )

    # Reward args
    parser.add_argument(
        "--reward",
        choices=["rdkit", "dft"],
        default="rdkit",
        help="How the energy is calculated",
    )
    parser.add_argument(
        "--minimize_on_every_step",
        default=True,
        choices=[True, False],
        metavar="True|False",
        type=str2bool,
        help="Whether to minimize conformation with rdkit on every step",
    )

    # Other args
    parser.add_argument("--project", type=str, help="Project name in wandb")
    parser.add_argument("--run_id", type=str, help="Run name in wandb project")
    parser.add_argument("--port_type", choices=['train', 'eval'], default='eval', type=str, help="DFT server port")
    parser.add_argument("--logging_tcp_client", default=False, choices=[True, False], metavar="True|False",
                        type=str2bool, help="Whether to log tcp communication events")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)

    evaluation_config_file = checkpoint_path / "evaluation_config.json"
    with open(evaluation_config_file, "w") as file_obj:
        json.dump(dict(args.__dict__), file_obj, indent=4)

    config_path = checkpoint_path / "config.json"
    # Read config and turn it into a class object with properties
    with open(config_path, "rb") as f:
        config = json.load(f)

    args.results_db_path = checkpoint_path / args.results_db
    args.evaluation_metrics_path = checkpoint_path / args.evaluation_metrics_file
    if not args.resume:
        assert (
            not args.results_db_path.exists()
        ), f"Database file {args.results_db_path} exists!"
        assert (
            not args.evaluation_metrics_path.exists()
        ), f"Evaluation metrics file {args.evaluation_metrics_path} exists!"

    config["db_path"] = "/".join(args.eval_db_path.split("/")[-3:])
    config["eval_db_path"] = "/".join(args.eval_db_path.split("/")[-3:])
    config["molecules_xyz_prefix"] = "env/molecules_xyz"
    config["n_parallel"] = args.n_parallel
    config["timelimit"] = args.timelimit + 1
    config["n_threads"] = args.n_threads
    config["terminate_on_negative_reward"] = args.terminate_on_negative_reward
    config["max_num_negative_rewards"] = args.max_num_negative_rewards
    config["reward"] = args.reward
    config["minimize_on_every_step"] = args.minimize_on_every_step
    config["sample_initial_conformations"] = False
    config["num_initial_conformations"] = -1
    config["deterministic"] = args.deterministic
    for argument in policy_arguments:
        value = getattr(args, argument, None)
        if value is not None:
            config[argument] = value

    config = Config(**config)
    if len(args.eval_early_stop_steps) == 0:
        args.eval_early_stop_steps = list(range(args.timelimit + 1))

    main(checkpoint_path, args, config)
