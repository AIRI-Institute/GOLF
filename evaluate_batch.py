import argparse
import collections
import datetime
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch

from pathlib import Path

from ase import Atoms
from ase.db import connect
from schnetpack import properties

from AL.make_policies import make_policies

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm

from AL.utils import recollate_batch, get_atoms_indices_range
from AL import DEVICE
from AL.eval import rdkit_minimize_until_convergence
from env.moldynamics_env import env_fn
from env.wrappers import RewardWrapper
from utils.arguments import str2bool


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def reserve_db_ids(db_path, db_ids):
    with connect(db_path) as conn:
        for db_id in db_ids:
            n = conn.count(selection=db_id)
            if n == 1:
                conn.update(id=db_id, data={"reserve_id": db_id})
            elif n == 0:
                with conn.managed_connection() as sqlite_conn:
                    cur = sqlite_conn.cursor()
                    cur.execute(f'INSERT INTO systems (id) VALUES ({db_id})')

                conn._write(atoms=Atoms(), key_value_pairs={}, data={"reserve_id": db_id}, id=db_id)
            else:
                assert False, f'{n} conformations with the same db_id={db_id}!'


def write_to_db(db_path, db_ids, atoms_list, smiles_list):
    assert len(db_ids) == len(atoms_list)
    with connect(db_path) as conn:
        for db_id, atoms, smiles in zip(db_ids, atoms_list, smiles_list):
            conn.update(id=db_id, atoms=atoms, delete_keys=["reserve_id"], smiles=smiles)


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
        lines.append(f"| {name}{name_space} | {mean}{mean_space} +/- {std}{std_space} |")
    lines.append(dashes)
    print("\n", "\n".join(lines), sep="", flush=True)


def read_metrics(path):
    metrics = []
    with open(path, 'r') as file_obj:
        for line in file_obj:
            line = line.strip()
            if line == '':
                continue

            metrics.append(json.loads(line))

    return metrics


def reconcile_results_db_and_evaluation_metrics(results_db_path, evaluation_metrics_path):
    metrics = sorted(read_metrics(evaluation_metrics_path), key=lambda metric: metric['conformation_id'])
    metrics_ids = []
    for metric in metrics:
        if len(metrics_ids) == 0 or metric['conformation_id'] == metrics_ids[-1] + 1:
            metrics_ids.append(metric['conformation_id'])
        else:
            break

    all_db_ids = []
    with connect(results_db_path) as conn:
        with conn.managed_connection() as sqlite_conn:
            cur = sqlite_conn.cursor()
            cur.execute('select id from systems order by id asc')
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
        metrics_ids = metrics_ids[:len(db_ids)]

    with connect(results_db_path) as conn:
        for db_id in db_ids_to_delete:
            del conn[db_id]

    metrics_ids = set(metrics_ids)
    with open(evaluation_metrics_path, 'w') as file_obj:
        for metric in metrics:
            if metric['conformation_id'] in metrics_ids:
                json.dump(metric, file_obj)
                file_obj.write('\r\n')


def make_envs(args):
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
        "molecules_xyz_prefix": args.molecules_xyz_prefix,
        "terminate_on_negative_reward": args.terminate_on_negative_reward,
        "max_num_negative_rewards": args.max_num_negative_rewards,
    }

    eval_env = env_fn(**env_kwargs)
    eval_env = RewardWrapper(eval_env, **reward_wrapper_kwargs)

    env_kwargs["n_parallel"] = 1
    env_kwargs["db_path"] = args.initial_db_path
    initial_env = env_fn(**env_kwargs)
    initial_env = RewardWrapper(initial_env, **reward_wrapper_kwargs)

    return eval_env, initial_env


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


def main(checkpoint_path, args, config):
    eval_env, initial_env = make_envs(config)
    eval_policy = make_policies(eval_env, initial_env, config)[0]

    if config.actor != 'rdkit':
        agent_path = checkpoint_path / args.agent_path
        eval_policy.actor.load_state_dict(torch.load(agent_path, map_location=torch.device(DEVICE)))
        eval_policy.actor.to(DEVICE)
        eval_policy.actor.eval()

    state = eval_env.reset()
    if args.resume:
        metrics_ids = set(metric['conformation_id'] for metric in read_metrics(args.evaluation_metrics_path))
        while set(eval_env.atoms_ids).issubset(metrics_ids):
            state = eval_env.reset()

    reserve_db_ids(args.results_db_path, eval_env.atoms_ids.copy())
    eval_policy.reset(state)

    finished = np.zeros(shape=eval_env.n_parallel, dtype=bool)
    molecules = []
    smiles = []
    for db_id in eval_env.atoms_ids:
        row = initial_env.get_molecule(db_id)
        molecules.append(row.toatoms().copy())
        if hasattr(row, 'smiles'):
            smiles.append(row.smiles)

    returns = np.zeros(shape=eval_env.n_parallel)
    negative_reward_steps = np.zeros(shape=eval_env.n_parallel)
    negative_reward_n_iters = np.zeros(shape=eval_env.n_parallel)
    negative_reward_energies = np.zeros(shape=eval_env.n_parallel)
    lbfgs_done_steps = np.full_like(negative_reward_steps, fill_value=-1)
    lbfgs_done_energies = np.zeros(shape=eval_env.n_parallel)
    previous_energies = np.full(shape=eval_env.n_parallel, fill_value=np.inf)

    convergence_thresholds = list(sorted(args.eval_energy_convergence_thresholds))[::-1]
    convergence_info = {
        thresh: {
            key: np.zeros(shape=eval_env.n_parallel)
            for key in ("convergence_energy", "convergence_step", "convergence_flag", "convergence_n_iter")
        }
        for thresh in convergence_thresholds
    }

    early_stop_steps = list(sorted(args.eval_early_stop_steps))
    early_stop_info = {
        early_stop_step: {
            key: np.zeros(shape=eval_env.n_parallel)
            for key in ("early_stop_flag", "early_stop_energy", "energy_mse", "force_mse", "early_stop_n_iter")
        }
        for early_stop_step in early_stop_steps}

    stats = collections.defaultdict(list)
    n_conf_processed_delta = None
    n_conf_processed_total = 0
    n_conf = args.conf_number if args.conf_number > 0 else eval_env.db_len
    assert n_conf >= args.n_parallel

    next_log_summary = args.summary_log_interval
    start_time = time.perf_counter()
    pbar = tqdm(total=n_conf, mininterval=10)
    n_iters = np.zeros(shape=eval_env.n_parallel)
    not_finite_action_steps = np.full(shape=eval_env.n_parallel, fill_value=-1)
    not_finite_action_energies = np.zeros(shape=eval_env.n_parallel)
    while not np.all(finished).item():
        # Get current timesteps
        episode_timesteps = eval_env.get_env_step()
        steps = np.asarray(episode_timesteps, dtype=np.float32)

        # Select next action
        select_action_result = eval_policy.select_action(episode_timesteps)
        actions = select_action_result["action"]
        energies = select_action_result["energy"]
        lbfgs_dones = select_action_result["done"]
        is_finite_action = select_action_result["is_finite_action"]
        n_iters_dones = select_action_result["n_iter"]
        forces = np.asarray(np.vsplit(
            select_action_result["anti_gradient"], np.cumsum(state[properties.n_atoms].cpu().numpy())[:-1]
        ), dtype=object)

        energies_ground_truth = np.full_like(previous_energies, fill_value=np.inf)
        forces_ground_truth = np.empty(previous_energies.shape[0], dtype=object)

        # Handle non-finite actions
        non_finite_action_mask = ~finished & (~is_finite_action)
        not_finite_action_steps[non_finite_action_mask] = steps[non_finite_action_mask]
        for i in np.where(np.isinf(energies_ground_truth) & non_finite_action_mask)[0]:
            _, energies_ground_truth[i], _ = eval_env.minimize_rdkit(i)
        not_finite_action_energies[non_finite_action_mask] = energies_ground_truth[non_finite_action_mask]

        # Handle convergence of energy predictions
        energies_delta = np.abs(energies - previous_energies)
        for threshold in convergence_thresholds:
            if np.all(energies_delta >= threshold).item():
                break

            convergence_mask = ~finished & (steps > 1) & (energies_delta < threshold) & (
                    convergence_info[threshold]["convergence_step"] == 0)

            for i in np.where(np.isinf(energies_ground_truth) & convergence_mask)[0]:
                _, energies_ground_truth[i], _ = eval_env.minimize_rdkit(i)

            convergence_info[threshold]["convergence_step"][convergence_mask] = steps[convergence_mask]
            convergence_info[threshold]["convergence_n_iter"][convergence_mask] = n_iters[convergence_mask]
            convergence_info[threshold]["convergence_energy"][convergence_mask] = energies_ground_truth[
                convergence_mask]
            convergence_info[threshold]["convergence_flag"][convergence_mask] = 1

        previous_energies = energies

        # Handle different time limits
        for early_stop_step in early_stop_steps:
            if np.all(steps < early_stop_step).item():
                break

            early_stop_step_mask = ~finished & (steps >= early_stop_step) & (
                    early_stop_info[early_stop_step]["early_stop_flag"] == 0)

            for i in np.where((np.isinf(energies_ground_truth) | (forces_ground_truth == None)) & early_stop_step_mask)[0]:
                _, energies_ground_truth[i], forces_ground_truth[i] = eval_env.minimize_rdkit(i)

            early_stop_info[early_stop_step]["early_stop_energy"][early_stop_step_mask] = energies_ground_truth[
                early_stop_step_mask]
            early_stop_info[early_stop_step]["early_stop_n_iter"][early_stop_step_mask] = n_iters[early_stop_step_mask]
            early_stop_info[early_stop_step]["early_stop_flag"][early_stop_step_mask] = 1
            early_stop_info[early_stop_step]["energy_mse"][early_stop_step_mask] = \
                (energies - energies_ground_truth)[early_stop_step_mask] ** 2

            for i in np.where(early_stop_step_mask)[0]:
                if non_finite_action_mask[i]:
                    early_stop_info[early_stop_step]["force_mse"][i] = -1
                    continue
                early_stop_info[early_stop_step]["force_mse"][i] = np.mean((forces[i] - forces_ground_truth[i]) ** 2)

        actions *= get_not_finished_mask(state, ~is_finite_action | finished)
        state, rewards, dones, info = eval_env.step(actions)
        dones = ~is_finite_action | np.asarray(dones)
        returns += rewards
        n_iters += np.asarray(n_iters_dones)
        steps = np.asarray(eval_env.get_env_step(), dtype=np.float32)

        # Handle lbfgs dones
        lbfgs_dones_mask = ~finished & lbfgs_dones & (lbfgs_done_steps == -1)
        lbfgs_done_steps[lbfgs_dones_mask] = steps[lbfgs_dones_mask]
        for i in np.where(lbfgs_dones_mask)[0]:
            _, energies_ground_truth[i], _ = eval_env.minimize_rdkit(i)
        lbfgs_done_energies[lbfgs_dones_mask] = energies_ground_truth[lbfgs_dones_mask]

        # Handle negative rewards
        if eval_env.minimize_on_every_step:
            energies_ground_truth = np.asarray(info["final_energy"])
            first_negative_reward_mask = ~finished & (rewards < 0) & (negative_reward_steps == 0)
            negative_reward_steps[first_negative_reward_mask] = steps[first_negative_reward_mask]
            negative_reward_n_iters[first_negative_reward_mask] = n_iters[first_negative_reward_mask]
            negative_reward_energies[first_negative_reward_mask] = energies_ground_truth[first_negative_reward_mask]

        # Handle episode termination
        done_envs_ids = []
        for i, done in enumerate(dones):
            with open(args.evaluation_metrics_path, 'a') as file_obj:
                if not finished[i] and done:
                    stats['conformation_id'].append(eval_env.atoms_ids[i])

                    stats['episode_length'].append(int(steps[i]))
                    stats['episode_length_n_iters'].append(int(n_iters[i]))

                    initial_energy, final_energy_rdkit = rdkit_minimize_until_convergence(initial_env, [molecules[i]],
                                                                                          [smiles[i]], max_its=0)
                    stats['rdkit_initial_energy'].append(float(initial_energy))
                    stats['rdkit_final_energy'].append(float(final_energy_rdkit))

                    pct = (initial_energy - info['final_energy'][i]) / (initial_energy - final_energy_rdkit)
                    stats['pct_of_minimized_energy'].append(pct)

                    negative_reward_step = negative_reward_steps[i]
                    negative_reward_n_iter = negative_reward_n_iters[i]
                    negative_reward_energy = negative_reward_energies[i]
                    # Didn't get a negative reward
                    if negative_reward_step == 0:
                        negative_reward_step = steps[i]
                        negative_reward_n_iter = n_iters[i]
                        negative_reward_energy = info['final_energy'][i]

                    stats['negative_reward_step'].append(int(negative_reward_step))
                    stats['negative_reward_n_iter'].append(int(negative_reward_n_iter))
                    stats['pct_of_minimized_energy_negative_reward'].append(
                        (initial_energy - negative_reward_energy) / (initial_energy - final_energy_rdkit)
                    )
                    negative_reward_steps[i] = 0
                    negative_reward_n_iters[i] = 0
                    negative_reward_energies[i] = 0

                    lbfgs_done_step = lbfgs_done_steps[i]
                    lbfgs_done_energy = lbfgs_done_energies[i]
                    if lbfgs_done_step == -1:
                        lbfgs_done_step = steps[i]
                        lbfgs_done_energy = info['final_energy'][i]

                    stats['lbfgs_done_step'].append(int(lbfgs_done_step))
                    stats['pct_of_minized_energy_lbfgs_done'].append(
                        (initial_energy - lbfgs_done_energy) / (initial_energy - final_energy_rdkit)
                    )
                    lbfgs_done_steps[i] = -1
                    lbfgs_done_energies[i] = 0

                    for threshold in convergence_thresholds:
                        convergence_step = convergence_info[threshold]['convergence_step'][i]
                        convergence_n_iter = convergence_info[threshold]['convergence_n_iter'][i]
                        convergence_energy = convergence_info[threshold]['convergence_energy'][i]
                        convergence_flag = convergence_info[threshold]['convergence_flag'][i]
                        # Didn't converge during the episode
                        if convergence_step == 0:
                            convergence_step = steps[i]
                            convergence_n_iter = n_iters[i]
                            convergence_energy = info['final_energy'][i]

                        stats[f'convergence_step@thresh:{threshold}'].append(int(convergence_step))
                        stats[f'convergence_n_iter@thresh:{threshold}'].append(int(convergence_n_iter))
                        stats[f'converged@thresh:{threshold}'].append(int(convergence_flag))
                        stats[f'pct_of_minimized_energy@thresh:{threshold}'].append(
                            (initial_energy - convergence_energy) / (initial_energy - final_energy_rdkit)
                        )
                        convergence_info[threshold]['convergence_step'][i] = 0
                        convergence_info[threshold]['convergence_n_iter'][i] = 0
                        convergence_info[threshold]['convergence_energy'][i] = 0
                        convergence_info[threshold]['convergence_flag'][i] = 0

                    previous_energies[i] = np.inf

                    for early_stop_step in early_stop_steps:
                        early_stop_energy = early_stop_info[early_stop_step]['early_stop_energy'][i]
                        early_stop_flag = early_stop_info[early_stop_step]['early_stop_flag'][i]
                        early_stop_n_iter = early_stop_info[early_stop_step]['early_stop_n_iter'][i]
                        energy_mse = early_stop_info[early_stop_step]['energy_mse'][i]
                        force_mse = early_stop_info[early_stop_step]['force_mse'][i]
                        # Didn't reach the early stop step
                        if early_stop_flag == 0:
                            early_stop_energy = info['final_energy'][i]
                            early_stop_n_iter = n_iters[i]
                            energy_mse = -1
                            force_mse = -1

                        stats[f'reached@step:{early_stop_step}'].append(int(early_stop_flag))
                        stats[f'n_iter@step:{early_stop_step}'].append(int(early_stop_n_iter))
                        stats[f'energy_mse@step:{early_stop_step}'].append(float(energy_mse))
                        stats[f'force_mse@step:{early_stop_step}'].append(float(force_mse))
                        stats[f'rdkit_energy@step:{early_stop_step}'].append(float(early_stop_energy))
                        stats[f'pct_of_minimized_energy@step:{early_stop_step}'].append(
                            (initial_energy - early_stop_energy) / (initial_energy - final_energy_rdkit)
                        )
                        early_stop_info[early_stop_step]['early_stop_energy'][i] = 0
                        early_stop_info[early_stop_step]['early_stop_n_iter'][i] = 0
                        early_stop_info[early_stop_step]['early_stop_flag'][i] = 0
                        early_stop_info[early_stop_step]['energy_mse'][i] = 0
                        early_stop_info[early_stop_step]['force_mse'][i] = 0

                    stats['not_finite_action_step'].append(int(not_finite_action_steps[i]))
                    not_finite_action_steps[i] = -1
                    n_iters[i] = 0

                    last_conformation_stats = {key: values[-1] for key, values in stats.items()}
                    json.dump(last_conformation_stats, file_obj)
                    file_obj.write('\r\n')

                    done_envs_ids.append(i)

        assert n_conf >= n_conf_processed_total

        if len(done_envs_ids) > 0:
            atoms_ids = [eval_env.atoms_ids[i] for i in done_envs_ids]
            atoms_list = [eval_env.atoms[i].copy() for i in done_envs_ids]
            smiles_list = [eval_env.smiles[i] for i in done_envs_ids]
            write_to_db(args.results_db_path, atoms_ids, atoms_list, smiles_list)

        n_conf_processed_delta = min(len(done_envs_ids), n_conf - n_conf_processed_total)
        n_conf_processed_total += n_conf_processed_delta
        envs_to_reset = done_envs_ids[:n_conf - n_conf_processed_total]
        if len(envs_to_reset) > 0:
            reset_states = eval_env.reset(indices=envs_to_reset)
            eval_policy.reset(reset_states, indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            atoms_ids = [eval_env.atoms_ids[i] for i in envs_to_reset]
            reserve_db_ids(args.results_db_path, atoms_ids)
            pbar.update(n_conf_processed_delta)

        for i in envs_to_reset:
            row = initial_env.get_molecule(eval_env.atoms_ids[i])
            molecules[i] = row.toatoms().copy()
            if hasattr(row, 'smiles'):
                smiles[i] = row.smiles

        for i in done_envs_ids[n_conf - n_conf_processed_total:]:
            finished[i] = True

        if n_conf_processed_total >= next_log_summary:
            print_stats_table(n_conf_processed_total, aggregate2string(stats, args.pct_min_threshold, args.pct_max_threshold))
            next_log_summary += args.summary_log_interval

    pbar.update(n_conf_processed_delta)
    pbar.close()
    time_elapsed = time.perf_counter() - start_time

    assert n_conf_processed_total == n_conf, f'Expected processed conformations: {n_conf}. Actual: {n_conf_processed_total}.'

    print(f'Time elapsed: {datetime.timedelta(seconds=time_elapsed)}. OPS: {round(n_conf / time_elapsed, 3)}')
    print_stats_table(n_conf_processed_total, aggregate2string(stats, args.pct_min_threshold, args.pct_max_threshold))

    if 'wandb' in sys.modules and os.environ.get('WANDB_API_KEY'):
        wandb.init(project=args.project, save_code=True, name=args.run_id)
        columns = list(stats.keys())
        table = wandb.Table(columns)
        for values in zip(*stats.values()):
            table.add_data(*values)

        wandb.log({'evaluation_metrics': table})
        wandb.finish()
    else:
        warnings.warn('Could not configure wandb access.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Batch evaluation args
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--pct_max_threshold", type=float, default=2)
    parser.add_argument("--pct_min_threshold", type=float, default=-math.inf)
    parser.add_argument("--summary_log_interval", type=int, default=1000)
    parser.add_argument(
        "--eval_energy_convergence_thresholds",
        nargs='*',
        type=float,
        default=[],
        help="Check evaluation metrics on energy prediction convergence"
    )
    parser.add_argument(
        "--eval_early_stop_steps",
        nargs='*',
        type=int,
        default=[],
        help="Evaluate at multiple time steps during episode"
    )
    parser.add_argument("--results_db", type=str, default='results.db', help="Path to database where results will be stored")
    parser.add_argument("--evaluation_metrics_file", type=str, default='evaluation_metrics.json')
    parser.add_argument(
        "--resume",
        default=False,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Resume evaluation")

    # Env args
    parser.add_argument('--initial_db_path',  default='', type=str, help="Path to database with initial conformations")
    parser.add_argument(
        "--eval_db_path", default="", type=str, help="Path to molecules database for evaluation"
    )
    parser.add_argument(
        "--n_parallel",
        default=1,
        type=int,
        help="Number of copies of env to run in parallel")
    parser.add_argument(
        "--n_threads",
        default=1,
        type=int,
        help="Number of parallel threads for DFT computations")

    policy_arguments = [
        "actor", "conformation_optimizer", "conf_opt_lr", "conf_opt_lr_scheduler", "max_iter", "lbfgs_device",
        "momentum", "lion_beta1", "lion_beta2"
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
        "--timelimit",
        default=100,
        type=int,
        help="Timelimit for MD env")
    parser.add_argument(
        "--terminate_on_negative_reward",
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Terminate the episode when enough negative rewards are encountered")
    parser.add_argument(
        "--max_num_negative_rewards",
        default=1,
        type=int,
        help="Max number of negative rewards to terminate the episode")

    # Reward args
    parser.add_argument(
        "--reward",
        choices=["rdkit", "dft"],
        default="rdkit",
        help="How the energy is calculated")
    parser.add_argument(
        "--minimize_on_every_step",
        default=True,
        choices=[True, False],
        metavar='True|False',
        type=str2bool,
        help="Whether to minimize conformation with rdkit on every step")

    # Other args
    parser.add_argument("--project", type=str, help="Project name in wandb")
    parser.add_argument("--run_id", type=str, help="Run name in wandb project")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)

    evaluation_config_file = checkpoint_path / "evaluation_config.json"
    with open(evaluation_config_file, 'w') as file_obj:
        json.dump(dict(args.__dict__), file_obj, indent=4)

    config_path = checkpoint_path / "config.json"
    # Read config and turn it into a class object with properties
    with open(config_path, "rb") as f:
        config = json.load(f)

    args.results_db_path = checkpoint_path / args.results_db
    args.evaluation_metrics_path = checkpoint_path / args.evaluation_metrics_file
    if not args.resume:
        assert not args.results_db_path.exists(), f'Database file {args.results_db_path} exists!'
        assert not args.evaluation_metrics_path.exists(), f'Evaluation metrics file {args.evaluation_metrics_path} exists!'

    if args.initial_db_path is None or args.initial_db_path == '':
        args.initial_db_path = args.eval_db_path

    config['db_path'] = '/'.join(args.eval_db_path.split('/')[-3:])
    config['eval_db_path'] = '/'.join(args.eval_db_path.split('/')[-3:])
    config['initial_db_path'] = '/'.join(args.initial_db_path.split('/')[-3:])
    config['molecules_xyz_prefix'] = "env/molecules_xyz"
    config['n_parallel'] = args.n_parallel
    config['timelimit'] = args.timelimit + 1
    config['n_threads'] = args.n_threads
    config['terminate_on_negative_reward'] = args.terminate_on_negative_reward
    config['max_num_negative_rewards'] = args.max_num_negative_rewards
    config['reward'] = args.reward
    config['minimize_on_every_step'] = args.minimize_on_every_step
    config['sample_initial_conformations'] = False
    config['num_initial_conformations'] = -1
    for argument in policy_arguments:
        value = getattr(args, argument, None)
        if value is not None:
            config[argument] = value

    config = Config(**config)
    if len(args.eval_early_stop_steps) == 0:
        args.eval_early_stop_steps = list(range(args.timelimit + 1))

    main(checkpoint_path, args, config)
