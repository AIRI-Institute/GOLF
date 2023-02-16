import argparse
import collections
import copy
import datetime
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import schnetpack
import torch

from pathlib import Path

from schnetpack import properties

try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm

from AL.utils import recollate_batch, get_cutoff_by_string, get_atoms_indices_range
from AL import DEVICE
from AL.AL_actor import Actor
from AL.eval import rdkit_minimize_until_convergence
from env.moldynamics_env import env_fn
from env.wrappers import RewardWrapper
from utils.arguments import str2bool


class Config():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def stats_mean_std_strings(values, min_threshold, max_threshold, digits=3):
    values = np.asarray(values)
    mask = (values >= min_threshold) & (values <= max_threshold)
    values = values[mask]
    mean_string = f'{round(values.mean(), digits)}'
    std_string = f'{round(values.std(ddof=0), digits)}'
    return mean_string, std_string


def print_stats_table(stats2strings):
    name_width = max(map(len, stats2strings.keys()))
    mean_width = max(map(len, (value[0] for value in stats2strings.values())))
    std_width = max(map(len, (value[1] for value in stats2strings.values())))
    dashes = "-" * (name_width + mean_width + std_width + 12)
    lines = [dashes]
    for name, (mean, std) in stats2strings.items():
        name_space = " " * (name_width - len(name))
        mean_space = " " * (mean_width - len(mean))
        std_space = " " * (std_width - len(std))
        lines.append(f"| {name}{name_space} | {mean}{mean_space} +/- {std}{std_space} |")
    lines.append(dashes)
    print('\n'.join(lines))


def make_envs(args):
    # Env kwargs
    env_kwargs = {
        'db_path': args.db_path,
        'n_parallel': args.n_parallel,
        'timelimit': args.timelimit,
        'sample_initial_conformations': args.sample_initial_conformations,
        'num_initial_conformations': args.num_initial_conformations,
    }

    # Reward wrapper kwargs
    reward_wrapper_kwargs = {
        'dft': args.reward == 'dft',
        'n_threads': args.n_threads,
        'minimize_on_every_step': args.minimize_on_every_step,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M,
        'terminate_on_negative_reward': args.terminate_on_negative_reward,
        'max_num_negative_rewards': args.max_num_negative_rewards
    }

    eval_env = env_fn(**env_kwargs)
    eval_env = RewardWrapper(eval_env, **reward_wrapper_kwargs)

    env_kwargs['n_parallel'] = 1
    auxiliary_env = env_fn(**env_kwargs)
    auxiliary_env = RewardWrapper(auxiliary_env, **reward_wrapper_kwargs)

    return eval_env, auxiliary_env


def get_not_finished_mask(state, finished):
    n_molecules = state[properties.n_atoms].size(0)
    n_atoms = get_atoms_indices_range(state).cpu().numpy()
    not_finished_mask = np.ones(shape=(state[properties.position].size(0),), dtype=np.float32)
    for i in range(n_molecules):
        if finished[i]:
            not_finished_mask[n_atoms[i]:n_atoms[i + 1]] = 0

    return np.expand_dims(not_finished_mask, axis=1)


def main(checkpoint_path, args, config):
    backbone_args = {
        'n_interactions': config.n_interactions,
        'n_atom_basis': config.n_atom_basis,
        'radial_basis': schnetpack.nn.BesselRBF(n_rbf=config.n_rbf, cutoff=config.cutoff),
        'cutoff_fn': get_cutoff_by_string('cosine')(config.cutoff),
    }

    actor = Actor(
        backbone=config.backbone,
        backbone_args=backbone_args,
        action_scale=config.action_scale,
        action_norm_limit=config.action_norm_limit,
    )
    agent_path = checkpoint_path / args.agent_path
    actor.load_state_dict(torch.load(agent_path, map_location=torch.device(DEVICE)))
    actor.to(DEVICE)
    actor.eval()

    eval_env, auxiliary_env = make_envs(config)
    state = eval_env.reset()

    convergence_thresholds = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    config.convergence_thresholds = convergence_thresholds.copy()

    finished = np.zeros(shape=eval_env.n_parallel, dtype=bool)
    molecules = copy.deepcopy(eval_env.unwrapped.atoms)
    smiles = copy.deepcopy(eval_env.unwrapped.smiles)
    returns = np.zeros(shape=eval_env.n_parallel)
    negative_reward_steps = np.zeros(shape=eval_env.n_parallel)
    negative_reward_energies = np.zeros(shape=eval_env.n_parallel)
    previous_energies = np.ones(shape=eval_env.n_parallel) * np.inf
    convergence_info = {
        thresh: {key: np.zeros(shape=eval_env.n_parallel)
                 for key in ('convergence_energy', 'convergence_step', 'convergence_flag')
                 } for thresh in convergence_thresholds
    }

    stats = collections.defaultdict(list)
    n_conf_processed_delta = None
    n_conf_processed_total = 0
    n_conf = args.conf_number if args.conf_number > 0 else eval_env.db_len
    assert n_conf >= args.n_parallel

    start_time = time.perf_counter()
    pbar = tqdm(total=n_conf, mininterval=10)
    while not np.all(finished).item():
        actions, energies = actor.select_action(state)
        actions *= get_not_finished_mask(state, finished)
        state, rewards, dones, info = eval_env.step(actions)
        returns += rewards
        energies = energies.squeeze()

        steps = np.asarray(eval_env.get_env_step())

        energies_ground_truth = None
        if eval_env.minimize_on_every_step:
            energies_ground_truth = np.asarray(info['final_energy'])
            first_negative_reward_mask = ~finished & (rewards < 0) & (negative_reward_steps == 0)
            negative_reward_steps[first_negative_reward_mask] = steps[first_negative_reward_mask]
            negative_reward_energies[first_negative_reward_mask] = energies_ground_truth[first_negative_reward_mask]

        energies_delta = np.abs(energies - previous_energies)

        for threshold in convergence_thresholds:
            if np.all(energies_delta >= threshold).item():
                break

            mask = ~finished & (steps > 1) & (energies_delta < threshold) & (
                        convergence_info[threshold]['convergence_step'] == 0)

            converged_indices, = np.where(mask)
            if converged_indices.size > 0:
                if energies_ground_truth is None:
                    energies_ground_truth = np.zeros(shape=eval_env.n_parallel, dtype=np.float32)
                    for i in converged_indices:
                        _, energies_ground_truth[i], _ = eval_env.minimize_rdkit(i)

                convergence_info[threshold]['convergence_step'][mask] = steps[mask]
                convergence_info[threshold]['convergence_energy'][mask] = energies_ground_truth[mask]
                convergence_info[threshold]['convergence_flag'][mask] = 1

        previous_energies = energies

        done_envs_ids = []
        for i, done in enumerate(dones):
            if not finished[i] and done:
                stats['delta_energy'].append(returns[i])
                returns[i] = 0

                stats['episode_length'].append(int(steps[i]))
                stats['final_energy'].append(info['final_energy'][i])
                stats['final_rl_energy'].append(info['final_rl_energy'][i])

                initial_energy, final_energy_rdkit = rdkit_minimize_until_convergence(auxiliary_env, [molecules[i]],
                                                                                      [smiles[i]], M=0)
                pct = (initial_energy - stats['final_energy'][-1]) / (initial_energy - final_energy_rdkit)
                stats['pct_of_minimized_energy'].append(pct)

                negative_reward_step = negative_reward_steps[i]
                negative_reward_energy = negative_reward_energies[i]
                if negative_reward_step == 0:
                    negative_reward_step = steps[i]
                    negative_reward_energy = info['final_energy'][i]

                stats['negative_reward_step'].append(int(negative_reward_step))
                stats['pct_of_minimized_energy_negative_reward'].append(
                    (initial_energy - negative_reward_energy) / (initial_energy - final_energy_rdkit)
                )
                negative_reward_steps[i] = 0
                negative_reward_energies[i] = 0

                for threshold in convergence_thresholds:
                    convergence_step = convergence_info[threshold]['convergence_step'][i]
                    convergence_energy = convergence_info[threshold]['convergence_energy'][i]
                    convergence_flag = convergence_info[threshold]['convergence_flag'][i]
                    if convergence_step == 0:
                        convergence_step = steps[i]
                        convergence_energy = info['final_energy'][i]

                    stats[f'convergence_step@thresh:{threshold}'].append(int(convergence_step))
                    stats[f'converged@thresh:{threshold}'].append(int(convergence_flag))
                    stats[f'pct_of_minimized_energy@thresh:{threshold}'].append(
                        (initial_energy - convergence_energy) / (initial_energy - final_energy_rdkit)
                    )
                    convergence_info[threshold]['convergence_step'][i] = 0
                    convergence_info[threshold]['convergence_energy'][i] = 0
                    convergence_info[threshold]['convergence_flag'][i] = 0

                previous_energies[i] = np.inf

                done_envs_ids.append(i)

        assert n_conf >= n_conf_processed_total

        n_conf_processed_delta = min(len(done_envs_ids), n_conf - n_conf_processed_total)
        n_conf_processed_total += n_conf_processed_delta
        envs_to_reset = done_envs_ids[:n_conf - n_conf_processed_total]
        if len(envs_to_reset) > 0:
            reset_states = eval_env.reset(indices=envs_to_reset)
            state = recollate_batch(state, envs_to_reset, reset_states)
            pbar.update(n_conf_processed_delta)

        for i in envs_to_reset:
            molecules[i] = copy.deepcopy(eval_env.atoms[i])
            smiles[i] = copy.deepcopy(eval_env.smiles[i])

        for i in done_envs_ids[n_conf - n_conf_processed_total:]:
            finished[i] = True

    pbar.update(n_conf_processed_delta)
    pbar.close()
    time_elapsed = time.perf_counter() - start_time

    assert n_conf_processed_total == n_conf, f'Expected processed conformations: {n_conf}. Actual: {n_conf_processed_total}.'

    stats_mean_std = {}
    for key, value in stats.items():
        min_threshold = -math.inf
        max_threshold = math.inf
        if 'pct' in key:
            min_threshold = args.pct_min_threshold
            max_threshold = args.pct_max_threshold

        stats_mean_std[key] = stats_mean_std_strings(value, min_threshold, max_threshold)

    print(f'Time elapsed: {datetime.timedelta(seconds=time_elapsed)}. OPS: {round(n_conf / time_elapsed, 3)}')
    print_stats_table(stats_mean_std)

    # Save the result
    evaluation_metrics_file = checkpoint_path / "evaluation_metrics.json"
    with open(evaluation_metrics_file, 'w') as file_obj:
        json.dump(dict(stats), file_obj, indent=4)

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

    # Env args
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
    parser.add_argument(
        "--M",
        type=int,
        default=10,
        help="Number of steps to run rdkit minimization for")

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

    config['db_path'] = '/'.join(args.eval_db_path.split('/')[-3:])
    config['eval_db_path'] = '/'.join(args.eval_db_path.split('/')[-3:])
    config['molecules_xyz_prefix'] = "env/molecules_xyz"
    config['n_parallel'] = args.n_parallel
    config['timelimit'] = args.timelimit
    config['n_threads'] = args.n_threads
    config['terminate_on_negative_reward'] = args.terminate_on_negative_reward
    config['max_num_negative_rewards'] = args.max_num_negative_rewards
    config['reward'] = args.reward
    config['minimize_on_every_step'] = args.minimize_on_every_step
    config['M'] = args.M
    config['sample_initial_conformations'] = False
    config['num_initial_conformations'] = -1

    config = Config(**config)

    main(checkpoint_path, args, config)
