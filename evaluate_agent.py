import argparse
import datetime
import json
import numpy as np
import os
import random
import torch

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

from env.make_envs import make_envs
from rl import DEVICE
from rl.actor_critics.tqc import Actor
from rl.utils import ActionScaleScheduler
from rl.eval import run_policy, rdkit_minimize_until_convergence
from utils.arguments import str2bool


class Config():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def run_policy(env, actor, fixed_atoms, smiles, max_timestamps):
    done = np.array([False])
    delta_energy = 0
    t = 0
    state = env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
    state = {k:v.to(DEVICE) for k, v in state.items()}
    while not done[0] and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        state, reward, done, info = env.step(action)
        state = {k:v.to(DEVICE) for k, v in state.items()}
        if not done[0]:
            delta_energy += reward[0]
        t += 1
    return delta_energy, info['final_energy'][0], info['final_rl_energy'][0], t

def rdkit_minimize_until_convergence(env, fixed_atoms, smiles):
    M_init = 1000
    env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
    initial_energy = env.initial_energy['rdkit'][0]
    not_converged, final_energy, _ = env.minimize_rdkit(idx=0, M=M_init)
    while not_converged:
        M_init *= 2
        not_converged, final_energy, _ = env.minimize_rdkit(idx=0, M=M_init)
        if M_init > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy

def rdkit_minimize_until_convergence_binary_search(env, fixed_atoms, smiles):
    # Binary search :/
    # A more efficient way requires messing with c++ code in rdkit
    left = 0
    right = 150
    mid = 0
    while left <= right:
        mid = (left + right) // 2
        env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
        not_converged_l, final_energy_left, _ = env.minimize_rdkit(M=mid, idx=0)
        env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
        not_converged_r, _, _ = env.minimize_rdkit(M=mid + 1, idx=0)
        if not_converged_l and not_converged_r:
            left = mid + 1
        elif not not_converged_l and not not_converged_r:
            right = mid - 1
        else:
            return mid + 1, final_energy_left
    return left, final_energy_left

def eval_agent(env, actor, n_confs, N, verbose):
    result = defaultdict(lambda: 0.0)
    for _ in tqdm(range(n_confs)):
        # Sample molecule from the test set
        env.reset()
        if hasattr(env.unwrapped, 'smiles'):
            smiles = env.unwrapped.smiles.copy()
        else:
            smiles = [None]
        fixed_atoms = env.unwrapped.atoms.copy()

        # Optimize molecule with RL
        _, eval_final_energy, _, eval_episode_len =\
            run_policy(env, actor, fixed_atoms, smiles, N)

        # Save result of RL optimization
        after_rl_atoms = env.unwrapped.atoms.copy()

        # Optimize with rdkit until convergence
        initial_energy, final_energy =\
            rdkit_minimize_until_convergence(env, fixed_atoms, smiles)

        # Get number of iterations untils convergence
        rdkit_num_iter, final_energy_rdkit =\
             rdkit_minimize_until_convergence_binary_search(env, fixed_atoms, smiles)
        rl_num_iter, final_energy_rl_rdkit =\
             rdkit_minimize_until_convergence_binary_search(env, after_rl_atoms, smiles)

        # Save results
        result['pct_final_energy_rl_better_rdkit'] += int(final_energy_rdkit < final_energy_rl_rdkit)
        if final_energy_rdkit < final_energy_rl_rdkit:
            result['rl_better_rdkit'] += final_energy_rl_rdkit - final_energy_rdkit
        else:
            result['rl_worse_rdkit'] += final_energy_rdkit - final_energy_rl_rdkit
        result['episode_len'] += eval_episode_len
        result['rdkit_num_iter'] += rdkit_num_iter
        result['rl_num_iter'] += rl_num_iter
        if initial_energy - final_energy < 1e-5:
            continue
        result['pct_of_minimized'] += (initial_energy - eval_final_energy) / (initial_energy - final_energy)

    result = {k: v / n_confs for k, v in result.items()}
    result['rl_better_rdkit'] *= (1 / result['pct_final_energy_rl_better_rdkit'])
    result['rl_worse_rdkit'] *= (1 / (1 - result['pct_final_energy_rl_better_rdkit']))
    
    if verbose:
        print(result)
    return result
        

def main(exp_folder, args, config):

    # Update config
    if args.M != -1:
        config.M = args.M
    config.done_when_not_improved = args.done_when_not_improved
   
    _, eval_env = make_envs(config)
    # Initialize action_scale scheduler

    backbone_args = {
        'n_interactions': config.n_interactions,
        'cutoff': config.cutoff,
        'n_gaussians': config.n_rbf,
        'n_rbf':  config.n_rbf,
        'use_cosine_between_vectors': config.use_cosine_between_vectors,
    }

    actor = Actor(
        backbone=config.backbone,
        backbone_args=backbone_args,
        generate_action_type=config.generate_action_type,
        out_embedding_size=config.out_embedding_size,
        action_scale=config.action_scale,
        cutoff_type=config.cutoff_type,
        use_activation=config.use_activation,
        limit_actions=config.limit_actions,
        summation_order=config.summation_order
    )
    actor.load_state_dict(torch.load(args.agent_path, map_location=torch.device('cpu')))
    actor.eval()

    result = eval_agent(eval_env, actor, args.conf_number, args.N, args.verbose)

    # Save the result
    result['config'] = args.__dict__
    output_file = exp_folder / "output.json"
    with open(output_file, 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--log_dir", default="evaluation_output", type=str, help="Which directory to store outputs to")
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--N", default=10, type=int, help="Run RL policy for maximum of N steps")
    parser.add_argument("--M", default=-1, type=int, help="Run rdkit minimization for M steps after RL.\
                                                           Set to -1 to use value from config")
    parser.add_argument("--done_when_not_improved", default=False, choices=[True, False], \
                        metavar='True|False', type=str2bool, help="Done on negative reward")
    parser.add_argument("--verbose", default=False, choices=[True, False], \
                        metavar='True|False', type=str2bool, help="Print results")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{random.randint(0, 1000000)}'
    if os.path.exists(exp_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
    os.makedirs(exp_folder)

    # Read config and turn it into a class object with properties
    with open(args.config_path, "rb") as f:
        config = json.load(f)

    # TMP
    config['db_path'] = '/'.join(config['db_path'].split('/')[-3:])
    config['eval_db_path'] = '/'.join(config['eval_db_path'].split('/')[-3:])
    config['molecules_xyz_prefix'] = "/Users/artem/Desktop/work/MARL/MolDynamics/env/molecules_xyz"

    config = Config(**config)
    
    main(exp_folder, args, config)
