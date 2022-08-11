import argparse
import datetime
import json
import numpy as np
import os
import pickle
import random
import torch

from collections import defaultdict
from pathlib import Path

from env.moldynamics_env import env_fn
from env.wrappers import rdkit_reward_wrapper

from tqc import DEVICE
from tqc.actor_critic import Actor
from tqc.utils import ActionScaleScheduler, rdkit_minimize_until_convergence

def run_policy(env, actor, fixed_atoms, max_timestamps):
    done = False
    delta_energy = 0
    t = 0
    action_norm_list = []
    state = env.set_initial_positions(fixed_atoms)
    while not done and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        action_norm_list.append(np.linalg.norm(action, axis=1).mean().item())
        state, reward, done, info = env.step(action)
        delta_energy += reward
        t += 1
    
    return np.array(action_norm_list), info['final_energy'], info['final_rl_energy']

def evaluate_action_norm(env, actor, args):
    all_action_norms = defaultdict(list)
    result = defaultdict(lambda: 0.0)
    for _ in range(args.conf_number):
        env.reset()
        fixed_atoms = env.atoms.copy()
        initial_energy, full_rdkit_final_energy = rdkit_minimize_until_convergence(env, fixed_atoms, M=0)
        # RL (N its) + rdkit (M its) minization
        if args.N > 0:
            action_norm, final_energy, rl_final_energy = run_policy(env, actor, fixed_atoms, args.N)
            all_action_norms[str(fixed_atoms.symbols)].append(action_norm)
            result['rl_delta_energy'] += (initial_energy - rl_final_energy)
            result['rdkit_after_rl_delta_energy'] += (rl_final_energy - final_energy)
            result['rl_rdkit_delta_energy'] += (initial_energy - final_energy)
            result['pct_minimized'] += (initial_energy - final_energy) / (initial_energy - full_rdkit_final_energy)
    result = {k: v / args.conf_number for k, v in result.items()}

    mean_action_norms = {k: np.array(v).mean(axis=0) for k, v in all_action_norms.items()}
    
    if args.verbose:
        print("\nRL + rdkit minimization")
        print("RL ({:d} its) ΔE = {:.3f}".format(args.N, result['rl_delta_energy']))
        print("Rdkit ({:d} its) ΔE = {:.3f}".format(args.M, result['rdkit_after_rl_delta_energy']))
        print("RL ({:d} its) + rdkit ({:d} its) ΔE = {:.3f}".format(args.N, args.M, result['rl_rdkit_delta_energy']))
        print("RL ({:d} its) + rdkit ({:d} its) % minimized  = {:.3f}".format(args.N, args.M, result['pct_minimized']))

    return result, mean_action_norms

def main(exp_folder, args):
    # Initialize env
    env_kwargs = {
        'db_path': args.db_path,
        'timelimit': args.N,
        'done_on_timelimit': False,
        'num_initial_conformations': args.conf_number,
        'sample_initial_conformations': False,
        'inject_noise': False,
        'remove_hydrogen': args.remove_hydrogen,
    }
    env = env_fn(DEVICE, **env_kwargs)
    # Initialize reward wrapper for training
    reward_wrapper_kwargs = {
        'env': env,
        'minimize_on_every_step': True,
        'remove_hydrogen': args.remove_hydrogen,
        'molecules_xyz_prefix': args.molecules_xyz_prefix,
        'M': args.M
    }
    env = rdkit_reward_wrapper(**reward_wrapper_kwargs)
    # Initialize action_scale scheduler
    action_scale_scheduler = ActionScaleScheduler(action_scale_init=args.action_scale, 
                                                  action_scale_end=args.action_scale,
                                                  n_step_end=0,
                                                  mode="constant")
    action_scale_scheduler.update(0)
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }
    actor = Actor(schnet_args, args.actor_out_embedding_size, action_scale_scheduler).to(DEVICE)
    actor.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    actor.eval()

    result, mean_action_norms = evaluate_action_norm(env, actor, args)

    # Save the result
    output_file = exp_folder / "actions_norm.pickle"
    with open(output_file, 'wb') as f:
        pickle.dump(mean_action_norms, f)
    #with open(output_file, 'w') as fp:
    #    json.dump(result, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--molecules_xyz_prefix", type=str, default="", help="Path to env/ folder. For cluster compatability")
    # Schnet args
    parser.add_argument("--n_interactions", default=3, type=int, help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument("--cutoff", default=20.0, type=float, help="Cutoff for Schnet in actor/critic")
    parser.add_argument("--n_gaussians", default=50, type=int, help="Number of Gaussians for Schnet in actor/critic")
    parser.add_argument("--remove_hydrogen", type=bool, default=False, help="Whether to remove hydrogen atoms from the molecule")
    # Agent args
    parser.add_argument("--actor_out_embedding_size", default=128, type=int, help="Output embedding size for actor")
    parser.add_argument("--action_scale", default=0.01, type=float, help="Bounds actions to [-action_scale, action_scale]")
    # Other args
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--log_dir", default="evaluation_output", type=str, help="Which directory to store outputs to")
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--N", default=10, type=int, help="Run RL policy for N steps")
    parser.add_argument("--M", default=5, type=int, help="Run rdkit minimization for M steps after RL")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}_{random.randint(0, 1000000)}'
    if os.path.exists(exp_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
    os.makedirs(exp_folder)
    
    main(exp_folder, args)