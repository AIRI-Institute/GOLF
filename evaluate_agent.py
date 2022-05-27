import argparse
import json
import numpy as np
import torch

from collections import defaultdict
from rdkit.Chem import AllChem

from env.moldynamics_env import env_fn
from env.wrappers import rdkit_reward_wrapper
from env.xyz2mol import parse_molecule, get_rdkit_energy, set_coordinates

from tqc import DEVICE
from tqc.actor_critic import Actor


def rl_minimize(policy, env, timelimit, action_scale=1.0):
    final_energy = 0.

    state, done = env.reset(), False
    initial_state = state

    t = 0
    while not done and t < timelimit:
        with torch.no_grad():
            action = policy.select_action(state)
        state, _, done, info = env.step(action * action_scale)
        t += 1
    final_energy += info['final_energy']
    positions = env.atoms.get_positions()
    return initial_state, final_energy, positions

def rdkit_minimize(molecule, max_iter=15):
    ff = AllChem.MMFFGetMoleculeForceField(molecule,
            AllChem.MMFFGetMoleculeProperties(molecule), confId=0)
    ff.Initialize()
    solved = ff.Minimize(maxIts=max_iter)
    final_energy = get_rdkit_energy(molecule)
    return solved, final_energy

def rdkit_minimize_until_convergence(molecule):
    # UGLY binary search :/ Need to fix it
    initial_positions = molecule.GetConformers()[0].GetPositions()
    left = 0
    right = 100
    mid = 0
    while left <= right:
        mid = (left + right) // 2
        set_coordinates(molecule, initial_positions)
        ff = AllChem.MMFFGetMoleculeForceField(molecule,
            AllChem.MMFFGetMoleculeProperties(molecule), confId=0)
        ff.Initialize()
        not_converged_l = ff.Minimize(maxIts=mid)
        set_coordinates(molecule, initial_positions)
        ff = AllChem.MMFFGetMoleculeForceField(molecule,
            AllChem.MMFFGetMoleculeProperties(molecule), confId=0)
        ff.Initialize()
        not_converged_r = ff.Minimize(maxIts=mid + 1)
        if not_converged_l and not_converged_r:
            left = mid + 1
        elif not not_converged_l and not not_converged_r:
            right = mid - 1
        else:
            return mid + 1
    return left

def evaluate_final_energy(env, actor, rdkit_molecule, args):
    result = defaultdict(list)
    for _ in range(args.conf_number):
        # RL + rdkit minization
        if args.N > 0:
            initial_state, rl_final_energy, positions = rl_minimize(actor, env, args.N, args.action_scale)
            result['rl_final_energy'].append(rl_final_energy)
        else:
            initial_state = env.reset()
            positions = initial_state['_positions'].double()[0].cpu().numpy()

        initial_state_positions = initial_state['_positions'].double()[0].cpu().numpy()
        set_coordinates(rdkit_molecule, initial_state_positions)
        initial_energy = get_rdkit_energy(rdkit_molecule)
        result['initial_energy'].append(initial_energy)
        if args.N > 0:
            result['rl_delta'].append(initial_energy - rl_final_energy)
        
        set_coordinates(rdkit_molecule, positions)
        solved, rdkit_final_energy = rdkit_minimize(rdkit_molecule, args.M)
        result['rdkit_solved'].append(solved)
        result['rdkit_final_energy'].append(rdkit_final_energy)
        if args.N > 0:
            result['rdkit_delta'].append(rl_final_energy - rdkit_final_energy)

        # Rdkit only minimization from the same initial state
        set_coordinates(rdkit_molecule, initial_state_positions)
        solved, rdkit_final_energy = rdkit_minimize(rdkit_molecule, args.M)
        result['only_rdkit_solved'].append(solved)
        result['only_rdkit_final_energy'].append(rdkit_final_energy)
        result['only_rdkit_delta'].append(initial_energy - rdkit_final_energy)

    if args.verbose:
        rl_delta_mean = np.array(result['rl_delta']).mean()
        rl_delta_std = np.array(result['rl_delta']).std()
        rdkit_delta_mean = np.array(result['rdkit_delta']).mean()
        rdkit_delta_std = np.array(result['rdkit_delta']).std()
        total_delta_mean = rl_delta_mean + rdkit_delta_mean
        rdkit_final_energy_mean = np.array(result['rdkit_final_energy']).mean()

        only_rdkit_delta_mean = np.array(result['only_rdkit_delta']).mean()
        only_rdkit_delta_std = np.array(result['only_rdkit_delta']).std()
        only_rdkit_final_energy_mean = np.array(result['only_rdkit_final_energy']).mean()

        print("RL + rdkit minimization")
        print("Delta RL, N={:d}: {:.3f} ± {:.3f}".format(args.N, rl_delta_mean, rl_delta_std))
        print("Delta rdkit, M={:d}: {:.3f} ± {:.3f}".format(args.M, rdkit_delta_mean, rdkit_delta_std))
        print("Delta total: {:.3f}".format(total_delta_mean))
        print("Final energy: {:.3f}".format(rdkit_final_energy_mean))
        print("\n Rdkit only minimization")
        print("Rdkit only delta, M={:d}: {:.3f} ± {:.3f}".format(args.M, only_rdkit_delta_mean, only_rdkit_delta_std))   
        print("Rdkit only final energy: {:.3f}".format(only_rdkit_final_energy_mean))

    return result

def evaluate_convergence(env, actor, rdkit_molecule, args):
    result = defaultdict(list)
    assert args.N > 0, "N must be greater than 0 to evalute convergence"
    for _ in range(args.conf_number):
        # RL
        initial_state, _, positions = rl_minimize(actor, env, args.N, args.action_scale)

        # RL + rdkit until convergence
        set_coordinates(rdkit_molecule, positions)
        num_iters = rdkit_minimize_until_convergence(rdkit_molecule)
        result['rl_rdkit_iterations'].append(num_iters)

        # rdkit until convergence
        initial_state_positions = initial_state['_positions'].double()[0].cpu().numpy()
        set_coordinates(rdkit_molecule, initial_state_positions)
        num_iters = rdkit_minimize_until_convergence(rdkit_molecule)
        result['rdkit_iterations'].append(num_iters)

    if args.verbose:
        rl_rdkit_iterations_mean = np.array(result['rl_rdkit_iterations']).mean()
        rl_rdkit_iterations_std = np.array(result['rl_rdkit_iterations']).std()

        rdkit_iterations_mean = np.array(result['rdkit_iterations']).mean()
        rdkit_iterations_std = np.array(result['rdkit_iterations']).std()

        print("RL + rdkit")
        print("Iterations until convergence: {:.3f} ± {:.3f}".format(rl_rdkit_iterations_mean, rl_rdkit_iterations_std))
        print("Rdkit")
        print("Iterations until convergence: {:.3f} ± {:.3f}".format(rdkit_iterations_mean, rdkit_iterations_std))

def main(args):
    env = env_fn(DEVICE, multiagent=False, db_path=args.db_path, timelimit=args.N,
                      done_on_timelimit=False, inject_noise=args.inject_noise, noise_std=args.noise_std,
                      calculate_mean_std=args.calculate_mean_std_energy, exp_folder='./')
    env = rdkit_reward_wrapper(env, multiagent=False, molecule_path=args.molecule_path,
                                        reward_delta=args.reward_delta)
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }
    actor = Actor(schnet_args, out_embedding_size=args.actor_out_embedding_size).to(DEVICE)
    actor.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    actor.train()
    rdkit_molecule = parse_molecule('env/molecules_xyz/malonaldehyde.xyz')

    if args.mode == "energy":
        result = evaluate_final_energy(env, actor, rdkit_molecule, args)
    elif args.mode == "convergence":
        result = evaluate_convergence(env, actor, rdkit_molecule, args)
    else:
        raise NotImplemented()

    # Save the result
    with open(args.output_file, 'w') as fp:
        json.dump(result, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--schnet_model_path", default="env/schnet_model/schnet_model_3_blocks", type=str, help="Path to trained schnet model")
    parser.add_argument("--molecule_path", default="env/molecules_xyz/malonaldehyde.xyz", type=str, help="Path to example .xyz file")
    parser.add_argument("--inject_noise", type=bool, default=False, help="Whether to inject random noise into initial states")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Std of the injected noise")
    parser.add_argument("--calculate_mean_std_energy", type=bool, default=False, help="Calculate mean, std of energy of database")
    parser.add_argument("--reward", default="both", choices=["schnet", "rdkit", "both"], help="Type of reward for MD env")
    parser.add_argument("--reward_delta", type=bool, default=False, help="Use delta of energy as reward")
    parser.add_argument("--done_on_timelimit", type=bool, default=False, help="Env returns done when timelimit is reached")
    # Schnet args
    parser.add_argument("--n_interactions", default=3, type=int, help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument("--cutoff", default=20.0, type=float, help="Cutoff for Schnet in actor/critic")
    parser.add_argument("--n_gaussians", default=50, type=int, help="Number of Gaussians for Schnet in actor/critic")
    # Agent args
    parser.add_argument("--actor_out_embedding_size", default=128, type=int, help="Output embedding size for actor")
    parser.add_argument("--action_scale", default=0.01, type=float, help="Bounds actions to [-action_scale, action_scale]")
    parser.add_argument("--n_nets", default=1, type=int)
    # Other args
    parser.add_argument("--mode", choices=["energy", "convergence"], help="Evaluation mode")
    parser.add_argument("--output_file", default="eval_output.json", type=str, help="Evaluation result file name")
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--N", default=10, type=int, help="Run RL policy for N steps")
    parser.add_argument("--M", default=5, type=int, help="Run RdKit minimization for M steps")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()

    main(args)