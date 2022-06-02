import argparse
import datetime
import os
import numpy as np
import torch

from ase.io import Trajectory
from pathlib import Path
from rdkit.Chem import AllChem

from env.moldynamics_env import env_fn
from env.wrappers import rdkit_reward_wrapper
from env.xyz2mol import get_rdkit_energy, parse_molecule, set_coordinates

from tqc import DEVICE
from tqc.actor_critic import Actor

def rl_minimize(file_name, policy, env, timelimit, action_scale=1.0):
    total_reward = 0.
    traj = Trajectory(file_name, mode='w')

    state, done = env.reset(), False
    traj.write(env.atoms)
    t = 0
    while not done and t < timelimit:
        with torch.no_grad():
            action = policy.select_action(state)
        state, reward, done, info = env.step(action * action_scale)
        total_reward += reward
        traj.write(env.atoms)
        t += 1
    final_energy = info['final_energy']
    return total_reward, final_energy

def rdkit_minimize(file_name, initial_posisitons, molecule, ase_atoms, M):
    traj = Trajectory(file_name, mode='a')
    for i in range(1, M + 1):
        set_coordinates(molecule, initial_posisitons)
        ff = AllChem.MMFFGetMoleculeForceField(molecule,
            AllChem.MMFFGetMoleculeProperties(molecule), confId=0)
        ff.Initialize()
        not_converged = ff.Minimize(maxIts=i)
        ase_atoms.set_positions(molecule.GetConformers()[0].GetPositions())
        traj.write(ase_atoms)
    return not_converged

def main(exp_folder, args):
    env = env_fn(DEVICE, multiagent=False, db_path=args.db_path, timelimit=args.N,
                      done_on_timelimit=False, inject_noise=args.inject_noise, noise_std=args.noise_std,
                      calculate_mean_std=args.calculate_mean_std_energy, exp_folder='./')
    env = rdkit_reward_wrapper(env, molecule_path=args.molecule_path,
                               minimize_on_every_step=False, M=args.M)
    schnet_args = {
        'n_interactions': args.n_interactions,
        'cutoff': args.cutoff,
        'n_gaussians': args.n_gaussians,
    }
    actor = Actor(schnet_args, out_embedding_size=args.actor_out_embedding_size).to(DEVICE)
    actor.load_state_dict(torch.load(args.load_model, map_location=DEVICE))
    actor.eval()
    rdkit_molecule = parse_molecule('env/molecules_xyz/malonaldehyde.xyz')

    total_rl_delta_energy = 0.
    total_final_energy = 0.
    total_not_converged = 0.
    for traj_num in range(args.traj_number):
        file_name = exp_folder / f'trajectory_{traj_num}'
        # Write state visited by the RL agent to a file
        if args.N > 0:
            rl_delta_energy, final_energy = rl_minimize(file_name, actor, env, args.N, args.action_scale)
            initial_positions = env.atoms.get_positions()
        else:
            env.reset()
            initial_positions = env.atoms.get_positions()
        not_converged = rdkit_minimize(file_name, initial_positions, rdkit_molecule, env.atoms, args.M)
        if args.N == 0:
            rl_delta_energy = 0
            final_energy = get_rdkit_energy(rdkit_molecule)
        total_final_energy += final_energy
        total_rl_delta_energy += rl_delta_energy
        total_not_converged += not_converged
    if args.verbose:
        print("Mean final energy: {:.3f}".format(total_final_energy / args.traj_number))
        print("Mean converged: {:.3f}".format(1 - total_not_converged / args.traj_number))
        print("Mean rl delta energy: {:.3f}".format(total_rl_delta_energy / args.traj_number))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Env args
    parser.add_argument("--db_path", default="env/data/malonaldehyde.db", type=str, help="Path to molecules database")
    parser.add_argument("--schnet_model_path", default="env/schnet_model/schnet_model_3_blocks", type=str, help="Path to trained schnet model")
    parser.add_argument("--molecule_path", default="env/molecules_xyz/malonaldehyde.xyz", type=str, help="Path to example .xyz file")
    parser.add_argument("--inject_noise", type=bool, default=False, help="Whether to inject random noise into initial states")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Std of the injected noise")
    parser.add_argument("--calculate_mean_std_energy", type=bool, default=False, help="Calculate mean, std of energy of database")
    parser.add_argument("--done_on_timelimit", type=bool, default=False, help="Env returns done when timelimit is reached")
    # Schnet args
    parser.add_argument("--n_interactions", default=3, type=int, help="Number of interaction blocks for Schnet in actor/critic")
    parser.add_argument("--cutoff", default=20.0, type=float, help="Cutoff for Schnet in actor/critic")
    parser.add_argument("--n_gaussians", default=50, type=int, help="Number of Gaussians for Schnet in actor/critic")
    # Agent args
    parser.add_argument("--actor_out_embedding_size", default=128, type=int, help="Output embedding size for actor")
    parser.add_argument("--action_scale", default=0.01, type=float, help="Bounds actions to [-action_scale, action_scale]")
    # Other args
    parser.add_argument("--exp_name", required=True, type=str, help="Name of the experiment")
    parser.add_argument("--log_dir", default="trajectories", type=str, help="Which directory to store trajectories in")
    parser.add_argument("--traj_number", default=int(1e5), type=int, help="Number of visualized trajectories")
    parser.add_argument("--N", default=10, type=int, help="Run RL policy for N steps")
    parser.add_argument("--M", default=10, type=int, help="Run RdKit minimization for M steps")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    exp_folder = log_dir / f'{args.exp_name}_{start_time}'
    if os.path.exists(exp_folder):
            raise Exception('Experiment folder exists, apparent seed conflict!')
    os.makedirs(exp_folder)

    main(exp_folder, args)