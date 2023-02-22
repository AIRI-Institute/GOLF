import argparse
import datetime
import json
import numpy as np
import pickle
import torch

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from schnetpack.nn import BesselRBF

from env.make_envs import make_envs
from AL import DEVICE
from AL.AL_actor import Actor
from AL.eval import run_policy, rdkit_minimize_until_convergence
from AL.utils import get_cutoff_by_string
from utils.arguments import str2bool


class Config():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def eval_episode(env, actor, fixed_atoms, smiles, max_timestamps):
    delta_energy = 0
    t = 0

    l2_forces = []
    l2_energy = []
    state = env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
    state = {k:v.to(DEVICE) for k, v in state.items()}
    current_forces = [np.array(force, dtype=np.float32) for force in env.force['rdkit']][0]
    current_energy = np.array(env.initial_energy['rdkit'], dtype=np.float32)
    while not t >= max_timestamps:
        action, energy = actor.select_action(state)
        l2_forces.append(np.sqrt(((action - current_forces) ** 2).mean()))
        l2_energy.append(np.sqrt(((energy - current_energy) ** 2).mean()))
        state, reward, _, info = env.step(action)
        state = {k:v.to(DEVICE) for k, v in state.items()}
        current_forces = [np.array(force, dtype=np.float32) for force in env.force['rdkit']][0]
        current_energy = np.array(env.initial_energy['rdkit'], dtype=np.float32)
        delta_energy += reward[0]
        t += 1

    return l2_forces, l2_energy

def eval_agent(env, actor, n_confs):
    result = []

    max_timestamps = env.unwrapped.TL
    if n_confs == -1:
        n_confs = env.unwrapped.get_db_length()

    for _ in tqdm(range(n_confs)):
        # Sample molecule from the test set
        env.reset()
        
        # Get initial energy
        initial_energy = env.initial_energy['rdkit'][0]
        if hasattr(env.unwrapped, 'smiles'):
            smiles = env.unwrapped.smiles.copy()
        else:
            smiles = [None]
        fixed_atoms = env.unwrapped.atoms.copy()

        l2_forces, l2_energy = eval_episode(env, actor, fixed_atoms, smiles, max_timestamps)
        result.append((l2_forces, l2_energy))
    
    return result
        

def main(checkpoint_path, args, config):

    # Update config
    # if args.conf_number == -1:
    #     config.sample_initial_conformation = False
    # else:
    #     config.sample_initial_conformation = True
    config.sample_initial_conformation = False
    config.timelimit_eval = args.timelimit_eval
   
    _, eval_env = make_envs(config)

    backbone_args = {
        'n_interactions': config.n_interactions,
        'n_atom_basis': config.n_atom_basis,
        'radial_basis': BesselRBF(n_rbf=config.n_rbf, cutoff=config.cutoff),
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

    result = eval_agent(eval_env, actor, args.conf_number)

    # Save the result
    result_file_name = checkpoint_path / "mse_forces_energy.pickle"
    with open(result_file_name, 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--timelimit_eval", default=500, type=int, help="Max len of episode on eval")
    args = parser.parse_args()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    checkpoint_path = Path(args.checkpoint_path)
    config_path = checkpoint_path / "config.json"
    # Read config and turn it into a class object with properties
    with open(config_path, "rb") as f:
        config = json.load(f)


    # TMP
    config['db_path'] = '/'.join(config['db_path'].split('/')[-3:])
    config['eval_db_path'] = '/'.join(config['eval_db_path'].split('/')[-3:])
    config['molecules_xyz_prefix'] = "env/molecules_xyz"

    config = Config(**config)
    
    main(checkpoint_path, args, config)
