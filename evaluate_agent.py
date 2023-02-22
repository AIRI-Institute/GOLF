import argparse
import datetime
import json
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

def rdkit_minimize(env, fixed_atoms, smiles, max_its):
    env.set_initial_positions(fixed_atoms, smiles, energy_list=[None], M=0)
    not_converged, final_energy, _ = env.minimize_rdkit(idx=0, M=max_its)
    return final_energy, not_converged

def eval_agent(env, actor, n_confs, evaluate_rdkit, evaluate_rl, eval_termination_mode):
    convergence_results = defaultdict(list)
    result = defaultdict(lambda: defaultdict(lambda: []))

    max_iter = 1000
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

        # Optimize with rdkit until convergence
        _, final_rdkit_energy =\
            rdkit_minimize_until_convergence(env, fixed_atoms, smiles, M=0)
        total_delta_rdkit = initial_energy - final_rdkit_energy
        if total_delta_rdkit < 0 or total_delta_rdkit < 1e-5:
            print("Rdkit optimization failure")
            continue

        # Optimize with rdkit until convergence after RL
        if evaluate_rl:
             # Optimize molecule with RL and save resulting conformation
            _ = run_policy(env, actor, fixed_atoms, smiles, max_timestamps, eval_termination_mode)
            after_rl_atoms = env.unwrapped.atoms.copy()
            _, final_rdkit_after_rl_energy = \
                rdkit_minimize_until_convergence(env, after_rl_atoms, smiles, M=0)
            total_delta_rdkit_after_rl = initial_energy - final_rdkit_after_rl_energy
            if total_delta_rdkit_after_rl < 0 or total_delta_rdkit_after_rl < 1e-5:
                print("Rdkit optimization failure")
                continue
        
        rdkit_converged_flag = False
        rdkit_after_rl_converged_flag = False
        for n_its in range(1, 1000):
            # If rdkit optimization converged add 1.0 and set
            # rdkit_converged_flag to True to stop further computations
            if not rdkit_converged_flag and evaluate_rdkit: 
                final_energy_rdkit, nc = rdkit_minimize(env, fixed_atoms, smiles, n_its)
                result[f'{n_its}']['rdkit'].append((initial_energy - final_energy_rdkit) / total_delta_rdkit)
                if not nc:
                    #print("rdkit", nc, n_its)
                    rdkit_converged_flag = True
                    convergence_results['rdkit'].append(n_its)
                    for i in range(n_its + 1, max_iter + 1):
                        result[f'{i}']['rdkit'].append(1.0)
                
            # If rdkit optimization converged add 1.0 and set
            # rdkit_after_rl_converged_flag to True to stop further computations
            if not rdkit_after_rl_converged_flag and evaluate_rl:
                final_energy_rdkit_after_rl, nc_after_rl = rdkit_minimize(env, after_rl_atoms, smiles, n_its)
                result[f'{n_its}']['rdkit_after_rl'].append((initial_energy - final_energy_rdkit_after_rl) / total_delta_rdkit_after_rl)
                result[f'{n_its}']['rdkit_after_rl_rel'].append((initial_energy - final_energy_rdkit_after_rl) / total_delta_rdkit)
                if not nc_after_rl:
                    rdkit_after_rl_converged_flag = True
                    convergence_results['rdkit_after_rl'].append(n_its)
                    for i in range(n_its + 1, max_iter + 1):
                        result[f'{i}']['rdkit_after_rl'].append(1.0)
                        result[f'{i}']['rdkit_after_rl_rel'].append((initial_energy - final_energy_rdkit_after_rl) / total_delta_rdkit)
                
            # If both rdkit and rdkit after rl converged break
            if (rdkit_converged_flag or not evaluate_rdkit) and\
               (rdkit_after_rl_converged_flag or not evaluate_rl):
                break
    
    real_max_iter = max([max(v) for v in convergence_results.values()])
    result = {k:v for k, v in result.items() if int(k) <= real_max_iter}
    return result, convergence_results
        

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

    result, convergence_results = eval_agent(eval_env, actor, args.conf_number, 
                                             args.evaluate_rdkit, args.evaluate_rl,
                                             eval_termination_mode="fixed_length")
    
    # Convert defaultdict into normal dict to pickle it
    normal_dict = {}
    for k, v in result.items():
        normal_dict[k] = {}
        for k2, v2 in v.items():
            normal_dict[k][k2] = v2

    # Save the result
    result_file_name = checkpoint_path / "result.pickle"
    convergence_results_file_name = checkpoint_path / "convergence_results.pickle"
    with open(result_file_name, 'wb') as f:
        pickle.dump(normal_dict, f)
    with open(convergence_results_file_name, 'wb') as f:
        pickle.dump(convergence_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--agent_path", type=str, required=True)
    parser.add_argument("--conf_number", default=int(1e5), type=int, help="Number of conformations to evaluate on")
    parser.add_argument("--timelimit_eval", default=500, type=int, help="Max len of episode on eval")
    parser.add_argument("--evaluate_rdkit", default=False, choices=[True, False],
                         metavar='True|False', type=str2bool, help="Evaluate initial state with rdkit")
    parser.add_argument("--evaluate_rl", default=False, choices=[True, False],
                         metavar='True|False', type=str2bool, help="Evaluate after-RL state with rdkit")
    args = parser.parse_args()

    start_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    checkpoint_path = Path(args.checkpoint_path)
    config_path = checkpoint_path / "config.json"
    # Read config and turn it into a class object with properties
    with open(config_path, "rb") as f:
        config = json.load(f)

    assert args.evaluate_rdkit or args.evaluate_rl , "Nothing to evaluate!"

    # TMP
    config['db_path'] = '/'.join(config['db_path'].split('/')[-3:])
    config['eval_db_path'] = '/'.join(config['eval_db_path'].split('/')[-3:])
    config['molecules_xyz_prefix'] = "env/molecules_xyz"

    config = Config(**config)
    
    main(checkpoint_path, args, config)
