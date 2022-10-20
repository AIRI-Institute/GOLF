import numpy as np
import torch

from collections import defaultdict

from rl import DEVICE
from rl.utils import recollate_batch


TIMELIMITS = [1, 5, 10, 50, 100]


def run_policy(env, actor, fixed_atoms, smiles, max_timestamps):
    done = False
    delta_energy = 0
    t = 0
    state = env.set_initial_positions(fixed_atoms, smiles, energy=None)
    state = {k:v.to(DEVICE) for k, v in state.items()}
    while not done and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        state, reward, done, info = env.step(action)
        state = {k:v.to(DEVICE) for k, v in state.items()}
        delta_energy += reward
        t += 1
    return delta_energy, info['final_energy'], info['final_rl_energy']

def rdkit_minimize_until_convergence(env, fixed_atoms, smiles):
    M_init = 1000
    env.set_initial_positions(fixed_atoms, smiles, energy=None)
    initial_energy = env.initial_energy['rdkit']
    not_converged, final_energy = env.minimize_rdkit(M=M_init)
    while not_converged:
        M_init *= 2
        not_converged, final_energy = env.minimize_rdkit(M=M_init)
        if M_init > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy


def eval_policy_dft(actor, env, max_timestamps, eval_episodes=10):
    result = defaultdict(list)
    state = env.reset()
    episode_returns = np.zeros(env.n_envs)
    actor.eval()
    while len(result['eval/delta_energy']) < eval_episodes:
        with torch.no_grad():
            actions = actor.select_action(state)
        # Obser reward and next obs
        state, rewards, dones, infos = env.step(actions)
        episode_timesteps = env.env_method("get_env_step")
        dones = [done or t > max_timestamps for done, t in zip(dones, episode_timesteps)]
        episode_returns += rewards

        envs_to_reset = []
        for i in range(env.n_envs):
            if dones[i]:
                envs_to_reset.append(i)
                result['eval/delta_energy'].append(episode_returns[i])
                result['eval/final_energy'].append(infos[i]['final_energy'])
                result['eval/final_rl_energy'].append(infos[i]['final_rl_energy'])
                episode_returns[i] = 0

        reset_states = [{k:v.squeeze() for k, v in s.items()} for s in env.env_method("reset", indices=envs_to_reset)]
        # Recollate state_batch after resets as atomic numbers might have changed.
        # Execute only if at least one env has reset.
        if len(envs_to_reset) > 0:
            state = recollate_batch(state, envs_to_reset, reset_states)
    actor.train()
    result = {k: np.array(v).mean() for k, v in result.items()}
    return result


def eval_policy_rdkit(actor, env, max_timestamps, eval_episodes=10,
                      n_explore_runs=5, evaluate_multiple_timesteps=True):
    result = defaultdict(lambda: 0.0)
    for _ in range(eval_episodes):
        env.reset()
        if hasattr(env.unwrapped, 'smiles'):
            smiles = env.unwrapped.smiles
        else:
            smiles = None
        fixed_atoms = env.unwrapped.atoms.copy()

        # Evaluate policy in eval mode
        actor.eval()
        eval_delta_energy, eval_final_energy, eval_final_rl_energy = run_policy(env, actor, fixed_atoms, smiles, max_timestamps)
        result['eval/delta_energy'] += eval_delta_energy
        result['eval/final_energy'] += eval_final_energy
        result['eval/final_rl_energy'] += eval_final_rl_energy

        # Compute minimal energy of the molecule
        initial_energy, final_energy = rdkit_minimize_until_convergence(env, fixed_atoms, smiles)
        result['eval/pct_of_minimized_energy'] += (initial_energy - eval_final_energy) / (initial_energy - final_energy)

        # Evaluate policy at multiple timelimits
        if evaluate_multiple_timesteps:
            for timelimit in TIMELIMITS:
                # Set env's TL to current timelimit
                env.update_timelimit(timelimit)
                delta_energy_at, final_energy_at, _ = run_policy(env, actor, fixed_atoms, smiles, timelimit)
                result[f'eval/delta_energy_at_{timelimit}'] += delta_energy_at

                # If reward is given by rdkit we know the optimal energy for the conformation.
                result[f'eval/pct_of_minimized_energy_at_{timelimit}'] += (initial_energy - final_energy_at)  / (initial_energy - final_energy)
            # Set env's TL to original value
            env.update_timelimit(max_timestamps)

        # Evaluate policy in explore mode
        actor.train()
        if n_explore_runs > 0:
            explore_results = np.array([run_policy(env, actor, fixed_atoms, smiles, max_timestamps) for _ in range(n_explore_runs)])
            explore_delta_energy, explore_final_energy, explore_final_rl_energy = explore_results.mean(axis=0)
            result['explore/delta_energy'] += explore_delta_energy
            result['explore/final_energy'] += explore_final_energy
            result['explore/final_rl_energy'] += explore_final_rl_energy
        
    result = {k: v / eval_episodes for k, v in result.items()}
    return result