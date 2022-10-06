import numpy as np
import torch

from collections import defaultdict

from rl import DEVICE


TIMELIMITS = [1, 5, 10, 50, 100]


def run_policy(env, actor, fixed_atoms, max_timestamps):
    done = False
    delta_energy = 0
    t = 0
    state = env.set_initial_positions(fixed_atoms)
    state = {k:v.to(DEVICE) for k, v in state.items()}
    while not done and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        state, reward, done, info = env.step(action)
        state = {k:v.to(DEVICE) for k, v in state.items()}
        delta_energy += reward
        t += 1
    return delta_energy, info['final_energy'], info['final_rl_energy']

def rdkit_minimize_until_convergence(env, fixed_atoms, M=None):
    M_init = 1000
    env.set_initial_positions(fixed_atoms, M=M)
    initial_energy = env.initial_energy['rdkit']
    not_converged, final_energy = env.minimize_rdkit(M=M_init)
    while not_converged:
        M *= 2
        not_converged, final_energy = env.minimize_rdkit(M=M_init)
        if M > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy

def eval_policy(actor, env, max_timestamps, eval_episodes=10,
                n_explore_runs=5, rdkit=True, evaluate_multiple_timesteps=True):

    result = defaultdict(lambda: 0.0)
    for _ in range(eval_episodes):
        env.reset()
        fixed_atoms = env.unwrapped.atoms.copy()

        # Evaluate policy in eval mode
        actor.eval()
        eval_delta_energy, eval_final_energy, eval_final_rl_energy = run_policy(env, actor, fixed_atoms, max_timestamps=max_timestamps)
        result['eval/delta_energy'] += eval_delta_energy
        result['eval/final_energy'] += eval_final_energy
        result['eval/final_rl_energy'] += eval_final_rl_energy

        # Compute minimal energy of the molecule
        if rdkit:
            initial_energy, final_energy = rdkit_minimize_until_convergence(env, fixed_atoms)
            result['eval/pct_of_minimized_energy'] += (initial_energy - eval_final_energy) / (initial_energy - final_energy)

        # Evaluate policy at multiple timelimits
        if evaluate_multiple_timesteps:
            for timelimit in TIMELIMITS:

                # Set env's TL to current timelimit
                env.update_timelimit(timelimit)
                delta_energy_at, final_energy_at, _ = run_policy(env, actor, fixed_atoms, max_timestamps=timelimit)
                result[f'eval/delta_energy_at_{timelimit}'] += delta_energy_at

                # If reward is given by rdkit we know the optimal energy for the conformation.
                if rdkit:
                    result[f'eval/pct_of_minimized_energy_at_{timelimit}'] += (initial_energy - final_energy_at)  / (initial_energy - final_energy)

            # Set env's TL to original value
            env.update_timelimit(max_timestamps)

        # Evaluate policy in explore mode
        actor.train()
        if n_explore_runs > 0:
            explore_results = np.array([run_policy(env, actor, fixed_atoms, max_timestamps=max_timestamps) for _ in range(n_explore_runs)])
            explore_delta_energy, explore_final_energy, explore_final_rl_energy = explore_results.mean(axis=0)
            result['explore/delta_energy'] += explore_delta_energy
            result['explore/final_energy'] += explore_final_energy
            result['explore/final_rl_energy'] += explore_final_rl_energy
        
    result = {k: v / eval_episodes for k, v in result.items()}
    return result