import numpy as np
import torch

from collections import defaultdict

from AL import DEVICE
from AL.utils import recollate_batch


TIMELIMITS = [1, 5, 10, 50, 100]
CONVERGENCE_THRESHOLD = 1e-5


def run_policy(env, actor, fixed_atoms, smiles, max_timestamps, eval_termination_mode):
    teminate_episode_condition = False
    delta_energy = 0
    t = 0

    # Reset initial state in actor
    state = env.set_initial_positions(fixed_atoms, smiles, energy_list=[None])
    actor.reset({k: v.to(DEVICE) for k, v in state.items()})

    # Get initial final energies in case of an optimization failure
    initial_energy = env.get_energies()

    while not teminate_episode_condition:
        action, _, done = actor.select_action([t])
        state, reward, _, info = env.step(action)
        state = {k: v.to(DEVICE) for k, v in state.items()}
        delta_energy += reward[0]
        t += 1
        if eval_termination_mode == "grad_norm":
            teminate_episode_condition = done[0]
        elif eval_termination_mode == "negative_reward":
            teminate_episode_condition = reward[0] < 0
        # Terminate if max len is reached
        teminate_episode_condition = teminate_episode_condition or t >= max_timestamps

    if delta_energy < 0:
        final_energy = initial_energy[0]
        delta_energy = 0
    else:
        final_energy = info["final_energy"][0]

    return delta_energy, final_energy, t


def rdkit_minimize_until_convergence(env, fixed_atoms, smiles, M=None):
    M_init = 1000
    env.set_initial_positions(fixed_atoms, smiles, energy_list=[None], M=M)
    initial_energy = env.initial_energy["rdkit"][0]
    not_converged, final_energy, _ = env.minimize_rdkit(idx=0, M=M_init)
    while not_converged:
        M_init *= 2
        not_converged, final_energy, _ = env.minimize_rdkit(idx=0, M=M_init)
        if M_init > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy


def eval_policy_dft(actor, env, eval_episodes=10):
    max_timestamps = env.unwrapped.TL
    result = defaultdict(list)
    state = env.reset()
    # Reset initial states in actor
    actor.reset(state)
    episode_returns = np.zeros(env.unwrapped.n_parallel)
    actor.eval()
    while len(result["eval/delta_energy"]) < eval_episodes:
        episode_timesteps = env.unwrapped.get_env_step()
        # TODO incorporate actor dones into DFT evaluation
        action, _, actor_dones = actor.select_action(episode_timesteps)
        # Obser reward and next obs
        state, rewards, dones, infos = env.step(action)
        dones = [
            done or (t + 1) > max_timestamps
            for done, t in zip(dones, episode_timesteps)
        ]
        episode_returns += rewards

        envs_to_reset = []
        for i in range(env.unwrapped.n_parallel):
            if dones[i]:
                envs_to_reset.append(i)
                result["eval/delta_energy"].append(episode_returns[i])
                result["eval/final_energy"].append(infos["final_energy"][i])
                result["eval/final_rl_energy"].append(infos["final_rl_energy"][i])
                result["eval/episode_len"].append(episode_timesteps[i])
                episode_returns[i] = 0

        if len(envs_to_reset) > 0:
            reset_states = env.reset(indices=envs_to_reset)

            # Recollate state_batch after resets as atomic numbers might have changed.
            state = recollate_batch(state, envs_to_reset, reset_states)

            # Reset initial states in policy
            actor.reset(reset_states, indices=envs_to_reset)
    actor.train()
    result = {k: np.array(v).mean() for k, v in result.items()}
    return result


def eval_policy_rdkit(
    actor,
    env,
    eval_episodes=10,
    evaluate_multiple_timesteps=True,
    eval_termination_mode=False,
):
    assert env.n_parallel == 1, "Eval env is supposed to have n_parallel=1."

    max_timestamps = env.unwrapped.TL
    result = defaultdict(lambda: 0.0)
    for _ in range(eval_episodes):
        env.reset()
        if hasattr(env.unwrapped, "smiles"):
            smiles = env.unwrapped.smiles.copy()
        else:
            smiles = [None]
        fixed_atoms = env.unwrapped.atoms.copy()

        # Evaluate policy in eval mode
        actor.eval()
        eval_delta_energy, eval_final_energy, eval_episode_len = run_policy(
            env, actor, fixed_atoms, smiles, max_timestamps, eval_termination_mode
        )
        result["eval/delta_energy"] += eval_delta_energy
        result["eval/final_energy"] += eval_final_energy
        result["eval/episode_len"] += eval_episode_len

        # Compute minimal energy of the molecule
        initial_energy, final_energy = rdkit_minimize_until_convergence(
            env, fixed_atoms, smiles, M=0
        )
        pct = (initial_energy - eval_final_energy) / (initial_energy - final_energy)
        result["eval/pct_of_minimized_energy"] += pct
        if pct > 1.0 or pct < -100:
            print(
                "Strange conformation encountered: pct={:.3f} \nSmiles: {} \
                \n Coords: \n{}".format(
                    result["eval/pct_of_minimized_energy"],
                    smiles,
                    fixed_atoms[0].get_positions(),
                )
            )

        # Evaluate policy at multiple timelimits
        if evaluate_multiple_timesteps:
            for timelimit in TIMELIMITS:
                # Set env's TL to current timelimit
                env.update_timelimit(timelimit)
                delta_energy_at, final_energy_at, _ = run_policy(
                    env,
                    actor,
                    fixed_atoms,
                    smiles,
                    max_timestamps,
                    eval_termination_mode,
                )
                result[f"eval/delta_energy_at_{timelimit}"] += delta_energy_at

                # If reward is given by rdkit we know the optimal energy for the conformation.
                result[f"eval/pct_of_minimized_energy_at_{timelimit}"] += (
                    initial_energy - final_energy_at
                ) / (initial_energy - final_energy)
            # Set env's TL to original value
            env.update_timelimit(max_timestamps)

        # Switch actor back to training mode
        actor.train()

    result = {k: v / eval_episodes for k, v in result.items()}
    return result
