import numpy as np
import time

from collections import defaultdict

from GOLF import DEVICE
from GOLF.utils import recollate_batch

CONVERGENCE_THRESHOLD = 1e-5


def run_policy(env, actor, fixed_atoms, smiles, max_timestamps, eval_termination_mode):
    teminate_episode_condition = False
    delta_energy = 0
    t = 0

    # Reset initial state in actor
    state = env.set_initial_positions(
        fixed_atoms, smiles, energy_list=[None], force_list=[None]
    )
    actor.reset({k: v.to(DEVICE) for k, v in state.items()})

    # Get initial final energies in case of an optimization failure
    initial_energy = env.get_energies()

    while not teminate_episode_condition:
        select_action_result = actor.select_action([t])
        action = select_action_result["action"]
        done = select_action_result["done"]
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
        # Reset env to initial state
        state = env.set_initial_positions(
            fixed_atoms, smiles, energy_list=[None], force_list=[None]
        )
    else:
        final_energy = info["final_energy"][0]

    return delta_energy, final_energy, t


def rdkit_minimize_until_convergence(env, fixed_atoms, smiles, max_its=0):
    M_init = 1000
    env.set_initial_positions(
        fixed_atoms, smiles, energy_list=[None], force_list=[None], max_its=max_its
    )
    initial_energy = env.initial_energy["rdkit"][0]
    not_converged, final_energy, _ = env.minimize_rdkit(idx=0, max_its=M_init)
    while not_converged:
        M_init *= 2
        not_converged, final_energy, _ = env.minimize_rdkit(idx=0, max_its=M_init)
        if M_init > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy


def eval_policy_dft(actor, env, eval_episodes=10):
    start = time.perf_counter()
    max_timestamps = env.unwrapped.TL
    result = defaultdict(list)
    episode_returns = np.zeros(env.n_parallel)
    dft_pct_of_minimized_energy = []

    # Reset env and actor
    state = env.reset()
    actor.reset(state)
    actor.eval()

    # Calculate optimal delta energy DFT
    optimized_delta_energy_dft = np.zeros(env.n_parallel)
    for i in range(env.n_parallel):
        optimized_delta_energy_dft[i] = (
            env.unwrapped.energy[i] - env.unwrapped.optimal_energy[i]
        )

    # Calculate optimal delta energy Rdkit
    initial_energy_rdkit = env.rdkit_oracle.initial_energies

    # First save all the data
    molecules = [molecule.copy() for molecule in env.unwrapped.atoms]
    smiles = env.unwrapped.smiles.copy()
    dft_initial_energies = env.unwrapped.energy.copy()
    dft_forces = env.unwrapped.force.copy()

    # Get optimial rdkit energy and calculate delta
    _, optimal_energy_rdkit, _ = env.rdkit_oracle.calculate_energies_forces(
        max_its=5000
    )
    optimized_delta_energy_rdkit = initial_energy_rdkit - optimal_energy_rdkit

    # Reset the environment again
    env.set_initial_positions(molecules, smiles, dft_initial_energies, dft_forces)

    while len(result["eval/dft_delta_energy"]) < eval_episodes:
        episode_timesteps = env.unwrapped.get_env_step()
        # TODO incorporate actor dones into DFT evaluation
        select_action_result = actor.select_action(episode_timesteps)
        action = select_action_result["action"]
        # actor_dones = select_action_result["done"]

        # Obser reward and next obs
        state, rdkit_rewards, _, info = env.step(action)
        dones = [(t + 1) > max_timestamps for t in episode_timesteps]
        episode_returns += rdkit_rewards

        if "calculate_dft_energy_env_ids" in info:
            dft_pct_of_minimized_energy.extend(
                optimized_delta_energy_dft[
                    info["calculate_dft_energy_env_ids"]
                ].tolist()[: env.n_parallel - len(dft_pct_of_minimized_energy)]
            )

        # If task queue is full wait for all tasks to finish
        if env.dft_oracle.task_queue_full_flag:
            _, _, _, episode_total_delta_energies = env.dft_oracle.get_data(eval=True)
            # Log total delta energy and pct of optimized energy
            result["eval/dft_delta_energy"].extend(
                episode_total_delta_energies.tolist()
            )
            dft_pct_of_minimized_energy = episode_total_delta_energies / np.array(
                dft_pct_of_minimized_energy
            )
            result["eval/dft_pct_of_minimized_energy"].extend(
                dft_pct_of_minimized_energy.tolist()
            )

        # All trajectories terminate at the same time
        if np.all(dones):
            rdkit_pct_of_minimized_energy = (
                episode_returns / optimized_delta_energy_rdkit
            )
            rdkit_delta_energy = episode_returns
            final_energies = info["final_energy"]

            # Optimization failure
            optimization_failure_mask = episode_returns < 0
            rdkit_pct_of_minimized_energy[optimization_failure_mask] = 0.0
            rdkit_delta_energy[optimization_failure_mask] = 0.0
            final_energies[optimization_failure_mask] = initial_energy_rdkit[
                optimization_failure_mask
            ]

            # Log results
            result["eval/rdkit_pct_of_minimized_energy"].extend(
                rdkit_pct_of_minimized_energy.tolist()
            )
            result["eval/rdkit_delta_energy"].extend(rdkit_delta_energy.tolist())
            result["eval/final_energy"].extend(final_energies.tolist())
            result["eval/episode_len"].extend(episode_timesteps)

            # Reset episode returns
            episode_returns = np.zeros(env.n_parallel)

            # Reset env and actor
            state = env.reset()
            actor.reset(state)
            actor.eval()

            # Update optimal delta energy DFT
            for i in range(env.n_parallel):
                optimized_delta_energy_dft[i] = (
                    env.unwrapped.energy[i] - env.unwrapped.optimal_energy[i]
                )

            # Update optimal delta energy Rdkit
            initial_energy_rdkit = env.rdkit_oracle.initial_energies

            # First save all the data
            molecules = [molecule.copy() for molecule in env.unwrapped.atoms]
            smiles = env.unwrapped.smiles.copy()
            dft_initial_energies = env.unwrapped.energy.copy()
            dft_forces = env.unwrapped.force.copy()

            # Get optimial rdkit energy and calculate delta
            _, optimal_energy_rdkit, _ = env.rdkit_oracle.calculate_energies_forces(
                max_its=5000
            )
            optimized_delta_energy_rdkit = initial_energy_rdkit - optimal_energy_rdkit

            # Reset the environment again
            env.set_initial_positions(
                molecules, smiles, dft_initial_energies, dft_forces
            )

    actor.train()
    result = {k: np.array(v).mean() for k, v in result.items()}
    print(
        "Full Evaluation time: {:.3f}, results: {}".format(
            time.perf_counter() - start, result
        )
    )
    return result


def eval_policy_rdkit(
    actor,
    env,
    eval_episodes=10,
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
            env, fixed_atoms, smiles, max_its=0
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

        # Switch actor back to training mode
        actor.train()

    result = {k: v / eval_episodes for k, v in result.items()}
    return result
