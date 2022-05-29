import torch

from rdkit.Chem import AllChem

from env.xyz2mol import set_coordinates, get_rdkit_energy
from tqc import DEVICE


TIMELIMITS = [1, 5, 10, 50, 100]


def eval_policy(policy, eval_env, max_episode_steps, action_scale=1.0, eval_episodes=10):
    policy.eval()
    avg_delta_energy = 0.
    avg_final_energy = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        initial_energy = eval_env.initial_energy
        while not done and t < max_episode_steps:
            with torch.no_grad():
                action = policy.select_action(state)
            state, _, done, info = eval_env.step(action * action_scale)
            t += 1
        avg_delta_energy += initial_energy - info['final_energy']
        avg_final_energy += info['final_energy']
    avg_delta_energy /= eval_episodes
    avg_final_energy /= eval_episodes
    policy.train()
    return avg_delta_energy, avg_final_energy


def eval_policy_multiple_timelimits(policy, eval_env, M, action_scale=1.0, eval_episodes=10):
    policy.eval()
    avg_reward_timelimits = {f'avg_reward_at_{timelimit}' : 0 for timelimit in TIMELIMITS}
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        initial_energy = eval_episodes.initial_energy
        t = 0
        while not done and t < max(TIMELIMITS):
            with torch.no_grad():
                action = policy.select_action(state)
            state, _, done, _ = eval_env.step(action * action_scale)
            if (t + 1 in TIMELIMITS):
                set_coordinates(eval_env.molecule, state['_positions'].double()[0].cpu().numpy())
                ff = AllChem.MMFFGetMoleculeForceField(eval_env.molecule,
                        AllChem.MMFFGetMoleculeProperties(eval_env.molecule), confId=0)
                ff.Initialize()
                ff.Minimize(maxIts=M)
                # Get energy after minimization
                final_energy = get_rdkit_energy(eval_env.molecule)
                avg_reward_timelimits[f'avg_reward_at_{t + 1}'] += initial_energy - final_energy
            t += 1
    avg_reward_timelimits = {k: v / eval_episodes for k, v in avg_reward_timelimits.items()}
    policy.train()
    return avg_reward_timelimits



def quantile_huber_loss_f(quantiles, samples):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
