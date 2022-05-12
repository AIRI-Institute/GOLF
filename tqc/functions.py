import torch

from tqc import DEVICE


TIMELIMITS = [1, 5, 10, 50, 100]


def eval_policy(policy, eval_env, max_episode_steps, action_scale=1.0, eval_episodes=10):
    policy.eval()
    avg_reward = 0.
    avg_info_reward = 0.
    avg_final_energy = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max_episode_steps:
            with torch.no_grad():
                action = policy.select_action(state)
            state, reward, done, info = eval_env.step(action * action_scale)
            avg_reward += reward
            if 'rdkit_reward' in info:
                avg_info_reward += info['rdkit_reward']
            t += 1
        avg_final_energy += info['final_energy']
    avg_reward /= eval_episodes
    avg_info_reward /= eval_episodes
    avg_final_energy /= eval_episodes
    policy.train()
    return avg_reward, avg_info_reward, avg_final_energy


def eval_policy_multiple_timelimits(policy, eval_env, action_scale=1.0, eval_episodes=10):
    policy.eval()
    cur_avg_reward = 0.
    avg_reward_timelimits = {f'avg_reward_at_{timelimit}' : 0 for timelimit in TIMELIMITS}
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        t = 0
        while not done and t < max(TIMELIMITS):
            with torch.no_grad():
                action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action * action_scale)
            cur_avg_reward += reward
            if (t + 1 in TIMELIMITS):
                avg_reward_timelimits[f'avg_reward_at_{t + 1}'] += cur_avg_reward
            t += 1
        cur_avg_reward = 0.
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
