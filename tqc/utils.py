import torch

import numpy as np

from math import floor
from rdkit.Chem import rdmolops

from env.xyz2mol import set_coordinates, get_rdkit_energy
from tqc import DEVICE


TIMELIMITS = [1, 5, 10, 50, 100]


class ActionScaleScheduler():
    def __init__(self,  action_scale_init, action_scale_end, n_step_end, mode="discrete"):
        self.as_init = action_scale_init
        self.as_end = action_scale_end
        self.n_step_end = n_step_end

        assert mode in ["constant", "discrete", "continuous"], "Unknown ActionScaleSheduler mode!"
        self.mode  = mode
        # For discrete mode
        if mode == "discrete":
            n_updates = (self.as_end - self.as_init) / 0.01
            self.update_interval = self.n_step_end / n_updates

    def update(self, n_step):
        if self.mode == "constant":
            current_action_scale = self.as_init
        elif self.mode == "discrete":
            current_action_scale = self.as_init + floor(n_step / self.update_interval) * 0.01
        else:
            p = max((self.n_step_end - n_step) / self.n_step_end, 0)
            current_action_scale = p * (self.as_init - self.as_end) + self.as_end
        self.current_action_scale = current_action_scale

    def get_action_scale(self):
        return self.current_action_scale

def run_policy(env, actor, state, fixed_positions, max_timestamps):
    done = False
    delta_energy = 0
    t = 0
    env.set_initial_positions(fixed_positions)
    state['_positions'] = torch.FloatTensor(fixed_positions).unsqueeze(0)
    while not done and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        state, reward, done, info = env.step(action)
        delta_energy += reward
        t += 1
    return delta_energy, info['final_energy'], info['final_rl_energy']

def run_policy_eval_and_explore(actor, env, max_timestamps, eval_episodes=10, n_explore_runs=10):
    result = {
        'avg_eval_delta_energy': 0.,
        'avg_eval_final_energy': 0.,
        'avg_eval_final_rl_energy': 0.,
        'avg_explore_delta_energy': 0.,
        'avg_explore_final_energy': 0.,
        'avg_explore_final_rl_energy': 0.
    }
    for _ in range(eval_episodes):
        state = env.reset()
        fixed_positions = state['_positions'][0].double().cpu().detach().numpy()
        actor.eval()
        eval_delta_energy, eval_final_energy, eval_final_rl_energy = run_policy(env, actor, state, fixed_positions, max_timestamps=max_timestamps)
        actor.train()
        explore_results = np.array([run_policy(env, actor, state, fixed_positions, max_timestamps=max_timestamps) for _ in range(n_explore_runs)])
        explore_delta_energy, explore_final_energy, explore_final_rl_energy = explore_results.mean(axis=0)
        
        result['avg_eval_delta_energy'] += eval_delta_energy
        result['avg_eval_final_energy'] += eval_final_energy
        result['avg_eval_final_rl_energy'] += eval_final_rl_energy
        result['avg_explore_delta_energy'] += explore_delta_energy
        result['avg_explore_final_energy'] += explore_final_energy
        result['avg_explore_final_rl_energy'] += explore_final_rl_energy
    
    result = {k: v / eval_episodes for k, v in result.items()}
    return result

def eval_policy_multiple_timelimits(actor, eval_env, M, eval_episodes=10):
    actor.eval()
    avg_delta_energy_timelimits = {f'avg_reward_at_{timelimit}' : 0 for timelimit in TIMELIMITS}
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        delta_energy = 0
        t = 0
        while not done and t < max(TIMELIMITS):
            with torch.no_grad():
                action = actor.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            delta_energy += reward
            if (t + 1 in TIMELIMITS):
                avg_delta_energy_timelimits[f'avg_reward_at_{t + 1}'] += delta_energy
            t += 1
    avg_delta_energy_timelimits = {k: v / eval_episodes for k, v in avg_delta_energy_timelimits.items()}
    actor.train()
    return avg_delta_energy_timelimits

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
