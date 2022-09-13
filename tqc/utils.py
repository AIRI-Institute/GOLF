import torch
import numpy as np

from collections import defaultdict
from math import floor

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


class TimelimitScheduler():
    def __init__(self,  timelimit_init=1, step=10, interval=100000, constant=True):
        self.init_tl = timelimit_init
        self.step = step
        self.interval = interval
        self.constant = constant

    def update(self, current_step):
        if not self.constant:
            self.tl = self.init_tl + self.step * (current_step // self.interval)
        else:
            self.tl = self.init_tl

    def get_timelimit(self):
        return self.tl


def run_policy(env, actor, fixed_atoms, max_timestamps):
    done = False
    delta_energy = 0
    t = 0
    state = env.set_initial_positions(fixed_atoms)
    while not done and t < max_timestamps:
        with torch.no_grad():
            action = actor.select_action(state)
        state, reward, done, info = env.step(action)
        delta_energy += reward
        t += 1
    
    return delta_energy, info['final_energy'], info['final_rl_energy']

def rdkit_minimize_until_convergence(env, fixed_atoms, M=None):
    M_init = 1000
    env.set_initial_positions(fixed_atoms, M=M)
    initial_energy = env.initial_energy
    not_converged, final_energy = env.minimize(M=M_init)
    while not_converged:
        M *= 2
        not_converged, final_energy = env.minimize(M=M_init)
        if M > 5000:
            print("Minimization did not converge!")
            return initial_energy, final_energy
    return initial_energy, final_energy

def eval_policy(actor, env, max_timestamps, eval_episodes=10, n_explore_runs=5):
    result = defaultdict(lambda: 0.0)
    for _ in range(eval_episodes):
        env.reset()
        fixed_atoms = env.atoms.copy()
        # Compute minimal energy of the molecule
        initial_energy, final_energy = rdkit_minimize_until_convergence(env, fixed_atoms)
        # Evaluate policy in eval mode
        actor.eval()
        eval_delta_energy, eval_final_energy, eval_final_rl_energy = run_policy(env, actor, fixed_atoms, max_timestamps=max_timestamps)
        result['eval/delta_energy'] += eval_delta_energy
        result['eval/final_energy'] += eval_final_energy
        result['eval/final_rl_energy'] += eval_final_rl_energy
        result['eval/pct_of_minimized_energy'] += (initial_energy - eval_final_energy) / (initial_energy - final_energy)
        # Evaluate policy at multiple timelimits
        for timelimit in TIMELIMITS:
            # Set env's TL to current timelimit
            env.env.TL = timelimit
            delta_energy_at, final_energy_at, _ = run_policy(env, actor, fixed_atoms, max_timestamps=timelimit)
            result[f'eval/delta_energy_at_{timelimit}'] += delta_energy_at
            result[f'eval/pct_of_minimized_energy_at_{timelimit}'] += (initial_energy - final_energy_at)  / (initial_energy - final_energy)
        # Set env's TL to original value
        env.env.TL = max_timestamps
        # Evaluate policy in explore mode
        actor.train()
        explore_results = np.array([run_policy(env, actor, fixed_atoms, max_timestamps=max_timestamps) for _ in range(n_explore_runs)])
        explore_delta_energy, explore_final_energy, explore_final_rl_energy = explore_results.mean(axis=0)
        result['explore/delta_energy'] += explore_delta_energy
        result['explore/final_energy'] += explore_final_energy
        result['explore/final_rl_energy'] += explore_final_rl_energy
        
    result = {k: v / eval_episodes for k, v in result.items()}
    return result

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

def calculate_gradient_norm(model):
    total_norm = 0.0
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    return total_norm
