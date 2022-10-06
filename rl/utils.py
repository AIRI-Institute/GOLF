import torch
import numpy as np

from math import floor
from schnetpack.data.loader import _collate_aseatoms

from rl import DEVICE


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

def recollate_batch(state_batch, indices, new_states):
    # Unpads states in batch.
    # Replaces some states with new ones and collates them into batch.
    num_atoms = state_batch['_atom_mask'].sum(-1).long()
    states = [{k:v[i, :num_atoms[i]].cpu() for k, v in state_batch.items()  if k != "representation"}\
              for i in range(len(num_atoms))]
    for i, ind in enumerate(indices):
        states[ind] = new_states[i]
    return {k:v.to(DEVICE) for k, v in _collate_aseatoms(states).items()}

def calculate_gradient_norm(model):
    total_norm = 0.0
    params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    return total_norm

def calculate_action_norm(actions, atom_mask):
    num_atoms = atom_mask.sum(-1).long()
    actions_list = [action[:num_atoms[i]] for i, action in enumerate(actions)]
    mean_norm = np.array([np.linalg.norm(action, axis=1).mean() for action in actions_list]).mean()
    return mean_norm
