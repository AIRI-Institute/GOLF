import torch
from torch.nn.functional import mse_loss

from rl.utils import calculate_gradient_norm


class GD(object):
    def __init__(self, policy, actor_lr, batch_size=256, actor_clip_value=None, energy_loss_coef=0.01, force_loss_coef=1.):
        self.actor = policy.actor
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.batch_size = batch_size
        self.actor_clip_value = actor_clip_value
        self.energy_loss_coef = energy_loss_coef
        self.force_loss_coef = force_loss_coef
        self.total_it = 0

    def update(self, replay_buffer, *args):
        metrics = dict()
        state, force, energy = replay_buffer.sample(self.batch_size)

        output = self.actor(state, train=True)
        predicted_energy = output['energy']
        anti_gradient = output['anti_gradient']

        energy_loss = mse_loss(predicted_energy, energy)
        force_loss = torch.mean(
            mse_loss(anti_gradient, force, reduction='none').sum(dim=(1, 2)) / state['_atom_mask'].sum(-1)
        )
        loss = self.force_loss_coef * force_loss + self.energy_loss_coef * energy_loss
        metrics['loss'] = loss.item()
        metrics['energy_loss'] = energy_loss.item()
        metrics['force_loss'] = force_loss.item()
        metrics['energy_loss_contrib'] = energy_loss.item() * self.energy_loss_coef
        metrics['force_loss_contrib'] = force_loss.item() * self.force_loss_coef

        self.optimizer.zero_grad()
        loss.backward()
        if self.actor_clip_value is not None:
            metrics['grad_norm'] = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip_value).item()
        else:
            metrics['grad_norm'] = calculate_gradient_norm(self.actor).item()

        self.optimizer.step()

        self.total_it += 1
        return metrics

    def save(self, filename):
        self.light_save(filename)
        torch.save(self.optimizer.state_dict(), f'{filename}_optimizer')

    def light_save(self, filename):
        torch.save(self.actor.state_dict(), f'{filename}_actor')

    def load(self, filename):
        self.light_load(filename)
        self.optimizer.load_state_dict(torch.load(f'{filename}_optimizer'))

    def light_load(self, filename):
        self.actor.load_state_dict(torch.load(f'{filename}_actor'))
