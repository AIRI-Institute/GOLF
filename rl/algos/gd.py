import collections

import torch
from torch.nn.functional import mse_loss

from rl import DEVICE
from rl.utils import calculate_gradient_norm


class GD(object):
    def __init__(self, policy, actor_lr, batch_size=256, actor_clip_value=None, energy_loss_coef=0.01, force_loss_coef=1., group_by_n_atoms=False):
        self.actor = policy.actor
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.batch_size = batch_size
        self.actor_clip_value = actor_clip_value
        self.energy_loss_coef = energy_loss_coef
        self.force_loss_coef = force_loss_coef
        self.group_by_n_atoms = group_by_n_atoms
        self.total_it = 0

    def update(self, replay_buffer, *args):
        metrics = dict()
        state, force, energy = replay_buffer.sample(self.batch_size)

        if self.group_by_n_atoms:
            n_atoms_array = state['_atom_mask'].sum(-1).to(torch.int32).cpu().numpy()
            groups = collections.defaultdict(list)
            for idx, n_atoms in enumerate(n_atoms_array):
                groups[n_atoms].append(idx)

            energy_losses = []
            force_losses = []
            group_size = []
            for n_atoms, group in groups.items():
                group = torch.tensor(group, dtype=torch.int64, device=DEVICE)
                group_state = {}
                for key, value in state.items():
                    if key == 'cell':
                        group_state[key] = value[group].clone()
                    elif key in ('_atomic_numbers', '_positions', '_atom_mask'):
                        group_state[key] = value[group][:, :n_atoms].clone()
                    else:
                        group_state[key] = value[group][:, :n_atoms, :n_atoms - 1].clone()

                group_force = force[group][:, :n_atoms]
                group_energy = energy[group]

                output = self.actor(group_state, train=True)
                predicted_energy = output['energy']
                anti_gradient = output['anti_gradient']

                energy_loss = mse_loss(predicted_energy, group_energy)
                force_loss = mse_loss(anti_gradient, group_force)
                energy_losses.append(energy_loss)
                force_losses.append(force_loss)
                group_size.append(len(group))

            energy_losses = torch.stack(energy_losses)
            force_losses = torch.stack(force_losses)
            group_size = torch.tensor(group_size, device=DEVICE)

            energy_loss = torch.sum(energy_losses * group_size) / group_size.sum()
            force_loss = torch.sum(force_losses * group_size) / group_size.sum()
            metrics['n_batch_groups'] = len(groups)
        else:
            output = self.actor(state, train=True)
            predicted_energy = output['energy']
            anti_gradient = output['anti_gradient']

            energy_loss = mse_loss(predicted_energy, energy)
            force_loss = torch.mean(
                mse_loss(anti_gradient, force, reduction='none').sum(dim=(1, 2)) / state['_atom_mask'].sum(-1)
            )
            metrics['n_batch_groups'] = 1

        loss = self.force_loss_coef * force_loss + self.energy_loss_coef * energy_loss
        if not torch.all(torch.isfinite(loss)):
            print(f'Nan in GD loss')
            return metrics

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
