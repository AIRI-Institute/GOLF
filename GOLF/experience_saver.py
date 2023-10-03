import torch
import numpy as np

from torch.nn.functional import mse_loss
from schnetpack import properties
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.nn import scatter_add

from GOLF.utils import unpad_state


class BaseSaver:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer

    def get_forces(self, indices=None):
        return self.env.get_forces(indices=indices)

    def save(self, states, envs_to_store):
        if len(envs_to_store) > 0:
            energies = self.env.get_energies(indices=envs_to_store)
            forces = self.env.get_forces(indices=envs_to_store)
            state_list = unpad_state(states)
            state_list = [state_list[i] for i in envs_to_store]
            self.replay_buffer.add(_atoms_collate_fn(state_list), forces, energies)


class RewardThresholdSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, states, rewards, _):
        # Only store states with reward > REWARD_THRESHOLD
        envs_to_store = [
            i for i, reward in enumerate(rewards) if reward > self.reward_threshold
        ]
        super().save(states, envs_to_store)


class InitialAndLastSaver(BaseSaver):
    def __init__(self, env, replay_buffer, reward_threshold):
        super().__init__(env, replay_buffer)
        self.reward_threshold = reward_threshold

    def __call__(self, states, rewards, dones):
        # Track last states of trajectories
        envs_to_store = [
            i
            for i, (done, reward) in enumerate(zip(dones, rewards))
            if done and reward > self.reward_threshold
        ]
        super().save(states, envs_to_store)


class GradientMissmatchSaver(BaseSaver):
    def __init__(
        self, env, replay_buffer, actor, reward_threshold, gradient_dif_threshold
    ):
        super().__init__(env, replay_buffer)
        self.actor = actor
        self.gradient_dif_threshold = gradient_dif_threshold
        self.reward_threshold = reward_threshold

    def __call__(self, states, rewards, dones):
        _, predicted_F = self.actor._get_last_output()
        gt_F = torch.tensor(
            np.concatenate(super().get_forces()), device=predicted_F.device
        )

        # Check if there is a shape missmatch (due to LBFGS optimization)
        if predicted_F.size(0) != gt_F.size(0):
            envs_to_store = list(range(len(dones)))
        else:
            with torch.no_grad():
                per_molecule_force_mse = (
                    scatter_add(
                        mse_loss(predicted_F, gt_F, reduction="none").mean(-1),
                        idx_i=states[properties.idx_m],
                        dim_size=len(dones),
                    )
                    / states[properties.n_atoms]
                )
            envs_to_store = [
                i
                for i, (done, reward, mse) in enumerate(
                    zip(dones, rewards, per_molecule_force_mse)
                )
                if (done or mse > self.gradient_dif_threshold)
                and reward > self.reward_threshold
            ]
        super().save(states, envs_to_store)
