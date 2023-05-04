from typing import Any
from schnetpack.data.loader import _atoms_collate_fn

from AL.utils import unpad_state


class BaseSaver:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer

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
    def __init__(self, env, replay_buffer):
        super().__init__(env, replay_buffer)

    def __call__(self, states, _, dones):
        # Track last states of trajectories
        envs_to_store = [i for i, done in enumerate(dones) if done]
        super().save(states, envs_to_store)
